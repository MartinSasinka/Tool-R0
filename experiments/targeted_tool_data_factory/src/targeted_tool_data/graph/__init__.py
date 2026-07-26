"""Program-first task graph synthesis.

Builds a typed semantic DAG per generation cell BEFORE any surface exists
(DESIGN.md §1). Nodes reference primitives from the registry; edges are
typed references.

Engine v1 (pilot1): motif-driven numeric chains.
Engine v2 (pilot2): additionally unit-aware (semantic plausibility),
answer-kind targeted (int/bool/string/list/numeric-string sinks calibrated
against the executed body) and builds genuine independent branches.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .. import registry as reg
from ..schemas import GenerationCell, GraphNode, SemanticProgram
from ..util import short_hash

REF = "__ref__"

# chainable numeric primitives by role
# NOTE: max_two/min_two/clamp are excluded from chains: their output always
# equals one of their inputs, which breaks path-invariant value-based
# correctness (duplicate observations / shorter value-reachable path). They
# remain in the registry as high-quality hard distractors.
_CHAIN_BIN = ["add", "subtract", "multiply", "divide", "floor_divide", "modulo",
              "percent_of", "ratio_of", "abs_difference", "average_two",
              "increase_by_percent", "decrease_by_percent"]
_CHAIN_UN = ["sqrt", "negate", "inverse", "floor_value", "ceil_value", "square",
             "seconds_to_minutes", "hours_to_minutes", "km_to_meters",
             "celsius_to_fahrenheit", "round_places"]
_START = _CHAIN_BIN + ["sum_values", "mean_values", "max_values", "min_values",
                       "range_spread", "count_values"] + _CHAIN_UN
_MERGE_BIN = ["add", "subtract", "multiply", "divide", "ratio_of", "abs_difference",
              "average_two"]
_MERGE_SEL = ["max_two", "min_two"]
_AGG = ["sum_values", "mean_values", "max_values", "min_values", "range_spread"]
_NS_OUT = ["format_fixed", "number_to_string"]


def _num_param_slots(p: reg.Primitive) -> List[int]:
    return [i for i, (_, t, _s) in enumerate(p.params) if t in (reg.NUM,)]


def _sample_consts(p: reg.Primitive, rng: random.Random) -> List[Any]:
    return p.sampler(rng)


class GraphBuildError(Exception):
    pass


def _mk_node(nid: str, sid: str, inputs: Dict[str, Any]) -> GraphNode:
    p = reg.get(sid)
    return GraphNode(node_id=nid, semantic_id=sid, inputs=inputs, output_type=p.out_type)


def _chain_step(prev_id: str, nid: str, rng: random.Random,
                unary_bias: float = 0.35) -> GraphNode:
    """One step consuming the previous node's numeric output."""
    if rng.random() < unary_bias:
        sid = rng.choice(_CHAIN_UN)
        p = reg.get(sid)
        consts = _sample_consts(p, rng)
        inputs = {p.params[0][0]: {REF: prev_id}}
        for (name, _t, _s), v in list(zip(p.params, consts))[1:]:
            inputs[name] = v
        return _mk_node(nid, sid, inputs)
    sid = rng.choice(_CHAIN_BIN)
    p = reg.get(sid)
    consts = _sample_consts(p, rng)
    slot = rng.choice(_num_param_slots(p))
    inputs = {}
    for i, ((name, _t, _s), v) in enumerate(zip(p.params, consts)):
        inputs[name] = {REF: prev_id} if i == slot else v
    return _mk_node(nid, sid, inputs)


def _start_node(nid: str, rng: random.Random, cell: GenerationCell,
                allow_list: bool = True, ns_input: bool = False) -> GraphNode:
    if ns_input:
        return _mk_node(nid, "parse_number", {"text": str(rng.randint(137, 8971))})
    pool = list(_START) if allow_list else list(_CHAIN_BIN + _CHAIN_UN)
    sid = rng.choice(pool)
    p = reg.get(sid)
    consts = _sample_consts(p, rng)
    inputs = {name: v for (name, _t, _s), v in zip(p.params, consts)}
    return _mk_node(nid, sid, inputs)


def build_program(cell: GenerationCell, rng: random.Random) -> SemanticProgram:
    """Engine v1 (pilot1). Exactly cell.call_count nodes, every node on a path
    to the sink."""
    n = cell.call_count
    motif = cell.motif
    ns_in = cell.numeric_string and rng.random() < 0.5
    ns_out = cell.numeric_string and not ns_in
    nodes: List[GraphNode] = []

    if motif == "linear" or n == 2:
        body = n - 1 if ns_out else n
        nodes.append(_start_node("n1", rng, cell, ns_input=ns_in))
        for i in range(2, body + 1):
            nodes.append(_chain_step(f"n{i - 1}", f"n{i}", rng))
        depth = body
    elif motif == "fan_in" or motif == "selection":
        merge_pool = _MERGE_SEL if motif == "selection" else _MERGE_BIN
        body = (n - 1) if ns_out else n
        if body < 3:
            raise GraphBuildError("fan_in needs >=3 calls")
        left_len = rng.randint(1, body - 2)
        right_len = body - 1 - left_len
        idx = 1
        for i in range(left_len):
            nid = f"n{idx}"
            nodes.append(_start_node(nid, rng, cell, allow_list=False, ns_input=(ns_in and i == 0))
                         if i == 0 else _chain_step(f"n{idx - 1}", nid, rng))
            idx += 1
        left_sink = f"n{idx - 1}"
        for i in range(right_len):
            nid = f"n{idx}"
            nodes.append(_start_node(nid, rng, cell, allow_list=False)
                         if i == 0 else _chain_step(f"n{idx - 1}", nid, rng))
            idx += 1
        right_sink = f"n{idx - 1}"
        sid = rng.choice(merge_pool)
        p = reg.get(sid)
        inputs = {p.params[0][0]: {REF: left_sink}, p.params[1][0]: {REF: right_sink}}
        nodes.append(_mk_node(f"n{idx}", sid, inputs))
        depth = max(left_len, right_len) + 1
    elif motif == "branch_aggregate":
        body = (n - 1) if ns_out else n
        if body < 3:
            raise GraphBuildError("branch_aggregate needs >=3 calls")
        k = body - 1
        for i in range(1, k + 1):
            nodes.append(_start_node(f"n{i}", rng, cell, allow_list=False,
                                     ns_input=(ns_in and i == 1)))
        sid = rng.choice(_AGG)
        inputs = {"values": [{REF: f"n{i}"} for i in range(1, k + 1)]}
        nodes.append(_mk_node(f"n{k + 1}", sid, inputs))
        depth = 2
    else:
        raise GraphBuildError(f"unknown motif {motif}")

    if ns_out:
        sink_prev = nodes[-1].node_id
        sid = rng.choice(_NS_OUT)
        p = reg.get(sid)
        inputs: Dict[str, Any] = {p.params[0][0]: {REF: sink_prev}}
        if sid == "format_fixed":
            inputs["places"] = rng.randint(1, 2)
        nodes.append(_mk_node(f"n{len(nodes) + 1}", sid, inputs))
        depth += 1

    prog = SemanticProgram(nodes=nodes, sink=nodes[-1].node_id, motif=motif, depth=depth)
    _assert_all_reach_sink(prog)
    return prog


# ══════════════════════════════════════════════════════════════════════════
#  Engine v2 (pilot2): unit-aware, answer-kind targeted, real branches
# ══════════════════════════════════════════════════════════════════════════

_V2_CHAIN = _CHAIN_BIN + _CHAIN_UN + ["meters_to_km", "minutes_to_seconds",
                                      "fahrenheit_to_celsius", "round_to_int"]
_V2_LIST_START = ["sum_values", "mean_values", "max_values", "min_values",
                  "range_spread", "count_values", "index_of_max"]
_V2_MERGE = ["add", "subtract", "multiply", "divide", "ratio_of",
             "abs_difference", "average_two", "percent_of"]
_V2_AGG3 = ["sum_three", "mean_three", "range_three"]

# answer-kind sinks: (sid, how many extra constants the sink needs)
_SINK_INT = ["round_to_int", "digit_sum"]
_SINK_BOOL = ["is_greater", "is_within_range", "is_divisible_by"]
_SINK_NUMSTR = ["format_fixed", "number_to_string"]
_SINK_STRING = ["format_with_unit", "tag_value"]
_SINK_LIST = ["scale_list", "filter_above", "append_value"]


def _v2_start(nid: str, rng: random.Random, *, ns_input: bool,
              allow_list: bool, neutral_only: bool = False
              ) -> Tuple[GraphNode, str, str]:
    """Returns (node, unit, out_type). `neutral_only` keeps the branch
    dimensionless, so branches aggregated later are actually comparable."""
    if ns_input:
        return (_mk_node(nid, "parse_number", {"text": str(rng.randint(137, 8971))}),
                reg.U_ABSTRACT, reg.NUM)
    pool = list(_V2_CHAIN)
    if allow_list:
        pool += _V2_LIST_START
    if neutral_only:
        pool = [s for s in pool
                if reg.get(s).unit_of_output(
                    [reg.unit_of_constant_expect(e)
                     for e in reg.get(s).param_units]) in reg.NEUTRAL_UNITS]
        if not pool:
            raise GraphBuildError("no unit-neutral start primitive")
    sid = rng.choice(pool)
    p = reg.get(sid)
    consts = _sample_consts(p, rng)
    inputs = {name: v for (name, _t, _s), v in zip(p.params, consts)}
    in_units = [reg.unit_of_constant_expect(e) for e in p.param_units]
    return _mk_node(nid, sid, inputs), p.unit_of_output(in_units), p.out_type


def _v2_step(prev_id: str, prev_unit: str, prev_type: str, nid: str,
             rng: random.Random) -> Tuple[GraphNode, str, str]:
    """One unit-compatible step consuming the previous node's output."""
    from ..plausibility import transition_class, ARTIFICIAL

    options: List[Tuple[str, int]] = []
    for sid in _V2_CHAIN:
        p = reg.get(sid)
        for i, ((_pn, ptype, semantic), expect) in enumerate(
                zip(p.params, p.param_units)):
            if not reg.type_accepts(ptype, prev_type):
                continue
            if transition_class(expect, prev_unit, semantic) == ARTIFICIAL:
                continue
            options.append((sid, i))
    if not options:
        raise GraphBuildError(f"no unit-compatible consumer for {prev_unit}")
    sid, slot = options[rng.randrange(len(options))]
    p = reg.get(sid)
    consts = _sample_consts(p, rng)
    inputs: Dict[str, Any] = {}
    in_units: List[str] = []
    for i, ((name, _t, _s), v) in enumerate(zip(p.params, consts)):
        if i == slot:
            inputs[name] = {REF: prev_id}
            in_units.append(prev_unit)
        else:
            inputs[name] = v
            in_units.append(reg.unit_of_constant_expect(p.param_units[i]))
    return _mk_node(nid, sid, inputs), p.unit_of_output(in_units), p.out_type


def _v2_merge(left: Tuple[str, str, str], right: Tuple[str, str, str],
              nid: str, rng: random.Random) -> Tuple[GraphNode, str, str]:
    """Combine two independent branches; both are required for the result."""
    from ..plausibility import transition_class, ARTIFICIAL

    l_id, l_unit, l_type = left
    r_id, r_unit, r_type = right
    viable = []
    for sid in _V2_MERGE:
        p = reg.get(sid)
        if len(p.params) != 2:
            continue
        (n0, t0, s0), (n1, t1, s1) = p.params
        e0, e1 = p.param_units
        if not (reg.type_accepts(t0, l_type) and reg.type_accepts(t1, r_type)):
            continue
        if transition_class(e0, l_unit, s0) == ARTIFICIAL:
            continue
        if transition_class(e1, r_unit, s1) == ARTIFICIAL:
            continue
        viable.append(sid)
    if not viable:
        raise GraphBuildError(f"no unit-compatible merge for {l_unit}+{r_unit}")
    sid = viable[rng.randrange(len(viable))]
    p = reg.get(sid)
    inputs = {p.params[0][0]: {REF: l_id}, p.params[1][0]: {REF: r_id}}
    return (_mk_node(nid, sid, inputs),
            p.unit_of_output([l_unit, r_unit]), p.out_type)


def _sink_node(kind: str, prev_id: str, prev_value: Any, prev_type: str,
               nid: str, rng: random.Random) -> GraphNode:
    """Answer-kind sink, calibrated against the executed body value so the
    result is genuinely typed and never degenerate."""
    if kind == "int":
        sid = "round_to_int" if prev_type == reg.NUM else "digit_sum"
        if prev_type == reg.INT:
            sid = "digit_sum"
        return _mk_node(nid, sid, {"a": {REF: prev_id}})
    if kind == "bool":
        v = float(prev_value)
        integral = abs(v - round(v)) < 1e-9 and abs(v) < 1e7
        # a divisibility check on a non-integral value is always False, which
        # would make the label predictable without solving the task
        sid = rng.choice(_SINK_BOOL if integral else _SINK_BOOL[:2])
        if sid == "is_divisible_by":
            iv = int(round(v))
            divisors = [k for k in (3, 4, 5, 6, 7, 8, 9, 11) if iv != 0 and iv % k == 0]
            nondiv = [k for k in (3, 4, 5, 6, 7, 8, 9, 11) if iv == 0 or iv % k != 0]
            pool = (divisors or nondiv) if rng.random() < 0.5 else (nondiv or divisors)
            return _mk_node(nid, "is_divisible_by",
                            {"a": {REF: prev_id}, "k": rng.choice(pool)})
        if sid == "is_greater":
            delta = abs(v) * rng.choice([0.2, 0.35, 0.5]) + 3
            b = round(v - delta) if rng.random() < 0.5 else round(v + delta)
            return _mk_node(nid, "is_greater", {"a": {REF: prev_id}, "b": int(b)})
        if sid == "is_within_range":
            if rng.random() < 0.5:                      # true case
                lo, hi = round(v - abs(v) * 0.4 - 5), round(v + abs(v) * 0.4 + 5)
            else:                                        # false case
                lo, hi = round(v + abs(v) * 0.2 + 7), round(v + abs(v) * 0.9 + 40)
            return _mk_node(nid, "is_within_range",
                            {"a": {REF: prev_id}, "lo": int(lo), "hi": int(hi)})
        raise GraphBuildError(f"unreachable bool sink {sid}")
    if kind == "numeric_string":
        sid = rng.choice(_SINK_NUMSTR)
        if sid == "format_fixed":
            return _mk_node(nid, "format_fixed",
                            {"a": {REF: prev_id}, "places": rng.randint(1, 2)})
        return _mk_node(nid, "number_to_string", {"a": {REF: prev_id}})
    if kind == "string":
        if rng.random() < 0.5:
            unit = rng.choice(["kg", "units", "items", "points", "boxes", "litres"])
            return _mk_node(nid, "format_with_unit",
                            {"a": {REF: prev_id}, "unit": unit})
        prefix = rng.choice(["batch", "order", "run", "lot", "ticket"])
        return _mk_node(nid, "tag_value", {"prefix": prefix, "a": {REF: prev_id}})
    if kind == "list":
        v = float(prev_value)
        sid = rng.choice(_SINK_LIST)
        if sid == "scale_list":
            values = sorted({rng.randint(3, 89) for _ in range(rng.randint(3, 5))})
            return _mk_node(nid, "scale_list",
                            {"values": values, "factor": {REF: prev_id}})
        if sid == "append_value":
            values = sorted({rng.randint(3, 89) for _ in range(rng.randint(3, 5))})
            return _mk_node(nid, "append_value",
                            {"values": values, "value": {REF: prev_id}})
        # filter list is calibrated around the body value so exactly the
        # entries above it survive — never empty, never the whole list
        base = max(abs(v), 6.0)
        values = sorted({int(round(base * f)) for f in
                         (0.35, 0.62, 0.88, 1.15, 1.4, 1.75)})
        if len(values) < 5:
            raise GraphBuildError("degenerate filter list")
        return _mk_node(nid, "filter_above",
                        {"values": values, "threshold": {REF: prev_id}})
    raise GraphBuildError(f"unknown answer kind {kind}")


def build_program_v2(cell: GenerationCell, rng: random.Random) -> SemanticProgram:
    """Engine v2. Unit-aware body + calibrated answer-kind sink.

    The sink (when the answer is not a plain float) consumes one call, so the
    body has call_count-1 nodes. The motif is classified from the built graph
    afterwards, so reported motifs are measured, not declared.
    """
    from ..executor import ExecutionError, execute

    n = cell.call_count
    kind = cell.answer_kind or "float"
    ns_in = kind == "numeric_string" and rng.random() < 0.4
    needs_sink = kind != "float"
    # a k-sink consuming an int (top_k) needs an integer-typed body result
    body_n = n - 1 if needs_sink else n
    if body_n < 1:
        raise GraphBuildError("call_count too small for typed sink")

    motif = cell.motif
    nodes: List[GraphNode] = []
    if motif == "linear" or body_n < 3:
        node, unit, otype = _v2_start("n1", rng, ns_input=ns_in, allow_list=True)
        nodes.append(node)
        for i in range(2, body_n + 1):
            node, unit, otype = _v2_step(f"n{i - 1}", unit, otype, f"n{i}", rng)
            nodes.append(node)
        depth = body_n
        tail = (nodes[-1].node_id, unit, otype)
    elif motif == "fan_in":
        left_len = rng.randint(1, body_n - 2)
        right_len = body_n - 1 - left_len
        idx = 1
        node, unit, otype = _v2_start(f"n{idx}", rng, ns_input=ns_in, allow_list=False)
        nodes.append(node)
        idx += 1
        for _ in range(left_len - 1):
            node, unit, otype = _v2_step(f"n{idx - 1}", unit, otype, f"n{idx}", rng)
            nodes.append(node)
            idx += 1
        left = (f"n{idx - 1}", unit, otype)
        node, r_unit, r_type = _v2_start(f"n{idx}", rng, ns_input=False,
                                         allow_list=False)
        nodes.append(node)
        idx += 1
        for _ in range(right_len - 1):
            node, r_unit, r_type = _v2_step(f"n{idx - 1}", r_unit, r_type,
                                            f"n{idx}", rng)
            nodes.append(node)
            idx += 1
        right = (f"n{idx - 1}", r_unit, r_type)
        node, unit, otype = _v2_merge(left, right, f"n{idx}", rng)
        nodes.append(node)
        depth = max(left_len, right_len) + 1
        tail = (f"n{idx}", unit, otype)
    elif motif == "branch_aggregate":
        # exactly three independent branches merged by a three-argument
        # scalar aggregator (indegree 3). No reference ever lands inside an
        # array argument: NESTFUL has none and the trainer cannot resolve one.
        if body_n < 4:
            raise GraphBuildError("branch_aggregate needs >=4 body calls")
        lens = [1, 1, 1]
        for j in range(body_n - 4):
            lens[j % 3] += 1
        idx = 1
        branch_tails = []
        units = []
        for bi, blen in enumerate(lens):
            node, u, t = _v2_start(f"n{idx}", rng, ns_input=(ns_in and bi == 0),
                                   allow_list=False, neutral_only=True)
            nodes.append(node)
            idx += 1
            for _ in range(blen - 1):
                node, u, t = _v2_step(f"n{idx - 1}", u, t, f"n{idx}", rng)
                nodes.append(node)
                idx += 1
            branch_tails.append(f"n{idx - 1}")
            units.append(u)
        if any(u not in reg.NEUTRAL_UNITS for u in units):
            raise GraphBuildError("branches carry incompatible units")
        sid = rng.choice(_V2_AGG3)
        p = reg.get(sid)
        inputs = {pn: {REF: bt} for (pn, _t, _s), bt in zip(p.params, branch_tails)}
        nodes.append(_mk_node(f"n{idx}", sid, inputs))
        depth = max(lens) + 1
        tail = (f"n{idx}", p.unit_of_output(units), p.out_type)
    else:
        raise GraphBuildError(f"unknown motif {motif}")

    if needs_sink:
        body_prog = SemanticProgram(nodes=list(nodes), sink=tail[0],
                                    motif=motif, depth=depth)
        try:
            _obs, body_value = execute(body_prog)
        except ExecutionError as exc:
            raise GraphBuildError(f"body execution failed: {exc}") from exc
        if kind in ("list", "bool", "int", "string", "numeric_string") and \
                not isinstance(body_value, (int, float)):
            raise GraphBuildError(f"{kind} sink needs a numeric body value")
        sink = _sink_node(kind, tail[0], body_value, tail[2],
                          f"n{len(nodes) + 1}", rng)
        nodes.append(sink)
        depth += 1

    prog = SemanticProgram(nodes=nodes, sink=nodes[-1].node_id,
                           motif=motif, depth=depth)
    _assert_all_reach_sink(prog)
    prog.motif = classify_program_motif(prog)
    return prog


def classify_program_motif(prog: SemanticProgram) -> str:
    """Motif measured from the built graph, using the same rule as the target
    profiler (profile.classify_motif)."""
    idx = {nd.node_id: i for i, nd in enumerate(prog.nodes)}
    edges: Dict[int, List[int]] = {}
    for nd in prog.nodes:
        srcs = []
        for v in nd.inputs.values():
            srcs.extend(_refs_in(v))
        if srcs:
            edges[idx[nd.node_id]] = [idx[s] for s in srcs]
    n = len(prog.nodes)
    if not edges:
        return "independent"
    indeg = {i: len(set(edges.get(i, []))) for i in range(n)}
    consumed: Dict[int, int] = {}
    for srcs in edges.values():
        for s in set(srcs):
            consumed[s] = consumed.get(s, 0) + 1
    if any(v > 2 for v in indeg.values()):
        return "branch_aggregate"
    if all(v <= 1 for v in indeg.values()) and all(
            consumed.get(i, 0) <= 1 for i in range(n)):
        return "linear" if len(edges) == n - 1 else "mixed"
    if any(v == 2 for v in indeg.values()):
        return "fan_in"
    return "mixed"


# ── shared helpers ────────────────────────────────────────────────────────
def _consumers(prog: SemanticProgram) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {nd.node_id: [] for nd in prog.nodes}
    for nd in prog.nodes:
        for v in nd.inputs.values():
            for ref in _refs_in(v):
                out[ref].append(nd.node_id)
    return out


def _refs_in(v: Any) -> List[str]:
    if isinstance(v, dict) and REF in v:
        return [v[REF]]
    if isinstance(v, list):
        return [r for item in v for r in _refs_in(item)]
    return []


def _assert_all_reach_sink(prog: SemanticProgram) -> None:
    cons = _consumers(prog)
    reach = {prog.sink}
    changed = True
    while changed:
        changed = False
        for nid, cs in cons.items():
            if nid not in reach and any(c in reach for c in cs):
                reach.add(nid)
                changed = True
    dangling = [nd.node_id for nd in prog.nodes if nd.node_id not in reach]
    if dangling:
        raise GraphBuildError(f"decorative nodes (no path to sink): {dangling}")


def is_acyclic(prog: SemanticProgram) -> bool:
    order = {nd.node_id: i for i, nd in enumerate(prog.nodes)}
    for nd in prog.nodes:
        for v in nd.inputs.values():
            for ref in _refs_in(v):
                if ref not in order or order[ref] >= order[nd.node_id]:
                    return False
    return True


def program_family(prog: SemanticProgram) -> str:
    sids = [nd.semantic_id for nd in prog.nodes]
    return "pf_" + short_hash({"motif": prog.motif, "sids": sids})


def graph_template_id(prog: SemanticProgram) -> str:
    shape = []
    for nd in prog.nodes:
        refs = sorted(r for v in nd.inputs.values() for r in _refs_in(v))
        shape.append({"cat": reg.get(nd.semantic_id).category, "refs": len(refs)})
    return "gt_" + short_hash({"motif": prog.motif, "shape": shape})


def argument_skeleton(prog: SemanticProgram) -> str:
    skel = []
    for nd in prog.nodes:
        row = []
        for name, v in sorted(nd.inputs.items()):
            row.append((name, "ref" if _refs_in(v) else type(v).__name__))
        skel.append((nd.semantic_id, tuple(row)))
    return short_hash(skel)
