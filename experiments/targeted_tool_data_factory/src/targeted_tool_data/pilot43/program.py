"""Program representation and deterministic executor for Pilot4.3.

The program is the *only* place a dependency edge can be expressed, and it is
built by the workflow plan (never annotated afterwards). Nodes are stored in
topological order; a reference to a later node is a hard error rather than a
warning, because the exported ``$var_k$`` labels must be resolvable by a
left-to-right tool executor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..repro import sha256_obj
from . import semtypes as st
from .ops import Op, build_ops

MAX_ABS = 1e12
MAX_LIST = 24
MAX_TEXT = 240


class ProgramError(Exception):
    """Structurally invalid program (cycles, unknown ops, missing arguments)."""


class ExecError(Exception):
    """The program is structurally fine but does not execute on these values."""


@dataclass(frozen=True)
class Ref:
    node_id: str

    def __repr__(self) -> str:            # pragma: no cover - debug aid
        return f"Ref({self.node_id})"


@dataclass
class Node:
    node_id: str
    op: str
    args: Dict[str, Any] = field(default_factory=dict)
    #: semantic type of each *literal* argument, taken from the role that
    #: supplied it. Without this a literal price would degrade to GenericScalar
    #: and the unit of the whole downstream chain would be lost.
    arg_sems: Dict[str, str] = field(default_factory=dict)

    def refs(self) -> List[str]:
        return _refs_in(self.args)

    def literals(self) -> Dict[str, Any]:
        return {k: v for k, v in self.args.items() if not _refs_in(v)}


@dataclass
class Program:
    nodes: List[Node]
    sink: str

    def index(self) -> Dict[str, int]:
        return {nd.node_id: i for i, nd in enumerate(self.nodes)}

    def node(self, node_id: str) -> Node:
        for nd in self.nodes:
            if nd.node_id == node_id:
                return nd
        raise ProgramError(f"unknown node {node_id}")

    def edges(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for nd in self.nodes:
            for src in dict.fromkeys(nd.refs()):
                out.append((src, nd.node_id))
        return out

    def op_sequence(self) -> Tuple[str, ...]:
        return tuple(nd.op for nd in self.nodes)

    def capability_sequence(self) -> Tuple[str, ...]:
        ops = build_ops()
        return tuple(ops[nd.op].capability for nd in self.nodes)

    def capability_family_sequence(self) -> Tuple[str, ...]:
        ops = build_ops()
        return tuple(ops[nd.op].family for nd in self.nodes)

    def program_id(self) -> str:
        payload = [(nd.node_id, nd.op,
                    {k: (f"@{v.node_id}" if isinstance(v, Ref) else v)
                     for k, v in sorted(nd.args.items())})
                   for nd in self.nodes]
        return "sp_" + sha256_obj({"nodes": payload, "sink": self.sink})[:20]

    def plan_id(self) -> str:
        """Identity of the *shape* (capabilities + wiring), values excluded."""
        idx = self.index()
        payload = [(build_ops()[nd.op].capability,
                    sorted(idx[r] for r in dict.fromkeys(nd.refs())))
                   for nd in self.nodes]
        return "pl_" + sha256_obj({"nodes": payload, "sink": idx[self.sink]})[:20]


def _refs_in(value: Any) -> List[str]:
    if isinstance(value, Ref):
        return [value.node_id]
    if isinstance(value, (list, tuple)):
        return [r for item in value for r in _refs_in(item)]
    if isinstance(value, dict):
        return [r for item in value.values() for r in _refs_in(item)]
    return []


def validate_structure(prog: Program) -> None:
    """Hard structural gate: ordering, arity, argument coverage, reachability."""
    ops = build_ops()
    seen: Dict[str, int] = {}
    for i, nd in enumerate(prog.nodes):
        if nd.node_id in seen:
            raise ProgramError(f"duplicate node id {nd.node_id}")
        seen[nd.node_id] = i
        if nd.op not in ops:
            raise ProgramError(f"unknown op {nd.op}")
        op = ops[nd.op]
        expected = {p.name for p in op.params}
        if set(nd.args) != expected:
            raise ProgramError(
                f"{nd.node_id}: args {sorted(nd.args)} != params {sorted(expected)}")
        for ref in nd.refs():
            if ref not in seen or seen[ref] >= i:
                raise ProgramError(f"{nd.node_id}: non-topological ref {ref}")
    if prog.sink not in seen:
        raise ProgramError("sink is not a node")
    # every node must be on a path to the sink; decorative nodes are rejected
    reach = {prog.sink}
    for nd in reversed(prog.nodes):
        if nd.node_id in reach:
            reach.update(nd.refs())
    dangling = [nd.node_id for nd in prog.nodes if nd.node_id not in reach]
    if dangling:
        raise ProgramError(f"nodes with no path to the sink: {dangling}")


def semantic_types(prog: Program) -> Dict[str, str]:
    """Semantic output type per node, resolving ``@preserve`` along the graph."""
    ops = build_ops()
    out: Dict[str, str] = {}
    for nd in prog.nodes:
        op = ops[nd.op]
        in_sems = []
        for p in op.params:
            value = nd.args[p.name]
            if isinstance(value, Ref):
                in_sems.append(out[value.node_id])
            else:
                in_sems.append(nd.arg_sems.get(p.name, p.sem))
        out[nd.node_id] = op.resolve_out_sem(in_sems)
    return out


def validate_semantic_edges(prog: Program) -> List[str]:
    """Every reference must be semantically admissible for its parameter."""
    ops = build_ops()
    sems = semantic_types(prog)
    errs: List[str] = []
    for nd in prog.nodes:
        op = ops[nd.op]
        for p in op.params:
            value = nd.args[p.name]
            if isinstance(value, Ref):
                produced = sems[value.node_id]
                if not st.compatible(p.sem, produced):
                    errs.append(f"{nd.node_id}.{p.name}: {produced} -> {p.sem}")
    return errs


# ── execution ────────────────────────────────────────────────────────────
def _resolve(value: Any, values: Dict[str, Any]) -> Any:
    if isinstance(value, Ref):
        if value.node_id not in values:
            raise ExecError(f"unresolved reference {value.node_id}")
        return values[value.node_id]
    if isinstance(value, list):
        return [_resolve(v, values) for v in value]
    if isinstance(value, dict):
        return {k: _resolve(v, values) for k, v in value.items()}
    return value


def _check(value: Any, node_id: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ExecError(f"{node_id}: NaN/Inf")
        try:
            magnitude = abs(float(value))
        except OverflowError as exc:      # a big-int power, e.g. 900 ** 400
            raise ExecError(f"{node_id}: magnitude overflow") from exc
        if magnitude > MAX_ABS:
            raise ExecError(f"{node_id}: magnitude overflow")
        return
    if isinstance(value, str):
        if not value or len(value) > MAX_TEXT:
            raise ExecError(f"{node_id}: degenerate text result")
        return
    if isinstance(value, list):
        if not value or len(value) > MAX_LIST:
            raise ExecError(f"{node_id}: degenerate list result")
        for item in value:
            _check(item, node_id)
        return
    if isinstance(value, dict):
        if not value or len(value) > MAX_LIST:
            raise ExecError(f"{node_id}: degenerate mapping result")
        for item in value.values():
            _check(item, node_id)
        return
    raise ExecError(f"{node_id}: unsupported value type {type(value).__name__}")


def execute(prog: Program) -> Tuple[Dict[str, Any], Any]:
    """Returns (node_id -> observation, final answer). The only oracle."""
    ops = build_ops()
    values: Dict[str, Any] = {}
    for nd in prog.nodes:
        op = ops[nd.op]
        kwargs = {}
        for p in op.params:
            if p.name not in nd.args:
                raise ExecError(f"{nd.node_id}: missing argument {p.name}")
            kwargs[p.name] = _resolve(nd.args[p.name], values)
        try:
            out = op.fn(**kwargs)
        except ExecError:
            raise
        except Exception as exc:                      # noqa: BLE001 - op guard
            raise ExecError(f"{nd.node_id}:{nd.op}: {exc}") from exc
        _check(out, nd.node_id)
        values[nd.node_id] = out
    return values, values[prog.sink]


def replay_identical(prog: Program, n: int = 3) -> bool:
    """Determinism gate: n independent executions must agree exactly."""
    runs = [execute(prog) for _ in range(n)]
    return all(run == runs[0] for run in runs[1:])


def observation_types(prog: Program) -> Dict[str, str]:
    values, _ = execute(prog)
    return {nid: st.value_kind(v) for nid, v in values.items()}


def check_value_types(prog: Program) -> List[str]:
    """Observed values must match the declared semantic types of their nodes."""
    values, _ = execute(prog)
    sems = semantic_types(prog)
    errs = []
    for nid, value in values.items():
        if not st.matches_value(sems[nid], value):
            errs.append(f"{nid}: {sems[nid]} vs observed {st.value_kind(value)}")
    return errs


def gold_calls(prog: Program, track: str,
               observations: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Exported NESTFUL-style calls: surface names, ``$var_k.output_0$`` refs.

    ``observations`` carries the oracle value each call produced. It is what lets a
    reviewer or a rollout grader award partial credit for reaching a correct
    intermediate result, so it travels with the record rather than being recomputed
    from a re-executed program.
    """
    ops = build_ops()
    if observations is None:
        observations, _answer = execute(prog)
    idx = prog.index()
    labels = {nd.node_id: f"$var_{i + 1}" for i, nd in enumerate(prog.nodes)}
    out: List[Dict[str, Any]] = []
    for nd in prog.nodes:
        op = ops[nd.op]
        surf = op.surface(track)
        arguments: Dict[str, Any] = {}
        for p, shown in zip(op.params, surf.param_names):
            value = nd.args[p.name]
            if isinstance(value, Ref):
                producer = prog.node(value.node_id)
                pfield = ops[producer.op].surface(track).output_field
                arguments[shown] = f"{labels[value.node_id]}.{pfield}$"
            else:
                arguments[shown] = _render_literal(value)
        out.append({
            "name": surf.name,
            "arguments": arguments,
            "label": labels[nd.node_id],
            "node_id": nd.node_id,
            "primitive_id": nd.op,
            "capability": op.capability,
            "capability_family": op.family,
            "coding_like": op.coding_like,
            "output_field": surf.output_field,
            "call_index": idx[nd.node_id] + 1,
            "observation": _render_literal(observations.get(nd.node_id)),
        })
    return out


def _render_literal(value: Any) -> Any:
    if isinstance(value, Ref):                        # pragma: no cover
        raise ProgramError("literal renderer received a reference")
    if isinstance(value, list):
        return [_render_literal(v) for v in value]
    if isinstance(value, dict):
        return {k: _render_literal(v) for k, v in value.items()}
    return value


def literal_constants(prog: Program) -> List[Any]:
    """Every value a solver could read straight out of the query."""
    out: List[Any] = []
    for nd in prog.nodes:
        for value in nd.args.values():
            if _refs_in(value):
                continue
            out.append(value)
    return out


def clone(prog: Program) -> Program:
    return Program(nodes=[Node(node_id=nd.node_id, op=nd.op, args=dict(nd.args),
                               arg_sems=dict(nd.arg_sems))
                          for nd in prog.nodes], sink=prog.sink)


def subprogram(prog: Program, keep: Iterable[str]) -> Program:
    """The program restricted to ``keep`` (used by necessity and V4 search)."""
    keep_set = set(keep)
    nodes = [Node(nd.node_id, nd.op, dict(nd.args), dict(nd.arg_sems))
             for nd in prog.nodes if nd.node_id in keep_set]
    sink = prog.sink if prog.sink in keep_set else (
        nodes[-1].node_id if nodes else "")
    return Program(nodes=nodes, sink=sink)


def answers_equal(a: Any, b: Any, *, tol: float = 1e-6) -> bool:
    """Answer equivalence used by every gate (V4, necessity, verifiers)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        try:
            return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)))
        except OverflowError:      # big-int intermediate from a search candidate
            return isinstance(a, int) and isinstance(b, int) and a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(answers_equal(x, y, tol=tol)
                                        for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return sorted(a) == sorted(b) and all(
            answers_equal(a[k], b[k], tol=tol) for k in a)
    return type(a) is type(b) and a == b


def program_summary(prog: Program) -> Dict[str, Any]:
    ops = build_ops()
    return {
        "call_count": len(prog.nodes),
        "primitive_sequence": list(prog.op_sequence()),
        "capability_sequence": list(prog.capability_sequence()),
        "capability_families": sorted({ops[nd.op].family for nd in prog.nodes}),
        "coding_call_count": sum(1 for nd in prog.nodes if ops[nd.op].coding_like),
        "semantic_program_id": prog.program_id(),
        "program_plan_id": prog.plan_id(),
        "edges": [[a, b] for a, b in prog.edges()],
    }
