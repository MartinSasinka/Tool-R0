"""Instantiate a workflow plan into an actual typed DAG.

Order of operations matters and is the whole point of Pilot4.3:

1. sample role values with the workflow's own generator,
2. bind each planned capability to a concrete op (semantic types must line up),
3. build the program from the plan's explicit edges,
4. execute it -- and only *then*, with the oracle values in hand, choose the
   comparison constants that decide a boolean or categorical answer,
5. recompute the structural pattern, answer type and capability usage from the
   built graph.

Nothing in this module trusts a label: :func:`instantiate` returns what the
graph turned out to be, and the caller rejects the instance when that does not
match what it asked for.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from ..repro import sha256_obj
from . import semtypes as st
from .blueprints import Blueprint, Plan, Role
from .ops import CODING_FAMILIES, Op, build_ops, ops_by_capability
from .patterns import classify, features, primary_pattern, satisfied_patterns
from .program import (ExecError, Node, Program, ProgramError, Ref, execute,
                      gold_calls, program_summary, replay_identical,
                      check_value_types, semantic_types, validate_semantic_edges,
                      validate_structure)
from .values import (Band, COMBINATORS, band_for, calibrate_predicate,
                     calibratable, coerce_constant, parent_targets, sample_hint)

CATEGORY_BANDS = ("low", "medium", "high")


class BuildError(Exception):
    """This (blueprint, plan, seed) triple did not yield a usable instance."""


@dataclass
class Instance:
    workflow_id: str
    domain: str
    plan_id: str
    program: Program
    role_values: Dict[str, Any]
    role_hints: Dict[str, str]
    role_descriptions: Dict[str, str]
    op_binding: Dict[str, str]
    observations: Dict[str, Any]
    answer: Any
    answer_type: str
    value_kinds: Dict[str, str]
    semantic_kinds: Dict[str, str]
    actual_patterns: List[str]
    actual_primary_pattern: str
    graph_features: Dict[str, Any]
    track: str
    boolean_band: str | None
    category_band: str | None
    seed: int

    @property
    def call_count(self) -> int:
        return len(self.program.nodes)

    def workflow_instance_id(self) -> str:
        return "wi_" + sha256_obj({
            "workflow": self.workflow_id, "plan": self.plan_id,
            "roles": {k: _hashable(v) for k, v in sorted(self.role_values.items())},
            "binding": dict(sorted(self.op_binding.items())),
        })[:20]

    def summary(self) -> Dict[str, Any]:
        ops = build_ops()
        used = [nd.op for nd in self.program.nodes]
        coding = [pid for pid in used if ops[pid].coding_like]
        return {
            **program_summary(self.program),
            "workflow_id": self.workflow_id,
            "domain": self.domain,
            "plan_id": self.plan_id,
            "workflow_instance_id": self.workflow_instance_id(),
            "answer_type": self.answer_type,
            "actual_patterns": list(self.actual_patterns),
            "actual_primary_pattern": self.actual_primary_pattern,
            "graph_features": self.graph_features,
            "coding_call_share": round(len(coding) / max(1, len(used)), 4),
            "coding_capability_families": sorted(
                {ops[pid].family for pid in coding}),
        }


def _hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _hashable(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_hashable(v) for v in value]
    return value


def _bind(plan: Plan, rng: random.Random,
          role_sems: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Choose a concrete op per step so that every edge is semantically valid."""
    caps = ops_by_capability()
    ops = build_ops()
    binding: Dict[str, str] = {}
    produced: Dict[str, str] = {}
    for step in plan.steps:
        candidates = [pid for pid in caps.get(step.capability, [])
                      if ops[pid].arity == len(step.args)]
        rng.shuffle(candidates)
        chosen: str | None = None
        for pid in candidates:
            op = ops[pid]
            in_sems: List[str] = []
            ok = True
            for arg, param in zip(step.args, op.params):
                if arg.startswith("@"):
                    src = produced[arg[1:]]
                    if not st.compatible(param.sem, src):
                        ok = False
                        break
                    in_sems.append(src)
                else:
                    if not st.compatible(param.sem, role_sems[arg]):
                        ok = False
                        break
                    in_sems.append(role_sems[arg])
            if ok:
                chosen = pid
                produced[step.node_id] = op.resolve_out_sem(in_sems)
                break
        if chosen is None:
            raise BuildError(f"{plan.plan_id}/{step.node_id}: no admissible op "
                             f"for {step.capability}")
        binding[step.node_id] = chosen
    return binding, produced


def _program_from(plan: Plan, binding: Dict[str, str],
                  role_values: Dict[str, Any],
                  role_sems: Dict[str, str]) -> Program:
    ops = build_ops()
    nodes: List[Node] = []
    for step in plan.steps:
        op = ops[binding[step.node_id]]
        args: Dict[str, Any] = {}
        arg_sems: Dict[str, str] = {}
        for arg, param in zip(step.args, op.params):
            if arg.startswith("@"):
                args[param.name] = Ref(arg[1:])
            else:
                args[param.name] = role_values[arg]
                arg_sems[param.name] = role_sems[arg]
        nodes.append(Node(node_id=step.node_id, op=binding[step.node_id],
                          args=args, arg_sems=arg_sems))
    return Program(nodes=nodes, sink=plan.sink)


#: role-name markers for the two ends of a period. Dates are sampled per role and
#: independently, which produced instances that hand a job over nine months before
#: it starts -- executable, and impossible for a reader to take seriously.
DATE_START_MARKERS = ("start", "begin", "from", "opened", "issued", "arrival",
                      "first")
DATE_END_MARKERS = ("deadline", "due", "end", "handover", "handed", "closes",
                    "closing", "target", "last", "expiry", "cutoff")


def _order_dates(plan: Plan, role_values: Dict[str, Any]) -> None:
    """Swap a start/end date pair that came out back to front."""
    starts, ends = [], []
    for role in plan.roles:
        value = role_values.get(role.name)
        if not isinstance(value, str) or role.sem != st.DATE:
            continue
        name = role.name.lower()
        if any(m in name for m in DATE_START_MARKERS):
            starts.append(role.name)
        elif any(m in name for m in DATE_END_MARKERS):
            ends.append(role.name)
    for start in starts:
        for end in ends:
            if role_values[end] < role_values[start]:
                role_values[start], role_values[end] = (role_values[end],
                                                        role_values[start])


def _calibrate_boolean(prog: Program, rng: random.Random, want: bool,
                       near: bool) -> bool:
    """Pick predicate constants against the executed values, not before them."""
    ops = build_ops()
    try:
        values, _ = execute(prog)
    except ExecError:
        return False
    targets: Dict[str, bool] = {prog.sink: want}
    for nd in reversed(prog.nodes):
        if nd.node_id not in targets:
            continue
        want_i = targets[nd.node_id]
        op = ops[nd.op]
        cap = op.capability
        if cap in COMBINATORS:
            kind, _n = COMBINATORS[cap]
            refs = [nd.args[p.name].node_id for p in op.params
                    if isinstance(nd.args[p.name], Ref)]
            if len(refs) != len(op.params):
                return False
            assign = parent_targets(kind, want_i, len(refs), rng)
            if assign is None:
                return False
            for ref, value in zip(refs, assign):
                if targets.get(ref, value) != value:
                    return False
                targets[ref] = value
            continue
        if not calibratable(cap):
            return False
        observed: Dict[str, Any] = {}
        for p in op.params:
            arg = nd.args[p.name]
            observed[p.name] = (values[arg.node_id] if isinstance(arg, Ref)
                                else arg)
        band = band_for(want_i, near and nd.node_id == prog.sink)
        overrides = calibrate_predicate(cap, [p.name for p in op.params],
                                        observed, band, rng)
        if overrides is None:
            return False
        for key, value in overrides.items():
            if key not in nd.args or isinstance(nd.args[key], Ref):
                return False
            nd.args[key] = coerce_constant(nd.arg_sems.get(key, st.GENERIC),
                                           value)
    try:
        _values, answer = execute(prog)
    except ExecError:
        return False
    return answer is want or answer == want


def _calibrate_category(prog: Program, rng: random.Random, band: str) -> bool:
    ops = build_ops()
    sink = prog.node(prog.sink)
    cap = ops[sink.op].capability
    try:
        values, _ = execute(prog)
    except ExecError:
        return False
    op = ops[sink.op]
    first = op.params[0].name
    arg = sink.args[first]
    observed = values[arg.node_id] if isinstance(arg, Ref) else arg
    if not isinstance(observed, (int, float)) or isinstance(observed, bool):
        return False
    v = float(observed)
    scale = max(abs(v), 1.0)
    names = [p.name for p in op.params]
    if cap == "classification.three_bands" and len(names) == 3:
        gap = round(scale * rng.uniform(0.12, 0.4) + 1.0, 2)
        if band == "low":
            low, high = round(v + gap, 2), round(v + gap * 2.4, 2)
        elif band == "high":
            low, high = round(v - gap * 2.4, 2), round(v - gap, 2)
        else:
            low, high = round(v - gap, 2), round(v + gap, 2)
        if any(isinstance(sink.args[n], Ref) for n in names[1:]):
            return False
        sink.args[names[1]], sink.args[names[2]] = low, high
    elif cap == "classification.ratio_band" and len(names) == 2:
        gap = round(scale * rng.uniform(0.05, 0.25) + 0.01, 4)
        if isinstance(sink.args[names[1]], Ref):
            return False
        sink.args[names[1]] = round(v - gap, 4) if band == "high" else round(
            v + gap, 4)
    else:
        return False
    try:
        execute(prog)
    except ExecError:
        return False
    return True


def instantiate(bp: Blueprint, plan: Plan, seed: int, *, track: str,
                want_bool: bool | None = None, near_boundary: bool = False,
                want_category: str | None = None) -> Instance:
    """Build one instance or raise :class:`BuildError`."""
    rng = random.Random(seed)
    role_sems = {r.name: r.sem for r in plan.roles}
    binding, _produced = _bind(plan, rng, role_sems)
    role_values = {r.name: sample_hint(r.hint, rng) for r in plan.roles}
    _order_dates(plan, role_values)
    prog = _program_from(plan, binding, role_values, role_sems)
    try:
        validate_structure(prog)
    except ProgramError as exc:
        raise BuildError(f"structure: {exc}") from exc
    edge_errs = validate_semantic_edges(prog)
    if edge_errs:
        raise BuildError("semantic edges: " + "; ".join(edge_errs[:3]))
    try:
        observations, answer = execute(prog)
    except ExecError as exc:
        raise BuildError(f"execution: {exc}") from exc

    sink_sem = semantic_types(prog)[prog.sink]
    # An uncalibrated boolean or category sink is the Pilot4.2 failure mode: the
    # threshold is sampled independently of the value it is compared against, so one
    # branch decides everything, the label collapses to one side and the other nodes
    # stop mattering. Every such sink is therefore calibrated, with the requested
    # outcome drawn from the instance seed when the caller does not ask for one.
    if want_bool is None and isinstance(answer, bool):
        want_bool = rng.random() < 0.5
        near_boundary = rng.random() < 0.5
    if want_category is None and sink_sem == st.CATEGORY:
        want_category = CATEGORY_BANDS[rng.randrange(len(CATEGORY_BANDS))]

    boolean_band: str | None = None
    category_band: str | None = None
    if want_bool is not None:
        if not isinstance(answer, bool):
            raise BuildError("boolean target requested for a non-boolean sink")
        if not _calibrate_boolean(prog, rng, want_bool, near_boundary):
            raise BuildError(f"boolean calibration failed (want={want_bool})")
        boolean_band = band_for(want_bool, near_boundary).name
    if want_category is not None:
        if not _calibrate_category(prog, rng, want_category):
            raise BuildError(f"category calibration failed (want={want_category})")
        category_band = want_category

    try:
        observations, answer = execute(prog)
    except ExecError as exc:
        raise BuildError(f"post-calibration execution: {exc}") from exc
    if not replay_identical(prog):
        raise BuildError("non-deterministic replay")
    type_errs = check_value_types(prog)
    if type_errs:
        raise BuildError("value/type mismatch: " + "; ".join(type_errs[:3]))

    # refresh role values that calibration rewrote so the query states the truth
    for step in plan.steps:
        node = prog.node(step.node_id)
        op = build_ops()[node.op]
        for arg, param in zip(step.args, op.params):
            if not arg.startswith("@"):
                role_values[arg] = node.args[param.name]

    kinds = {nid: st.value_kind(v) for nid, v in observations.items()}
    sat = satisfied_patterns(prog, kinds)
    feats = features(prog, kinds)
    sems = semantic_types(prog)
    answer_type = _answer_type(answer, sems[prog.sink])
    return Instance(
        workflow_id=bp.workflow_id, domain=bp.domain, plan_id=plan.plan_id,
        program=prog, role_values=role_values,
        role_hints={r.name: r.hint for r in plan.roles},
        role_descriptions={r.name: r.description for r in plan.roles},
        op_binding=binding, observations=observations, answer=answer,
        answer_type=answer_type, value_kinds=kinds, semantic_kinds=sems,
        actual_patterns=sorted(sat),
        actual_primary_pattern=primary_pattern(sat),
        graph_features=feats.as_dict(), track=track,
        boolean_band=boolean_band, category_band=category_band, seed=seed)


def _answer_type(answer: Any, sink_sem: str) -> str:
    kind = st.value_kind(answer)
    if kind == "boolean":
        return "boolean"
    if sink_sem == st.CATEGORY:
        return "category"
    if kind == "integer":
        return "integer"
    if kind == "float":
        return "float"
    if kind == "list":
        return "list"
    if kind == "object":
        return "object"
    return "string"


def describe_plan(bp: Blueprint, plan: Plan, *, trials: int = 12,
                  base_seed: int = 991) -> Dict[str, Any]:
    """Derived plan properties: what the plan actually produces, not what it claims."""
    ok = 0
    patterns: Dict[str, int] = {}
    primaries: Dict[str, int] = {}
    answers: Dict[str, int] = {}
    errors: Dict[str, int] = {}
    families: set[str] = set()
    primitives: set[str] = set()
    for i in range(trials):
        try:
            inst = instantiate(bp, plan, base_seed + i * 7919, track="A_NATIVE")
        except BuildError as exc:
            key = str(exc)[:110]
            errors[key] = errors.get(key, 0) + 1
            continue
        ok += 1
        for p in inst.actual_patterns:
            patterns[p] = patterns.get(p, 0) + 1
        primaries[inst.actual_primary_pattern] = primaries.get(
            inst.actual_primary_pattern, 0) + 1
        answers[inst.answer_type] = answers.get(inst.answer_type, 0) + 1
        ops = build_ops()
        for nd in inst.program.nodes:
            primitives.add(nd.op)
            families.add(ops[nd.op].family)
    return {
        "trials": trials, "instantiated": ok,
        "call_count": plan.call_count,
        "actual_patterns_observed": dict(sorted(patterns.items())),
        "actual_primary_patterns": dict(sorted(primaries.items())),
        "answer_types": dict(sorted(answers.items())),
        "capability_families": sorted(families),
        "primitives_bound": sorted(primitives),
        "coding_like": any(f in CODING_FAMILIES for f in families),
        "build_errors": dict(sorted(errors.items())),
    }
