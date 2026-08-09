"""Structural pattern families and composable graph transformations (Phase E).

A *shape* is the pure topology of a program: for each node, the list of
predecessor node indices. Shapes are generated independently of any primitive,
so the same topology can be realised with completely different capabilities.
Primitives are attached afterwards by a typed assignment pass that guarantees
the resulting DAG both typechecks and executes.

Transformations rewrite shapes, never programs, so DAG-ness and type validity
are re-established by re-running the same assignment pass.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .. import registry as reg
from ..capability import family_of
from ..executor import ExecutionError, execute, question_constants
from ..graph import REF
from ..schemas import GraphNode, SemanticProgram

PATTERN_FAMILIES = [
    "LINEAR_CHAIN", "FAN_IN_SINGLE", "FAN_IN_MULTIPLE", "FAN_OUT", "DIAMOND",
    "PARALLEL_THEN_MERGE", "REUSE_EARLY_OUTPUT", "LATE_REFERENCE",
    "TWO_STAGE_AGGREGATION", "MULTI_JOIN", "ALTERNATING_BRANCH_CHAIN",
    "MIXED_INDEPENDENT_DEPENDENT", "REPEATED_PRIMITIVE", "TYPE_TRANSITION_CHAIN",
    "NESTED_AGGREGATION",
]

TRANSFORMATIONS = [
    "INSERT_NODE_ON_EDGE", "SPLIT_BRANCH", "MERGE_BRANCHES", "REUSE_OUTPUT",
    "ADD_PARALLEL_BRANCH", "ADD_LATE_JOIN", "ADD_SECOND_JOIN",
    "REPEAT_PRIMITIVE_WITH_NEW_ARGS", "CHANGE_TYPE_PATH", "EXTEND_CRITICAL_PATH",
]

Shape = List[List[int]]     # parents per node, always strictly increasing indices


class PatternError(Exception):
    pass


# ── shape construction ────────────────────────────────────────────────────
def _linear(n: int) -> Shape:
    return [[] if i == 0 else [i - 1] for i in range(n)]


def _fan_in_single(n: int) -> Shape:
    """Two independent chains that merge exactly once, at the sink."""
    if n < 3:
        raise PatternError("FAN_IN_SINGLE needs >= 3 calls")
    left = (n - 1) // 2
    right = n - 1 - left
    shape: Shape = []
    for i in range(left):
        shape.append([] if i == 0 else [i - 1])
    for j in range(right):
        shape.append([] if j == 0 else [left + j - 1])
    shape.append([left - 1, n - 2])
    return shape


def _fan_in_multiple(n: int) -> Shape:
    """Chain whose every second node also pulls in a fresh independent root."""
    if n < 5:
        raise PatternError("FAN_IN_MULTIPLE needs >= 5 calls")
    shape: Shape = [[]]
    last = 0
    i = 1
    while i < n:
        if i + 1 < n and len(shape) + 1 < n:
            shape.append([])                    # fresh root
            root = len(shape) - 1
            shape.append([last, root])          # join
            last = len(shape) - 1
            i = len(shape)
        else:
            shape.append([last])
            last = len(shape) - 1
            i = len(shape)
    return shape[:n]


def _fan_out(n: int) -> Shape:
    """One early output feeds two different downstream nodes."""
    if n < 4:
        raise PatternError("FAN_OUT needs >= 4 calls")
    shape: Shape = [[], [0], [0]]
    for i in range(3, n - 1):
        shape.append([i - 1])
    shape.append([1, len(shape) - 1] if n > 4 else [1, 2])
    return shape[:n]


def _diamond(n: int) -> Shape:
    if n < 4:
        raise PatternError("DIAMOND needs >= 4 calls")
    shape: Shape = [[], [0], [0], [1, 2]]
    for i in range(4, n):
        shape.append([i - 1])
    return shape


def _parallel_then_merge(n: int) -> Shape:
    if n < 4:
        raise PatternError("PARALLEL_THEN_MERGE needs >= 4 calls")
    n_roots = min(3, n - 1)
    shape: Shape = [[] for _ in range(n_roots)]
    shape.append(list(range(n_roots)))
    for i in range(n_roots + 1, n):
        shape.append([i - 1])
    return shape


def _reuse_early_output(n: int) -> Shape:
    # at n == 3 this degenerates to the triangle [[], [0], [0, 1]], the only
    # 3-call shape with both a reused output and a join, so it is allowed
    if n < 3:
        raise PatternError("REUSE_EARLY_OUTPUT needs >= 3 calls")
    shape: Shape = [[], [0]]
    for i in range(2, n - 1):
        shape.append([i - 1])
    shape.append([0, n - 2])            # node 0 reused at the very end
    return shape


def _late_reference(n: int) -> Shape:
    if n < 4:
        raise PatternError("LATE_REFERENCE needs >= 4 calls")
    shape: Shape = [[], [], [0]]
    for i in range(3, n - 1):
        shape.append([i - 1])
    shape.append([1, n - 2])            # root 1 referenced far downstream
    return shape


def _two_stage_aggregation(n: int) -> Shape:
    if n < 5:
        raise PatternError("TWO_STAGE_AGGREGATION needs >= 5 calls")
    shape: Shape = [[], [], []]
    shape.append([0, 1])
    shape.append([2, 3])
    for i in range(5, n):
        shape.append([i - 1])
    return shape[:n]


def _multi_join(n: int) -> Shape:
    """Core with two joins, extended by alternating chain/join steps."""
    if n < 5:
        raise PatternError("MULTI_JOIN needs >= 5 calls")
    shape: Shape = [[], [], [0, 1], [], [2, 3]]
    while len(shape) < n:
        if len(shape) + 1 < n:
            shape.append([])                       # fresh root
            shape.append([len(shape) - 2, len(shape) - 1])
        else:
            shape.append([len(shape) - 1])
    return shape


def _alternating_branch_chain(n: int) -> Shape:
    if n < 5:
        raise PatternError("ALTERNATING_BRANCH_CHAIN needs >= 5 calls")
    shape: Shape = [[], [0]]
    while len(shape) < n:
        if len(shape) % 2 == 0 and len(shape) + 1 < n:
            shape.append([])
            shape.append([len(shape) - 2, len(shape) - 1])
        else:
            shape.append([len(shape) - 1])
    return shape[:n]


def _mixed_independent_dependent(n: int) -> Shape:
    if n < 4:
        raise PatternError("MIXED_INDEPENDENT_DEPENDENT needs >= 4 calls")
    shape: Shape = [[], [0], []]
    for i in range(3, n - 1):
        shape.append([i - 1])
    shape.append([1, n - 2])
    return shape[:n]


def _repeated_primitive(n: int) -> Shape:
    # topologically a chain; the assignment pass is told to repeat one primitive
    return _linear(n)


def _type_transition_chain(n: int) -> Shape:
    return _linear(n)


def _nested_aggregation(n: int) -> Shape:
    """Two independent aggregations whose results are aggregated again."""
    if n < 7:
        raise PatternError("NESTED_AGGREGATION needs >= 7 calls")
    shape: Shape = [[], [], [], [], [0, 1], [2, 3], [4, 5]]
    for i in range(7, n):
        shape.append([i - 1])
    return shape


_SHAPE_BUILDERS: Dict[str, Callable[[int], Shape]] = {
    "LINEAR_CHAIN": _linear,
    "FAN_IN_SINGLE": _fan_in_single,
    "FAN_IN_MULTIPLE": _fan_in_multiple,
    "FAN_OUT": _fan_out,
    "DIAMOND": _diamond,
    "PARALLEL_THEN_MERGE": _parallel_then_merge,
    "REUSE_EARLY_OUTPUT": _reuse_early_output,
    "LATE_REFERENCE": _late_reference,
    "TWO_STAGE_AGGREGATION": _two_stage_aggregation,
    "MULTI_JOIN": _multi_join,
    "ALTERNATING_BRANCH_CHAIN": _alternating_branch_chain,
    "MIXED_INDEPENDENT_DEPENDENT": _mixed_independent_dependent,
    "REPEATED_PRIMITIVE": _repeated_primitive,
    "TYPE_TRANSITION_CHAIN": _type_transition_chain,
    "NESTED_AGGREGATION": _nested_aggregation,
}

MIN_CALLS: Dict[str, int] = {
    "LINEAR_CHAIN": 2, "FAN_IN_SINGLE": 3, "FAN_IN_MULTIPLE": 5, "FAN_OUT": 4,
    "DIAMOND": 4, "PARALLEL_THEN_MERGE": 4, "REUSE_EARLY_OUTPUT": 3,
    "LATE_REFERENCE": 4, "TWO_STAGE_AGGREGATION": 5, "MULTI_JOIN": 5,
    "ALTERNATING_BRANCH_CHAIN": 5, "MIXED_INDEPENDENT_DEPENDENT": 4,
    "REPEATED_PRIMITIVE": 3, "TYPE_TRANSITION_CHAIN": 3, "NESTED_AGGREGATION": 7,
}


def build_shape(pattern: str, n_calls: int) -> Shape:
    builder = _SHAPE_BUILDERS.get(pattern)
    if builder is None:
        raise PatternError(f"unknown pattern family {pattern!r}")
    if n_calls < MIN_CALLS[pattern]:
        raise PatternError(f"{pattern} needs >= {MIN_CALLS[pattern]} calls, got {n_calls}")
    shape = builder(n_calls)
    if len(shape) != n_calls:
        raise PatternError(f"{pattern}: produced {len(shape)} nodes, wanted {n_calls}")
    validate_shape(shape)
    return shape


def patterns_for(n_calls: int) -> List[str]:
    return [p for p in PATTERN_FAMILIES if MIN_CALLS[p] <= n_calls]


def validate_shape(shape: Shape) -> None:
    """A shape is valid when it is a connected-enough DAG with no dead nodes."""
    n = len(shape)
    if n == 0:
        raise PatternError("empty shape")
    for i, parents in enumerate(shape):
        if len(set(parents)) != len(parents):
            raise PatternError(f"node {i}: duplicate parents")
        for p in parents:
            if not 0 <= p < i:
                raise PatternError(f"node {i}: parent {p} is not a strict predecessor")
    sink = n - 1
    reaching = {sink}
    for i in range(n - 1, -1, -1):
        if i in reaching:
            reaching.update(shape[i])
    dead = [i for i in range(n) if i not in reaching]
    if dead:
        raise PatternError(f"dead gold calls (unreachable from sink): {dead}")


def shape_signature(shape: Shape) -> str:
    return "|".join(",".join(str(p) for p in parents) for parents in shape)


# ── graph transformations ─────────────────────────────────────────────────
def _renumber_insert(shape: Shape, at: int) -> Shape:
    """Insert a placeholder node at index ``at``, shifting later references."""
    out: Shape = []
    for i, parents in enumerate(shape):
        idx = i if i < at else i + 1
        shifted = [p if p < at else p + 1 for p in parents]
        while len(out) < idx:
            out.append([])
        out.append(shifted)
    return out


def t_insert_node_on_edge(shape: Shape, rng: random.Random) -> Shape:
    edges = [(p, i) for i, parents in enumerate(shape) for p in parents]
    if not edges:
        raise PatternError("INSERT_NODE_ON_EDGE: no edge")
    a, b = rng.choice(edges)
    new = _renumber_insert(shape, b)
    new[b] = [a]                                    # the inserted node
    new[b + 1] = [x if x != a else b for x in new[b + 1]]
    return new


def t_extend_critical_path(shape: Shape, rng: random.Random) -> Shape:
    sink = len(shape) - 1
    new = [list(p) for p in shape]
    new.append([sink])
    return new


def t_add_parallel_branch(shape: Shape, rng: random.Random) -> Shape:
    new = [list(p) for p in shape]
    new.append([])                                  # new independent root
    root = len(new) - 1
    new.append([len(shape) - 1, root])              # merge with old sink
    return new


def t_split_branch(shape: Shape, rng: random.Random) -> Shape:
    """Take one node's output into a second, separate consumer."""
    n = len(shape)
    if n < 2:
        raise PatternError("SPLIT_BRANCH: too small")
    src = rng.randrange(max(n - 1, 1))
    new = [list(p) for p in shape]
    new.append([src])
    new.append([n - 1, len(new) - 1])
    return new


def t_merge_branches(shape: Shape, rng: random.Random) -> Shape:
    n = len(shape)
    roots = [i for i, p in enumerate(shape) if not p]
    if len(roots) < 2:
        raise PatternError("MERGE_BRANCHES: fewer than two roots")
    new = [list(p) for p in shape]
    new.append([n - 1, roots[0]])
    return new


def t_reuse_output(shape: Shape, rng: random.Random) -> Shape:
    n = len(shape)
    if n < 3:
        raise PatternError("REUSE_OUTPUT: too small")
    early = rng.randrange(max(n - 2, 1))
    new = [list(p) for p in shape]
    new.append([n - 1, early])
    return new


def t_add_late_join(shape: Shape, rng: random.Random) -> Shape:
    n = len(shape)
    if n < 3:
        raise PatternError("ADD_LATE_JOIN: too small")
    new = [list(p) for p in shape]
    new.append([])
    new.append([n - 1, len(new) - 1])
    return new


def t_add_second_join(shape: Shape, rng: random.Random) -> Shape:
    joins = [i for i, p in enumerate(shape) if len(p) >= 2]
    new = [list(p) for p in shape]
    if not joins:
        return t_add_late_join(shape, rng)
    new.append([])
    root = len(new) - 1
    new.append([len(shape) - 1, root])
    return new


def t_repeat_primitive_with_new_args(shape: Shape, rng: random.Random) -> Shape:
    """Structural half of the transform: append a sibling consumer chain."""
    n = len(shape)
    new = [list(p) for p in shape]
    new.append([n - 1])
    return new


def t_change_type_path(shape: Shape, rng: random.Random) -> Shape:
    """Insert one node on the sink's incoming edge so the type chain changes."""
    if len(shape) < 2:
        raise PatternError("CHANGE_TYPE_PATH: too small")
    sink = len(shape) - 1
    if not shape[sink]:
        raise PatternError("CHANGE_TYPE_PATH: sink has no parent")
    new = _renumber_insert(shape, sink)
    src = new[sink + 1][0]
    new[sink] = [src]
    new[sink + 1] = [sink] + [p for p in new[sink + 1] if p != src]
    return new


TRANSFORM_FNS: Dict[str, Callable[[Shape, random.Random], Shape]] = {
    "INSERT_NODE_ON_EDGE": t_insert_node_on_edge,
    "SPLIT_BRANCH": t_split_branch,
    "MERGE_BRANCHES": t_merge_branches,
    "REUSE_OUTPUT": t_reuse_output,
    "ADD_PARALLEL_BRANCH": t_add_parallel_branch,
    "ADD_LATE_JOIN": t_add_late_join,
    "ADD_SECOND_JOIN": t_add_second_join,
    "REPEAT_PRIMITIVE_WITH_NEW_ARGS": t_repeat_primitive_with_new_args,
    "CHANGE_TYPE_PATH": t_change_type_path,
    "EXTEND_CRITICAL_PATH": t_extend_critical_path,
}


def apply_transform(shape: Shape, name: str, rng: random.Random) -> Shape:
    fn = TRANSFORM_FNS.get(name)
    if fn is None:
        raise PatternError(f"unknown transformation {name!r}")
    new = fn([list(p) for p in shape], rng)
    validate_shape(new)
    return new


# ── typed primitive assignment ────────────────────────────────────────────
_ARR = reg.ARR


def _param_slots(prim: reg.Primitive) -> List[Tuple[int, str]]:
    """Parameter positions that may receive a reference (never arrays/enums)."""
    return [(i, t) for i, (_n, t, _s) in enumerate(prim.params)
            if t != _ARR and not t.startswith("enum:")]


def _can_host(prim: reg.Primitive, parent_types: Sequence[str]) -> Optional[List[int]]:
    """Greedy injective assignment of parent outputs onto parameter slots."""
    slots = _param_slots(prim)
    if len(slots) < len(parent_types):
        return None
    used: set = set()
    chosen: List[int] = []
    for ptype in parent_types:
        pick = None
        for idx, stype in slots:
            if idx in used:
                continue
            if reg.type_accepts(stype, ptype):
                pick = idx
                break
        if pick is None:
            return None
        used.add(pick)
        chosen.append(pick)
    return chosen


@dataclass
class AssignmentConfig:
    capability_mix: List[str] = field(default_factory=list)
    answer_kind: str = "float"
    repeat_primitive: bool = False
    force_type_transitions: bool = False
    max_attempts: int = 40


def _candidate_pool(parent_types: Sequence[str], cfg: AssignmentConfig,
                    is_sink: bool) -> List[str]:
    pool = []
    for sid, prim in reg.all_primitives().items():
        if _can_host(prim, parent_types) is None:
            continue
        if is_sink and cfg.answer_kind and prim.answer_kind != cfg.answer_kind:
            continue
        pool.append(sid)
    return sorted(pool)


def _weighted_pick(pool: Sequence[str], cfg: AssignmentConfig,
                   rng: random.Random, used_types: Sequence[str]) -> str:
    if not pool:
        raise PatternError("no primitive can host this node")
    weights = []
    for sid in pool:
        w = 1.0
        if cfg.capability_mix and family_of(sid) in cfg.capability_mix:
            w *= 6.0
        if cfg.force_type_transitions and used_types:
            if reg.get(sid).out_type != used_types[-1]:
                w *= 3.0
        weights.append(w)
    return rng.choices(list(pool), weights=weights, k=1)[0]


def assign_primitives(shape: Shape, rng: random.Random,
                      cfg: Optional[AssignmentConfig] = None) -> SemanticProgram:
    """Attach typed primitives + constants to a shape, producing a program."""
    cfg = cfg or AssignmentConfig()
    n = len(shape)
    out_types: List[str] = []
    sids: List[str] = []
    nodes: List[GraphNode] = []
    repeat_sid: Optional[str] = None

    for i, parents in enumerate(shape):
        ptypes = [out_types[p] for p in parents]
        is_sink = i == n - 1
        pool = _candidate_pool(ptypes, cfg, is_sink)
        if not pool and is_sink:
            pool = _candidate_pool(ptypes, AssignmentConfig(
                capability_mix=cfg.capability_mix, answer_kind=""), False)
        if cfg.repeat_primitive and repeat_sid and repeat_sid in pool and not is_sink:
            sid = repeat_sid
        else:
            sid = _weighted_pick(pool, cfg, rng, out_types)
            if cfg.repeat_primitive and repeat_sid is None and not is_sink:
                repeat_sid = sid
        prim = reg.get(sid)
        slot_for_parent = _can_host(prim, ptypes)
        if slot_for_parent is None:
            raise PatternError(f"node {i}: chosen primitive cannot host parents")
        sample = list(prim.sampler(rng))
        inputs: Dict[str, Any] = {}
        for pi, (pname, _t, _s) in enumerate(prim.params):
            inputs[pname] = sample[pi] if pi < len(sample) else 0
        for parent, slot in zip(parents, slot_for_parent):
            pname = prim.params[slot][0]
            inputs[pname] = {REF: f"n{parent}"}
        nodes.append(GraphNode(node_id=f"n{i}", semantic_id=sid, inputs=inputs,
                               output_type=prim.out_type))
        sids.append(sid)
        out_types.append(prim.out_type)

    depth = _depth_of(shape)
    return SemanticProgram(nodes=nodes, sink=f"n{n - 1}", motif="", depth=depth)


def _depth_of(shape: Shape) -> int:
    longest = [0] * len(shape)
    for i, parents in enumerate(shape):
        if parents:
            longest[i] = 1 + max(longest[p] for p in parents)
    return max(longest) if longest else 0


# ── full generation with executability guards ─────────────────────────────
@dataclass
class ProgramResult:
    program: SemanticProgram
    shape: Shape
    pattern_family: str
    transformations: List[str]
    observations: List[Any]
    answer: Any
    capability_families: List[str]


def generate_program(pattern: str, n_calls: int, rng: random.Random, *,
                     capability_mix: Optional[Sequence[str]] = None,
                     answer_kind: str = "float",
                     transformations: Sequence[str] = (),
                     max_attempts: int = 40) -> ProgramResult:
    """Build a shape, optionally transform it, then attach executable typing."""
    base_shape = build_shape(pattern, n_calls)
    shape = base_shape
    applied: List[str] = []
    for name in transformations:
        try:
            shape = apply_transform(shape, name, rng)
            applied.append(name)
        except PatternError:
            continue

    cfg = AssignmentConfig(
        capability_mix=list(capability_mix or []),
        answer_kind=answer_kind,
        repeat_primitive=pattern == "REPEATED_PRIMITIVE",
        force_type_transitions=pattern == "TYPE_TRANSITION_CHAIN",
    )
    last_err: Optional[Exception] = None
    for _ in range(max_attempts):
        try:
            prog = assign_primitives(shape, rng, cfg)
            observations, answer = execute(prog)
        except (PatternError, ExecutionError, ValueError, TypeError,
                ZeroDivisionError, OverflowError, IndexError, KeyError) as exc:
            last_err = exc
            continue
        if not _value_guards_ok(prog, observations):
            continue
        prog.motif = _motif_of_shape(shape)
        return ProgramResult(
            program=prog, shape=shape, pattern_family=pattern,
            transformations=applied, observations=observations, answer=answer,
            capability_families=[family_of(nd.semantic_id) for nd in prog.nodes])
    raise PatternError(f"{pattern}/{n_calls}: no executable assignment ({last_err})")


def _value_guards_ok(prog: SemanticProgram, observations: Sequence[Any]) -> bool:
    """No observation may collapse onto a question constant or onto another."""
    consts = {round(float(v), 9) for v in question_constants(prog)
              if isinstance(v, (int, float)) and not isinstance(v, bool)}
    nums = [round(float(o), 9) for o in observations
            if isinstance(o, (int, float)) and not isinstance(o, bool)]
    if any(o in consts for o in nums):
        return False
    return len(set(nums)) == len(nums)


def _motif_of_shape(shape: Shape) -> str:
    from collections import Counter

    indeg = [len(p) for p in shape]
    outdeg = Counter(p for parents in shape for p in parents)
    n_joins = sum(1 for d in indeg if d >= 2)
    if max(indeg) >= 3:
        return "branch_aggregate"
    if n_joins >= 2:
        return "multi_join"
    if any(c >= 2 for c in outdeg.values()):
        return "fan_out" if n_joins == 0 else "fan_in"
    if n_joins == 1:
        return "fan_in"
    if sum(1 for p in shape if not p) > 1:
        return "mixed"
    return "linear"
