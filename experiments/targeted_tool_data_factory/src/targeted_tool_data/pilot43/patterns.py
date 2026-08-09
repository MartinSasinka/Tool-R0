"""Structural patterns as properties of the actual dependency graph.

Pilot4.2 stored the structural pattern requested by the generation cell. This
module never trusts a request: it derives the graph from the program's own
references, computes features, evaluates the 15 pattern invariants and returns
the *set* of patterns the graph actually satisfies. The generator then keeps a
task only when the requested structural skill is in that set, and the exported
record carries the recomputed set plus a deterministic primary label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

from . import semtypes as st
from .ops import build_ops
from .program import Program

#: A single graph can legitimately satisfy several invariants at once; the
#: primary label is the most structurally specific satisfied one.
PATTERN_PRIORITY: Tuple[str, ...] = (
    "NESTED_AGGREGATION",
    "MULTI_JOIN",
    "TWO_STAGE_AGGREGATION",
    "DIAMOND",
    "PARALLEL_THEN_MERGE",
    "ALTERNATING_BRANCH_CHAIN",
    "REUSE_EARLY_OUTPUT",
    "LATE_REFERENCE",
    "FAN_IN_MULTIPLE",
    "FAN_OUT",
    "FAN_IN_SINGLE",
    "TYPE_TRANSITION_CHAIN",
    "REPEATED_PRIMITIVE",
    "MIXED_INDEPENDENT_DEPENDENT",
    "LINEAR_CHAIN",
)


#: Output kinds that count as a collection when deciding aggregation.
COLLECTION_KINDS: Tuple[str, ...] = ("list", "object")


def late_threshold_for(n_nodes: int) -> int:
    """A 'late' reference must skip more intermediate calls in longer programs."""
    return 4 if n_nodes >= 8 else 3


@dataclass
class GraphFeatures:
    n_nodes: int
    n_edges: int
    indegree: Dict[str, int]
    outdegree: Dict[str, int]
    parents: Dict[str, List[str]]
    children: Dict[str, List[str]]
    order: Dict[str, int]
    roots: List[str]
    leaves: List[str]
    depth: int
    critical_path: List[str]
    join_nodes: List[str]
    fan_out_nodes: List[str]
    reused_nodes: List[str]
    late_edges: List[Tuple[str, str]]
    reference_distances: List[int]
    n_parallel_branches: int
    n_type_transitions: int
    value_kinds: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_roots": len(self.roots),
            "n_leaves": len(self.leaves),
            "depth": self.depth,
            "critical_path": list(self.critical_path),
            "n_join_nodes": len(self.join_nodes),
            "n_multi_parent_nodes": sum(1 for v in self.indegree.values() if v >= 2),
            "n_fan_out_nodes": len(self.fan_out_nodes),
            "n_reused_outputs": len(self.reused_nodes),
            "n_late_edges": len(self.late_edges),
            "max_indegree": max(self.indegree.values()) if self.indegree else 0,
            "max_outdegree": max(self.outdegree.values()) if self.outdegree else 0,
            "reference_distances": list(self.reference_distances),
            "max_reference_distance": (max(self.reference_distances)
                                       if self.reference_distances else 0),
            "mean_reference_distance": (
                round(sum(self.reference_distances)
                      / len(self.reference_distances), 4)
                if self.reference_distances else 0.0),
            "n_parallel_branches": self.n_parallel_branches,
            "n_type_transitions": self.n_type_transitions,
        }


def features(prog: Program, value_kinds: Dict[str, str] | None = None
             ) -> GraphFeatures:
    ids = [nd.node_id for nd in prog.nodes]
    order = {nid: i for i, nid in enumerate(ids)}
    parents: Dict[str, List[str]] = {nid: [] for nid in ids}
    children: Dict[str, List[str]] = {nid: [] for nid in ids}
    edges: List[Tuple[str, str]] = []
    for nd in prog.nodes:
        for src in dict.fromkeys(nd.refs()):
            parents[nd.node_id].append(src)
            children[src].append(nd.node_id)
            edges.append((src, nd.node_id))
    indegree = {nid: len(parents[nid]) for nid in ids}
    outdegree = {nid: len(children[nid]) for nid in ids}
    roots = [nid for nid in ids if indegree[nid] == 0]
    leaves = [nid for nid in ids if outdegree[nid] == 0]

    # longest path (nodes are already topologically ordered)
    best: Dict[str, int] = {}
    prev: Dict[str, str | None] = {}
    for nid in ids:
        if not parents[nid]:
            best[nid], prev[nid] = 1, None
            continue
        cand = max(((best[p], p) for p in parents[nid]),
                   key=lambda t: (t[0], -order[t[1]]))
        best[nid], prev[nid] = cand[0] + 1, cand[1]
    end = max(ids, key=lambda nid: (best[nid], -order[nid]))
    path: List[str] = []
    cur: str | None = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    distances = [order[dst] - order[src] for src, dst in edges]
    threshold = late_threshold_for(len(ids))
    late = [(s, d) for s, d in edges if order[d] - order[s] >= threshold]

    kinds = value_kinds or {}
    transitions = 0
    for a, b in zip(path, path[1:]):
        ka, kb = kinds.get(a, "unknown"), kinds.get(b, "unknown")
        if ka != "unknown" and kb != "unknown" and ka != kb:
            transitions += 1

    return GraphFeatures(
        n_nodes=len(ids), n_edges=len(edges), indegree=indegree,
        outdegree=outdegree, parents=parents, children=children, order=order,
        roots=roots, leaves=leaves, depth=best[end], critical_path=path,
        join_nodes=[nid for nid in ids if indegree[nid] >= 2],
        fan_out_nodes=[nid for nid in ids if outdegree[nid] >= 2],
        reused_nodes=[nid for nid in ids if outdegree[nid] >= 2],
        late_edges=late, reference_distances=distances,
        n_parallel_branches=max(1, len(roots)), n_type_transitions=transitions,
        value_kinds=dict(kinds))


def _descendants(f: GraphFeatures, start: str) -> Set[str]:
    seen: Set[str] = set()
    stack = list(f.children[start])
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        stack.extend(f.children[nid])
    return seen


def _ancestors(f: GraphFeatures, start: str) -> Set[str]:
    seen: Set[str] = set()
    stack = list(f.parents[start])
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        stack.extend(f.parents[nid])
    return seen


def is_aggregator(prog: Program, node_id: str, f: GraphFeatures) -> bool:
    """A node that condenses several values into one.

    Either it structurally merges (two or more distinct producers) or it
    collapses a collection: it consumes one, as a literal or as a parent's
    output, and emits something that is not a collection. Decided from the
    program and the values it produced, never from a workflow label.

    The independent audit re-derives this same invariant from the exported
    observations, so the two sides can only agree on the aggregation patterns if
    they agree on where a collection may come from.
    """
    if f.indegree[node_id] >= 2:
        return True
    nd = prog.node(node_id)
    if f.value_kinds.get(node_id) in COLLECTION_KINDS:
        return False
    if any(isinstance(v, (list, tuple, dict)) for v in nd.literals().values()):
        return True
    if any(f.value_kinds.get(p) in COLLECTION_KINDS for p in f.parents[node_id]):
        return True
    if f.value_kinds:
        return False
    # no observations to consult: fall back to the declared signature
    ops = build_ops()
    op = ops[nd.op]
    return (any(p.sem in st.COLLECTIONS for p in op.params)
            and (op.out_sem == "@preserve" or op.out_sem not in st.COLLECTIONS))


def satisfied_patterns(prog: Program,
                       value_kinds: Dict[str, str] | None = None,
                       f: GraphFeatures | None = None) -> Set[str]:
    """The set of the 15 pattern invariants the actual graph satisfies."""
    f = f or features(prog, value_kinds)
    ids = [nd.node_id for nd in prog.nodes]
    out: Set[str] = set()

    if (len(f.roots) == 1 and len(f.leaves) == 1 and f.n_edges == f.n_nodes - 1
            and all(f.indegree[n] == 1 for n in ids if n not in f.roots)
            and all(f.outdegree[n] == 1 for n in ids if n not in f.leaves)):
        out.add("LINEAR_CHAIN")

    if len(f.join_nodes) == 1 and max(f.indegree.values(), default=0) < 3:
        out.add("FAN_IN_SINGLE")

    if max(f.indegree.values(), default=0) >= 3 or len(f.join_nodes) >= 2:
        out.add("FAN_IN_MULTIPLE")

    if f.fan_out_nodes:
        out.add("FAN_OUT")

    for s in f.fan_out_nodes:
        kids = f.children[s]
        merged = False
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                a, b = kids[i], kids[j]
                da, db = _descendants(f, a) | {a}, _descendants(f, b) | {b}
                if (da & db) - {s}:
                    merged = True
                    break
            if merged:
                break
        if merged:
            out.add("DIAMOND")
            break

    if len(f.roots) >= 2:
        out.add("MIXED_INDEPENDENT_DEPENDENT" if f.n_edges >= 1 else "")
        for nid in ids:
            anc = _ancestors(f, nid)
            if len([r for r in f.roots if r in anc]) >= 2:
                out.add("PARALLEL_THEN_MERGE")
                break
    out.discard("")

    for nid in ids:
        if f.outdegree[nid] >= 2 and any(
                f.order[c] - f.order[nid] >= 2 for c in f.children[nid]):
            out.add("REUSE_EARLY_OUTPUT")
            break

    if f.late_edges:
        out.add("LATE_REFERENCE")

    if len(f.join_nodes) >= 2:
        out.add("MULTI_JOIN")

    aggregators = [nid for nid in ids if is_aggregator(prog, nid, f)]
    for a in aggregators:
        downstream = _descendants(f, a)
        if any(b in downstream and b != a for b in aggregators):
            out.add("TWO_STAGE_AGGREGATION")
            break
    for a in aggregators:
        if f.indegree[a] >= 2 and any(p in aggregators for p in f.parents[a]):
            out.add("NESTED_AGGREGATION")
            break

    if len(f.join_nodes) >= 2 and f.roots:
        js = sorted(f.order[j] for j in f.join_nodes)
        if any(js[0] < f.order[r] < js[-1] for r in f.roots):
            out.add("ALTERNATING_BRANCH_CHAIN")

    provenance: Dict[str, Set[Tuple[Any, ...]]] = {}
    for nd in prog.nodes:
        key = (tuple(sorted(f.order[p] for p in f.parents[nd.node_id])),
               tuple(sorted((k, repr(v)) for k, v in nd.literals().items())))
        provenance.setdefault(nd.op, set()).add(key)
    if any(len(v) >= 2 for v in provenance.values()):
        out.add("REPEATED_PRIMITIVE")

    if f.n_type_transitions >= 2:
        out.add("TYPE_TRANSITION_CHAIN")

    return out


def primary_pattern(satisfied: Set[str]) -> str:
    for name in PATTERN_PRIORITY:
        if name in satisfied:
            return name
    return "UNCLASSIFIED"


def classify(prog: Program, value_kinds: Dict[str, str] | None = None
             ) -> Dict[str, Any]:
    f = features(prog, value_kinds)
    sat = satisfied_patterns(prog, value_kinds, f)
    return {
        "actual_patterns": sorted(sat),
        "actual_primary_pattern": primary_pattern(sat),
        "graph_features": f.as_dict(),
    }
