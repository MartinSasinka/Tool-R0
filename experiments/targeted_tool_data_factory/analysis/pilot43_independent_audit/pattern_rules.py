"""Independent structural-pattern classification.

The 15 invariants below are re-implemented from scratch over a reconstructed
:class:`~.graph_recon.Graph` plus the per-node output value kinds. No producer
side classifier is imported, so a mismatch between the label declared in the
export and the label computed here is real evidence of a mislabelled dataset.

Only the standard library is used.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence, Set, Tuple

from .graph_recon import Graph, literal_arguments

#: Patterns that cannot be decided when node output kinds are unknown.
KIND_DEPENDENT_PATTERNS: Tuple[str, ...] = ("TYPE_TRANSITION_CHAIN",)

#: Deterministic priority order used to pick a single primary label.
#: Most specific / most structurally informative patterns come first so that,
#: for example, a nested aggregation is not reported merely as ``FAN_OUT``.
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

#: Every pattern this module can decide.
ALL_PATTERNS: Tuple[str, ...] = tuple(sorted(PATTERN_PRIORITY))

UNCLASSIFIED = "UNCLASSIFIED"
UNKNOWN_KIND = "unknown"


def VALUE_KIND(v: Any) -> str:
    """Classify a JSON value into a coarse kind.

    ``bool`` is checked BEFORE ``int`` because Python booleans are integers and
    a boolean answer must never be counted as a numeric one.
    """
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "string"
    if isinstance(v, (list, tuple)):
        return "list"
    if isinstance(v, dict):
        return "object"
    return "string"


def late_threshold_for(n: int) -> int:
    """Reference distance at which an edge counts as a LATE_REFERENCE.

    The default threshold is 3 nodes. Longer programs have more room for
    incidental distance, so from 8 nodes onwards the threshold rises to 4; this
    keeps the invariant a statement about deliberate long-range reuse rather
    than a side effect of program length.
    """
    return 4 if n >= 8 else 3


COLLECTION_KINDS: Tuple[str, ...] = ("list", "object")


def _is_aggregation_node(graph: Graph, i: int, node_value_kinds: Sequence[str]) -> bool:
    """True when node ``i`` aggregates several values into one.

    A node aggregates if it merges at least two upstream results (indegree
    >= 2), or if it collapses a collection: it consumes one -- as a literal
    argument or as the output of a parent -- and produces something that is not
    a collection. Where the collection came from is irrelevant to the shape of
    the computation, so a reduction over a parent's list counts exactly like a
    reduction over a written-out list.
    """
    if len(graph.parents[i]) >= 2:
        return True
    kind = node_value_kinds[i] if i < len(node_value_kinds) else UNKNOWN_KIND
    if kind in COLLECTION_KINDS:
        return False
    args = graph.arguments[i] if i < len(graph.arguments) else {}
    if any(isinstance(v, (list, tuple, dict)) for v in args.values()):
        return True
    return any(node_value_kinds[p] in COLLECTION_KINDS
               for p in graph.parents[i] if p < len(node_value_kinds))


def _provenance_key(graph: Graph, i: int) -> str:
    """Fingerprint of a node's inputs: sorted parents plus sorted literals."""
    parents = sorted(graph.parents[i])
    args = graph.arguments[i] if i < len(graph.arguments) else {}
    literals = sorted(
        json.dumps(v, sort_keys=True, default=str) for v in literal_arguments(args)
    )
    return json.dumps([parents, literals], sort_keys=True)


def satisfied_patterns(
    graph: Graph,
    node_value_kinds: Sequence[str],
    late_threshold: Optional[int] = None,
) -> Set[str]:
    """Return every structural pattern the graph satisfies.

    Args:
        graph: reconstructed dependency DAG.
        node_value_kinds: output value kind per node (see :func:`VALUE_KIND`);
            use ``"unknown"`` where the exported record does not reveal it.
        late_threshold: override for the LATE_REFERENCE distance; defaults to
            :func:`late_threshold_for`.

    Invariants:
        LINEAR_CHAIN
            Exactly one root and one leaf, ``n_edges == n - 1``, every non-root
            node has indegree 1 and every non-leaf node has outdegree 1.
        FAN_IN_SINGLE
            Exactly one join node (indegree >= 2) and no node with indegree >= 3.
        FAN_IN_MULTIPLE
            Some node has indegree >= 3, or there are at least two join nodes.
        FAN_OUT
            Some node has outdegree >= 2.
        DIAMOND
            A source has >= 2 distinct children from both of which some common
            node (other than the source itself) is reachable.
        PARALLEL_THEN_MERGE
            At least two roots, and some node is reachable from >= 2 of them.
        REUSE_EARLY_OUTPUT
            A node has >= 2 consumers and at least one consumer sits at least
            two positions later in call order.
        LATE_REFERENCE
            Some edge spans at least ``late_threshold`` positions.
        TWO_STAGE_AGGREGATION
            An aggregation node transitively feeds a different aggregation node.
        MULTI_JOIN
            At least two distinct join nodes.
        ALTERNATING_BRANCH_CHAIN
            At least two join nodes, and a fresh independent input (a root) is
            introduced in call order strictly between two of them.
        MIXED_INDEPENDENT_DEPENDENT
            At least two roots and at least one edge.
        REPEATED_PRIMITIVE
            The same tool name occurs >= 2 times with distinct input provenance
            (differing sorted parent indices or literal arguments).
        TYPE_TRANSITION_CHAIN
            At least two positions along the critical path where consecutive
            node output kinds differ. Undecidable when kinds are unknown.
        NESTED_AGGREGATION
            An aggregation node with indegree >= 2 whose parent set contains at
            least one aggregation node.
    """
    n = graph.n
    out: Set[str] = set()
    if n == 0:
        return out

    kinds = list(node_value_kinds) + [UNKNOWN_KIND] * max(0, n - len(node_value_kinds))
    indeg = graph.indegrees()
    outdeg = graph.outdegrees()
    roots = graph.roots()
    leaves = graph.leaves()
    n_edges = len(graph.edges)
    join_nodes = [i for i in range(n) if indeg[i] >= 2]
    desc = graph.descendant_sets()
    threshold = late_threshold_for(n) if late_threshold is None else late_threshold

    if (
        len(roots) == 1
        and len(leaves) == 1
        and n_edges == n - 1
        and all(indeg[i] == 1 for i in range(n) if i not in roots)
        and all(outdeg[i] == 1 for i in range(n) if i not in leaves)
    ):
        out.add("LINEAR_CHAIN")

    if len(join_nodes) == 1 and not any(indeg[i] >= 3 for i in range(n)):
        out.add("FAN_IN_SINGLE")

    if any(indeg[i] >= 3 for i in range(n)) or len(join_nodes) >= 2:
        out.add("FAN_IN_MULTIPLE")

    if any(outdeg[i] >= 2 for i in range(n)):
        out.add("FAN_OUT")

    if len(join_nodes) >= 2:
        out.add("MULTI_JOIN")

    for s in range(n):
        kids = graph.children[s]
        if len(kids) < 2:
            continue
        found = False
        for a in range(len(kids)):
            for b in range(a + 1, len(kids)):
                c1, c2 = kids[a], kids[b]
                common = (desc[c1] | {c1}) & (desc[c2] | {c2})
                if any(d != s for d in common):
                    found = True
                    break
            if found:
                break
        if found:
            out.add("DIAMOND")
            break

    if len(roots) >= 2:
        reach_count = [0] * n
        for r in roots:
            for d in desc[r]:
                reach_count[d] += 1
        if any(c >= 2 for c in reach_count):
            out.add("PARALLEL_THEN_MERGE")
        if n_edges >= 1:
            out.add("MIXED_INDEPENDENT_DEPENDENT")

    for p in range(n):
        kids = graph.children[p]
        if len(kids) >= 2 and any(c - p >= 2 for c in kids):
            out.add("REUSE_EARLY_OUTPUT")
            break

    if any(dst - src >= threshold for src, dst in graph.edges):
        out.add("LATE_REFERENCE")

    agg = [_is_aggregation_node(graph, i, kinds) for i in range(n)]
    if any(
        agg[a] and any(agg[b] for b in desc[a] if b != a) for a in range(n)
    ):
        out.add("TWO_STAGE_AGGREGATION")

    if any(
        agg[i] and indeg[i] >= 2 and any(agg[p] for p in graph.parents[i])
        for i in range(n)
    ):
        out.add("NESTED_AGGREGATION")

    if len(join_nodes) >= 2:
        j_first, j_last = min(join_nodes), max(join_nodes)
        if any(j_first < r < j_last for r in roots):
            out.add("ALTERNATING_BRANCH_CHAIN")

    by_name: Dict[str, Set[str]] = {}
    counts: Dict[str, int] = {}
    for i in range(n):
        name = graph.names[i]
        counts[name] = counts.get(name, 0) + 1
        by_name.setdefault(name, set()).add(_provenance_key(graph, i))
    if any(counts[name] >= 2 and len(keys) >= 2 for name, keys in by_name.items()):
        out.add("REPEATED_PRIMITIVE")

    path = graph.critical_path()
    transitions = sum(
        1 for a, b in zip(path, path[1:]) if kinds[a] != kinds[b]
    )
    if transitions >= 2:
        out.add("TYPE_TRANSITION_CHAIN")

    return out


def undecidable_patterns(node_value_kinds: Sequence[str]) -> Set[str]:
    """Patterns that must be reported as undecidable, not as False.

    When the export does not carry per-node output values, kind-dependent
    invariants cannot be evaluated; reporting them as ``False`` would be a
    silent false negative, so they are reported as undecidable instead.
    """
    if any(k == UNKNOWN_KIND for k in node_value_kinds):
        return set(KIND_DEPENDENT_PATTERNS)
    return set()


def primary_pattern(satisfied: Set[str]) -> str:
    """Pick a single primary label from a satisfied set, by fixed priority."""
    for name in PATTERN_PRIORITY:
        if name in satisfied:
            return name
    return UNCLASSIFIED
