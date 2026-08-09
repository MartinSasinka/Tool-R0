"""Pilot4.3 pattern rules applied to raw NESTFUL dev programs.

Failure mode prevented: an apples-to-oranges pattern comparison. The dev-200
structural distribution stored in ``outputs/profiles/target_profile_v2.json``
was produced by the older ``profile_v2`` motif classifier (``linear`` /
``fan_in`` / ``multi_join`` / ...), which shares no vocabulary and no thresholds
with the Pilot4.3 15-invariant classifier. Measuring Pilot4.3 output with one
classifier and its target with another guarantees a meaningless deviation.

:mod:`.patterns` cannot be used directly on dev data because it needs a
:class:`~.program.Program`, and a Program requires every node's ``op`` to exist
in :func:`~.ops.build_ops`; NESTFUL dev calls name arbitrary Python-library
functions that the Pilot4.3 op table does not contain. This module therefore
mirrors ``patterns.py`` rule for rule over a reconstructed edge list, and
imports ``PATTERN_PRIORITY``, ``primary_pattern`` and ``late_threshold_for``
from ``patterns.py`` itself so those parts cannot drift. ``patterns.py`` is not
modified.

Two rules need an op table that dev data does not have, and both are resolved
from the dev tool schema instead. The substitution is exact about what it
replaces:

* ``patterns.is_aggregator`` asks whether an op consumes a collection
  (parameter semantic type in ``semtypes.COLLECTIONS``) and returns a scalar.
  Here a parameter counts as a collection when the dev tool schema declares it
  ``array`` or ``object``, and the output counts as scalar when the declared
  output type is neither. The literal-collection fallback (a literal list or
  dict of at least three elements) is kept verbatim.
* ``patterns.features`` takes per-node output kinds from executed observations.
  Dev programs are not executed here, so node output kinds are read from the
  tool's declared ``output_parameters`` type. Nodes whose type cannot be
  resolved stay ``"unknown"``, which is exactly the value ``patterns.py``
  already skips when counting type transitions.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

from .patterns import PATTERN_PRIORITY, late_threshold_for, primary_pattern

__all__ = [
    "CLASSIFIER_ID",
    "DevGraph",
    "PATTERN_PRIORITY",
    "classify_row",
    "dev_features",
    "primary_pattern",
    "reconstruct",
    "satisfied_patterns",
]

#: Stamped into the emitted profile so a consumer can tell which classifier
#: produced the pattern distribution.
CLASSIFIER_ID = "pilot43.patterns.v1 (mirrored over dev edge lists)"

#: Same reference grammar the dev profile builder used, so the reconstructed
#: edges are identical and only the classifier differs.
_REF_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z0-9_]+))?\$")

#: JSON-schema type -> Pilot4.3 value kind (see ``semtypes.value_kind``).
_SCHEMA_KIND: Dict[str, str] = {
    "integer": "integer",
    "number": "float",
    "float": "float",
    "string": "string",
    "boolean": "boolean",
    "array": "list",
    "object": "object",
}
_COLLECTION_KINDS = frozenset({"list", "object"})
UNKNOWN_KIND = "unknown"


def _label_key(label: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(label).lower())


def _refs_in(value: Any) -> List[str]:
    out: List[str] = []
    if isinstance(value, str):
        out.extend(_label_key(m.group(1)) for m in _REF_RE.finditer(value))
    elif isinstance(value, list):
        for item in value:
            out.extend(_refs_in(item))
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(_refs_in(item))
    return out


@dataclass
class DevGraph:
    """A reconstructed dev program: edges plus the schema facts the rules need."""

    n: int
    names: List[str]
    edges: List[Tuple[int, int]]
    parents: Dict[int, List[int]]
    children: Dict[int, List[int]]
    arguments: List[Dict[str, Any]]
    out_kinds: List[str]
    consumes_collection: List[bool]
    scalar_out: List[bool]
    unresolved_tools: List[str] = field(default_factory=list)

    def indegree(self, i: int) -> int:
        return len(self.parents[i])

    def outdegree(self, i: int) -> int:
        return len(self.children[i])

    def literals(self, i: int) -> Dict[str, Any]:
        return {k: v for k, v in self.arguments[i].items() if not _refs_in(v)}


def _tool_facts(tool: Dict[str, Any]) -> Tuple[str, bool, bool]:
    """(output kind, consumes a declared collection, produces a scalar)."""
    outs = tool.get("output_parameters")
    kind = UNKNOWN_KIND
    if isinstance(outs, dict) and outs:
        first = next(iter(outs.values()))
        if isinstance(first, dict):
            kind = _SCHEMA_KIND.get(str(first.get("type", "")), UNKNOWN_KIND)
    params = tool.get("parameters")
    collection_param = False
    if isinstance(params, dict):
        for spec in params.values():
            if isinstance(spec, dict) and str(spec.get("type", "")) in (
                    "array", "object"):
                collection_param = True
                break
    return kind, collection_param, kind not in _COLLECTION_KINDS


def reconstruct(calls: Sequence[Dict[str, Any]],
                tools: Sequence[Dict[str, Any]] = ()) -> DevGraph:
    """Rebuild the dependency DAG of one dev program from its own references.

    Edges come only from ``$label.field$`` references inside call arguments, so
    nothing producer-side is trusted. Backward and self references are dropped
    rather than raising, because a malformed dev row must not abort the profile
    build; the affected node simply keeps a lower indegree.
    """
    labels = [_label_key((c or {}).get("label") or f"var{i + 1}")
              for i, c in enumerate(calls)]
    pos = {lab: i for i, lab in enumerate(labels)}
    edge_set: Set[Tuple[int, int]] = set()
    arguments: List[Dict[str, Any]] = []
    for i, call in enumerate(calls):
        args = (call or {}).get("arguments")
        args = args if isinstance(args, dict) else {}
        arguments.append(args)
        for ref in _refs_in(args):
            j = pos.get(ref)
            if j is not None and j < i:
                edge_set.add((j, i))
    edges = sorted(edge_set)
    parents: Dict[int, List[int]] = {i: [] for i in range(len(calls))}
    children: Dict[int, List[int]] = {i: [] for i in range(len(calls))}
    for src, dst in edges:
        parents[dst].append(src)
        children[src].append(dst)

    by_name = {str(t.get("name")): t for t in tools if isinstance(t, dict)}
    names = [str((c or {}).get("name") or "") for c in calls]
    out_kinds: List[str] = []
    consumes: List[bool] = []
    scalar: List[bool] = []
    unresolved: List[str] = []
    for i, name in enumerate(names):
        tool = by_name.get(name)
        if tool is None:
            unresolved.append(name)
            out_kinds.append(UNKNOWN_KIND)
            # an unknown output type is not a declared collection, which is the
            # same branch patterns.py takes for a non-collection out_sem
            consumes.append(False)
            scalar.append(True)
            continue
        kind, collection_param, scalar_out = _tool_facts(tool)
        out_kinds.append(kind)
        consumes.append(collection_param)
        scalar.append(scalar_out)
    return DevGraph(n=len(calls), names=names, edges=edges, parents=parents,
                    children=children, arguments=arguments, out_kinds=out_kinds,
                    consumes_collection=consumes, scalar_out=scalar,
                    unresolved_tools=unresolved)


def _descendants(g: DevGraph, start: int) -> Set[int]:
    seen: Set[int] = set()
    stack = list(g.children[start])
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(g.children[node])
    return seen


def _ancestors(g: DevGraph, start: int) -> Set[int]:
    seen: Set[int] = set()
    stack = list(g.parents[start])
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(g.parents[node])
    return seen


def _critical_path(g: DevGraph) -> List[int]:
    """Longest path with ``patterns.features``' tie-break (latest-index parent)."""
    best: Dict[int, int] = {}
    prev: Dict[int, int | None] = {}
    for i in range(g.n):
        if not g.parents[i]:
            best[i], prev[i] = 1, None
            continue
        depth, parent = max(((best[p], p) for p in g.parents[i]),
                            key=lambda t: (t[0], -t[1]))
        best[i], prev[i] = depth + 1, parent
    if not best:
        return []
    end = max(range(g.n), key=lambda i: (best[i], -i))
    path: List[int] = []
    cur: int | None = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def is_aggregator(g: DevGraph, i: int) -> bool:
    """Mirror of :func:`patterns.is_aggregator` over dev schema facts."""
    if g.indegree(i) >= 2:
        return True
    consumes = g.consumes_collection[i]
    if not consumes:
        consumes = any(isinstance(v, (list, dict)) and len(v) >= 3
                       for v in g.literals(i).values())
    return bool(consumes and g.scalar_out[i])


def dev_features(g: DevGraph) -> Dict[str, Any]:
    """The subset of :class:`patterns.GraphFeatures` the profile reports."""
    indeg = [g.indegree(i) for i in range(g.n)]
    outdeg = [g.outdegree(i) for i in range(g.n)]
    distances = [dst - src for src, dst in g.edges]
    threshold = late_threshold_for(g.n)
    late = [(s, d) for s, d in g.edges if d - s >= threshold]
    path = _critical_path(g)
    transitions = 0
    for a, b in zip(path, path[1:]):
        ka, kb = g.out_kinds[a], g.out_kinds[b]
        if ka != UNKNOWN_KIND and kb != UNKNOWN_KIND and ka != kb:
            transitions += 1
    roots = [i for i in range(g.n) if indeg[i] == 0]
    return {
        "n_nodes": g.n,
        "n_edges": len(g.edges),
        "n_roots": len(roots),
        "n_leaves": sum(1 for d in outdeg if d == 0),
        "depth": len(path),
        "n_join_nodes": sum(1 for d in indeg if d >= 2),
        "n_fan_out_nodes": sum(1 for d in outdeg if d >= 2),
        "n_reused_outputs": sum(1 for d in outdeg if d >= 2),
        "n_late_edges": len(late),
        "max_indegree": max(indeg, default=0),
        "max_outdegree": max(outdeg, default=0),
        "max_reference_distance": max(distances, default=0),
        "mean_reference_distance": (round(sum(distances) / len(distances), 4)
                                    if distances else 0.0),
        "n_parallel_branches": max(1, len(roots)),
        "n_type_transitions": transitions,
        "n_unresolved_tools": len(g.unresolved_tools),
    }


def satisfied_patterns(g: DevGraph) -> Set[str]:
    """The 15 Pilot4.3 invariants, evaluated exactly as in ``patterns.py``."""
    if g.n == 0:
        return set()
    indeg = [g.indegree(i) for i in range(g.n)]
    outdeg = [g.outdegree(i) for i in range(g.n)]
    roots = [i for i in range(g.n) if indeg[i] == 0]
    leaves = [i for i in range(g.n) if outdeg[i] == 0]
    join_nodes = [i for i in range(g.n) if indeg[i] >= 2]
    fan_out_nodes = [i for i in range(g.n) if outdeg[i] >= 2]
    n_edges = len(g.edges)
    out: Set[str] = set()

    if (len(roots) == 1 and len(leaves) == 1 and n_edges == g.n - 1
            and all(indeg[i] == 1 for i in range(g.n) if i not in roots)
            and all(outdeg[i] == 1 for i in range(g.n) if i not in leaves)):
        out.add("LINEAR_CHAIN")

    if len(join_nodes) == 1 and max(indeg, default=0) < 3:
        out.add("FAN_IN_SINGLE")

    if max(indeg, default=0) >= 3 or len(join_nodes) >= 2:
        out.add("FAN_IN_MULTIPLE")

    if fan_out_nodes:
        out.add("FAN_OUT")

    for s in fan_out_nodes:
        kids = g.children[s]
        merged = False
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                a, b = kids[i], kids[j]
                if ((_descendants(g, a) | {a}) & (_descendants(g, b) | {b})) - {s}:
                    merged = True
                    break
            if merged:
                break
        if merged:
            out.add("DIAMOND")
            break

    if len(roots) >= 2:
        if n_edges >= 1:
            out.add("MIXED_INDEPENDENT_DEPENDENT")
        for i in range(g.n):
            anc = _ancestors(g, i)
            if len([r for r in roots if r in anc]) >= 2:
                out.add("PARALLEL_THEN_MERGE")
                break

    for i in range(g.n):
        if outdeg[i] >= 2 and any(c - i >= 2 for c in g.children[i]):
            out.add("REUSE_EARLY_OUTPUT")
            break

    threshold = late_threshold_for(g.n)
    if any(dst - src >= threshold for src, dst in g.edges):
        out.add("LATE_REFERENCE")

    if len(join_nodes) >= 2:
        out.add("MULTI_JOIN")

    aggregators = [i for i in range(g.n) if is_aggregator(g, i)]
    for a in aggregators:
        downstream = _descendants(g, a)
        if any(b in downstream and b != a for b in aggregators):
            out.add("TWO_STAGE_AGGREGATION")
            break
    for a in aggregators:
        if indeg[a] >= 2 and any(p in aggregators for p in g.parents[a]):
            out.add("NESTED_AGGREGATION")
            break

    if len(join_nodes) >= 2 and roots:
        first, last = min(join_nodes), max(join_nodes)
        if any(first < r < last for r in roots):
            out.add("ALTERNATING_BRANCH_CHAIN")

    provenance: Dict[str, Set[Tuple[Any, ...]]] = defaultdict(set)
    for i in range(g.n):
        key = (tuple(sorted(g.parents[i])),
               tuple(sorted((k, repr(v)) for k, v in g.literals(i).items())))
        provenance[g.names[i]].add(key)
    if any(len(keys) >= 2 for keys in provenance.values()):
        out.add("REPEATED_PRIMITIVE")

    path = _critical_path(g)
    transitions = 0
    for a, b in zip(path, path[1:]):
        ka, kb = g.out_kinds[a], g.out_kinds[b]
        if ka != UNKNOWN_KIND and kb != UNKNOWN_KIND and ka != kb:
            transitions += 1
    if transitions >= 2:
        out.add("TYPE_TRANSITION_CHAIN")

    return out


def classify_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Classify one raw NESTFUL dev row with the Pilot4.3 invariants."""
    calls = row.get("output") or row.get("gold_calls") or []
    tools = row.get("tools") or []
    graph = reconstruct(calls, tools)
    sat = satisfied_patterns(graph)
    feats = dev_features(graph)
    return {
        "call_count": graph.n,
        "satisfied_patterns": sorted(sat),
        "primary_pattern": primary_pattern(sat),
        "offered_tool_count": len(tools),
        "features": feats,
        "join_count": feats["n_join_nodes"],
        "fan_out_count": feats["n_fan_out_nodes"],
        "reuse_count": feats["n_reused_outputs"],
        "late_reference": feats["n_late_edges"] > 0,
    }


def pattern_counts(rows: Sequence[Dict[str, Any]]) -> Counter:
    """Number of rows satisfying each invariant (rows satisfy several)."""
    counts: Counter = Counter()
    for row in rows:
        for name in row["satisfied_patterns"]:
            counts[name] += 1
    return counts
