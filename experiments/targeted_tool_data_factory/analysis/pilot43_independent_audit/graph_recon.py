"""Independent reconstruction of gold-call dependency DAGs from exported records.

This module is deliberately self-contained: it imports only the Python standard
library. Nothing here may import producer-side code (``targeted_tool_data.graph``,
``targeted_tool_data.pilot42``, ``targeted_tool_data.pilot43``, ...), because the
whole point of the audit is to re-derive dataset structure from the exported
JSONL content and disagree with the producer when the producer is wrong.

The only structural signal we trust is the exported ``gold_calls`` list: each
call has a ``name``, an ``arguments`` mapping and a ``label`` such as ``$var_1``.
Data dependencies are recovered from reference strings embedded in arguments.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

# Tool-call references appear in four surface forms:
#   $var_3.output_0$   $var3.output_0$   $var_3$   $var3$
# The optional field group captures the referenced output key when present.
REF_RE = re.compile(r"^\$var_?(\d+)(?:\.([A-Za-z0-9_]+))?\$$")

# Labels are normalised to the canonical ``var_<n>`` spelling.
_LABEL_RE = re.compile(r"^var_?(\d+)$", re.IGNORECASE)


class ReconError(ValueError):
    """Raised when ``gold_calls`` cannot be turned into a legal DAG.

    Reconstruction fails hard (rather than repairing silently) on unknown
    reference labels, forward or self references, and duplicate labels, since
    each of those is a real dataset defect that the audit must surface.
    """


def parse_ref(s: Any) -> Optional[Tuple[str, str]]:
    """Parse a reference string into ``(normalized_label, field)``.

    ``$var3.output_0$`` and ``$var_3.output_0$`` both yield ``("var_3", "output_0")``.
    The bare forms ``$var3$`` / ``$var_3$`` yield ``("var_3", "")``.
    Anything else (including non-strings) yields ``None``.
    """
    if not isinstance(s, str):
        return None
    m = REF_RE.match(s.strip())
    if m is None:
        return None
    return f"var_{int(m.group(1))}", (m.group(2) or "")


def normalize_label(label: Any) -> str:
    """Normalise a call label such as ``$var1`` / ``var_1`` to ``var_1``.

    Labels that do not follow the ``var<n>`` convention are returned stripped of
    ``$`` and whitespace, so that non-standard labels remain distinguishable
    instead of collapsing onto each other.
    """
    s = str(label if label is not None else "").strip().lstrip("$").strip()
    m = _LABEL_RE.match(s)
    if m is not None:
        return f"var_{int(m.group(1))}"
    return s


def iter_refs(value: Any) -> Iterator[Tuple[str, str]]:
    """Yield every reference found in ``value``, walking nested lists and dicts."""
    if isinstance(value, str):
        parsed = parse_ref(value)
        if parsed is not None:
            yield parsed
    elif isinstance(value, dict):
        for key in value:
            yield from iter_refs(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_refs(item)


def literal_arguments(arguments: Any) -> List[Any]:
    """Return the non-reference (literal) argument values of one call.

    Nested containers are flattened; reference strings are dropped. Used for
    provenance fingerprinting and for numeric-realism measurements.
    """
    out: List[Any] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            if parse_ref(value) is None:
                out.append(value)
        elif isinstance(value, dict):
            for key in sorted(value):
                walk(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        else:
            out.append(value)

    walk(arguments)
    return out


@dataclass
class Graph:
    """A reconstructed gold-program DAG.

    Nodes are indexed by position in ``gold_calls``; edges always point from a
    lower index to a higher index because forward references are rejected, so
    the natural index order is a topological order.
    """

    n: int
    names: List[str]
    edges: List[Tuple[int, int]]
    parents: Dict[int, List[int]]
    children: Dict[int, List[int]]
    labels: List[str] = field(default_factory=list)
    arguments: List[Dict[str, Any]] = field(default_factory=list)

    def indegrees(self) -> List[int]:
        """Number of DISTINCT parent nodes per node."""
        return [len(self.parents[i]) for i in range(self.n)]

    def outdegrees(self) -> List[int]:
        """Number of DISTINCT child nodes per node."""
        return [len(self.children[i]) for i in range(self.n)]

    def roots(self) -> List[int]:
        """Nodes with no incoming data dependency (fresh inputs only)."""
        return [i for i in range(self.n) if not self.parents[i]]

    def leaves(self) -> List[int]:
        """Nodes whose output is not consumed by any other node."""
        return [i for i in range(self.n) if not self.children[i]]

    def descendant_sets(self) -> List[Set[int]]:
        """For each node, the set of nodes strictly reachable from it."""
        desc: List[Set[int]] = [set() for _ in range(self.n)]
        for i in range(self.n - 1, -1, -1):
            acc: Set[int] = set()
            for c in self.children[i]:
                acc.add(c)
                acc |= desc[c]
            desc[i] = acc
        return desc

    def longest_path_lengths(self) -> List[int]:
        """Length in nodes of the longest path ending at each node."""
        best = [1] * self.n
        for i in range(self.n):
            for p in self.parents[i]:
                if best[p] + 1 > best[i]:
                    best[i] = best[p] + 1
        return best

    def critical_path(self) -> List[int]:
        """One longest path, chosen deterministically.

        Tie-breaking rule: among all nodes that end a longest path, the node
        with the SMALLEST index is chosen; the path is then walked backwards,
        at each step taking the smallest-index parent that lies on a longest
        path. This yields a single reproducible path for any DAG.
        """
        if self.n == 0:
            return []
        best = self.longest_path_lengths()
        target = max(best)
        end = min(i for i in range(self.n) if best[i] == target)
        path = [end]
        cur = end
        while best[cur] > 1:
            candidates = [p for p in self.parents[cur] if best[p] == best[cur] - 1]
            cur = min(candidates)
            path.append(cur)
        path.reverse()
        return path

    def features(self) -> Dict[str, Any]:
        """Recomputed topology features.

        Definitions (all independent of any producer-side metadata):

        * ``indegree`` / ``outdegree``: counts of DISTINCT parents / children.
        * ``n_join_nodes`` / ``n_multi_parent_nodes``: nodes with indegree >= 2.
        * ``n_fan_out_nodes`` / ``n_reused_outputs``: nodes with outdegree >= 2.
        * ``n_late_edges``: edges whose index distance ``dst - src`` is >= 3.
        * ``depth``: longest path measured in nodes (a single node has depth 1).
        * ``critical_path``: see :meth:`critical_path`.
        * ``reference_distances``: ``dst - src`` for every deduplicated edge.
        * ``n_parallel_branches``: number of maximal input chains hanging off
          distinct roots. Exact rule: the smallest-index root is treated as the
          spine of the program, every OTHER root starts one additional parallel
          branch, so the value is ``(n_roots - 1) + 1 == max(n_roots, 1)`` for a
          non-empty graph and ``0`` for an empty one. A single-root DAG
          therefore always reports exactly one branch.
        * ``n_independent_roots``: number of roots (nodes with no parents).
        * ``has_cycle``: always ``False`` - forward and self references are
          rejected during reconstruction, so a reconstructed graph is acyclic
          by construction.
        """
        indeg = self.indegrees()
        outdeg = self.outdegrees()
        roots = self.roots()
        leaves = self.leaves()
        distances = [dst - src for src, dst in self.edges]
        best = self.longest_path_lengths()
        n_roots = len(roots)
        return {
            "n_nodes": self.n,
            "n_edges": len(self.edges),
            "indegree": indeg,
            "outdegree": outdeg,
            "n_roots": n_roots,
            "n_leaves": len(leaves),
            "depth": max(best) if self.n else 0,
            "critical_path": self.critical_path(),
            "n_join_nodes": sum(1 for d in indeg if d >= 2),
            "n_multi_parent_nodes": sum(1 for d in indeg if d >= 2),
            "n_fan_out_nodes": sum(1 for d in outdeg if d >= 2),
            "n_reused_outputs": sum(1 for d in outdeg if d >= 2),
            "n_late_edges": sum(1 for d in distances if d >= 3),
            "reference_distances": distances,
            "mean_reference_distance": (sum(distances) / len(distances)) if distances else 0.0,
            "max_reference_distance": max(distances) if distances else 0,
            "n_parallel_branches": max(n_roots, 1) if self.n else 0,
            "n_independent_roots": n_roots,
            "has_cycle": False,
        }


def reconstruct(gold_calls: Sequence[Dict[str, Any]]) -> Graph:
    """Rebuild the dependency DAG of a gold program from its exported calls.

    Each item of ``gold_calls`` must expose ``name``, ``arguments`` and
    ``label``. Reference labels are resolved against the calls' own labels, so
    the graph depends on nothing but the exported record.

    Raises:
        ReconError: on duplicate labels, references to unknown labels, or
            forward / self references (``src_index >= dst_index``).
    """
    calls = list(gold_calls)
    labels: List[str] = []
    label_to_index: Dict[str, int] = {}
    for idx, call in enumerate(calls):
        label = normalize_label(call.get("label"))
        if not label:
            label = f"var_{idx + 1}"
        if label in label_to_index:
            raise ReconError(f"duplicate call label {label!r} at index {idx}")
        label_to_index[label] = idx
        labels.append(label)

    names = [str(call.get("name", "")) for call in calls]
    arguments: List[Dict[str, Any]] = []
    for call in calls:
        args = call.get("arguments")
        arguments.append(args if isinstance(args, dict) else {})

    edge_set: Set[Tuple[int, int]] = set()
    for dst, args in enumerate(arguments):
        for ref_label, _field in iter_refs(args):
            if ref_label not in label_to_index:
                raise ReconError(
                    f"reference to unknown label {ref_label!r} in call index {dst}"
                )
            src = label_to_index[ref_label]
            if src >= dst:
                raise ReconError(
                    f"forward or self reference {ref_label!r} (src={src}) used by call index {dst}"
                )
            edge_set.add((src, dst))

    edges = sorted(edge_set)
    parents: Dict[int, List[int]] = {i: [] for i in range(len(calls))}
    children: Dict[int, List[int]] = {i: [] for i in range(len(calls))}
    for src, dst in edges:
        parents[dst].append(src)
        children[src].append(dst)

    return Graph(
        n=len(calls),
        names=names,
        edges=edges,
        parents=parents,
        children=children,
        labels=labels,
        arguments=arguments,
    )
