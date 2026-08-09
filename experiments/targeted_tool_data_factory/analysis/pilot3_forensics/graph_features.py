"""Gold-program DAG / topology features."""
from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .io import short_hash
from .statistics import call_bucket, counter_top_share, effective_n, shannon_entropy

# Inventory of observed / supported reference formats.
REF_PATTERNS = [
    re.compile(r"^\$var_?(\d+)\.([A-Za-z0-9_]+)\$$"),  # $var1.output_0$ / $var_1.result$
    re.compile(r"^\$var_?(\d+)\$$"),  # bare $var1$
    re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z0-9_]+)\$$"),  # $label.key$
    re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)\$$"),
]


def parse_reference(value: Any) -> Optional[Dict[str, str]]:
    if not isinstance(value, str):
        return None
    s = value.strip()
    for i, rx in enumerate(REF_PATTERNS):
        m = rx.match(s)
        if not m:
            continue
        if i <= 1:
            return {
                "raw": s,
                "kind": "var_index",
                "var_num": m.group(1),
                "label_norm": f"var{m.group(1)}",
                "output_key": m.group(2) if m.lastindex and m.lastindex >= 2 else "",
                "pattern_id": i,
            }
        return {
            "raw": s,
            "kind": "named_label",
            "var_num": "",
            "label_norm": m.group(1).replace("_", ""),
            "output_key": m.group(2) if m.lastindex and m.lastindex >= 2 else "",
            "pattern_id": i,
        }
    return None


def is_reference(value: Any) -> bool:
    return parse_reference(value) is not None


def label_norm(label: Any, fallback_idx: int) -> str:
    s = str(label or f"$var{fallback_idx + 1}")
    s = s.strip().strip("$")
    s = s.replace("_", "")
    # var1 / var_1 -> var1
    m = re.match(r"var(\d+)$", s, flags=re.I)
    if m:
        return f"var{m.group(1)}"
    return s


def iter_arg_values(args: Any) -> Iterable[Any]:
    if isinstance(args, dict):
        for v in args.values():
            yield from iter_arg_values(v)
    elif isinstance(args, list):
        for v in args:
            yield from iter_arg_values(v)
    else:
        yield args


def build_dag(calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build dependency DAG from gold/pred calls with reference args.

    Edge direction: parent -> child (producer -> consumer).
    """
    n = len(calls)
    label_to_idx: Dict[str, int] = {}
    for i, c in enumerate(calls):
        label_to_idx[label_norm(c.get("label"), i)] = i

    parents: Dict[int, List[int]] = defaultdict(list)
    ref_edges_meta: List[Dict[str, Any]] = []
    ref_formats: Counter = Counter()

    for i, c in enumerate(calls):
        args = c.get("arguments") or {}
        for v in iter_arg_values(args):
            ref = parse_reference(v)
            if not ref:
                continue
            ref_formats[ref["pattern_id"]] += 1
            src = None
            if ref["kind"] == "var_index" and ref["var_num"]:
                key = f"var{ref['var_num']}"
                src = label_to_idx.get(key)
            else:
                src = label_to_idx.get(ref["label_norm"])
            if src is None:
                continue
            parents[i].append(src)
            ref_edges_meta.append({
                "src": src,
                "dst": i,
                "output_key": ref["output_key"],
                "raw": ref["raw"],
            })

    parents = {k: sorted(set(v)) for k, v in parents.items()}
    children: Dict[int, List[int]] = defaultdict(list)
    for child, ps in parents.items():
        for p in ps:
            children[p].append(child)
    children = {k: sorted(set(v)) for k, v in children.items()}
    edge_list = sorted((p, c) for c, ps in parents.items() for p in ps)

    return {
        "n": n,
        "parents": parents,
        "children": children,
        "edges": edge_list,
        "ref_edges_meta": ref_edges_meta,
        "ref_format_counts": dict(ref_formats),
    }


def _longest_path(n: int, children: Dict[int, List[int]]) -> int:
    memo: Dict[int, int] = {}

    def dfs(u: int) -> int:
        if u in memo:
            return memo[u]
        ch = children.get(u, [])
        memo[u] = 1 + (max((dfs(v) for v in ch), default=0))
        return memo[u]

    return max((dfs(i) for i in range(n)), default=0)


def _weakly_connected_components(n: int, edges: Sequence[Tuple[int, int]]) -> int:
    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen = set()
    comps = 0
    for i in range(n):
        if i in seen:
            continue
        comps += 1
        q = deque([i])
        seen.add(i)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
    return comps


def topology_hash(calls: Sequence[Dict[str, Any]]) -> str:
    """Canonical topology hash independent of tool names, labels, constants.

    Representation: number of nodes + sorted parent→child edges on positional
    indices (call order). This distinguishes A→B→C vs fan-in vs fan-out for
    fixed call order programs. Limitation: isomorphic DAGs with different
    node orderings get different hashes (call order is part of the program
    serialization used in NESTFUL/factory data).
    """
    dag = build_dag(calls)
    payload = {"n": dag["n"], "e": dag["edges"]}
    return "topo_" + short_hash(payload)


def graph_features(calls: Sequence[Dict[str, Any]], *, sample_id: str = "", source: str = "") -> Dict[str, Any]:
    dag = build_dag(calls)
    n = dag["n"]
    edges = dag["edges"]
    parents = dag["parents"]
    children = dag["children"]
    indeg = [len(parents.get(i, [])) for i in range(n)]
    outdeg = [len(children.get(i, [])) for i in range(n)]
    n_edges = len(edges)
    max_indeg = max(indeg) if indeg else 0
    max_outdeg = max(outdeg) if outdeg else 0
    fan_in_nodes = sum(1 for d in indeg if d >= 2)
    fan_out_nodes = sum(1 for d in outdeg if d >= 2)
    reused_outputs = sum(1 for d in outdeg if d >= 2)
    roots = sum(1 for d in indeg if d == 0)
    leaves = sum(1 for d in outdeg if d == 0)

    n_ref_args = 0
    n_args = 0
    const_only_calls = 0
    for c in calls:
        args = list(iter_arg_values(c.get("arguments") or {}))
        if not args:
            const_only_calls += 1
            continue
        refs = sum(1 for v in args if is_reference(v))
        n_ref_args += refs
        n_args += len(args)
        if refs == 0:
            const_only_calls += 1

    ref_density = (n_ref_args / n_args) if n_args else 0.0
    max_edges = n * (n - 1) / 2 if n > 1 else 1.0
    edge_density = n_edges / max_edges if max_edges else 0.0
    # linearity: chain-like if edges==n-1 and max indeg/outdeg <=1
    linear = float(n_edges == max(0, n - 1) and max_indeg <= 1 and max_outdeg <= 1 and n > 0)
    branching = float(fan_out_nodes + fan_in_nodes) / n if n else 0.0
    agg_arity = float(max_indeg)

    depth = _longest_path(n, children)
    wcc = _weakly_connected_components(n, edges)

    # motif-like label
    if n == 0:
        motif = "empty"
    elif not edges:
        motif = "independent"
    elif linear >= 1.0:
        motif = "linear"
    elif fan_in_nodes and not fan_out_nodes:
        motif = "fan_in"
    elif fan_out_nodes and not fan_in_nodes:
        motif = "fan_out"
    elif max_indeg >= 3:
        motif = "branch_aggregate"
    else:
        motif = "mixed"

    return {
        "sample_id": sample_id,
        "source": source,
        "n_nodes": n,
        "n_edges": n_edges,
        "depth": depth,
        "critical_path_length": depth,
        "n_roots": roots,
        "n_leaves": leaves,
        "weakly_connected_components": wcc,
        "indegree_hist": dict(Counter(indeg)),
        "outdegree_hist": dict(Counter(outdeg)),
        "max_indegree": max_indeg,
        "max_outdegree": max_outdeg,
        "n_fan_in_nodes": fan_in_nodes,
        "n_fan_out_nodes": fan_out_nodes,
        "n_reused_outputs": reused_outputs,
        "n_constant_only_calls": const_only_calls,
        "n_calls_with_reference": n - const_only_calls,
        "reference_density": round(ref_density, 6),
        "edge_density": round(edge_density, 6),
        "linearity_score": linear,
        "branching_score": round(branching, 6),
        "aggregation_arity": agg_arity,
        "topology_hash": topology_hash(calls),
        "motif": motif,
        "call_bucket": call_bucket(n),
        "ref_format_counts": dag["ref_format_counts"],
        "tool_names": [str(c.get("name") or "") for c in calls],
    }


def inventory_reference_formats(rows_calls: Iterable[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    counts: Counter = Counter()
    examples: Dict[str, str] = {}
    for calls in rows_calls:
        for c in calls:
            for v in iter_arg_values(c.get("arguments") or {}):
                if not isinstance(v, str):
                    continue
                ref = parse_reference(v)
                if ref:
                    key = f"pattern_{ref['pattern_id']}"
                    counts[key] += 1
                    examples.setdefault(key, ref["raw"])
                elif "$" in v:
                    counts["unparsed_dollar_string"] += 1
                    examples.setdefault("unparsed_dollar_string", v)
    return {"counts": dict(counts), "examples": examples}


def summarize_topology_distribution(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    counter = Counter(f["topology_hash"] for f in features)
    n = sum(counter.values()) or 1
    return {
        "n_programs": len(features),
        "n_unique_topology_hashes": len(counter),
        "top1_share": counter_top_share(counter, 1),
        "top5_share": counter_top_share(counter, 5),
        "top10_share": counter_top_share(counter, 10),
        "shannon_entropy": shannon_entropy(counter),
        "effective_n_topologies": effective_n(counter),
        "singleton_rate": sum(1 for c in counter.values() if c == 1) / max(1, len(counter)),
        "top_topologies": [
            {"topology_hash": h, "count": c, "share": c / n}
            for h, c in counter.most_common(20)
        ],
    }


def topology_coverage(train_feats: List[Dict[str, Any]], diag_feats: List[Dict[str, Any]]) -> Dict[str, Any]:
    train_set = {f["topology_hash"] for f in train_feats}
    diag_hashes = [f["topology_hash"] for f in diag_feats]
    covered = sum(1 for h in diag_hashes if h in train_set)
    n = len(diag_hashes) or 1
    return {
        "train_unique": len(train_set),
        "diagnostic_unique": len(set(diag_hashes)),
        "diagnostic_exact_topology_coverage_rate": covered / n,
        "diagnostic_unseen_topology_rate": 1.0 - covered / n,
        "n_diagnostic_covered": covered,
        "n_diagnostic": len(diag_hashes),
    }
