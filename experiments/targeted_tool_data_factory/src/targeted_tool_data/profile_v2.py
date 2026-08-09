"""TargetProfile v2: conditional distributions, graph features, surface features.

Pilot3 matched the target on *marginal* distributions, which let a dataset be
"on profile" while every 6-call task shared one topology. v2 conditions each
structural feature on the call-count bucket and adds the graph/surface features
that marginals hid.

PROFILE_SAFE profiles may only be built from the dev split and generic factory
metadata. Diagnostic-informed statistics are computed into a separate object
that is never used as a generation target by default.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "ttdf.target_profile.v2"

_REF_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z0-9_]+))?\$")

CALL_BUCKETS = ["2", "3", "4", "5", "6+"]

CONDITIONAL_KEYS = [
    "motif", "depth", "join_count", "fan_out_count", "reuse_count",
    "reference_density", "answer_type", "offered_tool_count", "query_mode",
    "schema_complexity",
]

GRAPH_FEATURE_KEYS = [
    "n_nodes", "n_edges", "depth", "critical_path", "n_roots", "n_leaves",
    "n_joins", "max_indegree", "n_fan_out_nodes", "max_outdegree",
    "n_reused_outputs", "n_late_references", "mean_reference_distance",
    "max_reference_distance", "n_parallel_branches", "n_type_transitions",
]

SURFACE_FEATURE_KEYS = [
    "parameter_count", "required_parameter_count", "optional_parameter_count",
    "nested_schema_depth", "output_key_family", "repeated_tool_count",
    "same_family_tool_count",
]


def call_bucket(n: int) -> str:
    if n <= 2:
        return "2"
    if n >= 6:
        return "6+"
    return str(n)


# ── dependency graph from rendered calls ──────────────────────────────────
def _label_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(label).lower())


def _refs_in_value(value: Any) -> List[str]:
    out: List[str] = []
    if isinstance(value, str):
        out.extend(_label_key(m.group(1)) for m in _REF_RE.finditer(value))
    elif isinstance(value, list):
        for v in value:
            out.extend(_refs_in_value(v))
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(_refs_in_value(v))
    return out


def build_dag(calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Node/edge view of a rendered call sequence."""
    labels = [_label_key((c or {}).get("label") or f"var{i + 1}")
              for i, c in enumerate(calls)]
    pos = {lab: i for i, lab in enumerate(labels)}
    edges: List[Tuple[int, int]] = []
    for i, c in enumerate(calls):
        for ref in _refs_in_value((c or {}).get("arguments") or {}):
            j = pos.get(ref)
            if j is not None and j != i:
                edges.append((j, i))
    return {"n": len(calls), "labels": labels, "edges": sorted(set(edges))}


def graph_features(calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    dag = build_dag(calls)
    n = dag["n"]
    edges = dag["edges"]
    indeg = Counter(b for _a, b in edges)
    outdeg = Counter(a for a, _b in edges)
    preds: Dict[int, List[int]] = defaultdict(list)
    for a, b in edges:
        preds[b].append(a)

    longest = [0] * max(n, 1)
    for i in range(n):
        if preds[i]:
            longest[i] = 1 + max(longest[p] for p in preds[i])
    depth = max(longest) if n else 0

    distances = [b - a for a, b in edges]
    late = sum(1 for d in distances if d >= 2)
    reused = sum(1 for _i, c in outdeg.items() if c >= 2)
    joins = sum(1 for _i, c in indeg.items() if c >= 2)
    fan_out = sum(1 for _i, c in outdeg.items() if c >= 2)
    roots = sum(1 for i in range(n) if indeg.get(i, 0) == 0)
    leaves = sum(1 for i in range(n) if outdeg.get(i, 0) == 0)

    types = [_out_type_of(c) for c in calls]
    transitions = sum(1 for a, b in edges if types[a] != types[b])
    parallel = _count_parallel_branches(n, edges)

    return {
        "n_nodes": n,
        "n_edges": len(edges),
        "depth": depth,
        "critical_path": depth + 1 if n else 0,
        "n_roots": roots,
        "n_leaves": leaves,
        "n_joins": joins,
        "max_indegree": max(indeg.values()) if indeg else 0,
        "n_fan_out_nodes": fan_out,
        "max_outdegree": max(outdeg.values()) if outdeg else 0,
        "n_reused_outputs": reused,
        "n_late_references": late,
        "mean_reference_distance": round(sum(distances) / len(distances), 4)
                                   if distances else 0.0,
        "max_reference_distance": max(distances) if distances else 0,
        "n_parallel_branches": parallel,
        "n_type_transitions": transitions,
    }


def _out_type_of(call: Dict[str, Any]) -> str:
    """Coarse output-kind guess from the call's own argument shapes."""
    args = (call or {}).get("arguments") or {}
    for v in args.values():
        if isinstance(v, list):
            return "array"
    return "scalar"


def _count_parallel_branches(n: int, edges: Sequence[Tuple[int, int]]) -> int:
    """Number of maximal chains that start at a root and are not merged yet."""
    if n == 0:
        return 0
    succ: Dict[int, List[int]] = defaultdict(list)
    indeg = Counter(b for _a, b in edges)
    for a, b in edges:
        succ[a].append(b)
    roots = [i for i in range(n) if indeg.get(i, 0) == 0]
    if len(roots) <= 1:
        return max(len(roots), 0)
    # branches that reach a join independently
    joins = {i for i in range(n) if indeg.get(i, 0) >= 2}
    if not joins:
        return len(roots)
    count = 0
    for r in roots:
        seen, stack = {r}, [r]
        reached = False
        while stack:
            cur = stack.pop()
            if cur in joins and cur != r:
                reached = True
                break
            for nxt in succ.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        count += 1 if reached else 1
    return count


def motif_of(feats: Dict[str, Any]) -> str:
    if feats["n_nodes"] <= 1:
        return "single"
    if feats["n_edges"] == 0:
        return "independent"
    if feats["max_indegree"] >= 3:
        return "branch_aggregate"
    if feats["n_joins"] >= 2:
        return "multi_join"
    if feats["n_joins"] == 1:
        return "fan_in"
    if feats["n_fan_out_nodes"] >= 1:
        return "fan_out"
    if feats["n_roots"] > 1:
        return "mixed"
    return "linear"


# ── surface features ──────────────────────────────────────────────────────
def _schema_depth(obj: Any, level: int = 1) -> int:
    if isinstance(obj, dict):
        if not obj:
            return level
        return max(_schema_depth(v, level + 1) for v in obj.values())
    if isinstance(obj, list) and obj:
        return max(_schema_depth(v, level + 1) for v in obj)
    return level


def _params_of(tool: Dict[str, Any]) -> Dict[str, Any]:
    params = tool.get("parameters")
    if isinstance(params, dict) and isinstance(params.get("properties"), dict):
        return params["properties"]
    if isinstance(params, dict):
        return params
    if isinstance(params, list):
        return {str(p.get("name")): p for p in params if isinstance(p, dict)}
    return {}


def _required_of(tool: Dict[str, Any]) -> List[str]:
    params = tool.get("parameters")
    if isinstance(params, dict) and isinstance(params.get("required"), list):
        return [str(x) for x in params["required"]]
    props = _params_of(tool)
    return [k for k, v in props.items()
            if not isinstance(v, dict) or v.get("required", True)]


def surface_features(tools: Sequence[Dict[str, Any]],
                     calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_name = {str(t.get("name")): t for t in tools if isinstance(t, dict)}
    used = [by_name.get(str((c or {}).get("name")), {}) for c in calls]
    used = [t for t in used if t]
    pcounts = [len(_params_of(t)) for t in used] or [0]
    reqs = [len(_required_of(t)) for t in used] or [0]
    depths = [_schema_depth(t.get("parameters") or {}) for t in used] or [1]
    out_keys: List[str] = []
    for t in used:
        outs = t.get("output_parameters") or {}
        out_keys.extend(str(k) for k in outs) if isinstance(outs, dict) else None
    names = [str((c or {}).get("name") or "") for c in calls]
    repeated = len(names) - len(set(names))
    prefixes = Counter(n.split("_")[0] for n in names if n)
    same_family = sum(v - 1 for v in prefixes.values() if v > 1)
    key_family = Counter(out_keys).most_common(1)[0][0] if out_keys else ""
    total_params = sum(pcounts)
    return {
        "parameter_count": total_params,
        "required_parameter_count": sum(reqs),
        "optional_parameter_count": max(total_params - sum(reqs), 0),
        "nested_schema_depth": max(depths),
        "output_key_family": key_family,
        "repeated_tool_count": repeated,
        "same_family_tool_count": same_family,
        "schema_complexity": round(
            (total_params / max(len(used), 1)) * (max(depths) / 3.0), 4),
    }


# ── row featurization ─────────────────────────────────────────────────────
def _answer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, str):
        return "numeric_string" if re.fullmatch(r"-?\d+(\.\d+)?", value.strip()) else "string"
    return "other"


def featurize(row: Dict[str, Any], *, with_query_mode: bool = True) -> Dict[str, Any]:
    calls = row.get("gold_calls") or row.get("output") or row.get("canonical_calls") or []
    tools = row.get("tools") or row.get("offered_tools") or []
    gf = graph_features(calls)
    sf = surface_features(tools, calls)
    n_calls = len(calls)
    n_ref_args = 0
    n_args = 0
    for c in calls:
        for v in ((c or {}).get("arguments") or {}).values():
            n_args += 1
            if _refs_in_value(v):
                n_ref_args += 1
    feats: Dict[str, Any] = {
        "call_count": n_calls,
        "call_bucket": call_bucket(n_calls),
        "motif": motif_of(gf),
        "reference_density": round(n_ref_args / max(n_args, 1), 4),
        "offered_tool_count": len(tools),
        "answer_type": _answer_type(row.get("gold_answer")),
        "question_chars": len(str(row.get("question") or row.get("input") or "")),
        "join_count": gf["n_joins"],
        "fan_out_count": gf["n_fan_out_nodes"],
        "reuse_count": gf["n_reused_outputs"],
        **gf, **sf,
    }
    if with_query_mode:
        from . import query_realism as qr

        sids = qr.gold_sids_from_row(row) or [str((c or {}).get("name") or "")
                                              for c in calls]
        audit = qr.audit_task(str(row.get("question") or row.get("input") or ""), sids)
        feats["query_mode"] = audit["query_mode"]
        feats["operation_explicitness"] = audit["lexical_operation_coverage"]
        feats["sequence_leakage"] = audit["sequence_leakage"]
        feats["procedural_cue_count"] = audit["procedural_cue_count"]
    return feats


# ── distributions ─────────────────────────────────────────────────────────
def _bucketize(key: str, value: Any) -> str:
    if key in ("depth", "join_count", "fan_out_count", "reuse_count"):
        v = int(value)
        return str(v) if v <= 3 else "4+"
    if key == "reference_density":
        v = float(value)
        return ("0.0" if v <= 0.0 else "0-0.25" if v <= 0.25 else
                "0.25-0.5" if v <= 0.5 else "0.5-0.75" if v <= 0.75 else "0.75-1.0")
    if key == "offered_tool_count":
        v = int(value)
        return "<=9" if v <= 9 else "10-12" if v <= 12 else "13-18" if v <= 18 else "19+"
    if key == "schema_complexity":
        v = float(value)
        return "low" if v <= 0.7 else "medium" if v <= 1.2 else "high"
    return str(value)


def _dist(values: Iterable[Any]) -> Dict[str, float]:
    c = Counter(values)
    total = sum(c.values()) or 1
    return {str(k): round(v / total, 5) for k, v in sorted(c.items(),
                                                           key=lambda kv: str(kv[0]))}


def _num_stats(values: Sequence[float]) -> Dict[str, float]:
    xs = sorted(float(v) for v in values)
    if not xs:
        return {"mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "min": 0.0, "max": 0.0}

    def q(p: float) -> float:
        return xs[min(int(p * (len(xs) - 1)), len(xs) - 1)]

    return {"mean": round(sum(xs) / len(xs), 4), "p25": round(q(0.25), 4),
            "p50": round(q(0.5), 4), "p75": round(q(0.75), 4),
            "min": round(xs[0], 4), "max": round(xs[-1], 4)}


def conditional_distributions(feats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """P(feature | call_bucket) for every conditional key, plus two extras."""
    out: Dict[str, Any] = {}
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in feats:
        by_bucket[f["call_bucket"]].append(f)
    for key in CONDITIONAL_KEYS:
        cond: Dict[str, Dict[str, float]] = {}
        for bucket in CALL_BUCKETS:
            rows = by_bucket.get(bucket, [])
            if not rows:
                continue
            cond[bucket] = _dist(_bucketize(key, r.get(key, "")) for r in rows)
        out[f"P({key}|call_count)"] = cond
    # P(operation_explicitness | query_mode)
    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in feats:
        by_mode[str(f.get("query_mode", "UNKNOWN"))].append(f)
    out["P(operation_explicitness|query_mode)"] = {
        mode: _num_stats([r.get("operation_explicitness", 0.0) for r in rows])
        for mode, rows in sorted(by_mode.items())
    }
    out["P(sequence_leakage|query_mode)"] = {
        mode: _num_stats([r.get("sequence_leakage", 0.0) for r in rows])
        for mode, rows in sorted(by_mode.items())
    }
    return out


def topology_diversity(feats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Diversity measured INSIDE each call bucket, never globally."""
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in feats:
        by_bucket[f["call_bucket"]].append(f)
    out: Dict[str, Any] = {}
    for bucket in CALL_BUCKETS:
        rows = by_bucket.get(bucket, [])
        if not rows:
            continue
        sigs = Counter(topology_signature(r) for r in rows)
        n = len(rows)
        top1 = sigs.most_common(1)[0][1] / n
        out[bucket] = {
            "n": n,
            "n_distinct_topologies": len(sigs),
            "top1_topology_share": round(top1, 4),
            "normalized_entropy": round(_norm_entropy(sigs), 4),
            "motif_distribution": _dist(r["motif"] for r in rows),
            "join_rate": round(sum(1 for r in rows if r["join_count"] > 0) / n, 4),
            "fan_out_rate": round(sum(1 for r in rows if r["fan_out_count"] > 0) / n, 4),
            "reuse_rate": round(sum(1 for r in rows if r["reuse_count"] > 0) / n, 4),
            "multi_join_rate": round(sum(1 for r in rows if r["join_count"] >= 2) / n, 4),
            "late_reference_rate": round(
                sum(1 for r in rows if r["n_late_references"] > 0) / n, 4),
            "mean_reference_distance": round(
                sum(r["mean_reference_distance"] for r in rows) / n, 4),
            "mean_type_transitions": round(
                sum(r["n_type_transitions"] for r in rows) / n, 4),
        }
    return out


def topology_signature(f: Dict[str, Any]) -> str:
    return (f"n{f['n_nodes']}_e{f['n_edges']}_d{f['depth']}_j{f['n_joins']}"
            f"_fo{f['n_fan_out_nodes']}_ru{f['n_reused_outputs']}"
            f"_lr{f['n_late_references']}_mi{f['max_indegree']}")


def _norm_entropy(counts: Counter) -> float:
    import math

    total = sum(counts.values()) or 1
    if len(counts) <= 1:
        return 0.0
    h = -sum((v / total) * math.log(v / total) for v in counts.values())
    return h / math.log(len(counts))


def build_profile_v2(rows: Sequence[Dict[str, Any]], *, source: str, mode: str,
                     target: str = "nestful") -> Dict[str, Any]:
    """``mode`` is PROFILE_SAFE or DIAGNOSTIC_EXPLORATORY (never a default target)."""
    feats = [featurize(r) for r in rows]
    n = len(feats) or 1
    return {
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "source": source,
        "mode": mode,
        "n_rows": len(feats),
        "call_count_dist": _dist(f["call_bucket"] for f in feats),
        "marginal": {
            "motif": _dist(f["motif"] for f in feats),
            "answer_type": _dist(f["answer_type"] for f in feats),
            "query_mode": _dist(f.get("query_mode", "UNKNOWN") for f in feats),
            "offered_tool_count": _num_stats([f["offered_tool_count"] for f in feats]),
            "reference_density": _num_stats([f["reference_density"] for f in feats]),
            "question_chars": _num_stats([f["question_chars"] for f in feats]),
        },
        "conditional": conditional_distributions(feats),
        "topology_diversity_by_bucket": topology_diversity(feats),
        "graph_features": {k: _num_stats([f[k] for f in feats])
                           for k in GRAPH_FEATURE_KEYS},
        "surface_features": {
            **{k: _num_stats([f[k] for f in feats])
               for k in SURFACE_FEATURE_KEYS if k != "output_key_family"},
            "output_key_family": _dist(f["output_key_family"] for f in feats),
            "schema_complexity": _num_stats([f["schema_complexity"] for f in feats]),
        },
        "query_realism": _query_realism_block(feats),
    }


def _query_realism_block(feats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from . import query_realism as qr

    n = len(feats) or 1

    def band(x: float, edges: Sequence[float], labels: Sequence[str]) -> str:
        for e, lab in zip(edges, labels):
            if x <= e:
                return lab
        return labels[-1]

    return {
        "query_mode_distribution": _dist(f.get("query_mode", "UNKNOWN")
                                         for f in feats),
        "operation_explicitness_distribution": _dist(
            band(f.get("operation_explicitness", 0.0), *qr.EXPLICITNESS_BUCKETS)
            for f in feats),
        "sequence_leakage_distribution": _dist(
            band(f.get("sequence_leakage", 0.0), *qr.LEAKAGE_BUCKETS)
            for f in feats),
        "procedural_cue_distribution": _dist(
            str(min(int(f.get("procedural_cue_count", 0)), 5)) for f in feats),
        "intermediate_reference_explicitness": {
            "share_with_explicit_reference": round(
                sum(1 for f in feats
                    if int(f.get("procedural_cue_count", 0)) > 0) / n, 4),
            "mean_reference_density": round(
                sum(float(f.get("reference_density", 0.0)) for f in feats) / n, 4),
        },
        "mean_operation_explicitness": round(
            sum(f.get("operation_explicitness", 0.0) for f in feats) / n, 4),
        "mean_sequence_leakage": round(
            sum(f.get("sequence_leakage", 0.0) for f in feats) / n, 4),
        "mean_procedural_cue_count": round(
            sum(f.get("procedural_cue_count", 0) for f in feats) / n, 4),
        "plan_leak_rate": round(
            sum(1 for f in feats
                if f.get("query_mode") == "PROCEDURAL_EXPLICIT") / n, 4),
    }


def derive_topology_constraints(profile: Dict[str, Any], *,
                                floor: float = 0.6) -> Dict[str, Any]:
    """Turn measured per-bucket diversity into generation constraints.

    Values are derived from the profile, not hardcoded: each bucket asks for at
    least ``floor`` of the target's observed structural richness, and long
    buckets additionally get a minimum pattern-family count.
    """
    out: Dict[str, Any] = {}
    div = profile.get("topology_diversity_by_bucket", {})
    for bucket in CALL_BUCKETS:
        d = div.get(bucket)
        max_calls = 2 if bucket == "2" else (8 if bucket == "6+" else int(bucket))
        if not d:
            out[bucket] = {"minimum_pattern_families": 1 if max_calls <= 2 else 3}
            continue
        cons: Dict[str, Any] = {}
        if max_calls <= 2:
            cons["allowed_patterns"] = ["LINEAR_CHAIN"]
            cons["minimum_pattern_families"] = 1
        else:
            cons["minimum_pattern_families"] = max(
                3, min(10, int(round(d["n_distinct_topologies"] * floor))))
            cons["minimum_join_rate"] = round(d["join_rate"] * floor, 3)
            cons["maximum_top1_topology_share"] = round(
                min(0.6, max(d["top1_topology_share"], 0.15) * 1.2), 3)
        if max_calls >= 5:
            cons["minimum_multi_join_rate"] = round(max(d["multi_join_rate"] * floor,
                                                        0.15), 3)
            cons["minimum_reuse_rate"] = round(max(d["reuse_rate"] * floor, 0.15), 3)
        if max_calls >= 6:
            cons["minimum_fan_out_rate"] = round(max(d["fan_out_rate"] * floor, 0.20), 3)
            cons["minimum_late_reference_rate"] = round(
                max(d["late_reference_rate"] * floor, 0.15), 3)
            cons["minimum_pattern_families"] = max(cons["minimum_pattern_families"], 8)
        out[bucket] = cons
    return out


def markdown_report(profile: Dict[str, Any]) -> str:
    lines = [
        "# TARGET_PROFILE_V2", "",
        f"- source: `{profile['source']}`",
        f"- mode: **{profile['mode']}**",
        f"- rows: {profile['n_rows']}",
        f"- schema: `{profile['schema_version']}`", "",
        "## Call-count distribution", "",
    ]
    for k, v in profile["call_count_dist"].items():
        lines.append(f"- `{k}`: {v:.4f}")
    lines += ["", "## Topology diversity inside each call bucket", "",
              "| bucket | n | distinct topologies | top1 share | norm. entropy | "
              "join | multi-join | fan-out | reuse | late-ref |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for b, d in profile["topology_diversity_by_bucket"].items():
        lines.append(
            f"| {b} | {d['n']} | {d['n_distinct_topologies']} | {d['top1_topology_share']:.3f} "
            f"| {d['normalized_entropy']:.3f} | {d['join_rate']:.3f} | "
            f"{d['multi_join_rate']:.3f} | {d['fan_out_rate']:.3f} | {d['reuse_rate']:.3f} "
            f"| {d['late_reference_rate']:.3f} |")
    lines += ["", "## Conditional distributions", ""]
    for key, cond in profile["conditional"].items():
        lines.append(f"### `{key}`")
        lines.append("")
        for bucket, dist in cond.items():
            if isinstance(dist, dict) and dist and isinstance(
                    next(iter(dist.values())), (int, float)):
                inner = ", ".join(f"{k}={v}" for k, v in dist.items())
            else:
                inner = json.dumps(dist)
            lines.append(f"- `{bucket}`: {inner}")
        lines.append("")
    lines += ["## Graph features", "",
              "| feature | mean | p25 | p50 | p75 | max |", "|---|---:|---:|---:|---:|---:|"]
    for k in GRAPH_FEATURE_KEYS:
        s = profile["graph_features"][k]
        lines.append(f"| `{k}` | {s['mean']} | {s['p25']} | {s['p50']} | {s['p75']} | {s['max']} |")
    lines += ["", "## Surface features", ""]
    for k, v in profile["surface_features"].items():
        lines.append(f"- `{k}`: {json.dumps(v)}")
    lines += ["", "## Query realism", ""]
    for k, v in profile["query_realism"].items():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Safety note", "",
              "PROFILE_SAFE profiles are built from the dev split and generic factory",
              "metadata only. Diagnostic-informed statistics live in a separate file",
              "and are never used as a default generation target.", ""]
    return "\n".join(lines) + "\n"
