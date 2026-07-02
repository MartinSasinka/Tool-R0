#!/usr/bin/env python3
"""Shared motif analysis utilities for NESTFUL Synthetic Curriculum v3."""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_VAR_REF_RE = re.compile(r"^\$var_?(\d+)(?:\.([A-Za-z_][\w]*))?\$$", re.I)

VALID_MOTIF_TYPES = frozenset({
    "linear_dependency", "reference_reuse", "fan_in", "fan_out",
    "object_or_list_output", "argument_transformation", "distractor_tools",
    "long_chain", "alternative_valid_traces", "baseline_failure_inspired",
    "simple_fan_in", "independent_calls",
})

OUTPUT_TYPES = frozenset({
    "scalar", "string", "list", "object", "array", "boolean", "mixed", "unknown",
})

CALL_BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 8), (9, 999)]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_nestful_path() -> Path:
    return repo_root() / "experiments/nestful_mtgrpo_minimal/data/NESTFUL-main/data_v2/nestful_data.jsonl"


def default_dev_path() -> Path:
    return repo_root() / "experiments/nestful_mtgrpo_minimal/data/splits/nestful_dev.jsonl"


def default_test_path() -> Path:
    return repo_root() / "experiments/nestful_mtgrpo_minimal/data/splits/nestful_test.jsonl"


def coerce_json(val: Any) -> Any:
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return None
    return None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_task_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize raw NESTFUL or v3 synthetic row to a common dict."""
    task_id = row.get("task_id") or row.get("sample_id") or row.get("id") or ""
    question = row.get("question") or row.get("input") or row.get("prompt") or ""
    tools = coerce_json(row.get("tools")) or []
    gold_calls = (
        coerce_json(row.get("gold_calls"))
        or coerce_json(row.get("output"))
        or coerce_json(row.get("gold_output"))
        or []
    )
    gold_answer = row.get("gold_answer", row.get("answer", row.get("final_answer")))
    out = {
        "task_id": str(task_id),
        "question": str(question),
        "tools": tools if isinstance(tools, list) else [],
        "gold_calls": gold_calls if isinstance(gold_calls, list) else [],
        "gold_answer": gold_answer,
        "num_calls": len(gold_calls) if isinstance(gold_calls, list) else 0,
    }
    for k in ("motif_type", "dependency_graph", "reference_pattern", "output_type",
              "answer_type", "difficulty_score", "source_motif_cluster", "generation_seed"):
        if k in row:
            out[k] = row[k]
    return out


def extract_references_from_value(value: Any) -> List[Tuple[int, Optional[str]]]:
    refs: List[Tuple[int, Optional[str]]] = []
    if isinstance(value, str):
        m = _VAR_REF_RE.match(value.strip())
        if m:
            refs.append((int(m.group(1)), m.group(2)))
    elif isinstance(value, list):
        for item in value:
            refs.extend(extract_references_from_value(item))
    elif isinstance(value, dict):
        for v in value.values():
            refs.extend(extract_references_from_value(v))
    return refs


def refs_for_call(call_idx: int, call: Dict[str, Any]) -> List[Tuple[int, Optional[str]]]:
    args = call.get("arguments") or {}
    refs: List[Tuple[int, Optional[str]]] = []
    if isinstance(args, dict):
        for v in args.values():
            refs.extend(extract_references_from_value(v))
    return [(r, f) for r, f in refs if r < call_idx]


def build_dependency_graph(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    nodes = [{"id": i, "name": c.get("name", "")} for i, c in enumerate(calls, start=1)]
    edges: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, int]] = set()
    for i, call in enumerate(calls, start=1):
        for ref_idx, field in refs_for_call(i, call):
            key = (ref_idx, i)
            if key not in seen:
                seen.add(key)
                edges.append({"from": ref_idx, "to": i, "field": field})
    return {"nodes": nodes, "edges": edges}


def compute_dependency_depth(calls: List[Dict[str, Any]]) -> int:
    n = len(calls)
    if n == 0:
        return 0
    refs_by_call: Dict[int, Set[int]] = {}
    for i, call in enumerate(calls, start=1):
        preds = {r for r, _ in refs_for_call(i, call)}
        if preds:
            refs_by_call[i] = preds
    memo: Dict[int, int] = {}

    def depth(i: int) -> int:
        if i in memo:
            return memo[i]
        preds = refs_by_call.get(i, set())
        if not preds:
            memo[i] = 1
            return 1
        memo[i] = 1 + max(depth(p) for p in preds)
        return memo[i]

    return max((depth(i) for i in range(1, n + 1)), default=0)


def count_references(calls: List[Dict[str, Any]]) -> int:
    total = 0
    for i, call in enumerate(calls, start=1):
        total += len(refs_for_call(i, call))
    return total


def reference_pattern_stats(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    edges = build_dependency_graph(calls)["edges"]
    in_deg = Counter(e["to"] for e in edges)
    out_deg = Counter(e["from"] for e in edges)
    fan_in = sum(1 for d in in_deg.values() if d > 1)
    fan_out = sum(1 for d in out_deg.values() if d > 1)
    reuse = sum(1 for d in out_deg.values() if d > 1)
    nested = 0
    for i, call in enumerate(calls, start=1):
        for ref_idx, field in refs_for_call(i, call):
            if ref_idx < i - 1:
                nested += 1
    return {
        "num_references": len(edges),
        "fan_in_count": fan_in,
        "fan_out_count": fan_out,
        "reuse_count": reuse,
        "nested_reference_depth": nested,
    }


def is_linear_chain(calls: List[Dict[str, Any]]) -> bool:
    if len(calls) <= 1:
        return True
    for i, call in enumerate(calls, start=1):
        preds = {r for r, _ in refs_for_call(i, call)}
        if i == 1:
            if preds:
                return False
        elif preds != {i - 1}:
            return False
    return True


def has_independent_calls(calls: List[Dict[str, Any]]) -> bool:
    for i, call in enumerate(calls, start=1):
        if not refs_for_call(i, call):
            return True
    return len(calls) <= 1


def infer_value_type(val: Any) -> str:
    if val is None:
        return "unknown"
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return "scalar"
    if isinstance(val, str):
        return "string"
    if isinstance(val, list):
        if not val:
            return "list"
        return "list"
    if isinstance(val, dict):
        return "object"
    t = type(val).__name__
    if "ndarray" in t.lower() or "tensor" in t.lower():
        return "array"
    return "unknown"


def tool_families(calls: List[Dict[str, Any]]) -> List[str]:
    from synthetic_tool_registry import (
        BOOLEAN_TOOL_NAMES,
        LIST_TOOL_NAMES,
        MATH_TOOL_NAMES,
        OBJECT_TOOL_NAMES,
        STRING_TOOL_NAMES,
    )

    families = []
    for c in calls:
        name = str(c.get("name", "")).lower()
        if name in MATH_TOOL_NAMES:
            families.append("math")
        elif name in STRING_TOOL_NAMES:
            families.append("string")
        elif name in LIST_TOOL_NAMES:
            families.append("list")
        elif name in OBJECT_TOOL_NAMES:
            families.append("object")
        elif name in BOOLEAN_TOOL_NAMES:
            families.append("boolean")
        else:
            families.append("other")
    return families


def classify_motif_type(calls: List[Dict[str, Any]], ref_stats: Dict[str, Any]) -> str:
    n = len(calls)
    if n >= 5:
        return "long_chain"
    if ref_stats["fan_in_count"] > 0:
        return "fan_in" if ref_stats["fan_in_count"] > 0 else "simple_fan_in"
    if ref_stats["fan_out_count"] > 0:
        return "fan_out"
    if ref_stats["reuse_count"] > 0:
        return "reference_reuse"
    if is_linear_chain(calls) and n >= 2:
        return "linear_dependency"
    if has_independent_calls(calls) and n > 1:
        return "independent_calls"
    return "linear_dependency"


def difficulty_score(
    num_calls: int,
    ref_stats: Dict[str, Any],
    dep_depth: int,
    output_type: str,
    answer_type: str,
    max_calls_ref: int = 10,
) -> float:
    nc = min(1.0, num_calls / max(max_calls_ref, 1))
    nr = min(1.0, ref_stats["num_references"] / max(num_calls, 1))
    dd = min(1.0, dep_depth / max(max_calls_ref, 1))
    fan_bonus = min(0.2, 0.05 * (ref_stats["fan_in_count"] + ref_stats["fan_out_count"]))
    complex_types = {"object", "list", "array", "mixed"}
    ot = 0.15 if output_type in complex_types else 0.0
    at = 0.10 if answer_type in complex_types else 0.0
    raw = 0.30 * nc + 0.25 * nr + 0.25 * dd + fan_bonus + ot + at
    return round(max(0.0, min(1.0, raw)), 4)


def extract_motifs(task: Dict[str, Any]) -> Dict[str, Any]:
    t = load_task_row(task)
    calls = t["gold_calls"]
    ref_stats = reference_pattern_stats(calls)
    dep_graph = build_dependency_graph(calls)
    dep_depth = compute_dependency_depth(calls)
    out_type = t.get("output_type") or infer_value_type(t.get("gold_answer"))
    ans_type = t.get("answer_type") or infer_value_type(t.get("gold_answer"))
    motif = t.get("motif_type") or classify_motif_type(calls, ref_stats)
    tool_names = [str(c.get("name", "")) for c in calls]
    gold_set = set(tool_names)
    distractors = [tl.get("name") for tl in t.get("tools", []) if tl.get("name") not in gold_set]
    score = t.get("difficulty_score")
    if score is None:
        score = difficulty_score(len(calls), ref_stats, dep_depth, out_type, ans_type)
    return {
        "task_id": t["task_id"],
        "num_calls": len(calls),
        "dependency_graph": dep_graph,
        "dependency_depth": dep_depth,
        "linear_chain": is_linear_chain(calls),
        "independent_calls": has_independent_calls(calls),
        "fan_in": ref_stats["fan_in_count"] > 0,
        "fan_out": ref_stats["fan_out_count"] > 0,
        "reference_reuse": ref_stats["reuse_count"] > 0,
        "reference_pattern": ref_stats,
        "argument_complexity": ref_stats["num_references"],
        "output_type": out_type,
        "answer_type": ans_type,
        "tool_family": ",".join(tool_families(calls)),
        "tool_sequence": "->".join(tool_names),
        "tool_sequence_bigram": "|".join(
            f"{tool_names[i]}->{tool_names[i+1]}" for i in range(len(tool_names) - 1)
        ) if len(tool_names) >= 2 else "",
        "tool_sequence_trigram": "|".join(
            f"{tool_names[i]}->{tool_names[i+1]}->{tool_names[i+2]}"
            for i in range(len(tool_names) - 2)
        ) if len(tool_names) >= 3 else "",
        "distractor_tools": len(distractors),
        "motif_type": motif,
        "difficulty_score": float(score),
    }


def call_count_bucket(n: int) -> str:
    for lo, hi in CALL_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}" if lo == hi else f"{lo}-{hi}" if hi < 999 else "9+"
    return "9+"


def histogram(values: Iterable[Any]) -> Dict[str, int]:
    return dict(Counter(values))


def normalize_dist(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def l1_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    return sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def kl_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-9) -> float:
    keys = set(p) | set(q)
    s = 0.0
    for k in keys:
        pk = max(eps, p.get(k, 0.0))
        qk = max(eps, q.get(k, 0.0))
        s += pk * math.log(pk / qk)
    return s


def load_blocked_ids(dev_path: Optional[Path] = None, test_path: Optional[Path] = None) -> Set[str]:
    blocked: Set[str] = set()
    for p in (dev_path or default_dev_path(), test_path or default_test_path()):
        if p.is_file():
            for row in load_jsonl(p):
                tid = row.get("task_id") or row.get("sample_id") or row.get("id")
                if tid:
                    blocked.add(str(tid))
    return blocked


def validate_references(calls: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for i, call in enumerate(calls, start=1):
        args = call.get("arguments") or {}
        if not isinstance(args, dict):
            errors.append(f"call_{i}: arguments not dict")
            continue
        for key, val in args.items():
            for ref_idx, _ in extract_references_from_value(val):
                if ref_idx >= i:
                    errors.append(f"call_{i}: forward ref to var_{ref_idx}")
                if ref_idx < 1:
                    errors.append(f"call_{i}: invalid ref var_{ref_idx}")
    return errors


def graph_matches_refs(calls: List[Dict[str, Any]], graph: Dict[str, Any]) -> bool:
    expected = build_dependency_graph(calls)
    exp_edges = {(e["from"], e["to"]) for e in expected["edges"]}
    got_edges = {(e["from"], e["to"]) for e in (graph or {}).get("edges", [])}
    return exp_edges == got_edges


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate_distribution(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    motifs = [extract_motifs(t) for t in tasks]
    return {
        "num_tasks": len(masks := motifs),
        "num_calls": histogram(call_count_bucket(m["num_calls"]) for m in motifs),
        "motif_type": histogram(m["motif_type"] for m in motifs),
        "dependency_depth": histogram(str(m["dependency_depth"]) for m in motifs),
        "output_type": histogram(m["output_type"] for m in motifs),
        "answer_type": histogram(m["answer_type"] for m in motifs),
        "fan_in_rate": sum(1 for m in motifs if m["fan_in"]) / max(len(motifs), 1),
        "fan_out_rate": sum(1 for m in motifs if m["fan_out"]) / max(len(motifs), 1),
        "reference_reuse_rate": sum(1 for m in motifs if m["reference_reuse"]) / max(len(motifs), 1),
        "mean_difficulty": sum(m["difficulty_score"] for m in motifs) / max(len(motifs), 1),
        "tool_bigrams": histogram(m["tool_sequence_bigram"] for m in motifs if m["tool_sequence_bigram"]),
    }
