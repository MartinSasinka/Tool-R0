#!/usr/bin/env python3
"""Shared uniqueness signatures and stage-aware dedup registry (v3.1)."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from motif_lib import extract_references_from_value
from tool_registry_v3_1 import infer_answer_type, tool_family

STAGE_FILES = {
    "stage1_1call_atomic": "stage1_1call_atomic.jsonl",
    "stage2_2call_dependency": "stage2_2call_dependency.jsonl",
    "stage3_3call_composition": "stage3_3call_composition.jsonl",
    "stage4_4to6call_persistence": "stage4_4to6call_persistence.jsonl",
}

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_STR_RE = re.compile(r'"[^"]*"')
_LIST_RE = re.compile(r"\[[^\]]*\]")


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _normalize_question(q: str) -> str:
    return " ".join((q or "").strip().lower().split())


def _normalize_answer(ans: Any) -> Any:
    if isinstance(ans, float) and ans.is_integer():
        return int(ans)
    if isinstance(ans, list):
        return [_normalize_answer(v) for v in ans]
    if isinstance(ans, dict):
        return {k: _normalize_answer(v) for k, v in sorted(ans.items())}
    return ans


def _normalize_call(call: dict) -> dict:
    args = call.get("arguments") or {}
    norm_args = {}
    for k, v in sorted(args.items()):
        if isinstance(v, str) and extract_references_from_value(v):
            refs = extract_references_from_value(v)
            norm_args[k] = f"$ref:{refs[0][0]}:{refs[0][1] or 'result'}$"
        elif isinstance(v, list):
            norm_args[k] = ["list", len(v)]
        elif isinstance(v, dict):
            norm_args[k] = "object"
        else:
            norm_args[k] = v
    return {"name": call.get("name", ""), "arguments": norm_args}


def normalize_gold_calls(calls: List[dict]) -> List[dict]:
    return [_normalize_call(c) for c in calls]


def normalize_question_template(q: str) -> str:
    t = (q or "").lower()
    t = _LIST_RE.sub("<LIST>", t)
    t = _STR_RE.sub("<STR>", t)
    t = _NUM_RE.sub("<NUM>", t)
    return " ".join(t.split())


def _arg_pattern(val: Any) -> Tuple[str, str, Optional[int]]:
    if isinstance(val, str) and extract_references_from_value(val):
        refs = extract_references_from_value(val)
        return "reference", "ref", refs[0][0]
    if isinstance(val, bool):
        return "literal", "boolean", None
    if isinstance(val, int):
        return "literal", "int", None
    if isinstance(val, float):
        return "literal", "float", None
    if isinstance(val, str):
        return "literal", "string", None
    if isinstance(val, list):
        return "literal", "list", None
    if isinstance(val, dict):
        return "literal", "object", None
    return "literal", type(val).__name__, None


def argument_pattern_signature(calls: List[dict]) -> List[List[Any]]:
    out: List[List[Any]] = []
    for call in calls:
        row = [call.get("name", "")]
        for _, val in sorted((call.get("arguments") or {}).items()):
            kind, typ, ref_tgt = _arg_pattern(val)
            row.extend([kind, typ, ref_tgt])
        out.append(row)
    return out


def compute_signatures(sample: dict) -> Dict[str, Any]:
    q = sample.get("question", "")
    calls = sample.get("gold_calls") or []
    ans = sample.get("gold_answer")
    nq = _normalize_question(q)
    nc = normalize_gold_calls(calls)
    na = _normalize_answer(ans)
    tool_seq = tuple(c.get("name", "") for c in calls)
    q_tpl = normalize_question_template(q)
    arg_pat = argument_pattern_signature(calls)
    exact = _stable_hash((nq, nc, na))
    trace = _stable_hash((nc, na))
    return {
        "exact": exact,
        "trace": trace,
        "tool_sequence": tool_seq,
        "argument_pattern": _stable_hash(arg_pat),
        "question_template": q_tpl,
        "question_text": q,
    }


class StageDedupRegistry:
    """Stage-aware dedup tracking."""

    def __init__(
        self,
        stage: str,
        *,
        max_trace_count: int = 2,
        max_template_count: int = 8,
        max_tool_seq_ratio: float = 0.16,
        max_traj_per_stage: int = 5,
        stage_target: int = 800,
    ):
        self.stage = stage
        self.max_trace_count = max_trace_count
        self.max_template_count = max_template_count
        self.max_tool_seq_ratio = max_tool_seq_ratio
        self.max_traj_per_stage = max_traj_per_stage
        self.stage_target = stage_target
        self.exact: Set[str] = set()
        self.trace: Counter = Counter()
        self.templates: Counter = Counter()
        self.tool_seq: Counter = Counter()
        self.traj_id: Counter = Counter()
        self.warnings: List[str] = []

    def _tool_seq_limit(self) -> int:
        return max(1, int(self.stage_target * self.max_tool_seq_ratio))

    def can_add(self, sigs: Dict[str, Any], sample: Optional[dict] = None) -> bool:
        if sigs["exact"] in self.exact:
            return False
        if self.trace[sigs["trace"]] >= self.max_trace_count:
            return False
        if self.templates[sigs["question_template"]] >= self.max_template_count:
            return False
        seq_limit = self._tool_seq_limit()
        if self.tool_seq[sigs["tool_sequence"]] >= seq_limit:
            return False
        if sample and self.traj_id[sample.get("trajectory_id", "")] >= self.max_traj_per_stage:
            return False
        return True

    def register(self, sample: dict, sigs: Optional[Dict[str, Any]] = None) -> None:
        sigs = sigs or compute_signatures(sample)
        self.exact.add(sigs["exact"])
        self.trace[sigs["trace"]] += 1
        self.templates[sigs["question_template"]] += 1
        self.tool_seq[sigs["tool_sequence"]] += 1
        tid = sample.get("trajectory_id", "")
        if tid:
            self.traj_id[tid] += 1

    def note_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def analyze_stage_samples(samples: List[dict], stage: str) -> dict:
    exact_counts: Counter = Counter()
    trace_counts: Counter = Counter()
    tpl_counts: Counter = Counter()
    tool_seq_counts: Counter = Counter()
    traj_counts: Counter = Counter()
    questions: Set[str] = set()
    tools_used: Counter = Counter()
    families_used: Counter = Counter()
    output_types: Counter = Counter()
    clusters: Counter = Counter()
    motifs: Counter = Counter()

    dup_rows: List[dict] = []
    for s in samples:
        sigs = compute_signatures(s)
        exact_counts[sigs["exact"]] += 1
        trace_counts[sigs["trace"]] += 1
        tpl_counts[sigs["question_template"]] += 1
        tool_seq_counts[sigs["tool_sequence"]] += 1
        traj_counts[s.get("trajectory_id", "")] += 1
        questions.add(s.get("question", ""))
        for c in s.get("gold_calls") or []:
            name = c.get("name", "")
            tools_used[name] += 1
            families_used[tool_family(name)] += 1
        ot = s.get("answer_type") or infer_answer_type(s.get("gold_answer"))
        output_types[ot] += 1
        clusters[s.get("source_failure_cluster", "unknown")] += 1
        motifs[s.get("target_full_motif", "unknown")] += 1
        if exact_counts[sigs["exact"]] > 1 or trace_counts[sigs["trace"]] > 1:
            dup_rows.append({
                "sample_id": s.get("sample_id"),
                "stage": stage,
                "exact_count": exact_counts[sigs["exact"]],
                "trace_count": trace_counts[sigs["trace"]],
                "question": (s.get("question") or "")[:100],
                "tool_sequence": "->".join(sigs["tool_sequence"]),
                "trajectory_id": s.get("trajectory_id"),
            })

    total = len(samples)
    exact_dup = sum(c - 1 for c in exact_counts.values() if c > 1)
    trace_dup = sum(c - 1 for c in trace_counts.values() if c > 1)
    tpl_dup = sum(c - 1 for c in tpl_counts.values() if c > 1)
    uq_ratio = len(questions) / max(total, 1)
    trace_dup_ratio = trace_dup / max(total, 1)
    tpl_dup_ratio = tpl_dup / max(total, 1)

    top_exact = exact_counts.most_common(20)
    top_trace = trace_counts.most_common(20)
    top_tpl = tpl_counts.most_common(20)
    top_tool_seq = tool_seq_counts.most_common(20)
    max_traj = max(traj_counts.values()) if traj_counts else 0
    max_tool_seq_share = max((c / max(total, 1) for c in tool_seq_counts.values()), default=0)

    hard_fail = exact_dup > 0 or total < 800
    soft_warn = (
        trace_dup_ratio > 0.05
        or tpl_dup_ratio > 0.30
        or max_traj > 6
        or max_tool_seq_share > 0.15
        or len(tools_used) < 20
        or len(families_used) < 5
    )
    status = "FAIL" if hard_fail else ("WARN" if soft_warn else "PASS")

    return {
        "stage": stage,
        "status": status,
        "total_samples": total,
        "unique_questions": len(questions),
        "unique_question_ratio": round(uq_ratio, 4),
        "exact_duplicate_count": exact_dup,
        "trace_duplicate_count": trace_dup,
        "trace_duplicate_ratio": round(trace_dup_ratio, 4),
        "question_template_duplicate_count": tpl_dup,
        "question_template_duplicate_ratio": round(tpl_dup_ratio, 4),
        "unique_exact_signatures": len(exact_counts),
        "unique_trace_signatures": len(trace_counts),
        "unique_tool_sequences": len(tool_seq_counts),
        "max_trajectory_id_count": max_traj,
        "max_tool_sequence_share": round(max_tool_seq_share, 4),
        "used_tool_names": sorted(tools_used.keys()),
        "used_tool_count": len(tools_used),
        "used_tool_families": sorted(families_used.keys()),
        "used_tool_family_count": len(families_used),
        "output_type_distribution": dict(output_types),
        "source_failure_cluster_distribution": dict(clusters.most_common(15)),
        "target_full_motif_distribution": dict(motifs.most_common(15)),
        "top_exact_duplicates": [{"hash": h, "count": c} for h, c in top_exact if c > 1][:20],
        "top_trace_duplicates": [{"hash": h, "count": c} for h, c in top_trace if c > 1][:20],
        "top_template_duplicates": [{"template": t[:80], "count": c} for t, c in top_tpl if c > 1][:20],
        "top_tool_sequence_duplicates": [{"sequence": "->".join(k), "count": c} for k, c in top_tool_seq if c > 1][:20],
        "duplicate_sample_examples": dup_rows[:50],
    }


def analyze_all_stages(stage_samples: Dict[str, List[dict]]) -> dict:
    per_stage = {stage: analyze_stage_samples(samples, stage) for stage, samples in stage_samples.items()}
    overall = {
        "exact_duplicate_count": sum(p["exact_duplicate_count"] for p in per_stage.values()),
        "mean_unique_question_ratio": round(
            sum(p["unique_question_ratio"] for p in per_stage.values()) / max(len(per_stage), 1), 4
        ),
        "mean_trace_duplicate_ratio": round(
            sum(p["trace_duplicate_ratio"] for p in per_stage.values()) / max(len(per_stage), 1), 4
        ),
    }
    status = "FAIL" if any(p["status"] == "FAIL" for p in per_stage.values()) else (
        "WARN" if any(p["status"] == "WARN" for p in per_stage.values()) else "PASS"
    )
    return {"status": status, "overall": overall, "per_stage": per_stage}
