#!/usr/bin/env python3
"""NESTFUL / ENSTFULL rollout analyzer — task difficulty, failure typing, SFT/KD subsets."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Taxonomies (user spec)
# ---------------------------------------------------------------------------

FAILURE_TYPES = [
    "TOOL_FORMAT_ERROR",
    "WRONG_TOOL_OR_ARGUMENTS",
    "TYPE_OR_FORMAT_ERROR",
    "REASONING_ERROR",
    "INCOMPLETE_OR_LOOPING",
    "UNKNOWN_ERROR",
]

FAILURE_META: Dict[str, Dict[str, str]] = {
    "TOOL_FORMAT_ERROR": {
        "meaning": "Invalid tool-call JSON/schema, missing arguments, or result where arguments belong.",
        "train": "Schema-following SFT with valid name+arguments examples.",
    },
    "WRONG_TOOL_OR_ARGUMENTS": {
        "meaning": "Wrong tool selected or wrong argument values / variable resolution.",
        "train": "Tool-selection and argument-resolution distillation from gold trajectories.",
    },
    "TYPE_OR_FORMAT_ERROR": {
        "meaning": "Output type or format wrong (list/dict/bool/JSON expected, got text or scalar).",
        "train": "Typed-output SFT and format-constrained completions.",
    },
    "REASONING_ERROR": {
        "meaning": "Wrong multi-step plan, calculation, percentage/ratio/unit/time reasoning.",
        "train": "Plan-and-execute KD from successful rollouts on similar tasks.",
    },
    "INCOMPLETE_OR_LOOPING": {
        "meaning": "No final answer, repeated calls, step/context limit, or early stop.",
        "train": "Stop/continue policy SFT and anti-loop counter-examples.",
    },
    "UNKNOWN_ERROR": {
        "meaning": "Failure pattern did not match a known bucket.",
        "train": "Manual review and targeted counter-examples.",
    },
}

PROBLEM_TYPES = [
    "arithmetic",
    "multi_step_math",
    "percentages_ratios_units",
    "time_date",
    "statistics_probability",
    "string_text_regex",
    "list_dict_json",
    "boolean_logic",
    "structured_output",
    "tool_use",
    "unknown",
]

PROBLEM_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("percentages_ratios_units", [
        r"\bpercent\b", r"%", r"\bratio\b", r"\brate\b", r"\bunit\b", r"\bfraction\b",
    ]),
    ("statistics_probability", [
        r"\bprobability\b", r"\bprobabil", r"\bmean\b", r"\bmedian\b",
        r"\bvariance\b", r"\baverage\b", r"\brandom\b",
    ]),
    ("time_date", [
        r"\bdatetime\b", r"\btimestamp\b", r"\bdate\b", r"\btime\b",
        r"\bday\b", r"\bmonth\b", r"\byear\b",
    ]),
    ("string_text_regex", [
        r"\bregex\b", r"\bsubstring\b", r"\bstring\b", r"\burl\b", r"\bemail\b",
    ]),
    ("boolean_logic", [r"\bboolean\b", r"\btrue\b", r"\bfalse\b"]),
    ("multi_step_math", [
        r"\bthen\b", r"\bstep\b", r"\bfirst\b.*\bthen\b", r"\btotal\b", r"\bhow many\b",
    ]),
    ("arithmetic", [
        r"\bcalculate\b", r"\badd\b", r"\bsubtract\b", r"\bmultiply\b", r"\bdivide\b",
        r"\bsum\b", r"\bdifference\b",
    ]),
]

DIFFICULTY_BUCKETS = [
    "easy", "mostly_ok", "unstable", "mostly_failed", "hard_failed",
]

TRAINING_SUBSETS: Dict[str, Callable[[Dict[str, Any]], bool]] = {}
SUBSET_INFO: Dict[str, Dict[str, str]] = {
    "hard_tasks": {
        "why": "Model never passes — systematic gaps.",
        "use": "Priority KD / hard-example SFT.",
    },
    "mostly_failed_tasks": {
        "why": "Model rarely succeeds — high-value learning signal.",
        "use": "Distillation from gold tool trajectories.",
    },
    "unstable_tasks": {
        "why": "Model sometimes succeeds — consistency problem.",
        "use": "Robustness SFT and rejection sampling.",
    },
    "tool_format_tasks": {
        "why": "Schema/format errors block execution.",
        "use": "Tool-call format repair SFT.",
    },
    "tool_argument_tasks": {
        "why": "Wrong tools or args cause runtime failures.",
        "use": "API-usage and argument-resolution training.",
    },
    "structured_output_tasks": {
        "why": "Semantic success but wrong output type/format.",
        "use": "Typed-output and JSON/list SFT.",
    },
    "reasoning_tasks": {
        "why": "Wrong plans or calculations despite tool access.",
        "use": "Multi-step reasoning distillation.",
    },
}


@dataclass
class FieldMapping:
    task_id: str = "task_id"
    prompt: str = "question"
    gold_answer: str = "gold_answer"
    predicted_answer: str = "predicted_final"
    tool_calls: str = "predicted_calls"
    trace: str = "execution_trace"
    error: str = "error_category"
    rollout_id: str = "rollout_idx"
    tools: str = "tools"
    gold_calls: str = "gold_calls"
    status: str = "status"
    score: str = "score"
    stopped: str = "stopped"
    raw_completions: str = "raw_completions"
    execution_error: str = "execution_error"


FIELD_CANDIDATES: Dict[str, List[str]] = {
    "task_id": ["task_id", "id", "problem_id"],
    "prompt": ["question", "prompt", "problem", "input"],
    "gold_answer": ["gold_answer", "answer", "target", "expected_answer"],
    "predicted_answer": ["predicted_final", "predicted_answer", "prediction"],
    "tool_calls": ["predicted_calls", "tool_calls", "calls"],
    "trace": ["execution_trace", "trace"],
    "error": ["error_category", "execution_error", "error"],
    "rollout_id": ["rollout_idx", "rollout_id"],
    "tools": ["tools", "available_tools"],
    "gold_calls": ["gold_calls", "reference_calls"],
    "status": ["status"],
    "score": ["score"],
    "stopped": ["stopped", "stop_reason"],
    "raw_completions": ["raw_completions", "completions"],
    "execution_error": ["execution_error"],
}


# ---------------------------------------------------------------------------
# I/O + schema inference
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping line %d: %s", i, exc)
    return rows


def infer_fields(records: List[Dict[str, Any]]) -> FieldMapping:
    if not records:
        logger.warning("Empty input — using default field names.")
        return FieldMapping()
    keys: Set[str] = set()
    for r in records[:100]:
        keys.update(r.keys())

    def pick(role: str, default: str) -> str:
        for cand in FIELD_CANDIDATES.get(role, [default]):
            if cand in keys:
                return cand
        if default not in keys:
            logger.warning("Field %r not found; using %r anyway.", role, default)
        return default

    return FieldMapping(
        task_id=pick("task_id", "task_id"),
        prompt=pick("prompt", "question"),
        gold_answer=pick("gold_answer", "gold_answer"),
        predicted_answer=pick("predicted_answer", "predicted_final"),
        tool_calls=pick("tool_calls", "predicted_calls"),
        trace=pick("trace", "execution_trace"),
        error=pick("error", "error_category"),
        rollout_id=pick("rollout_id", "rollout_idx"),
        tools=pick("tools", "tools"),
        gold_calls=pick("gold_calls", "gold_calls"),
        status=pick("status", "status"),
        score=pick("score", "score"),
        stopped=pick("stopped", "stopped"),
        raw_completions=pick("raw_completions", "raw_completions"),
        execution_error=pick("execution_error", "execution_error"),
    )


def normalize_record(record: Dict[str, Any], mapping: FieldMapping) -> Dict[str, Any]:
    def g(name: str, default: Any = None) -> Any:
        key = getattr(mapping, name, name)
        return record.get(key, default) if key else default

    passed = record.get("passed")
    if passed is None:
        status = g("status", "")
        score = g("score", 0)
        verdict = record.get("verdict", "")
        passed = (
            status == "completed"
            or (isinstance(score, (int, float)) and float(score) >= 1.0)
            or verdict == "pass"
        )

    prompt = g("prompt", "") or ""
    if not prompt and record.get("messages"):
        for m in record["messages"]:
            if isinstance(m, dict) and m.get("role") == "user":
                prompt = m.get("content", "")
                break

    return {
        "task_id": str(g("task_id", "")),
        "question": prompt,
        "gold_answer": g("gold_answer"),
        "predicted_final": g("predicted_answer"),
        "passed": bool(passed),
        "status": "completed" if passed else "failed",
        "predicted_calls": g("tool_calls") or [],
        "execution_trace": g("trace") or [],
        "error_category": g("error") or g("execution_error") or "",
        "execution_error": g("execution_error"),
        "rollout_idx": g("rollout_id", 0),
        "tools": g("tools") or [],
        "gold_calls": g("gold_calls") or [],
        "stopped": g("stopped") or record.get("stopped", ""),
        "verdict_reason": record.get("verdict_reason", ""),
        "raw_completions": g("raw_completions") or [],
        "num_tool_calls": record.get("num_tool_calls") or len(g("tool_calls") or []),
        "num_steps": record.get("num_steps", 0),
        "messages": record.get("messages") or [],
    }


# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------


def _gold_type(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    if isinstance(val, str):
        return "str"
    if isinstance(val, list):
        return "list"
    if isinstance(val, dict):
        return "dict"
    return "unknown"


def _type_mismatch(pred: Any, gold: Any) -> bool:
    if pred is None or gold is None:
        return False
    pt, gt = _gold_type(pred), _gold_type(gold)
    if pt == gt:
        return False
    if pt in ("int", "float") and gt in ("int", "float"):
        return False
    return True


def _short(text: str, n: int = 120) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _tool_names(row: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for c in row.get("predicted_calls") or []:
        if isinstance(c, dict) and c.get("name"):
            names.append(str(c["name"]))
    for tr in row.get("execution_trace") or []:
        if isinstance(tr, dict) and tr.get("name"):
            names.append(str(tr["name"]))
    return list(dict.fromkeys(names))


def _gold_tool_names(row: Dict[str, Any]) -> List[str]:
    return [
        str(c["name"]) for c in (row.get("gold_calls") or [])
        if isinstance(c, dict) and c.get("name")
    ]


def _evidence_snippet(row: Dict[str, Any], failure_type: str) -> str:
    parts: List[str] = []
    err = row.get("error_category") or row.get("execution_error") or ""
    if err:
        parts.append(_short(str(err), 200))
    for tr in (row.get("execution_trace") or [])[:2]:
        if isinstance(tr, dict) and tr.get("error"):
            parts.append(f"{tr.get('name')}: {_short(str(tr['error']), 120)}")
    raw = row.get("raw_completions") or []
    if raw:
        parts.append(_short(str(raw[-1]), 250))
    if not parts:
        parts.append(
            f"pred={row.get('predicted_final')!r} gold={row.get('gold_answer')!r} "
            f"stopped={row.get('stopped')}"
        )
    return _short(" | ".join(parts), 400)


# ---------------------------------------------------------------------------
# Failure + problem classification
# ---------------------------------------------------------------------------


def detect_failure_type(row: Dict[str, Any]) -> str:
    if row.get("passed"):
        return ""

    cat = str(row.get("error_category") or "")
    stopped = str(row.get("stopped") or "")
    verdict = str(row.get("verdict_reason") or "")
    n_tools = int(row.get("num_tool_calls") or 0)
    gold_calls = row.get("gold_calls") or []

    if cat.startswith("malformed_tool_call"):
        return "TOOL_FORMAT_ERROR"

    if stopped in ("converged", "step_limit", "context_limit", "advance_error"):
        return "INCOMPLETE_OR_LOOPING"
    if verdict == "no_final_value" or row.get("predicted_final") is None:
        if stopped in ("converged", "step_limit", "context_limit"):
            return "INCOMPLETE_OR_LOOPING"
        if n_tools == 0 and gold_calls:
            return "REASONING_ERROR"

    if cat.startswith("unknown_function"):
        return "WRONG_TOOL_OR_ARGUMENTS"
    if "ibm_runtime_error:TypeError" in cat:
        return "TYPE_OR_FORMAT_ERROR"
    if cat.startswith("ibm_runtime_error"):
        return "WRONG_TOOL_OR_ARGUMENTS"

    if _type_mismatch(row.get("predicted_final"), row.get("gold_answer")):
        return "TYPE_OR_FORMAT_ERROR"

    pred_names = _tool_names(row)
    gold_names = _gold_tool_names(row)
    if gold_names and pred_names and pred_names[0] != gold_names[0]:
        return "WRONG_TOOL_OR_ARGUMENTS"

    if verdict == "executor_mismatch":
        if n_tools == 0 and gold_names:
            return "REASONING_ERROR"
        if n_tools > 0:
            return "REASONING_ERROR"
        return "REASONING_ERROR"

    if stopped == "no_more_calls" and n_tools == 0 and gold_names:
        return "REASONING_ERROR"

    if "json" in cat.lower() or "parse" in cat.lower():
        return "TOOL_FORMAT_ERROR"

    return "UNKNOWN_ERROR"


def detect_problem_types(
    question: str,
    gold: Any,
    tools: Sequence[Any],
    *,
    has_tool_calls: bool,
) -> List[str]:
    q = (question or "").lower()
    hits: List[str] = []
    for ptype, patterns in PROBLEM_KEYWORDS:
        if any(re.search(p, q, re.I) for p in patterns):
            hits.append(ptype)

    gt = _gold_type(gold)
    if gt == "list":
        hits.extend(["list_dict_json", "structured_output"])
    elif gt == "dict":
        hits.extend(["list_dict_json", "structured_output"])
    elif gt == "bool":
        hits.append("boolean_logic")

    if has_tool_calls or (tools and len(tools) > 0):
        hits.append("tool_use")

    if gt in ("list", "dict"):
        hits.append("structured_output")

    hits = list(dict.fromkeys(hits))
    if not hits:
        if re.search(r"\bcalculate\b|\bwhat is\b|\bhow many\b", q):
            hits.append("arithmetic")
        else:
            hits.append("unknown")
    return hits[:4]


def difficulty_bucket(pass_count: int, num_rollouts: int) -> str:
    if num_rollouts <= 0:
        return "unknown"
    if pass_count == 0:
        return "hard_failed"
    if pass_count == num_rollouts:
        return "easy"

    rate = pass_count / num_rollouts
    if rate >= 0.75:
        return "mostly_ok"
    if rate >= 0.375:
        return "unstable"
    return "mostly_failed"


def _recommended_action(
    bucket: str,
    main_failure: str,
    problem_types: List[str],
) -> str:
    actions: List[str] = []
    if bucket in ("hard_failed", "mostly_failed"):
        actions.append("Priority KD on gold trajectories")
    elif bucket == "unstable":
        actions.append("Robustness SFT / rejection sampling")
    if main_failure:
        actions.append(FAILURE_META.get(main_failure, {}).get("train", ""))
    if "tool_use" in problem_types and "REASONING_ERROR" in main_failure:
        actions.append("Teach tool use instead of mental math")
    return "; ".join(a for a in actions if a) or "Review manually"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_tasks(rows: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], Dict]:
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)

    rollout_records: List[Dict[str, Any]] = []
    task_records: List[Dict[str, Any]] = []

    for task_id, task_rows in sorted(by_task.items()):
        task_rows.sort(key=lambda x: x.get("rollout_idx", 0))
        first = task_rows[0]
        question = first.get("question") or ""
        gold = first.get("gold_answer")
        tools = first.get("tools") or []
        gtype = _gold_type(gold)
        n = len(task_rows)
        pass_count = sum(1 for r in task_rows if r.get("passed"))
        fail_count = n - pass_count
        pass_rate = pass_count / n if n else 0.0
        bucket = difficulty_bucket(pass_count, n)

        failure_counter: Counter = Counter()
        failed_examples: List[Dict] = []

        any_tools = any((r.get("num_tool_calls") or 0) > 0 for r in task_rows)
        problem_types = detect_problem_types(
            question, gold, tools, has_tool_calls=any_tools,
        )

        for r in task_rows:
            ft = detect_failure_type(r)
            if ft:
                failure_counter[ft] += 1
            rollout_records.append({
                "task_id": task_id,
                "rollout_idx": r.get("rollout_idx", 0),
                "passed": r.get("passed"),
                "failure_type": ft or "PASS",
                "problem_types": problem_types,
                "gold_answer_type": gtype,
                "prompt_short": _short(question, 140),
                "question": question,
                "gold_answer": gold,
                "predicted_final": r.get("predicted_final"),
                "num_tool_calls": r.get("num_tool_calls", 0),
                "num_steps": r.get("num_steps", 0),
                "stopped": r.get("stopped"),
                "error_category": r.get("error_category"),
                "evidence_snippet": _evidence_snippet(r, ft) if ft else "",
                "tool_names": _tool_names(r),
            })
            if not r.get("passed"):
                failed_examples.append(r)

        main_failure = failure_counter.most_common(1)[0][0] if failure_counter else ""
        rep = failed_examples[0] if failed_examples else first
        action = _recommended_action(bucket, main_failure, problem_types)

        task_records.append({
            "task_id": task_id,
            "num_rollouts": n,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_rate": round(pass_rate, 4),
            "difficulty_bucket": bucket,
            "problem_types": problem_types,
            "main_failure_type": main_failure,
            "gold_answer_type": gtype,
            "prompt_short": _short(question, 140),
            "question": question,
            "gold_answer": gold,
            "representative_failed_prediction": rep.get("predicted_final"),
            "representative_error": rep.get("error_category") or rep.get("verdict_reason"),
            "explanation": _short(
                f"{bucket}: {pass_count}/{n} passed. "
                + FAILURE_META.get(main_failure, {}).get("meaning", ""),
                400,
            ),
            "recommended_training_action": action,
        })

    overview = _build_overview(task_records, rollout_records)
    return task_records, rollout_records, overview


def _build_overview(tasks: List[Dict], rollouts: List[Dict]) -> Dict[str, Any]:
    n_tasks = len(tasks)
    n_roll = len(rollouts)
    passed = sum(1 for r in rollouts if r.get("passed"))
    overall = passed / n_roll if n_roll else 0.0
    buckets = Counter(t["difficulty_bucket"] for t in tasks)
    fail_rollouts = [r for r in rollouts if not r.get("passed")]
    ft_counter = Counter(r["failure_type"] for r in fail_rollouts if r.get("failure_type"))
    dom_fail = ft_counter.most_common(1)[0][0] if ft_counter else "N/A"

    pt_stats: Dict[str, Dict] = defaultdict(lambda: {"tasks": set(), "pass": 0, "total": 0})
    for t in tasks:
        for p in t["problem_types"]:
            pt_stats[p]["tasks"].add(t["task_id"])
            pt_stats[p]["pass"] += t["pass_count"]
            pt_stats[p]["total"] += t["num_rollouts"]

    hardest_pt = "N/A"
    lowest_rate = 2.0
    for p, st in pt_stats.items():
        rate = st["pass"] / st["total"] if st["total"] else 0
        if rate < lowest_rate:
            lowest_rate = rate
            hardest_pt = p

    return {
        "total_tasks": n_tasks,
        "total_rollouts": n_roll,
        "overall_pass_rate": round(overall, 4),
        "overall_pass_rate_percent": round(100 * overall, 2),
        "hard_failed_tasks": buckets.get("hard_failed", 0),
        "mostly_failed_tasks": buckets.get("mostly_failed", 0),
        "unstable_tasks": buckets.get("unstable", 0),
        "mostly_ok_tasks": buckets.get("mostly_ok", 0),
        "easy_tasks": buckets.get("easy", 0),
        "most_common_failure_type": dom_fail,
        "most_difficult_problem_type": hardest_pt,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def create_training_subsets(tasks: List[Dict]) -> Dict[str, List[Dict]]:
    def hard(t):
        return t["pass_count"] == 0

    def mostly_failed(t):
        n = t["num_rollouts"]
        thr = max(1, round(2 / 8 * n)) if n else 2
        return 0 < t["pass_count"] <= thr

    def unstable(t):
        n = t["num_rollouts"]
        lo = max(1, math.ceil(3 / 8 * n))
        hi = max(lo, math.floor(5 / 8 * n))
        return lo <= t["pass_count"] <= hi

    def by_failure(ft: str):
        return lambda t: t.get("main_failure_type") == ft

    subsets = {
        "hard_tasks": [t for t in tasks if hard(t)],
        "mostly_failed_tasks": [t for t in tasks if mostly_failed(t)],
        "unstable_tasks": [t for t in tasks if unstable(t)],
        "tool_format_tasks": [t for t in tasks if by_failure("TOOL_FORMAT_ERROR")(t)],
        "tool_argument_tasks": [t for t in tasks if by_failure("WRONG_TOOL_OR_ARGUMENTS")(t)],
        "structured_output_tasks": [t for t in tasks if by_failure("TYPE_OR_FORMAT_ERROR")(t)],
        "reasoning_tasks": [t for t in tasks if by_failure("REASONING_ERROR")(t)],
    }
    return subsets


def _summarize_subsets(subsets: Dict[str, List[Dict]]) -> List[Dict]:
    summaries = []
    for name, tasks in subsets.items():
        if not tasks:
            continue
        pt = Counter()
        ft = Counter()
        for t in tasks:
            for p in t["problem_types"]:
                pt[p] += 1
            if t["main_failure_type"]:
                ft[t["main_failure_type"]] += 1
        info = SUBSET_INFO.get(name, {})
        summaries.append({
            "subset": name,
            "num_tasks": len(tasks),
            "common_problem_types": [p for p, _ in pt.most_common(5)],
            "common_failure_types": [f for f, _ in ft.most_common(5)],
            "why_it_matters": info.get("why", ""),
            "recommended_use": info.get("use", ""),
            "avg_pass_rate": round(
                float(np.mean([t["pass_rate"] for t in tasks])), 4,
            ),
        })
    return summaries


def _problem_type_summary(tasks: List[Dict], rollouts: List[Dict]) -> List[Dict]:
    stats: Dict[str, Dict] = defaultdict(lambda: {
        "task_ids": set(), "pass": 0, "total": 0, "failures": Counter(),
    })
    task_map = {t["task_id"]: t for t in tasks}
    for r in rollouts:
        t = task_map.get(r["task_id"], {})
        for p in t.get("problem_types") or r.get("problem_types") or []:
            stats[p]["task_ids"].add(r["task_id"])
            stats[p]["total"] += 1
            if r.get("passed"):
                stats[p]["pass"] += 1
            elif r.get("failure_type"):
                stats[p]["failures"][r["failure_type"]] += 1

    rows = []
    for p in PROBLEM_TYPES:
        st = stats.get(p)
        if not st or st["total"] == 0:
            continue
        pr = st["pass"] / st["total"]
        top_fail = st["failures"].most_common(1)
        rows.append({
            "problem_type": p,
            "task_count": len(st["task_ids"]),
            "rollout_count": st["total"],
            "pass_rate": round(pr, 4),
            "most_common_failure": top_fail[0][0] if top_fail else "",
            "what_to_train": FAILURE_META.get(
                top_fail[0][0] if top_fail else "", {},
            ).get("train", "Review failures in this domain"),
        })
    rows.sort(key=lambda x: x["pass_rate"])
    return rows


def _failure_type_summary(tasks: List[Dict], rollouts: List[Dict]) -> List[Dict]:
    counter: Counter = Counter()
    task_sets: Dict[str, Set[str]] = defaultdict(set)
    for r in rollouts:
        ft = r.get("failure_type")
        if ft and ft != "PASS":
            counter[ft] += 1
            task_sets[ft].add(r["task_id"])

    total = sum(counter.values())
    rows = []
    for ft in FAILURE_TYPES:
        cnt = counter.get(ft, 0)
        if cnt == 0:
            continue
        meta = FAILURE_META[ft]
        rows.append({
            "failure_type": ft,
            "rollout_count": cnt,
            "affected_tasks": len(task_sets[ft]),
            "pct_of_failures": round(100 * cnt / total, 2) if total else 0,
            "what_it_means": meta["meaning"],
            "what_to_train": meta["train"],
        })
    rows.sort(key=lambda x: -x["rollout_count"])
    return rows


def _training_priorities(subset_summaries: List[Dict]) -> List[Dict]:
    priority_order = [
        "hard_tasks", "mostly_failed_tasks", "tool_format_tasks",
        "tool_argument_tasks", "reasoning_tasks", "structured_output_tasks",
        "unstable_tasks",
    ]
    by_name = {s["subset"]: s for s in subset_summaries}
    rows = []
    for i, name in enumerate(priority_order, 1):
        s = by_name.get(name)
        if not s:
            continue
        main_prob = s["common_problem_types"][0] if s["common_problem_types"] else "unknown"
        rows.append({
            "priority": i,
            "subset": name,
            "num_tasks": s["num_tasks"],
            "main_problem": main_prob,
            "recommended_action": s["recommended_use"],
        })
    return rows


# ---------------------------------------------------------------------------
# Report + export
# ---------------------------------------------------------------------------


def generate_report(
    path: str,
    overview: Dict,
    problem_summary: List[Dict],
    failure_summary: List[Dict],
    priorities: List[Dict],
    rollouts: List[Dict],
) -> None:
    lines = [
        "# ENSTFULL Rollout Analysis",
        "",
        f"Generated: {overview.get('generated_at', '')}",
        "",
        "## 1. Overview",
        "",
        f"- **Total tasks:** {overview['total_tasks']}",
        f"- **Total rollouts:** {overview['total_rollouts']}",
        f"- **Overall pass rate:** {overview['overall_pass_rate_percent']}%",
        f"- **Hard failed (0/{overview.get('expected_rollouts', 8)}):** "
        f"{overview['hard_failed_tasks']}",
        f"- **Most common failure:** {overview['most_common_failure_type']}",
        "",
        "## 2. What is the model bad at?",
        "",
    ]
    for row in problem_summary[:8]:
        lines.append(
            f"- **{row['problem_type']}** — pass rate {100*row['pass_rate']:.1f}%, "
            f"{row['task_count']} tasks"
        )
    lines.extend(["", "## 3. Why does the model fail?", ""])
    for row in failure_summary[:8]:
        lines.append(
            f"- **{row['failure_type']}** ({row['rollout_count']} rollouts, "
            f"{row['affected_tasks']} tasks): {row['what_it_means']}"
        )
    lines.extend(["", "### Evidence examples", ""])
    seen: Set[str] = set()
    for r in rollouts:
        if r.get("passed") or not r.get("evidence_snippet"):
            continue
        key = r["failure_type"]
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- `{r['failure_type']}` task `{r['task_id'][:8]}…`: "
            f"{_short(r['evidence_snippet'], 200)}"
        )
        if len(seen) >= 5:
            break

    lines.extend(["", "## 4. What should we teach next?", ""])
    for p in priorities:
        lines.append(
            f"{p['priority']}. **{p['subset']}** ({p['num_tasks']} tasks) — "
            f"main problem: {p['main_problem']}. {p['recommended_action']}"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _df_to_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def export_all(
    output_dir: str,
    tasks: List[Dict],
    rollouts: List[Dict],
    overview: Dict,
    problem_summary: List[Dict],
    failure_summary: List[Dict],
    subset_summaries: List[Dict],
    priorities: List[Dict],
    subsets: Dict[str, List[Dict]],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    subset_dir = os.path.join(output_dir, "training_subsets")
    os.makedirs(subset_dir, exist_ok=True)

    task_cols = [
        "task_id", "num_rollouts", "pass_count", "fail_count", "pass_rate",
        "difficulty_bucket", "problem_types", "main_failure_type",
        "gold_answer_type", "prompt_short", "gold_answer",
        "representative_failed_prediction", "representative_error",
        "explanation", "recommended_training_action",
    ]
    rollout_cols = [
        "task_id", "rollout_idx", "passed", "failure_type", "problem_types",
        "gold_answer_type", "prompt_short", "predicted_final", "gold_answer",
        "num_tool_calls", "stopped", "error_category", "evidence_snippet",
    ]

    def _serialize_lists(rows: List[Dict]) -> List[Dict]:
        out = []
        for r in rows:
            c = dict(r)
            for k, v in c.items():
                if isinstance(v, (list, dict)):
                    c[k] = json.dumps(v, ensure_ascii=False, default=str)
            out.append(c)
        return out

    _df_to_csv(pd.DataFrame(_serialize_lists(tasks)), os.path.join(output_dir, "task_level_analysis.csv"))
    _df_to_csv(pd.DataFrame(_serialize_lists(rollouts)), os.path.join(output_dir, "rollout_level_analysis.csv"))
    _df_to_csv(pd.DataFrame(problem_summary), os.path.join(output_dir, "problem_type_summary.csv"))
    _df_to_csv(pd.DataFrame(failure_summary), os.path.join(output_dir, "failure_type_summary.csv"))

    task_export_cols = [
        "task_id", "pass_count", "pass_rate", "difficulty_bucket",
        "problem_types", "main_failure_type", "prompt_short", "gold_answer",
        "recommended_training_action",
    ]
    for name, subset_tasks in subsets.items():
        if not subset_tasks:
            pd.DataFrame(columns=task_export_cols).to_csv(
                os.path.join(subset_dir, f"{name}.csv"), index=False,
            )
            continue
        _df_to_csv(
            pd.DataFrame(_serialize_lists(subset_tasks))[task_export_cols],
            os.path.join(subset_dir, f"{name}.csv"),
        )

    generate_report(
        os.path.join(output_dir, "analysis_report.md"),
        overview, problem_summary, failure_summary, priorities, rollouts,
    )

    readme = """# NESTFUL Rollout Analysis Outputs

Run:
```bash
python analyze_rollouts.py --input predictions.jsonl --output_dir analysis_outputs
```

Open `index.html` in a browser (double-click, no server needed).

Files:
- `task_level_analysis.csv` — one row per task
- `rollout_level_analysis.csv` — one row per rollout
- `training_subsets/` — CSV lists for SFT/KD
- `analysis_report.md` — human-readable summary
"""
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme)

    req = "pandas>=2.0\nnumpy>=1.24\n"
    with open(os.path.join(output_dir, "requirements.txt"), "w", encoding="utf-8") as fh:
        fh.write(req)

    dashboard_payload = {
        "overview": overview,
        "difficulty_buckets": [
            {"bucket": b, "count": sum(1 for t in tasks if t["difficulty_bucket"] == b)}
            for b in DIFFICULTY_BUCKETS
        ],
        "problem_type_summary": problem_summary,
        "failure_type_summary": failure_summary,
        "training_priorities": priorities,
        "subset_summaries": subset_summaries,
        "subsets": {k: [t["task_id"] for t in v] for k, v in subsets.items()},
        "tasks": tasks,
        "rollouts": rollouts,
    }
    generate_html_dashboard(
        os.path.join(output_dir, "index.html"),
        dashboard_payload,
    )


# ---------------------------------------------------------------------------
# Self-contained HTML dashboard
# ---------------------------------------------------------------------------


def generate_html_dashboard(path: str, data: Dict[str, Any]) -> None:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ENSTFULL Rollout Analysis</title>
<style>
:root{{--bg:#0f172a;--card:#1e293b;--text:#e2e8f0;--muted:#94a3b8;--accent:#3b82f6;--ok:#22c55e;--warn:#eab308;--bad:#ef4444}}
*{{box-sizing:border-box}}body{{margin:0;font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);line-height:1.5}}
header{{padding:1rem 1.5rem;background:#020617;border-bottom:1px solid #334155}}
h1{{margin:0;font-size:1.35rem}}h2{{margin:2rem 0 .75rem;font-size:1.1rem;color:#cbd5e1}}
.container{{max-width:1200px;margin:0 auto;padding:1rem 1.5rem 3rem}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:.75rem}}
.card{{background:var(--card);border-radius:8px;padding:.85rem;border:1px solid #334155}}
.card .label{{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}}
.card .value{{font-size:1.35rem;font-weight:700;margin-top:.25rem}}
.bar-row{{display:flex;align-items:center;gap:.5rem;margin:.35rem 0;font-size:.85rem}}
.bar-label{{width:110px;color:var(--muted)}}
.bar-track{{flex:1;height:22px;background:#334155;border-radius:4px;overflow:hidden}}
.bar-fill{{height:100%;background:var(--accent);display:flex;align-items:center;padding-left:6px;font-size:.72rem}}
table{{width:100%;border-collapse:collapse;font-size:.82rem;background:var(--card);border-radius:8px;overflow:hidden}}
th,td{{padding:.45rem .6rem;border-bottom:1px solid #334155;text-align:left;vertical-align:top}}
th{{background:#0f172a;color:var(--muted);cursor:pointer;user-select:none}}
tr:hover{{background:#273449}}
.filters{{display:flex;flex-wrap:wrap;gap:.5rem;margin:1rem 0}}
.filters input,.filters select{{background:#1e293b;color:var(--text);border:1px solid #475569;border-radius:6px;padding:.4rem .6rem}}
.btn{{background:var(--accent);color:#fff;border:none;border-radius:6px;padding:.45rem .75rem;cursor:pointer;font-size:.82rem;margin-right:.4rem}}
.btn:hover{{opacity:.9}}
.detail{{background:var(--card);border:1px solid #334155;border-radius:8px;padding:1rem;margin-top:1rem;display:none}}
.detail.open{{display:block}}
.detail pre{{white-space:pre-wrap;font-size:.78rem;background:#0f172a;padding:.75rem;border-radius:6px;overflow:auto;max-height:320px}}
.tag{{display:inline-block;background:#334155;padding:.1rem .45rem;border-radius:4px;font-size:.72rem;margin:.1rem}}
.pass{{color:var(--ok)}}.fail{{color:var(--bad)}}
.hint{{color:var(--muted);font-size:.85rem}}
</style>
</head>
<body>
<header><h1>What does the model fail on — and what to teach next?</h1></header>
<div class="container">
<section id="overview"><h2>A. Overview</h2><div class="cards" id="cards"></div></section>
<section id="priorities"><h2>B. Training priorities</h2><table id="prio-table"><thead><tr>
<th>Priority</th><th>Subset</th><th>Tasks</th><th>Main problem</th><th>Action</th>
</tr></thead><tbody></tbody></table></section>
<section id="difficulty"><h2>C. Difficulty distribution</h2><div id="diff-bars"></div></section>
<section id="problems"><h2>D. Problem types (lowest pass rate first)</h2>
<table id="prob-table"><thead><tr>
<th>Type</th><th>Tasks</th><th>Pass rate</th><th>Top failure</th><th>Train</th>
</tr></thead><tbody></tbody></table></section>
<section id="failures"><h2>E. Failure types</h2>
<table id="fail-table"><thead><tr>
<th>Type</th><th>Count</th><th>Tasks</th><th>Meaning</th><th>Train</th>
</tr></thead><tbody></tbody></table></section>
<section id="explorer"><h2>F. Task explorer</h2>
<p class="hint">Click a row for full prompt, gold answer, and all rollouts.</p>
<div class="filters">
<input id="search" placeholder="Search prompt…" style="min-width:200px"/>
<select id="f-bucket"><option value="">All buckets</option></select>
<select id="f-problem"><option value="">All problem types</option></select>
<select id="f-failure"><option value="">All failure types</option></select>
<select id="f-gold"><option value="">All gold types</option></select>
</div>
<button class="btn" onclick="exportCSV()">Export visible CSV</button>
<button class="btn" onclick="exportJSON()">Export visible JSON</button>
<span id="export-subsets"></span>
<table id="task-table"><thead><tr>
<th>task_id</th><th>pass</th><th>rate</th><th>bucket</th><th>problems</th>
<th>failure</th><th>prompt</th><th>gold</th><th>action</th>
</tr></thead><tbody></tbody></table>
<div id="detail" class="detail"></div></section>
</div>
<script>
const DATA = {payload};
const rolloutsByTask = {{}};
DATA.rollouts.forEach(r => {{
  (rolloutsByTask[r.task_id] ||= []).push(r);
}});
let visible = [...DATA.tasks];

function esc(s) {{
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}
function pct(x) {{ return (100 * x).toFixed(1) + '%'; }}

function renderOverview() {{
  const o = DATA.overview;
  const items = [
    ['Total tasks', o.total_tasks],
    ['Total rollouts', o.total_rollouts],
    ['Pass rate', o.overall_pass_rate_percent + '%'],
    ['Hard failed', o.hard_failed_tasks],
    ['Mostly failed', o.mostly_failed_tasks],
    ['Unstable', o.unstable_tasks],
    ['Top failure', o.most_common_failure_type],
    ['Hardest domain', o.most_difficult_problem_type],
  ];
  document.getElementById('cards').innerHTML = items.map(([l,v]) =>
    `<div class="card"><div class="label">${{esc(l)}}</div><div class="value">${{esc(v)}}</div></div>`
  ).join('');
}}

function renderBars() {{
  const max = Math.max(1, ...DATA.difficulty_buckets.map(b => b.count));
  const colors = {{easy:'#22c55e',mostly_ok:'#84cc16',unstable:'#eab308',mostly_failed:'#f97316',hard_failed:'#ef4444'}};
  document.getElementById('diff-bars').innerHTML = DATA.difficulty_buckets.map(b => {{
    const w = (100 * b.count / max).toFixed(1);
    return `<div class="bar-row"><div class="bar-label">${{esc(b.bucket)}}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${{w}}%;background:${{colors[b.bucket]||'#3b82f6'}}">${{b.count}}</div></div></div>`;
  }}).join('');
}}

function fillTable(id, rows, cols) {{
  document.querySelector(`#${{id}} tbody`).innerHTML = rows.map(r =>
    '<tr>' + cols.map(c => `<td>${{esc(typeof c==='function'?c(r):r[c])}}</td>`).join('') + '</tr>'
  ).join('');
}}

function renderTables() {{
  fillTable('prio-table', DATA.training_priorities, ['priority','subset','num_tasks','main_problem','recommended_action']);
  fillTable('prob-table', DATA.problem_type_summary, [
    'problem_type','task_count', r=>pct(r.pass_rate),'most_common_failure','what_to_train'
  ]);
  fillTable('fail-table', DATA.failure_type_summary, [
    'failure_type','rollout_count','affected_tasks','what_it_means','what_to_train'
  ]);
}}

function populateFilters() {{
  const buckets = [...new Set(DATA.tasks.map(t => t.difficulty_bucket))];
  const probs = [...new Set(DATA.tasks.flatMap(t => t.problem_types))];
  const fails = [...new Set(DATA.tasks.map(t => t.main_failure_type).filter(Boolean))];
  const golds = [...new Set(DATA.tasks.map(t => t.gold_answer_type))];
  const add = (id, vals) => vals.sort().forEach(v => {{
    const o = document.createElement('option'); o.value = v; o.textContent = v;
    document.getElementById(id).appendChild(o);
  }});
  add('f-bucket', buckets); add('f-problem', probs); add('f-failure', fails); add('f-gold', golds);
}}

function applyFilters() {{
  const q = document.getElementById('search').value.toLowerCase();
  const b = document.getElementById('f-bucket').value;
  const p = document.getElementById('f-problem').value;
  const f = document.getElementById('f-failure').value;
  const g = document.getElementById('f-gold').value;
  visible = DATA.tasks.filter(t => {{
    if (b && t.difficulty_bucket !== b) return false;
    if (p && !(t.problem_types||[]).includes(p)) return false;
    if (f && t.main_failure_type !== f) return false;
    if (g && t.gold_answer_type !== g) return false;
    if (q && !(t.prompt_short||'').toLowerCase().includes(q) && !(t.question||'').toLowerCase().includes(q)) return false;
    return true;
  }});
  renderTaskTable();
}}

function renderTaskTable() {{
  document.querySelector('#task-table tbody').innerHTML = visible.map(t => `
    <tr data-id="${{esc(t.task_id)}}" style="cursor:pointer">
      <td>${{esc(t.task_id.slice(0,8))}}…</td>
      <td>${{t.pass_count}}/${{t.num_rollouts}}</td>
      <td>${{pct(t.pass_rate)}}</td>
      <td>${{esc(t.difficulty_bucket)}}</td>
      <td>${{(t.problem_types||[]).map(p=>`<span class="tag">${{esc(p)}}</span>`).join('')}}</td>
      <td>${{esc(t.main_failure_type)}}</td>
      <td>${{esc(t.prompt_short)}}</td>
      <td>${{esc(JSON.stringify(t.gold_answer))}}</td>
      <td>${{esc(t.recommended_training_action)}}</td>
    </tr>`).join('');
  document.querySelectorAll('#task-table tbody tr').forEach(tr => {{
    tr.onclick = () => showDetail(tr.dataset.id);
  }});
}}

function showDetail(taskId) {{
  const t = DATA.tasks.find(x => x.task_id === taskId);
  const rolls = rolloutsByTask[taskId] || [];
  const el = document.getElementById('detail');
  el.className = 'detail open';
  el.innerHTML = `<h3>${{esc(taskId)}}</h3>
    <p><b>Prompt:</b></p><pre>${{esc(t.question)}}</pre>
    <p><b>Gold:</b> <code>${{esc(JSON.stringify(t.gold_answer))}}</code></p>
    <h4>Rollouts (${{rolls.length}})</h4>
    ${{rolls.map(r => `<div style="margin:.75rem 0;padding:.5rem;border-left:3px solid #475569">
      <div>#${{r.rollout_idx}} <span class="${{r.passed?'pass':'fail'}}">${{r.passed?'PASS':'FAIL'}}</span>
      — ${{esc(r.failure_type)}} — tools:${{r.num_tool_calls}} stopped:${{esc(r.stopped)}}</div>
      <div>pred: <code>${{esc(JSON.stringify(r.predicted_final))}}</code></div>
      <div class="hint">${{esc(r.evidence_snippet)}}</div>
    </div>`).join('')}}`;
  el.scrollIntoView({{behavior:'smooth', block:'nearest'}});
}}

function exportCSV() {{
  const cols = ['task_id','pass_count','pass_rate','difficulty_bucket','problem_types','main_failure_type','prompt_short','gold_answer','recommended_training_action'];
  const lines = [cols.join(',')];
  visible.forEach(t => {{
    lines.push(cols.map(c => `"${{String(typeof t[c]==='object'?JSON.stringify(t[c]):t[c]??'').replace(/"/g,'""')}}"`).join(','));
  }});
  download('visible_tasks.csv', lines.join('\\n'));
}}

function exportJSON() {{
  download('visible_tasks.json', JSON.stringify(visible, null, 2));
}}

function download(name, text) {{
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], {{type:'text/plain'}}));
  a.download = name; a.click();
}}

function renderSubsetExports() {{
  const el = document.getElementById('export-subsets');
  Object.keys(DATA.subsets||{{}}).forEach(name => {{
    const btn = document.createElement('button');
    btn.className = 'btn'; btn.textContent = name;
    btn.onclick = () => {{
      const ids = new Set(DATA.subsets[name]);
      const rows = DATA.tasks.filter(t => ids.has(t.task_id));
      download(name + '.json', JSON.stringify(rows, null, 2));
    }};
    el.appendChild(btn);
  }});
}}

['search','f-bucket','f-problem','f-failure','f-gold'].forEach(id =>
  document.getElementById(id).addEventListener('input', applyFilters)
);
renderOverview(); renderBars(); renderTables(); populateFilters(); applyFilters(); renderSubsetExports();
</script>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Analyze NESTFUL multiturn rollouts for SFT/KD.")
    p.add_argument("--input", required=True, help="Path to *_multiturn_predictions.jsonl")
    p.add_argument("--output_dir", default="analysis_outputs", help="Output directory")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        logger.error("Input not found: %s", args.input)
        return 1

    logger.info("Loading %s", args.input)
    raw = load_jsonl(args.input)
    if not raw:
        logger.error("No records loaded.")
        return 1

    mapping = infer_fields(raw)
    rows = [normalize_record(r, mapping) for r in raw]
    logger.info("Loaded %d rollouts (%d unique tasks)", len(rows), len({r['task_id'] for r in rows}))

    tasks, rollouts, overview = aggregate_tasks(rows)
    subsets = create_training_subsets(tasks)
    subset_summaries = _summarize_subsets(subsets)
    problem_summary = _problem_type_summary(tasks, rollouts)
    failure_summary = _failure_type_summary(tasks, rollouts)
    priorities = _training_priorities(subset_summaries)

    export_all(
        args.output_dir, tasks, rollouts, overview,
        problem_summary, failure_summary, subset_summaries, priorities, subsets,
    )

    logger.info("Done. Open %s", os.path.abspath(os.path.join(args.output_dir, "index.html")))
    print(f"\nOverview: {overview['total_tasks']} tasks, "
          f"{overview['overall_pass_rate_percent']}% pass rate")
    print(f"Hard failed: {overview['hard_failed_tasks']} | "
          f"Top failure: {overview['most_common_failure_type']}")
    print(f"Report: {os.path.join(args.output_dir, 'analysis_report.md')}")
    print(f"Dashboard: {os.path.join(args.output_dir, 'index.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
