#!/usr/bin/env python3
"""NESTFUL results analyzer — CSV/PNG/report + dashboard_data.json."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Failure mode metadata
# ---------------------------------------------------------------------------

FAILURE_MODES = [
    "TOOL_SCHEMA_ERROR",
    "TOOL_SELECTION_ERROR",
    "TOOL_ARGUMENT_ERROR",
    "RUNTIME_TYPE_ERROR",
    "FINAL_ANSWER_ERROR",
    "STRUCTURED_OUTPUT_ERROR",
    "CONTROL_FLOW_ERROR",
    "REASONING_PLANNING_ERROR",
    "UNKNOWN_ERROR",
]

FAILURE_MODE_META: Dict[str, Dict[str, str]] = {
    "TOOL_SCHEMA_ERROR": {
        "explanation": (
            "The model knows that a tool should be used, but does not follow "
            "the required tool-call schema. This suggests a schema-following "
            "problem rather than a pure reasoning problem."
        ),
        "training_strategy": "Schema repair SFT with valid name+arguments examples.",
    },
    "TOOL_SELECTION_ERROR": {
        "explanation": (
            "The model invoked a tool name that is not available in the task "
            "toolbox or hallucinated a function."
        ),
        "training_strategy": "Tool-selection distillation from gold trajectories.",
    },
    "TOOL_ARGUMENT_ERROR": {
        "explanation": (
            "The model chose a plausible tool but passed wrong arguments or "
            "misresolved variables, causing runtime failures."
        ),
        "training_strategy": "Argument-resolution and API-usage fine-tuning.",
    },
    "RUNTIME_TYPE_ERROR": {
        "explanation": (
            "Arguments or intermediate values had the wrong Python type "
            "(e.g. list instead of string), causing TypeError at execution."
        ),
        "training_strategy": "Type-aware API training with typed examples.",
    },
    "FINAL_ANSWER_ERROR": {
        "explanation": (
            "Tool use may have worked, but the final numeric or semantic "
            "answer does not match gold."
        ),
        "training_strategy": "Multi-step reasoning distillation on hard tasks.",
    },
    "STRUCTURED_OUTPUT_ERROR": {
        "explanation": (
            "The model may solve the task semantically but fails to return "
            "the answer in the required type or format."
        ),
        "training_strategy": "Typed output and format-constrained SFT.",
    },
    "CONTROL_FLOW_ERROR": {
        "explanation": (
            "The rollout ended due to loops (converged), step limits, or "
            "context overflow before a clean final answer."
        ),
        "training_strategy": "Robustness training and better stop/continue policies.",
    },
    "REASONING_PLANNING_ERROR": {
        "explanation": (
            "The model fails to decompose the task correctly, skips tool use "
            "when needed, or loses track of intermediate steps."
        ),
        "training_strategy": "Plan-and-execute distillation from successful rollouts.",
    },
    "UNKNOWN_ERROR": {
        "explanation": "Failure pattern did not match a known diagnostic bucket.",
        "training_strategy": "Manual review and targeted counter-examples.",
    },
}

DOMAIN_RULES: List[Tuple[str, List[str], List[str]]] = [
    ("string_text_regex", [
        r"\bregex\b", r"\bsubstring\b", r"\bcamel\b", r"\bunderscore\b",
        r"\bstring\b", r"\btext\b", r"\bparse.*string\b", r"\bencode\b",
        r"\bdecode\b", r"\bhtml\b", r"\burl\b",
    ], [
        "string_", "regex", "replace_substring", "find_first", "wrap_text",
        "format_html", "camel", "underscore", "parse_string",
    ]),
    ("list_array_matrix", [
        r"\blist\b", r"\barray\b", r"\bmatrix\b", r"\bvector\b",
        r"\bgrid\b", r"\b2d\b", r"\btranspose\b",
    ], [
        "list_", "matrix", "vector", "numpy", "linear_search", "contains_value",
        "find_diff", "list_operations",
    ]),
    ("datetime_time", [
        r"\bdatetime\b", r"\btimestamp\b", r"\bdate\b", r"\btime\b",
        r"\bday\b", r"\bmonth\b", r"\byear\b",
    ], ["datetime", "date_", "time_", "timestamp"]),
    ("dict_object_json", [
        r"\bdict\b", r"\bjson\b", r"\bobject\b", r"\bdictionary\b",
        r"\bkey\b", r"\bvalue\b",
    ], ["dict", "json", "prefix_dict", "lookup_", "add_if_exists"]),
    ("probability_statistics", [
        r"\bprobability\b", r"\bprobabil", r"\brandom\b", r"\bpercent\b",
        r"\bmean\b", r"\bmedian\b", r"\bvariance\b", r"\bdistribution\b",
    ], ["probability", "percent", "choose", "permutation", "calculate_error"]),
    ("geometry_spatial", [
        r"\bcube\b", r"\btriangle\b", r"\bcircle\b", r"\barea\b",
        r"\bvolume\b", r"\bperimeter\b", r"\bangle\b", r"\bcoordinate\b",
    ], ["point", "distance", "mutate_point", "area", "perimeter"]),
    ("formatting_structured_output", [
        r"\bformat\b", r"\bpercentage\b", r"\bconvert\b", r"\boutput\b",
        r"\breturn.*as\b", r"\btype\b",
    ], ["format_", "convert_", "wrap_", "generate_label"]),
    ("arithmetic_numeric", [
        r"\bsum\b", r"\bproduct\b", r"\bdivide\b", r"\bmultiply\b",
        r"\bprime\b", r"\bgcd\b", r"\blcm\b", r"\binteger\b", r"\bnumber\b",
        r"\bcalculate\b", r"\bequation\b",
    ], [
        "add", "subtract", "multiply", "divide", "gcd", "lcm", "power",
        "mod", "factorial", "sqrt",
    ]),
    ("multi_step_reasoning", [
        r"\bthen\b", r"\bstep\b", r"\bfirst\b.*\bthen\b", r"\bafter\b",
        r"\bchain\b", r"\bsequence\b",
    ], []),
    ("tool_orchestration", [
        r"\btool\b", r"\bfunction\b", r"\busing\b.*\band\b",
    ], []),
]

DIFFICULTY_BUCKETS = [
    "hard_fail",
    "mostly_fail",
    "unstable",
    "mostly_pass",
    "easy",
]

TRAINING_SUBSETS = {
    "hard_fail_subset": lambda t: t["pass_count"] == 0,
    "mostly_fail_subset": lambda t: 0 < t["pass_count"] <= 2,
    "unstable_subset": lambda t: 3 <= t["pass_count"] <= 5,
    "schema_repair_subset": lambda t: "TOOL_SCHEMA_ERROR" in t["all_failure_modes"],
    "typed_api_subset": lambda t: bool(
        {"RUNTIME_TYPE_ERROR", "STRUCTURED_OUTPUT_ERROR"} & set(t["all_failure_modes"])
    ),
    "reasoning_subset": lambda t: bool(
        {"FINAL_ANSWER_ERROR", "REASONING_PLANNING_ERROR"} & set(t["all_failure_modes"])
    ),
}

SUBSET_RECOMMENDATIONS = {
    "hard_fail_subset": "Targeted SFT/KD on consistently failing tasks.",
    "mostly_fail_subset": "Priority distillation — model rarely succeeds.",
    "unstable_subset": "Robustness and self-consistency training.",
    "schema_repair_subset": "Tool-call format repair SFT.",
    "typed_api_subset": "Type-aware API and structured output training.",
    "reasoning_subset": "Multi-step reasoning distillation.",
}


def _gold_type(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int) and not isinstance(val, bool):
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


def _infer_problem_domains(question: str, tools: Sequence[Dict[str, Any]]) -> List[str]:
    q = (question or "").lower()
    tool_names = " ".join(
        (t.get("name") or "") for t in (tools or []) if isinstance(t, dict)
    ).lower()
    hits: List[str] = []
    for domain, q_patterns, tool_patterns in DOMAIN_RULES:
        score = 0
        for pat in q_patterns:
            if re.search(pat, q, re.I):
                score += 2
        for pat in tool_patterns:
            if pat in tool_names:
                score += 1
        if score >= 2:
            hits.append(domain)
    if not hits:
        if re.search(r"\bhow many\b|\bwhat is\b|\bfind\b", q):
            hits.append("general_word_problem")
        else:
            hits.append("unknown")
    return hits[:3]


def _malformed_detected(row: Dict[str, Any]) -> bool:
    cat = row.get("error_category") or ""
    if cat.startswith("malformed_tool_call"):
        return True
    err = row.get("execution_error") or ""
    return err.startswith("malformed_tool_call")


def classify_failure_modes(row: Dict[str, Any]) -> List[str]:
    if row.get("status") == "completed" or (row.get("score") or 0) >= 1.0:
        return []

    modes: List[str] = []
    cat = row.get("error_category") or ""
    stopped = row.get("stopped") or ""
    verdict = row.get("verdict_reason") or ""
    n_tools = row.get("num_tool_calls") or 0

    if cat.startswith("malformed_tool_call"):
        modes.append("TOOL_SCHEMA_ERROR")
    if cat.startswith("unknown_function"):
        modes.append("TOOL_SELECTION_ERROR")
    if "ibm_runtime_error:TypeError" in cat:
        modes.append("RUNTIME_TYPE_ERROR")
    elif cat.startswith("ibm_runtime_error"):
        modes.append("TOOL_ARGUMENT_ERROR")
    if stopped in ("converged", "step_limit", "context_limit"):
        modes.append("CONTROL_FLOW_ERROR")
    if stopped == "no_more_calls" and n_tools == 0:
        modes.append("REASONING_PLANNING_ERROR")
    if verdict == "executor_mismatch":
        if _type_mismatch(row.get("predicted_final"), row.get("gold_answer")):
            modes.append("STRUCTURED_OUTPUT_ERROR")
        elif "FINAL_ANSWER_ERROR" not in modes:
            modes.append("FINAL_ANSWER_ERROR")
    if verdict == "no_final_value" and "STRUCTURED_OUTPUT_ERROR" not in modes:
        modes.append("STRUCTURED_OUTPUT_ERROR")

    if not modes:
        modes.append("UNKNOWN_ERROR")
    return list(dict.fromkeys(modes))


def difficulty_bucket(pass_count: int, num_rollouts: int) -> str:
    if num_rollouts <= 0:
        return "unknown"
    if pass_count == 0:
        return "hard_fail"
    if pass_count <= max(1, num_rollouts // 4):
        return "mostly_fail"
    rate = pass_count / num_rollouts
    if rate >= 1.0:
        return "easy"
    if rate >= 0.75:
        return "mostly_pass"
    if rate > 0.25:
        return "unstable"
    return "mostly_fail"


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


def _trace_snippet(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    for tr in (row.get("execution_trace") or [])[:3]:
        if not isinstance(tr, dict):
            continue
        if tr.get("error"):
            parts.append(f"{tr.get('name')}: {tr['error']}")
        elif tr.get("result") is not None:
            parts.append(f"{tr.get('name')} -> {tr['result']}")
    raw = row.get("raw_completions") or []
    if raw:
        parts.append(_short(str(raw[-1]), 300))
    return "\n".join(parts)[:800]


def _dominant_mode(modes: List[str]) -> str:
    if not modes:
        return ""
    priority = FAILURE_MODES
    for m in priority:
        if m in modes:
            return m
    return modes[0]


def load_rollouts(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze_rollouts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        domains = _infer_problem_domains(question, tools)
        gtype = _gold_type(gold)
        num_rollouts = len(task_rows)
        pass_count = sum(1 for r in task_rows if r.get("status") == "completed")
        fail_count = num_rollouts - pass_count
        pass_rate = pass_count / num_rollouts if num_rollouts else 0.0
        bucket = difficulty_bucket(pass_count, num_rollouts)

        all_modes: Set[str] = set()
        malformed_count = 0
        tool_call_counts: List[int] = []
        failed_examples: List[Dict[str, Any]] = []

        for r in task_rows:
            modes = classify_failure_modes(r)
            all_modes.update(modes)
            if _malformed_detected(r):
                malformed_count += 1
            tool_call_counts.append(r.get("num_tool_calls") or 0)

            rollout_records.append({
                "task_id": task_id,
                "rollout_id": r.get("rollout_idx", 0),
                "passed": r.get("status") == "completed",
                "detected_failure_modes": modes,
                "dominant_failure_mode": _dominant_mode(modes),
                "problem_domains": domains,
                "gold_answer_type": gtype,
                "predicted_answer": r.get("predicted_final"),
                "gold_answer": gold,
                "tool_names_used": _tool_names(r),
                "tool_calls": r.get("predicted_calls") or [],
                "malformed_tool_call_detected": _malformed_detected(r),
                "error_message": r.get("error_category") or r.get("execution_error") or "",
                "stopped": r.get("stopped"),
                "verdict_reason": r.get("verdict_reason"),
                "evidence_snippet": _short(r.get("error_category") or "", 200),
                "trace_snippet": _trace_snippet(r),
                "num_tool_calls": r.get("num_tool_calls") or 0,
                "num_steps": r.get("num_steps") or 0,
            })

            if r.get("status") != "completed":
                failed_examples.append({
                    "prediction": r.get("predicted_final"),
                    "error": r.get("error_category") or r.get("verdict_reason"),
                    "modes": modes,
                })

        rep_fail = failed_examples[0] if failed_examples else {}
        mode_counter = Counter()
        for r in task_rows:
            for m in classify_failure_modes(r):
                mode_counter[m] += 1
        dom = mode_counter.most_common(1)[0][0] if mode_counter else ""

        explanation_parts = []
        if bucket == "hard_fail":
            explanation_parts.append("Fails on every rollout — systematic difficulty.")
        elif bucket == "unstable":
            explanation_parts.append("Mixed pass/fail — model sometimes succeeds.")
        if dom:
            explanation_parts.append(FAILURE_MODE_META.get(dom, {}).get("explanation", ""))

        subsets: List[str] = []
        task_stub = {
            "pass_count": pass_count,
            "all_failure_modes": list(all_modes),
        }
        for name, pred in TRAINING_SUBSETS.items():
            if pred(task_stub):
                subsets.append(name)

        task_records.append({
            "task_id": task_id,
            "num_rollouts": num_rollouts,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_rate": round(pass_rate, 4),
            "difficulty_score": round(1.0 - pass_rate, 4),
            "difficulty_bucket": bucket,
            "dominant_failure_mode": dom,
            "all_failure_modes": sorted(all_modes),
            "problem_domains": domains,
            "gold_answer_type": gtype,
            "avg_tool_calls": round(
                sum(tool_call_counts) / len(tool_call_counts), 2
            ) if tool_call_counts else 0.0,
            "malformed_tool_call_count": malformed_count,
            "task_prompt": question,
            "task_prompt_short": _short(question, 140),
            "gold_answer": gold,
            "representative_failed_prediction": rep_fail.get("prediction"),
            "representative_error": rep_fail.get("error"),
            "explanation": " ".join(explanation_parts)[:500],
            "recommended_training_subset": subsets,
        })

    return _build_aggregates(task_records, rollout_records, rows)


def _build_aggregates(
    tasks: List[Dict[str, Any]],
    rollouts: List[Dict[str, Any]],
    raw_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    num_tasks = len(tasks)
    num_rollouts = len(rollouts)
    passed = sum(1 for r in rollouts if r["passed"])
    overall_pass = passed / num_rollouts if num_rollouts else 0.0

    bucket_counts = Counter(t["difficulty_bucket"] for t in tasks)
    failed_rollouts = [r for r in rollouts if not r["passed"]]
    mode_rollout = Counter()
    mode_tasks: Dict[str, Set[str]] = defaultdict(set)
    for r in failed_rollouts:
        for m in r["detected_failure_modes"]:
            mode_rollout[m] += 1
            mode_tasks[m].add(r["task_id"])

    dom_mode = mode_rollout.most_common(1)[0][0] if mode_rollout else "N/A"

    domain_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "task_ids": set(), "rollout_count": 0, "passed": 0, "modes": Counter(),
    })
    for t in tasks:
        for d in t["problem_domains"]:
            domain_stats[d]["task_ids"].add(t["task_id"])
    for r in rollouts:
        for d in r["problem_domains"]:
            domain_stats[d]["rollout_count"] += 1
            if r["passed"]:
                domain_stats[d]["passed"] += 1
            for m in r["detected_failure_modes"]:
                domain_stats[d]["modes"][m] += 1

    problem_domain_summary = []
    for domain, st in sorted(domain_stats.items()):
        rc = st["rollout_count"]
        pr = st["passed"] / rc if rc else 0.0
        problem_domain_summary.append({
            "problem_domain": domain,
            "task_count": len(st["task_ids"]),
            "rollout_count": rc,
            "pass_rate": round(pr, 4),
            "fail_rate": round(1 - pr, 4),
            "most_common_failure_modes": [
                m for m, _ in st["modes"].most_common(3)
            ],
            "interpretation": (
                f"Pass rate {100*pr:.1f}% across {len(st['task_ids'])} tasks."
            ),
        })
    problem_domain_summary.sort(key=lambda x: x["pass_rate"])

    hardest_domain = (
        problem_domain_summary[0]["problem_domain"]
        if problem_domain_summary else "N/A"
    )

    failure_mode_summary = []
    total_failed = len(failed_rollouts)
    for mode in FAILURE_MODES:
        cnt = mode_rollout.get(mode, 0)
        if cnt == 0:
            continue
        affected = len(mode_tasks[mode])
        task_pass_rates = [
            t["pass_rate"] for t in tasks if t["task_id"] in mode_tasks[mode]
        ]
        avg_pr = sum(task_pass_rates) / len(task_pass_rates) if task_pass_rates else 0.0
        meta = FAILURE_MODE_META[mode]
        failure_mode_summary.append({
            "failure_mode": mode,
            "failed_rollout_count": cnt,
            "affected_task_count": affected,
            "percentage_of_failed_rollouts": round(
                100 * cnt / total_failed, 2
            ) if total_failed else 0.0,
            "average_task_pass_rate": round(avg_pr, 4),
            "interpretation": meta["explanation"],
            "recommended_training_strategy": meta["training_strategy"],
        })
    failure_mode_summary.sort(key=lambda x: -x["failed_rollout_count"])

    gold_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"p": 0, "t": 0})
    for r in rollouts:
        gt = r["gold_answer_type"]
        gold_stats[gt]["t"] += 1
        if r["passed"]:
            gold_stats[gt]["p"] += 1
    gold_answer_type_summary = [
        {
            "gold_answer_type": gt,
            "rollout_count": st["t"],
            "pass_rate": round(st["p"] / st["t"], 4) if st["t"] else 0.0,
        }
        for gt, st in sorted(gold_stats.items())
    ]

    heatmap: List[Dict[str, Any]] = []
    domain_mode: Dict[Tuple[str, str], int] = Counter()
    for r in failed_rollouts:
        for d in r["problem_domains"]:
            for m in r["detected_failure_modes"]:
                domain_mode[(d, m)] += 1
    for (d, m), cnt in domain_mode.items():
        heatmap.append({
            "problem_domain": d,
            "failure_mode": m,
            "failed_rollout_count": cnt,
        })

    tool_malformed: Counter = Counter()
    tool_failed: Counter = Counter()
    for r in rollouts:
        if not r["passed"]:
            for tn in r["tool_names_used"]:
                tool_failed[tn] += 1
        if r["malformed_tool_call_detected"]:
            for tn in r["tool_names_used"] or ["(unknown)"]:
                tool_malformed[tn] += 1

    passed_tc = [r["num_tool_calls"] for r in rollouts if r["passed"]]
    failed_tc = [r["num_tool_calls"] for r in rollouts if not r["passed"]]
    avg_pass_tc = sum(passed_tc) / len(passed_tc) if passed_tc else 0.0
    avg_fail_tc = sum(failed_tc) / len(failed_tc) if failed_tc else 0.0

    malformed_total = sum(1 for r in rollouts if r["malformed_tool_call_detected"])
    malformed_examples: List[Dict[str, str]] = []
    seen_malformed: Set[str] = set()
    for r in rollouts:
        if not r["malformed_tool_call_detected"]:
            continue
        key = (r.get("error_message") or "")[:120]
        if key in seen_malformed:
            continue
        seen_malformed.add(key)
        tools = r.get("tool_names_used") or []
        malformed_examples.append({
            "tool_name": tools[0] if tools else "(unknown)",
            "error_message": (r.get("error_message") or "")[:300],
            "task_id": r["task_id"],
            "rollout_id": str(r["rollout_id"]),
        })
        if len(malformed_examples) >= 25:
            break

    training_subsets = []
    for name, pred in TRAINING_SUBSETS.items():
        subset_tasks = [t for t in tasks if pred({
            "pass_count": t["pass_count"],
            "all_failure_modes": t["all_failure_modes"],
        })]
        if not subset_tasks:
            continue
        tids = {t["task_id"] for t in subset_tasks}
        sub_rollouts = [r for r in rollouts if r["task_id"] in tids]
        dom_counter = Counter()
        mode_counter = Counter()
        for t in subset_tasks:
            for d in t["problem_domains"]:
                dom_counter[d] += 1
            for m in t["all_failure_modes"]:
                mode_counter[m] += 1
        training_subsets.append({
            "subset_name": name,
            "task_count": len(subset_tasks),
            "rollout_count": len(sub_rollouts),
            "average_pass_rate": round(
                sum(t["pass_rate"] for t in subset_tasks) / len(subset_tasks), 4
            ),
            "top_problem_domains": [d for d, _ in dom_counter.most_common(5)],
            "top_failure_modes": [m for m, _ in mode_counter.most_common(5)],
            "recommended_use": SUBSET_RECOMMENDATIONS.get(name, ""),
            "task_ids": [t["task_id"] for t in subset_tasks],
        })

    difficulty_bucket_summary = [
        {"bucket": b, "task_count": bucket_counts.get(b, 0)}
        for b in DIFFICULTY_BUCKETS
    ]

    pass_count_hist: Counter = Counter(t["pass_count"] for t in tasks)

    overview = {
        "total_tasks": num_tasks,
        "total_rollouts": num_rollouts,
        "overall_pass_rate": round(overall_pass, 4),
        "overall_pass_rate_percent": round(100 * overall_pass, 2),
        "hard_fail_tasks": bucket_counts.get("hard_fail", 0),
        "mostly_fail_tasks": bucket_counts.get("mostly_fail", 0),
        "unstable_tasks": bucket_counts.get("unstable", 0),
        "easy_tasks": bucket_counts.get("easy", 0),
        "mostly_pass_tasks": bucket_counts.get("mostly_pass", 0),
        "most_common_failure_mode": dom_mode,
        "most_problematic_domain": hardest_domain,
        "malformed_tool_call_percent": round(
            100 * malformed_total / num_rollouts, 2
        ) if num_rollouts else 0.0,
        "malformed_tool_call_count": malformed_total,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "overview": overview,
        "interpretations": {
            "task_level": (
                "Task-level pass rate is more informative than rollout-level "
                "accuracy because it reveals whether a task is systematically "
                "hard or merely unstable."
            ),
            "hard_fail": (
                "Hard-fail tasks are good candidates for targeted distillation "
                "because the model fails consistently across rollouts."
            ),
            "unstable": (
                "Unstable tasks are useful for robustness training because the "
                "model sometimes finds the correct solution but does not do so "
                "reliably."
            ),
            "schema_vs_reasoning": (
                "Tool schema errors should be separated from reasoning errors "
                "because they require different training data."
            ),
            "structured_output": (
                "Structured output failures indicate that the model may "
                "understand the task but fails to match the expected type "
                "or format."
            ),
        },
        "difficulty_bucket_summary": difficulty_bucket_summary,
        "pass_count_histogram": [
            {"pass_count": k, "task_count": v}
            for k, v in sorted(pass_count_hist.items())
        ],
        "problem_domain_summary": problem_domain_summary,
        "failure_mode_summary": failure_mode_summary,
        "gold_answer_type_summary": gold_answer_type_summary,
        "tool_error_summary": {
            "malformed_by_tool": dict(tool_malformed.most_common(30)),
            "failed_rollout_by_tool": dict(tool_failed.most_common(30)),
            "avg_tool_calls_passed": round(avg_pass_tc, 2),
            "avg_tool_calls_failed": round(avg_fail_tc, 2),
            "malformed_examples": malformed_examples,
        },
        "domain_failure_heatmap": heatmap,
        "training_subset_recommendations": training_subsets,
        "tasks": tasks,
        "rollouts": rollouts,
        "_raw_row_count": len(raw_rows),
    }


def _write_csv(path: str, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            flat = dict(row)
            for k, v in flat.items():
                if isinstance(v, (list, dict)):
                    flat[k] = json.dumps(v, ensure_ascii=False)
            w.writerow(flat)


def _make_plots(data: Dict[str, Any], plot_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not installed — skipping PNG plots")
        return

    os.makedirs(plot_dir, exist_ok=True)

    # Difficulty buckets
    buckets = data["difficulty_bucket_summary"]
    if buckets:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = [b["bucket"] for b in buckets]
        vals = [b["task_count"] for b in buckets]
        colors = ["#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e"]
        ax.bar(names, vals, color=colors[: len(names)])
        ax.set_title("Task Difficulty Distribution")
        ax.set_ylabel("Number of tasks")
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "difficulty_bucket_distribution.png"), dpi=120)
        plt.close(fig)

    # Failure modes
    fm = data["failure_mode_summary"]
    if fm:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(
            [x["failure_mode"] for x in fm[:9]][::-1],
            [x["failed_rollout_count"] for x in fm[:9]][::-1],
            color="#a855f7",
        )
        ax.set_title("Failure Mode Distribution (failed rollouts)")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "failure_mode_distribution.png"), dpi=120)
        plt.close(fig)

    # Problem domains
    pd = data["problem_domain_summary"]
    if pd:
        fig, ax = plt.subplots(figsize=(10, 6))
        doms = [x["problem_domain"] for x in pd]
        rates = [100 * x["pass_rate"] for x in pd]
        ax.barh(doms, rates, color="#3b82f6")
        ax.set_xlabel("Pass rate (%)")
        ax.set_title("Pass Rate by Problem Domain")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "pass_rate_by_problem_domain.png"), dpi=120)
        plt.close(fig)

    # Gold answer types
    gt = data["gold_answer_type_summary"]
    if gt:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(
            [x["gold_answer_type"] for x in gt],
            [100 * x["pass_rate"] for x in gt],
            color="#06b6d4",
        )
        ax.set_ylabel("Pass rate (%)")
        ax.set_title("Pass Rate by Gold Answer Type")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "pass_rate_by_gold_answer_type.png"), dpi=120)
        plt.close(fig)

    # Malformed by tool
    mal = data["tool_error_summary"]["malformed_by_tool"]
    if mal:
        items = list(mal.items())[:20]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color="#ec4899")
        ax.set_title("Malformed Tool Calls by Tool Name")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "malformed_tool_calls_by_tool.png"), dpi=120)
        plt.close(fig)

    # Heatmap
    heat = data["domain_failure_heatmap"]
    if heat:
        domains = sorted({h["problem_domain"] for h in heat})
        modes = FAILURE_MODES
        mat = [[0] * len(modes) for _ in domains]
        mi = {m: i for i, m in enumerate(modes)}
        di = {d: i for i, d in enumerate(domains)}
        for h in heat:
            if h["problem_domain"] in di and h["failure_mode"] in mi:
                mat[di[h["problem_domain"]]][mi[h["failure_mode"]]] = h["failed_rollout_count"]
        fig, ax = plt.subplots(figsize=(12, max(4, len(domains) * 0.45)))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(modes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(domains)))
        ax.set_yticklabels(domains, fontsize=8)
        ax.set_title("Problem Domain vs Failure Mode (failed rollouts)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "heatmap_domain_vs_failure_mode.png"), dpi=120)
        plt.close(fig)


def _write_report(data: Dict[str, Any], path: str) -> None:
    ov = data["overview"]
    lines = [
        "# NESTFUL Evaluation Analysis Report",
        "",
        f"Generated: {ov.get('generated_at', '')}",
        "",
        "## Overview",
        "",
        f"- **Total tasks:** {ov['total_tasks']}",
        f"- **Total rollouts:** {ov['total_rollouts']}",
        f"- **Overall pass rate:** {ov['overall_pass_rate_percent']}%",
        f"- **Hard-fail tasks:** {ov['hard_fail_tasks']}",
        f"- **Mostly-fail tasks:** {ov['mostly_fail_tasks']}",
        f"- **Unstable tasks:** {ov['unstable_tasks']}",
        f"- **Easy tasks:** {ov['easy_tasks']}",
        f"- **Most common failure mode:** {ov['most_common_failure_mode']}",
        f"- **Most problematic domain:** {ov['most_problematic_domain']}",
        f"- **Malformed tool calls:** {ov['malformed_tool_call_percent']}%",
        "",
        "## Key interpretations",
        "",
    ]
    for k, v in data.get("interpretations", {}).items():
        lines.append(f"- **{k}:** {v}")
    lines.extend(["", "## Top failure modes", ""])
    for fm in data["failure_mode_summary"][:8]:
        lines.append(
            f"- **{fm['failure_mode']}** — {fm['failed_rollout_count']} failed rollouts, "
            f"{fm['affected_task_count']} tasks ({fm['percentage_of_failed_rollouts']}% of failures)"
        )
    lines.extend(["", "## Hardest problem domains", ""])
    for pd in data["problem_domain_summary"][:8]:
        lines.append(
            f"- **{pd['problem_domain']}** — pass rate {100*pd['pass_rate']:.1f}%, "
            f"{pd['task_count']} tasks"
        )
    lines.extend(["", "## Training subset recommendations", ""])
    for sub in data["training_subset_recommendations"]:
        lines.append(
            f"- **{sub['subset_name']}** — {sub['task_count']} tasks, "
            f"avg pass {100*sub['average_pass_rate']:.1f}%: {sub['recommended_use']}"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def build_standalone_dashboard(
    data: Dict[str, Any],
    template_dir: str,
    out_path: str,
) -> None:
    """Single self-contained HTML file (inline CSS, JS, embedded JSON)."""
    with open(os.path.join(template_dir, "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    with open(os.path.join(template_dir, "dashboard.css"), encoding="utf-8") as fh:
        css = fh.read()
    with open(os.path.join(template_dir, "dashboard.js"), encoding="utf-8") as fh:
        js = fh.read()

    html = html.replace(
        '<link rel="stylesheet" href="dashboard.css"/>',
        f"<style>\n{css}\n</style>",
    )
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html = html.replace(
        '<script src="dashboard.js"></script>',
        f"<script>window.DASHBOARD_DATA={payload};</script>\n<script>\n{js}\n</script>",
    )
    html = html.replace(
        "open via local HTTP server for best results",
        "self-contained — open directly in a browser (double-click)",
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)


def export_outputs(
    data: Dict[str, Any],
    output_dir: str,
    dashboard_src: str,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots")
    paths: Dict[str, str] = {}

    # Dashboard data (may be large)
    paths["dashboard_data"] = os.path.join(output_dir, "dashboard_data.json")
    with open(paths["dashboard_data"], "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))

    # Copy static dashboard assets
    for name in ("index.html", "dashboard.js", "dashboard.css"):
        src = os.path.join(dashboard_src, name)
        dst = os.path.join(output_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            paths[name] = dst

    # Schema overview
    paths["schema_overview"] = os.path.join(output_dir, "schema_overview.json")
    with open(paths["schema_overview"], "w", encoding="utf-8") as fh:
        json.dump({
            "failure_modes": FAILURE_MODE_META,
            "difficulty_buckets": DIFFICULTY_BUCKETS,
            "training_subsets": SUBSET_RECOMMENDATIONS,
        }, fh, indent=2)

    tasks = data["tasks"]
    rollouts = data["rollouts"]

    paths["task_level_analysis"] = os.path.join(output_dir, "task_level_analysis.csv")
    _write_csv(paths["task_level_analysis"], tasks, [
        "task_id", "num_rollouts", "pass_count", "fail_count", "pass_rate",
        "difficulty_bucket", "dominant_failure_mode", "all_failure_modes",
        "problem_domains", "gold_answer_type", "avg_tool_calls",
        "malformed_tool_call_count", "task_prompt_short", "gold_answer",
        "representative_failed_prediction", "representative_error",
        "recommended_training_subset",
    ])

    paths["rollout_level_analysis"] = os.path.join(output_dir, "rollout_level_analysis.csv")
    _write_csv(paths["rollout_level_analysis"], rollouts, [
        "task_id", "rollout_id", "passed", "dominant_failure_mode",
        "detected_failure_modes", "problem_domains", "gold_answer_type",
        "predicted_answer", "gold_answer", "tool_names_used",
        "malformed_tool_call_detected", "error_message", "stopped",
    ])

    paths["problematic_tasks"] = os.path.join(output_dir, "problematic_tasks_pass_leq2.csv")
    _write_csv(
        paths["problematic_tasks"],
        [t for t in tasks if t["pass_count"] <= 2],
        ["task_id", "pass_count", "pass_rate", "difficulty_bucket",
         "dominant_failure_mode", "problem_domains", "task_prompt_short"],
    )

    paths["unstable_tasks"] = os.path.join(output_dir, "unstable_tasks_pass_3_to_5.csv")
    _write_csv(
        paths["unstable_tasks"],
        [t for t in tasks if 3 <= t["pass_count"] <= 5],
        ["task_id", "pass_count", "pass_rate", "dominant_failure_mode", "task_prompt_short"],
    )

    paths["failure_mode_summary"] = os.path.join(output_dir, "failure_mode_summary.csv")
    _write_csv(paths["failure_mode_summary"], data["failure_mode_summary"], [
        "failure_mode", "failed_rollout_count", "affected_task_count",
        "percentage_of_failed_rollouts", "average_task_pass_rate",
        "recommended_training_strategy",
    ])

    paths["problem_domain_summary"] = os.path.join(output_dir, "problem_domain_summary.csv")
    _write_csv(paths["problem_domain_summary"], data["problem_domain_summary"], [
        "problem_domain", "task_count", "rollout_count", "pass_rate", "fail_rate",
        "most_common_failure_modes",
    ])

    paths["gold_answer_type_summary"] = os.path.join(output_dir, "gold_answer_type_summary.csv")
    _write_csv(paths["gold_answer_type_summary"], data["gold_answer_type_summary"], [
        "gold_answer_type", "rollout_count", "pass_rate",
    ])

    tool_rows = [
        {"tool_name": k, "malformed_count": v}
        for k, v in data["tool_error_summary"]["malformed_by_tool"].items()
    ]
    paths["tool_error_summary"] = os.path.join(output_dir, "tool_error_summary.csv")
    _write_csv(paths["tool_error_summary"], tool_rows, ["tool_name", "malformed_count"])

    err_counter = Counter()
    for r in rollouts:
        if not r["passed"] and r["error_message"]:
            err_counter[r["error_message"]] += 1
    err_rows = [{"error_message": k, "count": v} for k, v in err_counter.most_common(50)]
    paths["most_common_errors"] = os.path.join(output_dir, "most_common_errors.csv")
    _write_csv(paths["most_common_errors"], err_rows, ["error_message", "count"])

    paths["analysis_report"] = os.path.join(output_dir, "analysis_report.md")
    _write_report(data, paths["analysis_report"])

    _make_plots(data, plot_dir)
    paths["plots_dir"] = plot_dir
    paths["index_html"] = os.path.join(output_dir, "index.html")

    paths["dashboard_html"] = os.path.join(output_dir, "dashboard.html")
    tpl_dir = dashboard_src if os.path.isdir(dashboard_src) else output_dir
    build_standalone_dashboard(data, tpl_dir, paths["dashboard_html"])
    return paths


def _find_predictions(path: Optional[str]) -> str:
    if path:
        return path
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "nestful_results"),
        os.path.join(os.path.dirname(__file__), "nestful_results"),
    ]
    for base in candidates:
        if not os.path.isdir(base):
            continue
        for fn in sorted(os.listdir(base), reverse=True):
            if fn.endswith("_multiturn_predictions.jsonl"):
                return os.path.join(base, fn)
    raise FileNotFoundError("No *_multiturn_predictions.jsonl found — pass --predictions")


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze NESTFUL multiturn results.")
    p.add_argument("--predictions", default=None, help="Path to predictions JSONL.")
    p.add_argument(
        "--output", default=None,
        help="Output directory (default: nestful_evaluation/analysis_outputs).",
    )
    args = p.parse_args()

    pred_path = _find_predictions(args.predictions)
    out_dir = args.output or os.path.join(os.path.dirname(__file__), "analysis_outputs")
    dashboard_src = os.path.join(os.path.dirname(__file__), "dashboard")

    print(f"[analyze] loading {pred_path}")
    rows = load_rollouts(pred_path)
    print(f"[analyze] {len(rows)} rollouts loaded")

    data = analyze_rollouts(rows)
    ov = data["overview"]
    print(f"[analyze] {ov['total_tasks']} tasks, pass rate {ov['overall_pass_rate_percent']}%")

    paths = export_outputs(data, out_dir, dashboard_src)

    print()
    print("Analysis complete.")
    print()
    print(f"Output directory: {os.path.abspath(out_dir)}")
    print(f"Report:           {paths['analysis_report']}")
    print(f"Dashboard (1 file): {paths['dashboard_html']}")
    print(f"Multi-file view:    {paths['index_html']}")
    print()
    print("Open the dashboard:")
    print(f"  {os.path.abspath(paths['dashboard_html'])}")
    print("  (double-click dashboard.html — no server needed)")
    print()
    print("Main findings:")
    print(f"  - Overall pass rate: {ov['overall_pass_rate_percent']}%")
    print(f"  - Hard-fail tasks: {ov['hard_fail_tasks']}")
    print(f"  - Mostly-fail tasks: {ov['mostly_fail_tasks']}")
    print(f"  - Unstable tasks: {ov['unstable_tasks']}")
    print(f"  - Most common failure mode: {ov['most_common_failure_mode']}")
    print(f"  - Most difficult domain: {ov['most_problematic_domain']}")
    print(f"  - Malformed tool calls: {ov['malformed_tool_call_percent']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
