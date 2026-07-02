#!/usr/bin/env python3
"""Re-score rollouts with alternative answer extraction vs gold."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nestful_evaluation"))

from run import (  # noqa: E402
    _FINAL_ANSWER_RE,
    _NUMBER_IN_TEXT_RE,
    _RESULT_ONLY_RE,
    _extract_numeric,
    _loads_relaxed,
    _matches_gold,
    coerce_numeric,
)

PROFILES = {
    "baseline": ROOT / "curricullum/evaluation/results/curriculum_baseline_multiturn_predictions.jsonl",
    "stage1": ROOT / "curricullum/evaluation/results/curriculum_stage1_1call_multiturn_predictions.jsonl",
    "stage2": ROOT / "curricullum/evaluation/results/curriculum_stage2_2call_multiturn_predictions.jsonl",
    "stage3": ROOT / "curricullum/evaluation/results/curriculum_stage3_3call_multiturn_predictions.jsonl",
}


def last_completion(row: dict) -> str:
    comps = row.get("raw_completions") or []
    return comps[-1] if comps else ""


def extract_result_tag(text: str):
    m = _RESULT_ONLY_RE.search(text)
    if not m:
        return None
    obj = _loads_relaxed(m.group(1))
    if isinstance(obj, dict) and "result" in obj and "name" not in obj:
        return coerce_numeric(obj["result"])
    return None


def extract_final_answer_tag(text: str):
    m = _FINAL_ANSWER_RE.search(text)
    if not m:
        return None
    return coerce_numeric(m.group(1).strip())


def last_trace_result(row: dict):
    trace = row.get("execution_trace") or []
    if not trace:
        return None
    return trace[-1].get("result")


def gold_appears_in_text(text: str, gold) -> bool:
    if not text or gold is None:
        return False
    if _matches_gold(_extract_numeric(text), gold):
        return True
    gold_s = str(gold).strip()
    if gold_s and gold_s in text:
        return True
    # int gold: also match "400.0" style in text
    g = coerce_numeric(gold)
    if isinstance(g, (int, float)):
        for m in _NUMBER_IN_TEXT_RE.findall(text):
            if _matches_gold(coerce_numeric(m), g):
                return True
    return False


def any_number_matches(text: str, gold) -> bool:
    if not text or gold is None:
        return False
    for m in _NUMBER_IN_TEXT_RE.findall(text):
        if _matches_gold(coerce_numeric(m), gold):
            return True
    return False


def score_row(row: dict) -> dict[str, bool]:
    gold = row.get("gold_answer")
    text = last_completion(row)
    official = (row.get("score") or 0) >= 1
    predicted = row.get("predicted_final")
    last_num = _extract_numeric(text)
    result_tag = extract_result_tag(text)
    final_tag = extract_final_answer_tag(text)
    trace_last = last_trace_result(row)

    metrics = {
        "official": official,
        "last_number_in_text": _matches_gold(last_num, gold),
        "result_tag_value": _matches_gold(result_tag, gold) if result_tag is not None else False,
        "final_answer_tag": _matches_gold(final_tag, gold) if final_tag is not None else False,
        "last_tool_trace_result": _matches_gold(trace_last, gold) if trace_last is not None else False,
        "gold_anywhere_in_text": gold_appears_in_text(text, gold),
        "any_number_in_text": any_number_matches(text, gold),
    }

    # Rescue: if no tools, trust prose last number; else keep official logic
    no_tools = (row.get("num_tool_calls") or 0) == 0
    metrics["prose_if_no_tools_else_official"] = (
        _matches_gold(last_num, gold) if no_tools else official
    )

    # Best generous: pass if ANY credible model answer matches
    metrics["best_of_prose_or_official"] = any(
        [
            official,
            metrics["last_number_in_text"],
            metrics["result_tag_value"],
            metrics["final_answer_tag"],
            metrics["gold_anywhere_in_text"],
        ]
    )

    # Union official + last_number (minimal fix for wrong last-number heuristic when tools ran)
    metrics["official_or_last_number"] = official or metrics["last_number_in_text"]

    return metrics


def analyze_path(path: Path) -> dict:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    metric_names = None
    counts: dict[str, int] = defaultdict(int)
    rescued = defaultdict(int)  # official fail -> alt pass
    lost = defaultdict(int)  # official pass -> alt fail
    by_stop = defaultdict(lambda: defaultdict(int))

    for row in rows:
        m = score_row(row)
        if metric_names is None:
            metric_names = list(m.keys())
        for k, v in m.items():
            if v:
                counts[k] += 1
            if k != "official":
                if not m["official"] and v:
                    rescued[k] += 1
                if m["official"] and not v:
                    lost[k] += 1
        stop = row.get("stopped") or "unknown"
        if not m["official"] and m["prose_if_no_tools_else_official"]:
            by_stop[stop]["rescued"] += 1
        if not m["official"]:
            by_stop[stop]["failed_official"] += 1

    n = len(rows)
    return {
        "n": n,
        "counts": counts,
        "metric_names": metric_names or [],
        "rescued": rescued,
        "lost": lost,
        "by_stop": dict(by_stop),
    }


def pct(x: int, n: int) -> float:
    return 100.0 * x / n if n else 0.0


def main() -> None:
    print("=== Alternative pass metrics (rollout-level) ===\n")
    print(
        "Metriky:\n"
        "  official              = current eval (predicted_final vs gold)\n"
        "  last_number_in_text   = last number in last completion\n"
        "  result_tag_value      = tool_call_answer result JSON tag\n"
        "  final_answer_tag      = final_answer tag\n"
        "  last_tool_trace_result= last tool execution result\n"
        "  gold_anywhere_in_text = gold in text (last number or substring)\n"
        "  any_number_in_text    = any number in text matches gold\n"
        "  prose_if_no_tools     = no tools -> last number; else official\n"
        "  best_of_prose_or_official = pass if any prose/official signal matches\n"
        "  official_or_last_number   = official OR last number in text\n"
    )

    all_results = {}
    for name, path in PROFILES.items():
        if not path.exists():
            print(f"SKIP {name}: missing {path}")
            continue
        all_results[name] = analyze_path(path)

    metric_order = all_results["baseline"]["metric_names"] if "baseline" in all_results else []

    header = f"{'metric':32}" + "".join(f"{p:>12}" for p in all_results)
    print(header)
    print("-" * len(header))

    for metric in metric_order:
        line = f"{metric:32}"
        for name in all_results:
            r = all_results[name]
            c = r["counts"][metric]
            delta = c - r["counts"]["official"]
            line += f"{pct(c, r['n']):6.2f}%({delta:+4d})"
        print(line)

    print("\n=== Rescue analysis (baseline): official FAIL -> alt PASS ===")
    if "baseline" in all_results:
        r = all_results["baseline"]
        n = r["n"]
        off_fails = n - r["counts"]["official"]
        for metric in metric_order:
            if metric == "official":
                continue
            print(
                f"  {metric:32} +{r['rescued'][metric]:5d} rollouts "
                f"({100*r['rescued'][metric]/off_fails:.1f}% of official fails), "
                f"-{r['lost'][metric]} false flip"
            )

    print("\n=== Kde by prose_if_no_tools zachranil fail (baseline) ===")
    if "baseline" in all_results:
        for stop, d in sorted(all_results["baseline"]["by_stop"].items()):
            failed = d.get("failed_official", 0)
            rescued = d.get("rescued", 0)
            if failed:
                print(f"  {stop:20} rescued {rescued:5d} / {failed:5d} official fails ({100*rescued/failed:.1f}%)")

    print("\n=== Task pass@8 — official vs prose_if_no_tools (baseline) ===")
    if PROFILES["baseline"].exists():
        by_task: dict[str, list] = defaultdict(list)
        with PROFILES["baseline"].open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                m = score_row(row)
                by_task[row["task_id"]].append(m)
        n_tasks = len(by_task)
        off8 = sum(1 for rolls in by_task.values() if any(x["official"] for x in rolls))
        alt8 = sum(
            1 for rolls in by_task.values() if any(x["prose_if_no_tools_else_official"] for x in rolls)
        )
        best8 = sum(1 for rolls in by_task.values() if any(x["best_of_prose_or_official"] for x in rolls))
        print(f"  pass@8 official:              {100*off8/n_tasks:.2f}% ({off8}/{n_tasks})")
        print(f"  pass@8 prose_if_no_tools:     {100*alt8/n_tasks:.2f}% ({alt8}/{n_tasks})  (+{alt8-off8} tasks)")
        print(f"  pass@8 best_of_prose_or_official: {100*best8/n_tasks:.2f}% ({best8}/{n_tasks})  (+{best8-off8} tasks)")


if __name__ == "__main__":
    main()
