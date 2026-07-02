#!/usr/bin/env python3
"""Focused analysis: semantic / prose answer vs official scoring."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nestful_evaluation"))

from run import _extract_numeric, _matches_gold, _NUMBER_IN_TEXT_RE, coerce_numeric  # noqa: E402

PATH = ROOT / "curricullum/evaluation/results/curriculum_baseline_multiturn_predictions.jsonl"


def last_completion(row):
    comps = row.get("raw_completions") or []
    return comps[-1] if comps else ""


def any_number_matches(text, gold):
    for m in _NUMBER_IN_TEXT_RE.findall(text or ""):
        if _matches_gold(coerce_numeric(m), gold):
            return True
    return False


def main():
    rows = [json.loads(l) for l in PATH.open(encoding="utf-8")]

    n = len(rows)
    official_pass = sum(1 for r in rows if (r.get("score") or 0) >= 1)

    buckets = {
        "no_tools_fail": [],
        "no_tools_pass": [],
        "with_tools_fail": [],
        "with_tools_pass": [],
    }
    for r in rows:
        key = ("no_tools" if (r.get("num_tool_calls") or 0) == 0 else "with_tools") + (
            "_pass" if (r.get("score") or 0) >= 1 else "_fail"
        )
        buckets[key].append(r)

    print(f"Baseline rollouts: {n}, official pass: {official_pass} ({100*official_pass/n:.2f}%)\n")

    for bname, blist in buckets.items():
        if not blist:
            continue
        bn = len(blist)
        text = last_completion
        stats = {
            "last_num_ok": sum(1 for r in blist if _matches_gold(_extract_numeric(text(r)), r["gold_answer"])),
            "any_num_ok": sum(1 for r in blist if any_number_matches(text(r), r["gold_answer"])),
            "pred_ok": sum(1 for r in blist if _matches_gold(r.get("predicted_final"), r["gold_answer"])),
        }
        print(f"--- {bname} (n={bn}) ---")
        for k, v in stats.items():
            print(f"  {k:12} {v:5d} ({100*v/bn:5.1f}%)")

    # Official FAIL breakdown
    fails = [r for r in rows if (r.get("score") or 0) < 1]
    fn = len(fails)
    print(f"\n=== Official FAIL rollouts ({fn}) — co by pomohlo? ===")

    rescue = Counter()
    examples = defaultdict(list)

    for r in fails:
        gold = r["gold_answer"]
        txt = last_completion(r)
        pred = r.get("predicted_final")
        last_n = _extract_numeric(txt)
        no_tools = (r.get("num_tool_calls") or 0) == 0

        if _matches_gold(last_n, gold) and not _matches_gold(pred, gold):
            rescue["last_number_fixes_pred"] += 1
        if any_number_matches(txt, gold) and not _matches_gold(pred, gold):
            rescue["any_number_in_prose"] += 1
        if no_tools and any_number_matches(txt, gold):
            rescue["no_tools_gold_in_prose"] += 1
        if no_tools and _matches_gold(last_n, gold):
            rescue["no_tools_last_number_ok"] += 1

        # model "knew" answer in text but official used wrong pred
        if any_number_matches(txt, gold):
            rescue["semantic_knew_answer"] += 1
            if len(examples["semantic_knew_answer"]) < 3:
                examples["semantic_knew_answer"].append(
                    (r["task_id"][:8], pred, gold, r.get("stopped"), txt[-200:])
                )

    for k, v in rescue.most_common():
        print(f"  {k:28} {v:5d} ({100*v/fn:5.1f}% of fails)")

    print("\n  Priklady 'semantic_knew_answer' (gold v textu, official fail):")
    for ex in examples["semantic_knew_answer"]:
        print(f"    task={ex[0]} pred={ex[1]} gold={ex[2]} stop={ex[3]}")
        print(f"      ...{ex[4]!r}")

    # Counterfactual pass rates
    print("\n=== Counterfactual pass rates (baseline) ===")
    scenarios = {
        "official (current)": lambda r: (r.get("score") or 0) >= 1,
        "no_tools: any gold number in text, else official": lambda r: (
            any_number_matches(last_completion(r), r["gold_answer"])
            if (r.get("num_tool_calls") or 0) == 0
            else (r.get("score") or 0) >= 1
        ),
        "no_tools: last number in text, else official": lambda r: (
            _matches_gold(_extract_numeric(last_completion(r)), r["gold_answer"])
            if (r.get("num_tool_calls") or 0) == 0
            else (r.get("score") or 0) >= 1
        ),
        "official OR any gold number in text": lambda r: (
            (r.get("score") or 0) >= 1 or any_number_matches(last_completion(r), r["gold_answer"])
        ),
        "official OR last trace result matches": lambda r: (
            (r.get("score") or 0) >= 1
            or _matches_gold(
                (r.get("execution_trace") or [{}])[-1].get("result") if r.get("execution_trace") else None,
                r["gold_answer"],
            )
        ),
    }

    for name, fn_pass in scenarios.items():
        c = sum(1 for r in rows if fn_pass(r))
        delta = c - official_pass
        print(f"  {name:45} {100*c/n:6.2f}%  ({c}/{n}, {delta:+d})")

    # Task pass@8 for semantic metric
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)

    def task_pass8(fn_pass):
        return sum(1 for rolls in by_task.values() if any(fn_pass(r) for r in rolls))

    nt = len(by_task)
    off8 = task_pass8(lambda r: (r.get("score") or 0) >= 1)
    sem8 = task_pass8(
        lambda r: (r.get("score") or 0) >= 1
        or any_number_matches(last_completion(r), r["gold_answer"])
    )
    print(f"\n  pass@8 official:                    {100*off8/nt:.2f}% ({off8}/{nt})")
    print(f"  pass@8 official OR gold in prose:   {100*sem8/nt:.2f}% ({sem8}/{nt}, +{sem8-off8} tasks)")


if __name__ == "__main__":
    main()
