#!/usr/bin/env python3
"""
Compare baseline vs fine-tuned model eval results.

Usage:
    python -m eval.scripts.compare \
        --baseline eval/results/toolgym/baseline_summary.json \
        --finetuned eval/results/toolgym/finetuned_summary.json \
        --output eval/results/toolgym/comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def load_summary(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_primary_metric(summary: dict) -> float:
    """AST accuracy if available, otherwise mean_score_percent."""
    return summary.get("ast_accuracy_percent", summary.get("mean_score_percent", 0))


def compare(base: dict, tuned: dict) -> dict:
    base_score = _get_primary_metric(base)
    tuned_score = _get_primary_metric(tuned)

    has_ast = "ast_accuracy_percent" in base or "ast_accuracy_percent" in tuned
    metric_label = "ast_accuracy_%" if has_ast else "mean_score_%"

    comparison = {
        "benchmark": base.get("benchmark", tuned.get("benchmark", "unknown")),
        "metric": metric_label,
        "baseline_profile": base.get("model_profile", "baseline"),
        "finetuned_profile": tuned.get("model_profile", "finetuned"),
        "baseline_score": base_score,
        "finetuned_score": tuned_score,
        "delta_percent": round(tuned_score - base_score, 2),
        "baseline_total_tasks": base.get("total_tasks", 0),
        "finetuned_total_tasks": tuned.get("total_tasks", 0),
        "baseline_completed": base.get("completed", 0),
        "finetuned_completed": tuned.get("completed", 0),
    }

    if has_ast:
        comparison["baseline_soft_score"] = base.get("soft_accuracy_percent", base.get("mean_score_percent", 0))
        comparison["finetuned_soft_score"] = tuned.get("soft_accuracy_percent", tuned.get("mean_score_percent", 0))

    for key in ("avg_tool_calls", "avg_steps"):
        if key in base or key in tuned:
            comparison[f"baseline_{key}"] = base.get(key, "n/a")
            comparison[f"finetuned_{key}"] = tuned.get(key, "n/a")

    return comparison


def format_table(comp: dict) -> str:
    metric = comp.get("metric", "score_%")
    lines = [
        "| metric | baseline | finetuned | delta |",
        "|---|---:|---:|---:|",
        f"| {metric} | {comp['baseline_score']:.2f} | {comp['finetuned_score']:.2f} | {comp['delta_percent']:+.2f} |",
        f"| total_tasks | {comp['baseline_total_tasks']} | {comp['finetuned_total_tasks']} | |",
        f"| completed | {comp['baseline_completed']} | {comp['finetuned_completed']} | |",
    ]
    if "baseline_soft_score" in comp:
        delta_soft = round(comp["finetuned_soft_score"] - comp["baseline_soft_score"], 2)
        lines.insert(3, f"| soft_score_% | {comp['baseline_soft_score']:.2f} | {comp['finetuned_soft_score']:.2f} | {delta_soft:+.2f} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Compare baseline vs finetuned eval summaries.")
    p.add_argument("--baseline", required=True, help="Path to baseline summary JSON.")
    p.add_argument("--finetuned", required=True, help="Path to finetuned summary JSON.")
    p.add_argument("--output", default=None, help="Output comparison JSON path.")
    p.add_argument("--table", default=None, help="Output markdown table path.")
    args = p.parse_args()

    base = load_summary(args.baseline)
    tuned = load_summary(args.finetuned)
    comp = compare(base, tuned)

    table = format_table(comp)
    print("\n" + table + "\n")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(comp, f, indent=2)
        print(f"Comparison JSON: {args.output}")

    if args.table:
        os.makedirs(os.path.dirname(args.table) or ".", exist_ok=True)
        with open(args.table, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"Comparison table: {args.table}")


if __name__ == "__main__":
    main()
