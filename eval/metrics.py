"""
Shared metrics aggregation for eval benchmarks.

Produces per-task results and a final summary dict.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def save_predictions(results: List[Dict[str, Any]], path: str) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def save_summary(summary: Dict[str, Any], path: str) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)


def aggregate_summary(
    results: List[Dict[str, Any]],
    benchmark: str,
    model_profile: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a standard summary dict from per-task result dicts.

    Each result dict should have at minimum:
        task_id: str
        status: "completed" | "failed" | "error"
        score: float (0-1)
    Optional fields used when present:
        error_category: str
        num_tool_calls: int
        num_steps: int
    """
    total = len(results)
    completed = sum(1 for r in results if r.get("status") == "completed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    errors = sum(1 for r in results if r.get("status") == "error")

    scores = [r.get("score", 0.0) for r in results]
    mean_score = sum(scores) / total if total else 0.0

    error_cats = Counter(
        r.get("error_category", "none") for r in results if r.get("status") in ("failed", "error")
    )

    tool_calls = [r["num_tool_calls"] for r in results if "num_tool_calls" in r]
    steps = [r["num_steps"] for r in results if "num_steps" in r]

    summary: Dict[str, Any] = {
        "benchmark": benchmark,
        "model_profile": model_profile,
        "total_tasks": total,
        "completed": completed,
        "failed": failed,
        "errors": errors,
        "mean_score": round(mean_score, 4),
        "mean_score_percent": round(100.0 * mean_score, 2),
    }

    if error_cats:
        summary["error_categories"] = dict(error_cats)
    if tool_calls:
        summary["avg_tool_calls"] = round(sum(tool_calls) / len(tool_calls), 2)
    if steps:
        summary["avg_steps"] = round(sum(steps) / len(steps), 2)

    if extra:
        summary.update(extra)

    return summary


def print_summary_table(summary: Dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print(f"  Benchmark:     {summary.get('benchmark', '?')}")
    print(f"  Model:         {summary.get('model_profile', '?')}")
    print(f"  Total tasks:   {summary.get('total_tasks', 0)}")
    print(f"  Completed:     {summary.get('completed', 0)}")
    print(f"  Failed:        {summary.get('failed', 0)}")
    print(f"  Errors:        {summary.get('errors', 0)}")
    print(f"  Mean score:    {summary.get('mean_score_percent', 0):.2f}%")
    if "avg_tool_calls" in summary:
        print(f"  Avg tool calls: {summary['avg_tool_calls']}")
    if "avg_steps" in summary:
        print(f"  Avg steps:     {summary['avg_steps']}")
    if "error_categories" in summary:
        print(f"  Error categories: {summary['error_categories']}")
    print("=" * 60)
    print()
