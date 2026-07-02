"""
BFCL benchmark runner.

Evaluates a model on Berkeley Function Calling Leaderboard categories.

Supported categories (AST — ground-truth matching):
  - simple:            Single function, single call (400 tasks)
  - multiple:          Select one function from 2-4 candidates (200 tasks)
  - parallel:          Multiple concurrent calls from one query (200 tasks)
  - parallel_multiple: Multiple functions, multiple calls (200 tasks)
  - irrelevance:       No function should be called (240 tasks)

Supported categories (Exec — real function execution):
  - exec_simple:            Executable single call (100 tasks)
  - exec_multiple:          Executable function selection (50 tasks)
  - exec_parallel:          Executable parallel calls (50 tasks)
  - exec_parallel_multiple: Executable parallel + multiple (40 tasks)

Scoring: Uses AST-style checking against ground-truth function calls.
Exec categories use the same checker; with `bfcl-eval` installed,
the official package can additionally execute the functions.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from eval.benchmarks.bfcl.checker import HAS_BFCL_EVAL, check_task
from eval.benchmarks.bfcl.loader import build_user_content, load_tasks
from eval.metrics import aggregate_summary, print_summary_table, save_predictions, save_summary
from eval.model_adapter import build_chat_prompt, generate
from eval.parse_utils import parse_tool_calls


def _score_task(
    response: str,
    task: Dict[str, Any],
) -> Dict[str, Any]:
    """Score a single BFCL task."""
    category = task["category"]

    parsed_calls, parse_method = parse_tool_calls(response)

    result = check_task(
        category=category,
        model_output=parsed_calls,
        ground_truth=task["ground_truth"],
        func_docs=task["functions"],
    )

    return {
        "valid": result["valid"],
        "error": result.get("error", ""),
        "parse_method": parse_method,
        "parse_success": parsed_calls is not None,
        "num_predicted_calls": len(parsed_calls) if parsed_calls else 0,
    }


def run(
    model_cfg: Dict[str, Any],
    categories: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
    output_dir: str = "eval/results/bfcl",
    model_profile: str = "default",
    dry_run: bool = False,
    cache_dir: Optional[str] = None,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Run BFCL evaluation."""
    if categories is None:
        categories = ["simple", "multiple", "parallel", "irrelevance"]

    max_per_cat = max_tasks if max_tasks else None
    if dry_run:
        max_per_cat = 3

    print(f"[bfcl] Categories: {categories}")
    print(f"[bfcl] Checker: {'official bfcl-eval' if HAS_BFCL_EVAL else 'standalone'}")

    tasks = load_tasks(
        categories=categories,
        max_tasks_per_category=max_per_cat,
        cache_dir=cache_dir,
    )

    print(f"[bfcl] Building chat-templated prompts...")
    prompts = []
    for t in tasks:
        user_content = build_user_content(t)
        prompt = build_chat_prompt(user_content, model_cfg)
        prompts.append(prompt)

    print(f"[bfcl] Generating responses...")
    t0 = time.time()
    responses = generate(prompts, model_cfg, batch_size=batch_size)
    elapsed = time.time() - t0

    if responses:
        print(f"[bfcl] Sample response (first 500 chars):")
        print(f"  {responses[0][:500]}")

    results: List[Dict[str, Any]] = []
    cat_correct: Counter = Counter()
    cat_total: Counter = Counter()

    for task, response in zip(tasks, responses):
        scoring = _score_task(response, task)
        cat = task["category"]
        cat_total[cat] += 1
        if scoring["valid"]:
            cat_correct[cat] += 1

        status = "error" if response.startswith("[ERROR]") else (
            "completed" if scoring["valid"] else "failed"
        )
        error_cat = "none"
        if status == "error":
            error_cat = "api_error"
        elif not scoring["parse_success"] and cat != "irrelevance":
            error_cat = "parse_failure"
        elif not scoring["valid"]:
            error_cat = scoring.get("error", "check_failed")

        results.append({
            "task_id": task["task_id"],
            "category": cat,
            "question": task["question"][:300],
            "response": response[:2000],
            "status": status,
            "score": 1.0 if scoring["valid"] else 0.0,
            "valid": scoring["valid"],
            "error_detail": scoring.get("error", ""),
            "parse_method": scoring.get("parse_method", ""),
            "num_tool_calls": scoring.get("num_predicted_calls", 0),
            "error_category": error_cat,
        })

    os.makedirs(output_dir, exist_ok=True)

    pred_path = os.path.join(output_dir, f"{model_profile}_predictions.jsonl")
    save_predictions(results, pred_path)

    total_correct = sum(cat_correct.values())
    total_tasks = sum(cat_total.values())
    overall_acc = total_correct / total_tasks if total_tasks else 0.0

    per_category = {}
    for cat in categories:
        n = cat_total.get(cat, 0)
        c = cat_correct.get(cat, 0)
        per_category[cat] = {
            "total": n,
            "correct": c,
            "accuracy": round(c / n, 4) if n else 0.0,
            "accuracy_percent": round(100.0 * c / n, 2) if n else 0.0,
        }

    parse_methods = Counter(r.get("parse_method", "unknown") for r in results)

    summary = aggregate_summary(
        results,
        benchmark="bfcl",
        model_profile=model_profile,
        extra={
            "categories": categories,
            "overall_accuracy": round(overall_acc, 4),
            "overall_accuracy_percent": round(100.0 * overall_acc, 2),
            "per_category": per_category,
            "elapsed_seconds": round(elapsed, 2),
            "checker": "official_bfcl_eval" if HAS_BFCL_EVAL else "standalone",
            "parse_methods": dict(parse_methods),
        },
    )

    summary_path = os.path.join(output_dir, f"{model_profile}_summary.json")
    save_summary(summary, summary_path)

    print_summary_table(summary)
    print(f"\n  BFCL Per-Category Results:")
    print(f"  {'Category':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-' * 53}")
    for cat in categories:
        pc = per_category.get(cat, {})
        print(f"  {cat:<25} {pc.get('correct', 0):>8} {pc.get('total', 0):>8} {pc.get('accuracy_percent', 0):>9.2f}%")
    print(f"  {'-' * 53}")
    print(f"  {'OVERALL':<25} {total_correct:>8} {total_tasks:>8} {100.0 * overall_acc:>9.2f}%")
    print()
    print(f"  Parse methods: {dict(parse_methods)}")
    print(f"  Predictions: {pred_path}")
    print(f"  Summary:     {summary_path}")

    return summary
