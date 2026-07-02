"""
API-Bank benchmark runner.

Evaluates a model on API-Bank Level 1 (given-desc): the model is
provided with API descriptions and a conversation context, and must
generate the correct API call.

Dataset: 314 dialogues with 753 API call samples across 73 real-world
APIs (healthcare, smart home, calendar, search, finance, etc.).

Reference:
    Li et al., "API-Bank: A Comprehensive Benchmark for Tool-Augmented
    LLMs", EMNLP 2023.  https://arxiv.org/abs/2304.08244

Metrics:
    - api_name_accuracy:  correct API name selected
    - param_accuracy:     correct name AND all parameters match
    - param_f1:           average per-sample F1 of parameter key-value pairs
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from eval.ast_eval import ast_match_single
from eval.benchmarks.apibank.loader import build_user_content, load_tasks
from eval.metrics import aggregate_summary, print_summary_table, save_predictions, save_summary
from eval.model_adapter import build_chat_prompt, generate
from eval.parse_utils import parse_tool_calls


def _score_task(
    response: str,
    task: Dict[str, Any],
) -> Dict[str, Any]:
    """Score a single API-Bank task using AST matching (paper-comparable).

    Uses robust_value_match from the training reward code for type
    coercion and whitespace normalization — same as the paper's eval.
    """
    gt_name = task["ground_truth_name"]
    gt_args = task["ground_truth_args"]

    parsed_calls, parse_method = parse_tool_calls(response)

    if parsed_calls is None or len(parsed_calls) == 0:
        return {
            "ast_match": False,
            "name_correct": False,
            "parse_method": parse_method,
            "parse_success": False,
            "predicted_name": None,
            "predicted_args": {},
            "error": "parse_failure",
        }

    pred = parsed_calls[0]
    pred_name = pred.get("name", "")
    pred_args = pred.get("arguments", {})

    gt_call = {"name": gt_name, "arguments": gt_args}
    pred_call = {"name": pred_name, "arguments": pred_args}

    is_ast_match = ast_match_single(pred_call, gt_call)
    name_correct = pred_name.strip().lower() == gt_name.strip().lower()

    error = "none"
    if not name_correct:
        error = "name_mismatch"
    elif not is_ast_match:
        error = "param_mismatch"

    return {
        "ast_match": is_ast_match,
        "name_correct": name_correct,
        "parse_method": parse_method,
        "parse_success": True,
        "predicted_name": pred_name,
        "predicted_args": pred_args,
        "error": error,
    }


def run(
    model_cfg: Dict[str, Any],
    max_tasks: Optional[int] = None,
    output_dir: str = "eval/results/apibank",
    model_profile: str = "default",
    dry_run: bool = False,
    cache_dir: Optional[str] = None,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Run API-Bank Level 1 evaluation."""

    limit = max_tasks
    if dry_run:
        limit = 5

    print(f"[apibank] Loading API-Bank Level 1 tasks...")
    tasks = load_tasks(max_tasks=limit, cache_dir=cache_dir)
    print(f"[apibank] Loaded {len(tasks)} tasks")

    print(f"[apibank] Building chat-templated prompts...")
    prompts = []
    for t in tasks:
        user_content = build_user_content(t)
        prompt = build_chat_prompt(user_content, model_cfg)
        prompts.append(prompt)

    print(f"[apibank] Generating responses...")
    t0 = time.time()
    responses = generate(prompts, model_cfg, batch_size=batch_size)
    elapsed = time.time() - t0

    if responses:
        print(f"[apibank] Sample response (first 500 chars):")
        print(f"  {responses[0][:500]}")

    results: List[Dict[str, Any]] = []
    ast_match_count = 0
    name_correct_count = 0
    parse_success_count = 0

    error_counter: Counter = Counter()

    for task, response in zip(tasks, responses):
        scoring = _score_task(response, task)

        if scoring["ast_match"]:
            ast_match_count += 1
        if scoring["name_correct"]:
            name_correct_count += 1
        if scoring["parse_success"]:
            parse_success_count += 1

        error_counter[scoring["error"]] += 1

        status = "completed" if scoring["ast_match"] else "failed"
        if response.startswith("[ERROR]"):
            status = "error"

        results.append({
            "task_id": task["task_id"],
            "source_file": task["source_file"],
            "question": task["question"][:300],
            "response": response[:2000],
            "status": status,
            "score": 1.0 if scoring["ast_match"] else 0.0,
            "ast_match": scoring["ast_match"],
            "name_correct": scoring["name_correct"],
            "gt_name": task["ground_truth_name"],
            "gt_args": task["ground_truth_args"],
            "predicted_name": scoring["predicted_name"],
            "predicted_args": scoring["predicted_args"],
            "parse_method": scoring["parse_method"],
            "parse_success": scoring["parse_success"],
            "error_category": scoring["error"],
            "num_tool_calls": 1 if scoring["parse_success"] else 0,
        })

    total = len(tasks)
    ast_acc = ast_match_count / total if total else 0.0
    name_acc = name_correct_count / total if total else 0.0
    parse_rate = parse_success_count / total if total else 0.0

    parse_methods = Counter(r.get("parse_method", "unknown") for r in results)

    os.makedirs(output_dir, exist_ok=True)

    pred_path = os.path.join(output_dir, f"{model_profile}_predictions.jsonl")
    save_predictions(results, pred_path)

    summary = aggregate_summary(
        results,
        benchmark="apibank",
        model_profile=model_profile,
        extra={
            "level": "level-1-given-desc",
            "ast_accuracy": round(ast_acc, 4),
            "ast_accuracy_percent": round(100.0 * ast_acc, 2),
            "api_name_accuracy": round(name_acc, 4),
            "api_name_accuracy_percent": round(100.0 * name_acc, 2),
            "parse_success_rate": round(parse_rate, 4),
            "parse_success_rate_percent": round(100.0 * parse_rate, 2),
            "elapsed_seconds": round(elapsed, 2),
            "error_breakdown": dict(error_counter),
            "parse_methods": dict(parse_methods),
        },
    )

    summary_path = os.path.join(output_dir, f"{model_profile}_summary.json")
    save_summary(summary, summary_path)

    print_summary_table(summary)
    print(f"\n  API-Bank Level 1 Metrics:")
    print(f"  {'Metric':<35} {'Value':>10}")
    print(f"  {'-' * 47}")
    print(f"  {'AST Accuracy (paper-comparable)':<35} {100.0 * ast_acc:>9.2f}%")
    print(f"  {'API Name Accuracy':<35} {100.0 * name_acc:>9.2f}%")
    print(f"  {'Parse Success Rate':<35} {100.0 * parse_rate:>9.2f}%")
    print(f"  {'-' * 47}")
    print(f"  Error breakdown: {dict(error_counter)}")
    print(f"  Parse methods:   {dict(parse_methods)}")
    print(f"  Predictions:     {pred_path}")
    print(f"  Summary:         {summary_path}")

    return summary
