"""
ToolTalk benchmark runner.

Evaluates a model on ToolTalk multi-turn conversations with real
(simulated) API backends.  Uses a ground-truth oracle approach:
after the model predicts tool call(s) for each turn, the ground-truth
API responses are fed back for subsequent turns.  This isolates
per-turn accuracy from cascading errors.

Dataset: 78 conversations (28 easy, 50 hard) with 28 tools in 7 suites.

Metrics:
    - turn_accuracy:    fraction of turns with all tool calls correct
    - name_accuracy:    fraction of individual calls with correct API name
    - param_accuracy:   fraction of calls with correct name + params
    - conversation_success: fraction of conversations with ALL turns correct
    - precision/recall: over predicted vs ground-truth tool calls
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from eval.benchmarks.tooltalk.loader import build_user_content, load_tasks
from eval.metrics import aggregate_summary, print_summary_table, save_predictions, save_summary
from eval.model_adapter import build_chat_prompt, generate
from eval.parse_utils import parse_tool_calls


IGNORE_PARAMS = {"session_token"}


def _normalize_value(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip().strip("'\"").lower()


def _match_call(pred: Dict[str, Any], gt_api: Dict[str, Any]) -> Dict[str, Any]:
    """Compare a single predicted call against a ground truth API call.

    Returns dict with name_match, param_match, and details.
    """
    gt_req = gt_api["request"]
    gt_name = gt_req["api_name"]
    gt_params = {
        k: v for k, v in gt_req.get("parameters", {}).items()
        if k not in IGNORE_PARAMS
    }

    pred_name = pred.get("name", "")
    pred_args = {
        k: v for k, v in pred.get("arguments", {}).items()
        if k not in IGNORE_PARAMS
    }

    name_match = pred_name.strip().lower() == gt_name.strip().lower()

    if not name_match:
        return {"name_match": False, "param_match": False}

    gt_normalized = {k: _normalize_value(v) for k, v in gt_params.items()}
    pred_normalized = {k: _normalize_value(v) for k, v in pred_args.items()}

    param_match = gt_normalized == pred_normalized

    return {"name_match": True, "param_match": param_match}


def _score_turn(
    response: str,
    task: Dict[str, Any],
) -> Dict[str, Any]:
    """Score a single ToolTalk turn (may have multiple API calls)."""
    gt_apis = task["ground_truth_apis"]
    num_gt = len(gt_apis)

    parsed_calls, parse_method = parse_tool_calls(response)

    if parsed_calls is None or len(parsed_calls) == 0:
        return {
            "parse_success": False,
            "parse_method": parse_method,
            "num_predicted": 0,
            "num_ground_truth": num_gt,
            "name_matches": 0,
            "param_matches": 0,
            "turn_correct": False,
            "error": "parse_failure",
        }

    name_matches = 0
    param_matches = 0
    gt_matched = [False] * num_gt

    for pred_call in parsed_calls:
        best_match_idx = -1
        best_is_param = False
        best_is_name = False

        for j, gt_api in enumerate(gt_apis):
            if gt_matched[j]:
                continue
            result = _match_call(pred_call, gt_api)
            if result["param_match"] and not best_is_param:
                best_match_idx = j
                best_is_param = True
                best_is_name = True
            elif result["name_match"] and not best_is_param and not best_is_name:
                best_match_idx = j
                best_is_name = True

        if best_match_idx >= 0:
            gt_matched[best_match_idx] = True
            if best_is_name:
                name_matches += 1
            if best_is_param:
                param_matches += 1

    turn_correct = param_matches == num_gt and len(parsed_calls) == num_gt

    precision = param_matches / len(parsed_calls) if parsed_calls else 0.0
    recall = param_matches / num_gt if num_gt else 0.0

    error = "none"
    if not turn_correct:
        if name_matches == 0:
            error = "name_mismatch"
        elif param_matches < num_gt:
            error = "param_mismatch"
        elif len(parsed_calls) != num_gt:
            error = "wrong_count"

    return {
        "parse_success": True,
        "parse_method": parse_method,
        "num_predicted": len(parsed_calls),
        "num_ground_truth": num_gt,
        "name_matches": name_matches,
        "param_matches": param_matches,
        "turn_correct": turn_correct,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "error": error,
    }


def run(
    model_cfg: Dict[str, Any],
    max_tasks: Optional[int] = None,
    output_dir: str = "eval/results/tooltalk",
    model_profile: str = "default",
    dry_run: bool = False,
    cache_dir: Optional[str] = None,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Run ToolTalk multi-turn evaluation."""

    limit = max_tasks
    if dry_run:
        limit = 3

    print(f"[tooltalk] Loading ToolTalk conversations...")
    tasks, tool_schemas = load_tasks(max_tasks=limit, cache_dir=cache_dir)
    n_easy = sum(1 for t in tasks if t["difficulty"] == "easy")
    n_hard = sum(1 for t in tasks if t["difficulty"] == "hard")
    conv_ids = set(t["conversation_id"] for t in tasks)
    print(f"[tooltalk] Loaded {len(tasks)} turns from {len(conv_ids)} conversations "
          f"({n_easy} easy turns, {n_hard} hard turns)")

    print(f"[tooltalk] Building chat-templated prompts...")
    prompts = []
    for t in tasks:
        user_content = build_user_content(t)
        prompt = build_chat_prompt(user_content, model_cfg)
        prompts.append(prompt)

    print(f"[tooltalk] Generating responses...")
    t0 = time.time()
    responses = generate(prompts, model_cfg, batch_size=batch_size)
    elapsed = time.time() - t0

    if responses:
        print(f"[tooltalk] Sample response (first 500 chars):")
        print(f"  {responses[0][:500]}")

    results: List[Dict[str, Any]] = []
    total_name_matches = 0
    total_param_matches = 0
    total_gt_calls = 0
    total_pred_calls = 0
    turn_correct_count = 0

    conv_turns: Dict[str, List[bool]] = defaultdict(list)
    diff_counters: Dict[str, Counter] = {
        "easy": Counter(),
        "hard": Counter(),
    }

    error_counter: Counter = Counter()

    for task, response in zip(tasks, responses):
        scoring = _score_turn(response, task)

        total_name_matches += scoring["name_matches"]
        total_param_matches += scoring["param_matches"]
        total_gt_calls += scoring["num_ground_truth"]
        total_pred_calls += scoring["num_predicted"]

        if scoring["turn_correct"]:
            turn_correct_count += 1

        conv_turns[task["conversation_id"]].append(scoring["turn_correct"])
        error_counter[scoring["error"]] += 1

        diff = task["difficulty"]
        diff_counters[diff]["turns"] += 1
        if scoring["turn_correct"]:
            diff_counters[diff]["correct_turns"] += 1

        status = "completed" if scoring["turn_correct"] else "failed"
        if response.startswith("[ERROR]"):
            status = "error"

        results.append({
            "task_id": task["task_id"],
            "conversation_id": task["conversation_id"],
            "conversation_name": task["conversation_name"],
            "difficulty": diff,
            "turn_index": task["turn_index"],
            "source_file": task["source_file"],
            "response": response[:2000],
            "status": status,
            "score": 1.0 if scoring["turn_correct"] else 0.0,
            "num_ground_truth": scoring["num_ground_truth"],
            "num_predicted": scoring["num_predicted"],
            "name_matches": scoring["name_matches"],
            "param_matches": scoring["param_matches"],
            "turn_correct": scoring["turn_correct"],
            "parse_method": scoring["parse_method"],
            "parse_success": scoring["parse_success"],
            "error_category": scoring["error"],
            "num_tool_calls": scoring["num_predicted"],
        })

    total_turns = len(tasks)
    turn_acc = turn_correct_count / total_turns if total_turns else 0.0
    overall_precision = total_param_matches / total_pred_calls if total_pred_calls else 0.0
    overall_recall = total_param_matches / total_gt_calls if total_gt_calls else 0.0
    name_recall = total_name_matches / total_gt_calls if total_gt_calls else 0.0

    conv_success = sum(1 for turns in conv_turns.values() if all(turns))
    conv_total = len(conv_turns)
    conv_success_rate = conv_success / conv_total if conv_total else 0.0

    easy_turns = diff_counters["easy"]["turns"]
    easy_correct = diff_counters["easy"]["correct_turns"]
    hard_turns = diff_counters["hard"]["turns"]
    hard_correct = diff_counters["hard"]["correct_turns"]

    parse_methods = Counter(r.get("parse_method", "unknown") for r in results)

    os.makedirs(output_dir, exist_ok=True)

    pred_path = os.path.join(output_dir, f"{model_profile}_predictions.jsonl")
    save_predictions(results, pred_path)

    summary = aggregate_summary(
        results,
        benchmark="tooltalk",
        model_profile=model_profile,
        extra={
            "turn_accuracy": round(turn_acc, 4),
            "turn_accuracy_percent": round(100.0 * turn_acc, 2),
            "conversation_success_rate": round(conv_success_rate, 4),
            "conversation_success_percent": round(100.0 * conv_success_rate, 2),
            "conversations_succeeded": conv_success,
            "conversations_total": conv_total,
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "name_recall": round(name_recall, 4),
            "per_difficulty": {
                "easy": {
                    "turns": easy_turns,
                    "correct": easy_correct,
                    "accuracy_percent": round(100.0 * easy_correct / easy_turns, 2) if easy_turns else 0.0,
                },
                "hard": {
                    "turns": hard_turns,
                    "correct": hard_correct,
                    "accuracy_percent": round(100.0 * hard_correct / hard_turns, 2) if hard_turns else 0.0,
                },
            },
            "elapsed_seconds": round(elapsed, 2),
            "error_breakdown": dict(error_counter),
            "parse_methods": dict(parse_methods),
        },
    )

    summary_path = os.path.join(output_dir, f"{model_profile}_summary.json")
    save_summary(summary, summary_path)

    print_summary_table(summary)
    print(f"\n  ToolTalk Detailed Metrics:")
    print(f"  {'Metric':<35} {'Value':>10}")
    print(f"  {'-' * 47}")
    print(f"  {'Turn Accuracy':<35} {100.0 * turn_acc:>9.2f}%")
    print(f"  {'Conv. Success Rate':<35} {100.0 * conv_success_rate:>9.2f}%")
    print(f"  {'Precision':<35} {100.0 * overall_precision:>9.2f}%")
    print(f"  {'Recall':<35} {100.0 * overall_recall:>9.2f}%")
    print(f"  {'Name Recall':<35} {100.0 * name_recall:>9.2f}%")
    print(f"  {'-' * 47}")
    if easy_turns:
        print(f"  {'Easy Turn Accuracy':<35} {100.0 * easy_correct / easy_turns:>9.2f}%  ({easy_correct}/{easy_turns})")
    if hard_turns:
        print(f"  {'Hard Turn Accuracy':<35} {100.0 * hard_correct / hard_turns:>9.2f}%  ({hard_correct}/{hard_turns})")
    print(f"  {'-' * 47}")
    print(f"  Error breakdown: {dict(error_counter)}")
    print(f"  Parse methods:   {dict(parse_methods)}")
    print(f"  Predictions:     {pred_path}")
    print(f"  Summary:         {summary_path}")

    return summary
