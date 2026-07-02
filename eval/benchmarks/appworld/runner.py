"""
AppWorld benchmark runner — real API execution.

Evaluates a model on AppWorld tasks by executing predicted tool calls
against AppWorld's real API backend (local, in-process). This is the
only benchmark in our suite that performs actual API execution with
database state changes.

Flow:
    1. Load tasks + API docs from AppWorld (Phase 1 — metadata)
    2. Batch-generate model predictions (Phase 2 — inference)
    3. Per task: init AppWorld, execute predicted calls, evaluate (Phase 3)

AppWorld evaluation is state-based: unit tests check whether the
database state after execution matches the expected final state.

Metrics:
    - task_success_rate (TGC):  AppWorld's primary metric — task passed all tests
    - execution_success_rate:   all predicted API calls executed without error
    - api_name_recall:          predicted API names vs ground truth required APIs
    - parse_success_rate:       model outputs that parsed into valid tool calls
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from eval.benchmarks.appworld.loader import build_user_content, load_tasks
from eval.metrics import aggregate_summary, print_summary_table, save_predictions, save_summary
from eval.model_adapter import build_chat_prompt, generate
from eval.parse_utils import parse_tool_calls


def _split_tool_name(name: str) -> Tuple[str, str]:
    """Split 'app__api_name' or 'app.api_name' into (app, api_name)."""
    if "__" in name:
        parts = name.split("__", 1)
        return parts[0], parts[1]
    if "." in name:
        parts = name.split(".", 1)
        return parts[0], parts[1]
    return "", name


def _build_execute_code(calls: List[Dict[str, Any]]) -> List[str]:
    """Convert parsed tool calls into AppWorld execute() code strings."""
    code_lines = []
    for i, call in enumerate(calls):
        name = call.get("name", "")
        args = call.get("arguments", {})

        app_name, api_name = _split_tool_name(name)
        if not app_name or not api_name:
            code_lines.append(None)
            continue

        arg_parts = []
        for k, v in args.items():
            arg_parts.append(f"{k}={repr(v)}")
        args_str = ", ".join(arg_parts)

        code = f"_result_{i} = apis.{app_name}.{api_name}({args_str})\nprint(_result_{i})"
        code_lines.append(code)

    return code_lines


def _execute_and_evaluate(
    task_id: str,
    parsed_calls: Optional[List[Dict[str, Any]]],
    experiment_name: str,
    appworld_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute predicted calls against AppWorld and evaluate."""
    from appworld import AppWorld

    if appworld_root:
        os.environ["APPWORLD_ROOT"] = appworld_root

    exec_results = []
    all_success = True

    try:
        with AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
        ) as world:

            if parsed_calls:
                code_lines = _build_execute_code(parsed_calls)

                for i, (call, code) in enumerate(zip(parsed_calls, code_lines)):
                    if code is None:
                        exec_results.append({
                            "call": call,
                            "success": False,
                            "error": "invalid_name_format",
                            "output": "",
                        })
                        all_success = False
                        continue

                    try:
                        output = world.execute(code)
                        is_error = output.strip().startswith("Execution failed")
                        exec_results.append({
                            "call": call,
                            "success": not is_error,
                            "error": "exec_error" if is_error else "none",
                            "output": output[:500],
                        })
                        if is_error:
                            all_success = False
                    except Exception as e:
                        exec_results.append({
                            "call": call,
                            "success": False,
                            "error": str(e)[:200],
                            "output": "",
                        })
                        all_success = False

            try:
                evaluation = world.evaluate()
                eval_dict = evaluation.to_dict()
                task_success = eval_dict.get("success", False)
                num_passed = len(eval_dict.get("passes", []))
                num_failed = len(eval_dict.get("fails", []))
            except Exception as e:
                task_success = False
                num_passed = 0
                num_failed = -1
                eval_dict = {"error": str(e)[:200]}

    except Exception as e:
        return {
            "exec_results": [],
            "all_exec_success": False,
            "task_success": False,
            "num_tests_passed": 0,
            "num_tests_failed": -1,
            "eval_detail": {"error": str(e)[:200]},
            "appworld_error": str(e)[:300],
        }

    return {
        "exec_results": exec_results,
        "all_exec_success": all_success,
        "task_success": task_success,
        "num_tests_passed": num_passed,
        "num_tests_failed": num_failed,
        "eval_detail": eval_dict,
        "appworld_error": None,
    }


def _compute_api_name_recall(
    predicted_calls: Optional[List[Dict[str, Any]]],
    required_apis: List[str],
) -> float:
    """What fraction of required APIs did the model predict?"""
    if not required_apis:
        return 1.0
    if not predicted_calls:
        return 0.0

    predicted_names = set()
    for call in predicted_calls:
        name = call.get("name", "")
        _, api_name = _split_tool_name(name)
        predicted_names.add(name.lower())
        predicted_names.add(api_name.lower())

    required_set = {api.lower() for api in required_apis}
    hits = sum(1 for r in required_set if any(r in p for p in predicted_names))
    return hits / len(required_set)


def run(
    model_cfg: Dict[str, Any],
    max_tasks: Optional[int] = None,
    output_dir: str = "eval/results/appworld",
    model_profile: str = "default",
    dry_run: bool = False,
    cache_dir: Optional[str] = None,
    batch_size: int = 8,
    appworld_root: Optional[str] = None,
    dataset_name: str = "train",
    max_difficulty: Optional[int] = None,
    max_apis: Optional[int] = None,
) -> Dict[str, Any]:
    """Run AppWorld evaluation with real API execution."""

    limit = max_tasks
    if dry_run:
        limit = 3

    # Phase 1: Load tasks and collect metadata
    print(f"[appworld] === Phase 1: Loading task metadata ===")
    tasks = load_tasks(
        dataset_name=dataset_name,
        max_tasks=limit,
        appworld_root=appworld_root,
        max_difficulty=max_difficulty,
        max_apis=max_apis,
    )

    if not tasks:
        print("[appworld] No tasks loaded. Check AppWorld installation.")
        return {}

    # Phase 2: Batch generate model predictions
    print(f"\n[appworld] === Phase 2: Generating model predictions ===")
    prompts = []
    for t in tasks:
        user_content = build_user_content(t)
        prompt = build_chat_prompt(user_content, model_cfg)
        prompts.append(prompt)

    t0 = time.time()
    responses = generate(prompts, model_cfg, batch_size=batch_size)
    gen_elapsed = time.time() - t0

    if responses:
        print(f"[appworld] Sample response (first 500 chars):")
        print(f"  {responses[0][:500]}")

    # Phase 3: Execute and evaluate each task
    print(f"\n[appworld] === Phase 3: Executing against real APIs ===")
    results: List[Dict[str, Any]] = []
    task_success_count = 0
    exec_success_count = 0
    parse_success_count = 0
    api_recall_sum = 0.0
    error_counter: Counter = Counter()

    t1 = time.time()

    for idx, (task, response) in enumerate(zip(tasks, responses)):
        if idx % 10 == 0:
            print(f"[appworld] Executing task {idx + 1}/{len(tasks)}...")

        parsed_calls, parse_method = parse_tool_calls(response)

        if parsed_calls:
            parse_success_count += 1

        api_recall = _compute_api_name_recall(parsed_calls, task["required_apis"])
        api_recall_sum += api_recall

        eval_result = _execute_and_evaluate(
            task_id=task["task_id"],
            parsed_calls=parsed_calls,
            experiment_name=f"_tool_r0_eval_{model_profile}",
            appworld_root=appworld_root,
        )

        if eval_result["task_success"]:
            task_success_count += 1
        if eval_result["all_exec_success"] and parsed_calls:
            exec_success_count += 1

        if eval_result.get("appworld_error"):
            error_counter["appworld_init_error"] += 1
        elif not parsed_calls:
            error_counter["parse_failure"] += 1
        elif not eval_result["all_exec_success"]:
            error_counter["exec_failure"] += 1
        elif not eval_result["task_success"]:
            error_counter["task_eval_failed"] += 1
        else:
            error_counter["success"] += 1

        status = "completed" if eval_result["task_success"] else "failed"
        if response.startswith("[ERROR]"):
            status = "error"

        exec_successes = sum(1 for r in eval_result["exec_results"] if r["success"])
        exec_total = len(eval_result["exec_results"])

        results.append({
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
            "num_apis": task["num_apis"],
            "question": task["instruction"][:300],
            "response": response[:3000],
            "status": status,
            "score": 1.0 if eval_result["task_success"] else 0.0,
            "task_success": eval_result["task_success"],
            "all_exec_success": eval_result["all_exec_success"],
            "exec_calls_success": exec_successes,
            "exec_calls_total": exec_total,
            "num_predicted_calls": len(parsed_calls) if parsed_calls else 0,
            "gt_num_api_calls": task["gt_num_api_calls"],
            "api_name_recall": api_recall,
            "num_tests_passed": eval_result["num_tests_passed"],
            "num_tests_failed": eval_result["num_tests_failed"],
            "parse_method": parse_method,
            "parse_success": parsed_calls is not None,
            "error_category": (
                "success" if eval_result["task_success"]
                else "parse_failure" if not parsed_calls
                else "exec_failure" if not eval_result["all_exec_success"]
                else "task_eval_failed"
            ),
            "num_tool_calls": len(parsed_calls) if parsed_calls else 0,
        })

    exec_elapsed = time.time() - t1

    total = len(tasks)
    task_success_rate = task_success_count / total if total else 0.0
    exec_success_rate = exec_success_count / total if total else 0.0
    parse_rate = parse_success_count / total if total else 0.0
    mean_api_recall = api_recall_sum / total if total else 0.0

    parse_methods = Counter(r.get("parse_method", "unknown") for r in results)

    os.makedirs(output_dir, exist_ok=True)

    pred_path = os.path.join(output_dir, f"{model_profile}_predictions.jsonl")
    save_predictions(results, pred_path)

    summary = aggregate_summary(
        results,
        benchmark="appworld",
        model_profile=model_profile,
        extra={
            "dataset": dataset_name,
            "task_success_rate": round(task_success_rate, 4),
            "task_success_rate_percent": round(100.0 * task_success_rate, 2),
            "execution_success_rate": round(exec_success_rate, 4),
            "execution_success_rate_percent": round(100.0 * exec_success_rate, 2),
            "mean_api_name_recall": round(mean_api_recall, 4),
            "mean_api_name_recall_percent": round(100.0 * mean_api_recall, 2),
            "parse_success_rate": round(parse_rate, 4),
            "parse_success_rate_percent": round(100.0 * parse_rate, 2),
            "generation_elapsed_seconds": round(gen_elapsed, 2),
            "execution_elapsed_seconds": round(exec_elapsed, 2),
            "error_breakdown": dict(error_counter),
            "parse_methods": dict(parse_methods),
            "max_difficulty_filter": max_difficulty,
            "max_apis_filter": max_apis,
        },
    )

    summary_path = os.path.join(output_dir, f"{model_profile}_summary.json")
    save_summary(summary, summary_path)

    print_summary_table(summary)
    print(f"\n  AppWorld Metrics (REAL API EXECUTION):")
    print(f"  {'Metric':<45} {'Value':>10}")
    print(f"  {'-' * 57}")
    print(f"  {'Task Success Rate (TGC -- primary)':<45} {100.0 * task_success_rate:>9.2f}%")
    print(f"  {'Execution Success Rate':<45} {100.0 * exec_success_rate:>9.2f}%")
    print(f"  {'API Name Recall':<45} {100.0 * mean_api_recall:>9.2f}%")
    print(f"  {'Parse Success Rate':<45} {100.0 * parse_rate:>9.2f}%")
    print(f"  {'-' * 57}")
    print(f"  Error breakdown:    {dict(error_counter)}")
    print(f"  Parse methods:      {dict(parse_methods)}")
    print(f"  Generation time:    {gen_elapsed:.1f}s")
    print(f"  Execution time:     {exec_elapsed:.1f}s")
    print(f"  Predictions:        {pred_path}")
    print(f"  Summary:            {summary_path}")

    return summary
