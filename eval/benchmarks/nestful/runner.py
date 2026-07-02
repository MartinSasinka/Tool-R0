"""
NESTFUL benchmark runner.

Three evaluation modes:

structural (default, paper-aligned)
    Single-shot generate, parse the predicted call list, score it
    structurally against ``gold_calls`` (per-call name + args). Emits the
    same metrics as before — keeps backwards compatibility for any
    existing comparisons.

execute
    Single-shot generate, parse the predicted call list, *execute* it
    locally via :mod:`eval.benchmarks.nestful.executor` (math primitives
    + IBM/NESTFUL Python implementations from
    :mod:`eval.benchmarks.nestful.ibm_loader`), then compare the final
    value with ``gold_answer``. The LLM judge
    (:mod:`eval.benchmarks.nestful.judge`) is **off by default** — opt
    in via ``--nestful-use-judge`` for debugging only.

multiturn
    Interactive agent loop driven by
    :mod:`eval.benchmarks.nestful.multiturn` — the model emits one call
    at a time, we execute it (primitives → IBM funcs), return the result,
    and continue until the model is done or hits the step limit. Final
    value comparison and judge-fallback policy are the same as
    ``execute``.

Output files
------------
``structural`` keeps the historical names ``{profile}_predictions.jsonl``
and ``{profile}_summary.json``. The new modes append a suffix
(``_execute`` / ``_multiturn``) so reruns never overwrite each other.

Reference:
    Basu et al., "NESTFUL: A Benchmark for Evaluating LLMs on Nested
    Sequences of API Calls", 2024. https://arxiv.org/abs/2409.03797
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from eval.ast_eval import robust_value_match
from eval.benchmarks.nestful.executor import (
    ExecutionResult,
    coerce_numeric,
    execute_call_sequence,
)
from eval.benchmarks.nestful.ibm_loader import IBMFunctionRegistry
from eval.benchmarks.nestful.judge import JudgeResult, evaluate_with_llm
from eval.benchmarks.nestful.loader import build_user_content, load_tasks
from eval.benchmarks.nestful.multiturn import MultiTurnResult, run_multiturn_tasks
from eval.metrics import aggregate_summary, print_summary_table, save_predictions, save_summary
from eval.model_adapter import build_chat_prompt, generate
from eval.parse_utils import parse_tool_calls

_VAR_REF_RE = re.compile(r"^\$\w+(\.\w+)?\$$")


# ---------------------------------------------------------------------------
# Structural scoring (paper-aligned, kept for backwards compat)
# ---------------------------------------------------------------------------


def _is_variable_ref(v: Any) -> bool:
    return isinstance(v, str) and _VAR_REF_RE.match(v) is not None


def _match_arguments(
    pred_args: Dict[str, Any],
    gold_args: Dict[str, Any],
) -> Tuple[float, int, int]:
    if not gold_args and not pred_args:
        return 1.0, 0, 0

    total = len(gold_args)
    if total == 0:
        return (0.0 if pred_args else 1.0), 0, 0

    matched = 0
    for key, gold_val in gold_args.items():
        if key not in pred_args:
            continue

        pred_val = pred_args[key]

        if _is_variable_ref(str(gold_val)):
            if _is_variable_ref(str(pred_val)):
                matched += 1
            elif robust_value_match(pred_val, gold_val):
                matched += 1
        else:
            if robust_value_match(pred_val, gold_val):
                matched += 1

    return matched / total if total > 0 else 1.0, matched, total


def _score_single_call(
    pred_call: Dict[str, Any],
    gold_call: Dict[str, Any],
) -> Dict[str, Any]:
    pred_name = (pred_call.get("name") or "").strip()
    gold_name = (gold_call.get("name") or "").strip()
    name_match = pred_name == gold_name

    pred_args = pred_call.get("arguments", {})
    gold_args = gold_call.get("arguments", {})
    if not isinstance(pred_args, dict):
        pred_args = {}
    if not isinstance(gold_args, dict):
        gold_args = {}

    arg_match_ratio, arg_matched, arg_total = _match_arguments(pred_args, gold_args)

    exact_match = name_match and arg_match_ratio == 1.0

    return {
        "name_match": name_match,
        "arg_match_ratio": arg_match_ratio,
        "arg_matched": arg_matched,
        "arg_total": arg_total,
        "exact_match": exact_match,
    }


def _name_f1(pred_names: List[str], gold_names: List[str]) -> Tuple[float, float, float]:
    if not gold_names and not pred_names:
        return 1.0, 1.0, 1.0
    if not pred_names:
        return 0.0, 0.0, 0.0
    if not gold_names:
        return 0.0, 0.0, 0.0

    gold_counts: Counter = Counter(gold_names)
    pred_counts: Counter = Counter(pred_names)

    correct = 0
    for name in gold_counts:
        correct += min(gold_counts[name], pred_counts.get(name, 0))

    precision = correct / len(pred_names) if pred_names else 0.0
    recall = correct / len(gold_names) if gold_names else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def _score_task_structural(
    response: str,
    task: Dict[str, Any],
) -> Dict[str, Any]:
    gold_calls = task["gold_calls"]
    gold_names = [c["name"] for c in gold_calls]

    parsed_calls, parse_method = parse_tool_calls(response)

    if parsed_calls is None or len(parsed_calls) == 0:
        return {
            "parse_success": False,
            "parse_method": parse_method,
            "num_predicted_calls": 0,
            "num_gold_calls": len(gold_calls),
            "length_match": False,
            "full_match": False,
            "per_call_accuracy": 0.0,
            "name_precision": 0.0,
            "name_recall": 0.0,
            "name_f1": 0.0,
            "arg_match_ratio": 0.0,
            "error": "parse_failure",
        }

    pred_names = [c.get("name", "").strip() for c in parsed_calls]
    name_prec, name_rec, name_f1_val = _name_f1(pred_names, gold_names)

    length_match = len(parsed_calls) == len(gold_calls)

    per_call_scores = []
    n_compare = min(len(parsed_calls), len(gold_calls))
    for i in range(n_compare):
        sc = _score_single_call(parsed_calls[i], gold_calls[i])
        per_call_scores.append(sc)

    if n_compare > 0:
        per_call_accuracy = sum(1 for s in per_call_scores if s["exact_match"]) / len(gold_calls)
        arg_match_ratio = sum(s["arg_match_ratio"] for s in per_call_scores) / n_compare
    else:
        per_call_accuracy = 0.0
        arg_match_ratio = 0.0

    full_match = (
        length_match
        and n_compare == len(gold_calls)
        and all(s["exact_match"] for s in per_call_scores)
    )

    error = "none"
    if not length_match:
        error = "length_mismatch"
    elif not full_match:
        name_mismatches = sum(1 for s in per_call_scores if not s["name_match"])
        arg_mismatches = sum(1 for s in per_call_scores if s["name_match"] and not s["exact_match"])
        if name_mismatches > 0:
            error = "name_mismatch"
        elif arg_mismatches > 0:
            error = "arg_mismatch"

    return {
        "parse_success": True,
        "parse_method": parse_method,
        "num_predicted_calls": len(parsed_calls),
        "num_gold_calls": len(gold_calls),
        "length_match": length_match,
        "full_match": full_match,
        "per_call_accuracy": per_call_accuracy,
        "name_precision": name_prec,
        "name_recall": name_rec,
        "name_f1": name_f1_val,
        "arg_match_ratio": arg_match_ratio,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Final-answer scoring (shared by execute + multiturn)
# ---------------------------------------------------------------------------


def _final_answer_match(predicted: Any, gold_answer: Any) -> bool:
    """Compare a numeric/string final value with the dataset gold answer."""
    if predicted is None or gold_answer is None:
        return False
    if robust_value_match(predicted, gold_answer):
        return True
    pred_n = coerce_numeric(predicted)
    gold_n = coerce_numeric(gold_answer)
    if isinstance(pred_n, (int, float)) and isinstance(gold_n, (int, float)):
        try:
            return abs(float(pred_n) - float(gold_n)) < 1e-3
        except (TypeError, ValueError):
            return False
    return False


def _decide_with_judge(
    *,
    task: Dict[str, Any],
    predicted_calls: List[Dict[str, Any]],
    execution_trace: Optional[List[Dict[str, Any]]],
    use_judge_fallback: bool,
    cache_path: str,
) -> Optional[JudgeResult]:
    """Optionally consult the LLM judge for an undecidable task.

    When ``use_judge_fallback`` is False (the new default), returns a
    synthetic ``skip`` verdict so callers can rely on the same control
    flow regardless of whether the judge is on.
    """
    if not use_judge_fallback:
        return JudgeResult(verdict="skip", reason="judge_disabled")
    return evaluate_with_llm(
        task_id=task["task_id"],
        question=task["question"],
        gold_answer=task.get("gold_answer"),
        predicted_calls=predicted_calls,
        execution_trace=execution_trace,
        cache_path=cache_path,
    )


def _try_init_ibm_registry() -> Optional[IBMFunctionRegistry]:
    """Best-effort registry init; returns None and warns if the repo is missing.

    This lets the eval keep running on a host that hasn't cloned the IBM
    repo (e.g. a quick smoke test). The summary will show
    ``ibm_registry_stats.available = false`` so the absence is visible.
    """
    try:
        return IBMFunctionRegistry()
    except FileNotFoundError as exc:
        print(
            "[nestful] WARNING: IBM/NESTFUL functions unavailable — "
            "execution will fall back to math primitives only.\n"
            f"  Reason: {exc}\n"
            "  Run scripts/setup_nestful_funcs.sh to enable IBM funcs."
        )
        return None
    except Exception as exc:
        print(
            f"[nestful] WARNING: failed to initialise IBMFunctionRegistry "
            f"({type(exc).__name__}: {exc}); continuing without it."
        )
        return None


def _classify_trace_source(exec_res: ExecutionResult) -> str:
    """Return the per-task execution-class bucket from a successful sequence.

    For a fully-successful sequence we report whether the *last* call —
    the one whose value is compared against ``gold_answer`` — came from
    primitives or from the IBM registry. (A single sequence can mix both;
    the bucket reflects the producer of the final value.)
    """
    if not exec_res.success or not exec_res.per_call:
        return "executed_ok"
    last = exec_res.per_call[-1]
    src = getattr(last, "source", "unknown")
    if src == "primitive":
        return "executed_ok_primitive"
    if src == "ibm":
        return "executed_ok_ibm"
    return "executed_ok"


# ---------------------------------------------------------------------------
# Mode: structural (current paper-aligned behaviour)
# ---------------------------------------------------------------------------


def _run_structural(
    tasks: List[Dict[str, Any]],
    model_cfg: Dict[str, Any],
    *,
    output_dir: str,
    model_profile: str,
    batch_size: int,
) -> Dict[str, Any]:
    print(f"[nestful] mode=structural — building chat-templated prompts...")
    prompts = [build_chat_prompt(build_user_content(t), model_cfg) for t in tasks]

    print(f"[nestful] generating responses...")
    t0 = time.time()
    responses = generate(prompts, model_cfg, batch_size=batch_size)
    elapsed = time.time() - t0

    if responses:
        print(f"[nestful] sample response (first 500 chars):")
        print(f"  {responses[0][:500]}")

    results: List[Dict[str, Any]] = []
    full_match_count = 0
    length_match_count = 0
    parse_success_count = 0
    name_f1_sum = 0.0
    per_call_acc_sum = 0.0
    arg_match_sum = 0.0

    error_counter: Counter = Counter()
    length_dist: Counter = Counter()

    for task, response in zip(tasks, responses):
        scoring = _score_task_structural(response, task)

        if scoring["full_match"]:
            full_match_count += 1
        if scoring["length_match"]:
            length_match_count += 1
        if scoring["parse_success"]:
            parse_success_count += 1

        name_f1_sum += scoring["name_f1"]
        per_call_acc_sum += scoring["per_call_accuracy"]
        arg_match_sum += scoring["arg_match_ratio"]
        error_counter[scoring["error"]] += 1
        length_dist[task["num_gold_calls"]] += 1

        status = "completed" if scoring["full_match"] else "failed"
        if response.startswith("[ERROR]"):
            status = "error"

        results.append({
            "task_id": task["task_id"],
            "question": task["question"][:300],
            "response": response[:3000],
            "status": status,
            "score": 1.0 if scoring["full_match"] else scoring["per_call_accuracy"],
            "full_match": scoring["full_match"],
            "per_call_accuracy": scoring["per_call_accuracy"],
            "name_f1": scoring["name_f1"],
            "arg_match_ratio": scoring["arg_match_ratio"],
            "num_gold_calls": scoring["num_gold_calls"],
            "num_predicted_calls": scoring["num_predicted_calls"],
            "length_match": scoring["length_match"],
            "parse_method": scoring["parse_method"],
            "parse_success": scoring["parse_success"],
            "error_category": scoring["error"],
            "num_tool_calls": scoring["num_predicted_calls"],
        })

    total = len(tasks)
    full_match_acc = full_match_count / total if total else 0.0
    length_match_rate = length_match_count / total if total else 0.0
    parse_rate = parse_success_count / total if total else 0.0
    mean_name_f1 = name_f1_sum / total if total else 0.0
    mean_per_call_acc = per_call_acc_sum / total if total else 0.0
    mean_arg_match = arg_match_sum / total if total else 0.0

    parse_methods = Counter(r.get("parse_method", "unknown") for r in results)

    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, f"{model_profile}_predictions.jsonl")
    save_predictions(results, pred_path)

    summary = aggregate_summary(
        results,
        benchmark="nestful",
        model_profile=model_profile,
        extra={
            "mode": "structural",
            "full_match_accuracy": round(full_match_acc, 4),
            "full_match_accuracy_percent": round(100.0 * full_match_acc, 2),
            "partial_match_accuracy": round(mean_per_call_acc, 4),
            "partial_match_accuracy_percent": round(100.0 * mean_per_call_acc, 2),
            "mean_name_f1": round(mean_name_f1, 4),
            "mean_name_f1_percent": round(100.0 * mean_name_f1, 2),
            "mean_arg_match_ratio": round(mean_arg_match, 4),
            "mean_arg_match_ratio_percent": round(100.0 * mean_arg_match, 2),
            "length_match_rate": round(length_match_rate, 4),
            "length_match_rate_percent": round(100.0 * length_match_rate, 2),
            "parse_success_rate": round(parse_rate, 4),
            "parse_success_rate_percent": round(100.0 * parse_rate, 2),
            "elapsed_seconds": round(elapsed, 2),
            "error_breakdown": dict(error_counter),
            "parse_methods": dict(parse_methods),
            "sequence_length_distribution": dict(sorted(length_dist.items())),
        },
    )

    summary_path = os.path.join(output_dir, f"{model_profile}_summary.json")
    save_summary(summary, summary_path)

    print_summary_table(summary)
    print(f"\n  NESTFUL Metrics (paper-aligned):")
    print(f"  {'Metric':<40} {'Value':>10}")
    print(f"  {'-' * 52}")
    print(f"  {'Full Match Accuracy':<40} {100.0 * full_match_acc:>9.2f}%")
    print(f"  {'Partial Match Accuracy (per-call)':<40} {100.0 * mean_per_call_acc:>9.2f}%")
    print(f"  {'Name F1':<40} {100.0 * mean_name_f1:>9.2f}%")
    print(f"  {'Arg Match Ratio':<40} {100.0 * mean_arg_match:>9.2f}%")
    print(f"  {'Length Match Rate':<40} {100.0 * length_match_rate:>9.2f}%")
    print(f"  {'Parse Success Rate':<40} {100.0 * parse_rate:>9.2f}%")
    print(f"  {'-' * 52}")
    print(f"  Error breakdown: {dict(error_counter)}")
    print(f"  Parse methods:   {dict(parse_methods)}")
    print(f"  Predictions:     {pred_path}")
    print(f"  Summary:         {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Mode: execute
# ---------------------------------------------------------------------------


def _classify_execution(exec_res: ExecutionResult) -> str:
    """Map an :class:`ExecutionResult` onto a coarse execution-class bucket.

    On success we let :func:`_classify_trace_source` decide the bucket so
    the summary can split between primitive vs. IBM dispatch. Failure
    buckets mirror the new error categories produced by the executor
    (``ibm_unavailable``, ``ibm_runtime_error:*``, ``primitive_error:*``,
    ``unresolved_variable:*``).
    """
    if exec_res.success or exec_res.error is None:
        return _classify_trace_source(exec_res)
    err = exec_res.error
    if err.startswith("unknown_function"):
        return "unknown_function"
    if err.startswith("ibm_unavailable"):
        return "ibm_unavailable"
    if err.startswith("ibm_runtime_error"):
        return "ibm_runtime_error"
    if err.startswith("primitive_error"):
        return "primitive_error"
    if err.startswith("runtime_error"):
        # Legacy executor traces, kept for backwards compat.
        return "primitive_error"
    if err.startswith("unresolved_variable"):
        return "unresolved_variable"
    if err == "empty_call_sequence":
        return "empty_call_sequence"
    if err == "invalid_arguments_type":
        return "invalid_arguments_type"
    return "other_error"


def _run_execute(
    tasks: List[Dict[str, Any]],
    model_cfg: Dict[str, Any],
    *,
    output_dir: str,
    model_profile: str,
    batch_size: int,
    use_judge_fallback: bool,
    judge_cache_path: str,
) -> Dict[str, Any]:
    print(f"[nestful] mode=execute — building prompts...")
    prompts = [build_chat_prompt(build_user_content(t), model_cfg) for t in tasks]

    ibm_registry = _try_init_ibm_registry()

    print(f"[nestful] generating responses...")
    t0 = time.time()
    responses = generate(prompts, model_cfg, batch_size=batch_size)
    elapsed_gen = time.time() - t0

    results: List[Dict[str, Any]] = []
    pass_count = 0
    fail_count = 0
    skip_count = 0
    judge_used = 0
    judge_cache_hits = 0
    exec_class_counter: Counter = Counter()

    for task, response in zip(tasks, responses):
        parsed_calls, parse_method = parse_tool_calls(response)
        gold_answer = task.get("gold_answer")

        if not parsed_calls:
            results.append({
                "task_id": task["task_id"],
                "question": task["question"][:300],
                "response": response[:3000],
                "status": "failed",
                "score": 0.0,
                "verdict": "fail",
                "verdict_reason": "parse_failure",
                "execution_class": "parse_failure",
                "parse_method": parse_method,
                "num_predicted_calls": 0,
                "num_gold_calls": task["num_gold_calls"],
                "predicted_final": None,
                "gold_answer": gold_answer,
                "error_category": "parse_failure",
                "num_tool_calls": 0,
            })
            fail_count += 1
            exec_class_counter["parse_failure"] += 1
            continue

        exec_res = execute_call_sequence(parsed_calls, ibm_registry=ibm_registry)
        exec_class = _classify_execution(exec_res)
        exec_class_counter[exec_class] += 1

        verdict = "fail"
        verdict_reason = ""
        used_judge = False
        used_cache = False

        def _consult_judge() -> None:
            """Closure wrapping the judge call so both branches stay symmetric."""
            nonlocal verdict, verdict_reason, used_judge, used_cache
            jr = _decide_with_judge(
                task=task,
                predicted_calls=parsed_calls,
                execution_trace=[t.to_dict() for t in exec_res.per_call],
                use_judge_fallback=use_judge_fallback,
                cache_path=judge_cache_path,
            )
            used_judge = use_judge_fallback
            used_cache = jr.used_cache if jr else False
            if jr is None or jr.verdict == "skip":
                # Default path (judge off): treat undecidable as fail rather
                # than skip so the headline accuracy is grounded in
                # executor truth, not in API availability.
                if not use_judge_fallback:
                    verdict = "fail"
                    verdict_reason = "executor_mismatch"
                else:
                    verdict = "skip"
                    verdict_reason = jr.reason if jr else "judge_unavailable"
            elif jr.verdict == "error":
                verdict = "skip" if use_judge_fallback else "fail"
                verdict_reason = f"judge_error:{jr.reason}"
            else:
                verdict = jr.verdict
                verdict_reason = f"judge_{jr.verdict}:{jr.reason}"

        if exec_res.success:
            if _final_answer_match(exec_res.final_value, gold_answer):
                verdict = "pass"
                verdict_reason = "executor_match"
            else:
                _consult_judge()
        else:
            _consult_judge()

        if verdict == "pass":
            pass_count += 1
            status = "completed"
            score = 1.0
        elif verdict == "skip":
            skip_count += 1
            status = "skipped"
            score = 0.0
        else:
            fail_count += 1
            status = "failed"
            score = 0.0

        if used_judge:
            judge_used += 1
        if used_cache:
            judge_cache_hits += 1

        results.append({
            "task_id": task["task_id"],
            "question": task["question"][:300],
            "response": response[:3000],
            "status": status,
            "score": score,
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "execution_class": exec_class,
            "execution_error": exec_res.error,
            "predicted_final": exec_res.final_value,
            "gold_answer": gold_answer,
            "num_predicted_calls": len(parsed_calls),
            "num_gold_calls": task["num_gold_calls"],
            "parse_method": parse_method,
            "judge_used": used_judge,
            "judge_cache_hit": used_cache,
            "error_category": exec_class if verdict == "fail" else verdict,
            "num_tool_calls": len(parsed_calls),
        })

    total = len(tasks)
    elapsed = time.time() - t0
    final_acc = pass_count / total if total else 0.0
    skip_rate = skip_count / total if total else 0.0
    judge_rate = judge_used / total if total else 0.0

    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, f"{model_profile}_execute_predictions.jsonl")
    save_predictions(results, pred_path)

    ibm_stats = ibm_registry.stats() if ibm_registry is not None else {
        "available": False,
        "cached_imports": 0,
        "unavailable_funcs": 0,
    }

    summary = aggregate_summary(
        results,
        benchmark="nestful",
        model_profile=model_profile,
        extra={
            "mode": "execute",
            "final_answer_accuracy": round(final_acc, 4),
            "final_answer_accuracy_percent": round(100.0 * final_acc, 2),
            "passed": pass_count,
            "failed": fail_count,
            "skipped": skip_count,
            "skipped_rate_percent": round(100.0 * skip_rate, 2),
            "judge_used_count": judge_used,
            "judge_used_rate_percent": round(100.0 * judge_rate, 2),
            "judge_cache_hits": judge_cache_hits,
            "execution_class_breakdown": dict(exec_class_counter),
            "ibm_registry_stats": ibm_stats,
            "elapsed_generate_seconds": round(elapsed_gen, 2),
            "elapsed_total_seconds": round(elapsed, 2),
            "use_judge_fallback": use_judge_fallback,
        },
    )

    summary_path = os.path.join(output_dir, f"{model_profile}_execute_summary.json")
    save_summary(summary, summary_path)

    print_summary_table(summary)
    print(f"\n  NESTFUL Execute Metrics:")
    print(f"  {'-' * 52}")
    print(f"  {'Final Answer Accuracy':<40} {100.0 * final_acc:>9.2f}%")
    print(f"  {'Pass / Fail / Skip':<40} {pass_count}/{fail_count}/{skip_count}")
    print(f"  {'Judge Used':<40} {judge_used} ({100.0 * judge_rate:.1f}%)")
    print(f"  {'Judge Cache Hits':<40} {judge_cache_hits}")
    print(f"  Execution classes: {dict(exec_class_counter)}")
    print(
        f"  IBM registry:      available={ibm_stats.get('available')} "
        f"cached={ibm_stats.get('cached_imports')} "
        f"unavailable={ibm_stats.get('unavailable_funcs')}"
    )
    print(f"  Predictions:     {pred_path}")
    print(f"  Summary:         {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Mode: multiturn
# ---------------------------------------------------------------------------


def _run_multiturn(
    tasks: List[Dict[str, Any]],
    model_cfg: Dict[str, Any],
    *,
    output_dir: str,
    model_profile: str,
    batch_size: int,
    max_steps: int,
    use_judge_fallback: bool,
    judge_cache_path: str,
) -> Dict[str, Any]:
    print(f"[nestful] mode=multiturn — running interactive loop (max_steps={max_steps})...")
    ibm_registry = _try_init_ibm_registry()
    t0 = time.time()
    mt_results: List[MultiTurnResult] = run_multiturn_tasks(
        tasks,
        model_cfg,
        max_steps=max_steps,
        batch_size=batch_size,
        ibm_registry=ibm_registry,
    )
    elapsed_mt = time.time() - t0

    results: List[Dict[str, Any]] = []
    pass_count = 0
    fail_count = 0
    skip_count = 0
    judge_used = 0
    judge_cache_hits = 0
    stop_counter: Counter = Counter()
    step_counts: List[int] = []
    exec_class_counter: Counter = Counter()

    for task, mt in zip(tasks, mt_results):
        gold_answer = task.get("gold_answer")
        stop_counter[mt.stopped] += 1
        step_counts.append(mt.num_steps)

        predicted_calls = []
        execution_trace = []
        per_task_sources: Counter = Counter()
        for tr in mt.traces:
            predicted_calls.append({
                "name": tr.name,
                "arguments": tr.arguments_resolved,
                "label": tr.label,
            })
            execution_trace.append(tr.to_dict())
            per_task_sources[getattr(tr, "source", "unknown")] += 1

        used_judge = False
        used_cache = False
        verdict = "fail"
        verdict_reason = ""

        local_match = mt.final_value is not None and _final_answer_match(mt.final_value, gold_answer)
        if local_match:
            verdict = "pass"
            verdict_reason = "executor_match"
        else:
            jr = _decide_with_judge(
                task=task,
                predicted_calls=predicted_calls,
                execution_trace=execution_trace,
                use_judge_fallback=use_judge_fallback,
                cache_path=judge_cache_path,
            )
            used_judge = use_judge_fallback
            used_cache = jr.used_cache if jr else False
            if jr is None or jr.verdict == "skip":
                if not use_judge_fallback:
                    verdict = "fail"
                    verdict_reason = "executor_mismatch"
                else:
                    verdict = "skip"
                    verdict_reason = jr.reason if jr else "judge_unavailable"
            elif jr.verdict == "error":
                verdict = "skip" if use_judge_fallback else "fail"
                verdict_reason = f"judge_error:{jr.reason}"
            else:
                verdict = jr.verdict
                verdict_reason = f"judge_{jr.verdict}:{jr.reason}"

        # Bucket by what produced the *final* trace (or by error class).
        if mt.error:
            err = mt.error
            if err.startswith("ibm_runtime_error"):
                exec_class_counter["ibm_runtime_error"] += 1
            elif err.startswith("ibm_unavailable"):
                exec_class_counter["ibm_unavailable"] += 1
            elif err.startswith("primitive_error") or err.startswith("runtime_error"):
                exec_class_counter["primitive_error"] += 1
            elif err.startswith("unknown_function"):
                exec_class_counter["unknown_function"] += 1
            elif err.startswith("unresolved_variable"):
                exec_class_counter["unresolved_variable"] += 1
            else:
                exec_class_counter["other_error"] += 1
        elif mt.traces:
            last_src = getattr(mt.traces[-1], "source", "unknown")
            if last_src == "primitive":
                exec_class_counter["executed_ok_primitive"] += 1
            elif last_src == "ibm":
                exec_class_counter["executed_ok_ibm"] += 1
            else:
                exec_class_counter["executed_ok"] += 1
        else:
            exec_class_counter["no_calls_made"] += 1

        if verdict == "pass":
            pass_count += 1
            status = "completed"
            score = 1.0
        elif verdict == "skip":
            skip_count += 1
            status = "skipped"
            score = 0.0
        else:
            fail_count += 1
            status = "failed"
            score = 0.0

        if used_judge:
            judge_used += 1
        if used_cache:
            judge_cache_hits += 1

        results.append({
            "task_id": task["task_id"],
            "question": task["question"][:300],
            "status": status,
            "score": score,
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "stopped": mt.stopped,
            "num_steps": mt.num_steps,
            "predicted_final": mt.final_value,
            "gold_answer": gold_answer,
            "predicted_calls": predicted_calls[:20],
            "execution_trace": execution_trace[:20],
            "raw_completions": [c[:1500] for c in mt.raw_completions[:6]],
            "execution_error": mt.error,
            "trace_source_counts": dict(per_task_sources),
            "judge_used": used_judge,
            "judge_cache_hit": used_cache,
            "num_tool_calls": len(mt.traces),
            "error_category": mt.error or mt.stopped,
        })

    total = len(tasks)
    elapsed = time.time() - t0
    final_acc = pass_count / total if total else 0.0
    skip_rate = skip_count / total if total else 0.0
    judge_rate = judge_used / total if total else 0.0
    avg_steps = sum(step_counts) / total if total else 0.0
    step_limit_rate = stop_counter.get("step_limit", 0) / total if total else 0.0
    context_limit_rate = stop_counter.get("context_limit", 0) / total if total else 0.0
    explicit_final_rate = stop_counter.get("explicit_final", 0) / total if total else 0.0

    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, f"{model_profile}_multiturn_predictions.jsonl")
    save_predictions(results, pred_path)

    ibm_stats = ibm_registry.stats() if ibm_registry is not None else {
        "available": False,
        "cached_imports": 0,
        "unavailable_funcs": 0,
    }

    summary = aggregate_summary(
        results,
        benchmark="nestful",
        model_profile=model_profile,
        extra={
            "mode": "multiturn",
            "final_answer_accuracy": round(final_acc, 4),
            "final_answer_accuracy_percent": round(100.0 * final_acc, 2),
            "passed": pass_count,
            "failed": fail_count,
            "skipped": skip_count,
            "skipped_rate_percent": round(100.0 * skip_rate, 2),
            "judge_used_count": judge_used,
            "judge_used_rate_percent": round(100.0 * judge_rate, 2),
            "judge_cache_hits": judge_cache_hits,
            "avg_steps": round(avg_steps, 2),
            "step_limit_hit_rate_percent": round(100.0 * step_limit_rate, 2),
            "context_limit_hit_rate_percent": round(100.0 * context_limit_rate, 2),
            "explicit_final_rate_percent": round(100.0 * explicit_final_rate, 2),
            "stop_reason_breakdown": dict(stop_counter),
            "execution_class_breakdown": dict(exec_class_counter),
            "ibm_registry_stats": ibm_stats,
            "elapsed_multiturn_seconds": round(elapsed_mt, 2),
            "elapsed_total_seconds": round(elapsed, 2),
            "max_steps_setting": max_steps,
            "use_judge_fallback": use_judge_fallback,
        },
    )

    summary_path = os.path.join(output_dir, f"{model_profile}_multiturn_summary.json")
    save_summary(summary, summary_path)

    print_summary_table(summary)
    print(f"\n  NESTFUL Multiturn Metrics:")
    print(f"  {'-' * 52}")
    print(f"  {'Final Answer Accuracy':<40} {100.0 * final_acc:>9.2f}%")
    print(f"  {'Pass / Fail / Skip':<40} {pass_count}/{fail_count}/{skip_count}")
    print(f"  {'Avg Steps':<40} {avg_steps:>9.2f}")
    print(f"  {'Step-limit hit rate':<40} {100.0 * step_limit_rate:>9.2f}%")
    print(f"  {'Context-limit hit rate':<40} {100.0 * context_limit_rate:>9.2f}%")
    print(f"  {'Explicit final-answer rate':<40} {100.0 * explicit_final_rate:>9.2f}%")
    print(f"  {'Judge Used':<40} {judge_used} ({100.0 * judge_rate:.1f}%)")
    print(f"  Stop reasons:      {dict(stop_counter)}")
    print(f"  Execution classes: {dict(exec_class_counter)}")
    print(
        f"  IBM registry:      available={ibm_stats.get('available')} "
        f"cached={ibm_stats.get('cached_imports')} "
        f"unavailable={ibm_stats.get('unavailable_funcs')}"
    )
    print(f"  Predictions:  {pred_path}")
    print(f"  Summary:      {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def run(
    model_cfg: Dict[str, Any],
    max_tasks: Optional[int] = None,
    output_dir: str = "eval/results/nestful",
    model_profile: str = "default",
    dry_run: bool = False,
    cache_dir: Optional[str] = None,
    batch_size: int = 8,
    *,
    mode: str = "structural",
    max_steps: int = 10,
    use_judge_fallback: bool = False,
    judge_cache_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run NESTFUL evaluation in the requested mode.

    Args:
        mode: ``structural`` (default, paper-aligned), ``execute``
            (single-shot generate + local execution incl. IBM funcs),
            or ``multiturn`` (interactive agent loop).
        max_steps: only for ``multiturn``. Hard cap on turns per task.
        use_judge_fallback: when True the LLM judge is consulted for
            executor mismatches/errors. Off by default — execution
            is now decisive thanks to the IBM function registry.
        judge_cache_path: where to persist judge verdicts (only used
            when ``use_judge_fallback=True``). Defaults to
            ``<output_dir>/_judge_cache.jsonl``.
    """

    limit = max_tasks
    if dry_run:
        limit = 5

    print(f"[nestful] mode={mode} — loading tasks...")
    tasks = load_tasks(max_tasks=limit, cache_dir=cache_dir)
    print(f"[nestful] loaded {len(tasks)} tasks")

    if judge_cache_path is None:
        judge_cache_path = os.path.join(output_dir, "_judge_cache.jsonl")

    if mode == "structural":
        return _run_structural(
            tasks,
            model_cfg,
            output_dir=output_dir,
            model_profile=model_profile,
            batch_size=batch_size,
        )
    if mode == "execute":
        return _run_execute(
            tasks,
            model_cfg,
            output_dir=output_dir,
            model_profile=model_profile,
            batch_size=batch_size,
            use_judge_fallback=use_judge_fallback,
            judge_cache_path=judge_cache_path,
        )
    if mode == "multiturn":
        return _run_multiturn(
            tasks,
            model_cfg,
            output_dir=output_dir,
            model_profile=model_profile,
            batch_size=batch_size,
            max_steps=max_steps,
            use_judge_fallback=use_judge_fallback,
            judge_cache_path=judge_cache_path,
        )
    raise ValueError(
        f"Unknown nestful mode {mode!r}. Choose from: structural, execute, multiturn."
    )
