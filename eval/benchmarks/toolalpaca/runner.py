"""
ToolAlpaca benchmark runner.

Evaluates a model on ToolAlpaca using canonical tool-call matching
with ground-truth tool calls from real API schemas.

This wraps the existing scoring infrastructure from scripts/toolalpaca_eval_utils.py
into the unified eval pipeline, using model_adapter.py for inference.

Scoring uses the same `compute_accuracy_score` as Tool-R0 training:
  - Name match (lambda=0.2): exact function name
  - Key F1 (lambda=0.3): argument key overlap
  - Value match (lambda=0.5): argument value correctness

Dataset: ToolAlpaca eval JSON (eval_simulated.json from tangqiaoyu/ToolAlpaca)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from eval.ast_eval import ast_match
from eval.metrics import aggregate_summary, print_summary_table, save_predictions, save_summary
from eval.model_adapter import build_chat_prompt, generate

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from scripts.toolalpaca_eval_utils import (  # noqa: E402
        evaluate_prediction,
        load_toolalpaca_examples,
        parse_model_prediction,
    )
    _HAS_TOOLALPACA_UTILS = True
except ImportError as e:
    _HAS_TOOLALPACA_UTILS = False
    _TOOLALPACA_IMPORT_ERROR = str(e)


def _resolve_dataset_path(dataset_path: Optional[str] = None) -> str:
    """Resolve ToolAlpaca dataset path, downloading if needed."""
    if dataset_path and os.path.isfile(dataset_path):
        return dataset_path

    standard_paths = [
        os.path.join(_REPO_ROOT, "data", "toolalpaca", "eval_simulated.json"),
        os.path.join(_REPO_ROOT, "data", "eval_simulated.json"),
    ]
    for p in standard_paths:
        if os.path.isfile(p):
            return p

    print("[toolalpaca] Dataset not found locally, downloading from HuggingFace...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="tangqiaoyu/ToolAlpaca",
        filename="eval_simulated.json",
        repo_type="dataset",
    )
    return path


def _build_user_content(example: Dict[str, Any]) -> str:
    """Build user prompt in Tool-R0 format from a ToolAlpaca example."""
    question = example["question"]
    tools_json = json.dumps(example["tools"], ensure_ascii=False)
    return f"User request:\n{question}\n\nAvailable tools (JSON):\n{tools_json}\n"


def run(
    model_cfg: Dict[str, Any],
    dataset_path: Optional[str] = None,
    max_tasks: Optional[int] = None,
    output_dir: str = "eval/results/toolalpaca",
    model_profile: str = "default",
    dry_run: bool = False,
    batch_size: int = 8,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Run ToolAlpaca evaluation."""
    if not _HAS_TOOLALPACA_UTILS:
        print(f"[toolalpaca] ERROR: Could not import toolalpaca_eval_utils: {_TOOLALPACA_IMPORT_ERROR}")
        print("[toolalpaca] This typically means wandb or another dependency is not installed.")
        print("[toolalpaca] Install with: pip install wandb  (or run on DGX where it's available)")
        sys.exit(1)

    resolved_path = _resolve_dataset_path(dataset_path)
    print(f"[toolalpaca] Loading dataset from {resolved_path}")

    examples = load_toolalpaca_examples(resolved_path)
    print(f"[toolalpaca] Loaded {len(examples)} examples")

    if max_tasks:
        examples = examples[:max_tasks]
    if dry_run:
        examples = examples[:3]
        print(f"[toolalpaca] Dry-run mode: using {len(examples)} examples")

    print(f"[toolalpaca] Building chat-templated prompts...")
    prompts = []
    for ex in examples:
        user_content = _build_user_content(ex)
        prompt = build_chat_prompt(user_content, model_cfg)
        prompts.append(prompt)

    print(f"[toolalpaca] Generating responses...")
    t0 = time.time()
    responses = generate(prompts, model_cfg, batch_size=batch_size)
    elapsed = time.time() - t0

    if responses:
        print(f"[toolalpaca] Sample response (first 500 chars):")
        print(f"  {responses[0][:500]}")

    results: List[Dict[str, Any]] = []
    total_soft = 0.0
    ast_matches = 0
    exact_matches = 0
    name_scores = []
    key_scores = []
    value_scores = []

    for ex, response in zip(examples, responses):
        predicted_calls, parse_reason = parse_model_prediction(response)
        metrics = evaluate_prediction(predicted_calls, ex["gold_calls"])

        soft_score = metrics["soft_score"]
        total_soft += soft_score

        is_ast_match = ast_match(predicted_calls, ex["gold_calls"])
        if is_ast_match:
            ast_matches += 1

        diag = metrics["diagnostics"]
        name_scores.append(diag["mean_name_score"])
        key_scores.append(diag["mean_key_score"])
        value_scores.append(diag["mean_value_score"])

        if metrics["exact_match"]:
            exact_matches += 1

        status = "error" if response.startswith("[ERROR]") else (
            "completed" if is_ast_match else "failed"
        )
        error_cat = "none"
        if status == "error":
            error_cat = "api_error"
        elif predicted_calls is None:
            error_cat = "parse_failure"
        elif not is_ast_match:
            if diag["mean_name_score"] < 1.0:
                error_cat = "wrong_function"
            elif diag["mean_key_score"] < 1.0:
                error_cat = "wrong_params"
            else:
                error_cat = "wrong_values"

        results.append({
            "task_id": ex["example_id"],
            "api_name": ex["api_name"],
            "question": ex["question"][:300],
            "response": response[:2000],
            "status": status,
            "score": 1.0 if is_ast_match else 0.0,
            "ast_match": is_ast_match,
            "soft_score": soft_score,
            "exact_match": metrics["exact_match"],
            "parse_reason": parse_reason,
            "num_tool_calls": len(predicted_calls) if predicted_calls else 0,
            "diagnostics": diag,
            "error_category": error_cat,
        })

    os.makedirs(output_dir, exist_ok=True)

    pred_path = os.path.join(output_dir, f"{model_profile}_predictions.jsonl")
    save_predictions(results, pred_path)

    n = len(results)
    ast_acc = ast_matches / n if n else 0.0
    mean_soft = total_soft / n if n else 0.0
    mean_name = sum(name_scores) / n if n else 0.0
    mean_key = sum(key_scores) / n if n else 0.0
    mean_value = sum(value_scores) / n if n else 0.0

    from collections import Counter
    parse_reasons = Counter(r["parse_reason"] for r in results)

    summary = aggregate_summary(
        results,
        benchmark="toolalpaca",
        model_profile=model_profile,
        extra={
            "dataset_path": resolved_path,
            "ast_accuracy": round(ast_acc, 4),
            "ast_accuracy_percent": round(100.0 * ast_acc, 2),
            "soft_accuracy": round(mean_soft, 4),
            "soft_accuracy_percent": round(100.0 * mean_soft, 2),
            "exact_match_accuracy": round(exact_matches / n, 4) if n else 0.0,
            "exact_match_percent": round(100.0 * exact_matches / n, 2) if n else 0.0,
            "mean_name_match": round(mean_name, 4),
            "mean_key_match": round(mean_key, 4),
            "mean_value_match": round(mean_value, 4),
            "parse_reasons": dict(parse_reasons),
            "elapsed_seconds": round(elapsed, 2),
        },
    )

    summary_path = os.path.join(output_dir, f"{model_profile}_summary.json")
    save_summary(summary, summary_path)

    print_summary_table(summary)
    print(f"\n  ToolAlpaca Scores:")
    print(f"  {'Metric':<35} {'Value':>10}")
    print(f"  {'-' * 47}")
    print(f"  {'AST Accuracy (paper-comparable)':<35} {100.0 * ast_acc:>9.2f}%")
    print(f"  {'Soft Score (training reward)':<35} {100.0 * mean_soft:>9.2f}%")
    print(f"  {'Exact Match (strict JSON)':<35} {100.0 * exact_matches / n if n else 0:>9.2f}%")
    print(f"  {'-' * 47}")
    print(f"  {'Name Match':<35} {100.0 * mean_name:>9.2f}%")
    print(f"  {'Key Match (F1)':<35} {100.0 * mean_key:>9.2f}%")
    print(f"  {'Value Match':<35} {100.0 * mean_value:>9.2f}%")
    print(f"  {'-' * 47}")
    print(f"  Parse reasons: {dict(parse_reasons)}")
    print(f"  Predictions: {pred_path}")
    print(f"  Summary:     {summary_path}")

    return summary
