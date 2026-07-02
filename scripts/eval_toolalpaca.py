#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from step2_genverify import build_solver_prompt  # noqa: E402
from toolalpaca_eval_utils import (  # noqa: E402
    evaluate_prediction,
    load_toolalpaca_examples,
    parse_model_prediction,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate a model on ToolAlpaca using canonical tool-call matching.")
    ap.add_argument("--model_path", type=str, required=True, help="HF model path or local checkpoint directory.")
    ap.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to ToolAlpaca eval JSON. Supports official format or flattened examples.",
    )
    ap.add_argument("--output_path", type=str, required=True, help="Path to write the summary JSON report.")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for vLLM generation.")
    ap.add_argument("--max_new_tokens", type=int, default=256, help="Maximum generated tokens per example.")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    ap.add_argument("--top_p", type=float, default=1.0, help="Top-p for generation.")
    ap.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size.")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="vLLM GPU memory utilization.")
    ap.add_argument("--max_model_len", type=int, default=4096, help="vLLM max model length.")
    ap.add_argument("--predictions_path", type=str, default=None, help="Optional path for per-example JSONL predictions.")
    ap.add_argument("--table_path", type=str, default=None, help="Optional path to write a one-model markdown table report.")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of examples.")
    return ap.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def infer_split_metadata(dataset_path: str) -> Dict[str, Any]:
    name = os.path.basename(dataset_path).lower()
    if "train_data" in name:
        raise ValueError(
            "Refusing to evaluate on train_data.json. "
            "Use eval_simulated.json as PRIMARY and eval_real.json as optional supplementary split."
        )
    if "eval_simulated" in name:
        return {"split_name": "eval_simulated", "split_role": "primary"}
    if "eval_real" in name:
        return {"split_name": "eval_real", "split_role": "supplementary"}
    return {"split_name": "unknown", "split_role": "unspecified"}


def build_summary(results: List[Dict[str, Any]], model_path: str, dataset_path: str, elapsed_seconds: float) -> Dict[str, Any]:
    total = len(results)
    parseable = sum(1 for r in results if r["predicted_calls"] is not None)
    exact_matches = sum(1 for r in results if r["exact_match"])

    reason_counts = Counter(r["parse_reason"] for r in results)
    mean_name = sum(r["diagnostics"]["mean_name_score"] for r in results) / total if total else 0.0
    mean_key = sum(r["diagnostics"]["mean_key_score"] for r in results) / total if total else 0.0
    mean_value = sum(r["diagnostics"]["mean_value_score"] for r in results) / total if total else 0.0
    mean_soft = sum(r["soft_score"] for r in results) / total if total else 0.0

    split_meta = infer_split_metadata(dataset_path)
    return {
        "benchmark": "ToolAlpaca",
        "dataset_path": dataset_path,
        "split_name": split_meta["split_name"],
        "split_role": split_meta["split_role"],
        "model_path": model_path,
        "total_examples": total,
        "parseable_predictions": parseable,
        "exact_canonical_matches": exact_matches,
        "exact_match_accuracy": exact_matches / total if total else 0.0,
        "exact_match_accuracy_percent": 100.0 * exact_matches / total if total else 0.0,
        "final_accuracy": mean_soft,
        "final_accuracy_percent": 100.0 * mean_soft,
        "mean_soft_score": mean_soft,
        "mean_name_match_rate": mean_name,
        "mean_key_match_rate": mean_key,
        "mean_value_match_rate": mean_value,
        "parse_reason_counts": dict(reason_counts),
        "elapsed_seconds": elapsed_seconds,
    }


def warn_if_eval_real(dataset_path: str) -> None:
    ds_name = os.path.basename(dataset_path).lower()
    if "eval_real" not in ds_name:
        return

    print()
    print("WARNING: eval_real.json mode")
    print("  - This split may involve APIs with authentication / live external dependencies in the original ToolAlpaca setup.")
    print("  - This script evaluates tool-call prediction accuracy only (canonical AST-style matching), not real tool execution success.")
    print("  - Compare eval_real scores carefully with papers that run full execution environments.")
    print()


def build_single_model_table(summary: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "| metric | value |",
            "|---|---:|",
            f"| model | `{summary['model_path']}` |",
            f"| dataset | `{summary['dataset_path']}` |",
            f"| total_examples | {summary['total_examples']} |",
            f"| parseable_predictions | {summary['parseable_predictions']} |",
            f"| **ast_accuracy_percent** | **{summary['final_accuracy_percent']:.2f}** |",
            f"| exact_match_percent | {summary['exact_match_accuracy_percent']:.2f} |",
            f"| name_match_percent | {100.0 * summary['mean_name_match_rate']:.2f} |",
            f"| key_match_percent | {100.0 * summary['mean_key_match_rate']:.2f} |",
            f"| value_match_percent | {100.0 * summary['mean_value_match_rate']:.2f} |",
            f"| elapsed_seconds | {summary['elapsed_seconds']:.1f} |",
        ]
    )


def main() -> None:
    args = parse_args()
    t0 = time.time()
    split_meta = infer_split_metadata(args.dataset_path)
    warn_if_eval_real(args.dataset_path)

    examples = load_toolalpaca_examples(args.dataset_path)
    if args.limit is not None:
        examples = examples[: args.limit]

    print(f"[load] dataset={args.dataset_path}")
    print(f"[load] split={split_meta['split_name']} ({split_meta['split_role']})")
    print(f"[load] examples={len(examples)}")
    print(f"[model] model_path={args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )

    prompts = [build_solver_prompt(ex["question"], ex["tools"], tokenizer) for ex in examples]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        n=1,
        stop=["</tool_call_answer>"],
        include_stop_str_in_output=True,
    )

    results: List[Dict[str, Any]] = []
    for start in range(0, len(examples), args.batch_size):
        end = min(start + args.batch_size, len(examples))
        print(f"[infer] {start}-{end}/{len(examples)}")
        outputs = llm.generate(prompts[start:end], sampling_params)

        for ex, req in zip(examples[start:end], outputs):
            raw_text = req.outputs[0].text if req.outputs else ""
            predicted_calls, parse_reason = parse_model_prediction(raw_text)
            metrics = evaluate_prediction(predicted_calls, ex["gold_calls"])
            results.append(
                {
                    "example_id": ex["example_id"],
                    "api_name": ex["api_name"],
                    "question": ex["question"],
                    "tools": ex["tools"],
                    "gold_calls": ex["gold_calls"],
                    "predicted_calls": predicted_calls,
                    "raw_output": raw_text,
                    "parse_reason": parse_reason,
                    "exact_match": metrics["exact_match"],
                    "soft_score": metrics["soft_score"],
                    "diagnostics": metrics["diagnostics"],
                }
            )

    elapsed = time.time() - t0
    summary = build_summary(results, args.model_path, args.dataset_path, elapsed)

    ensure_parent_dir(args.output_path)
    predictions_path = args.predictions_path or os.path.splitext(args.output_path)[0] + ".predictions.jsonl"
    ensure_parent_dir(predictions_path)

    with open(predictions_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "summary": summary,
        "predictions_path": predictions_path,
        "config": {
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "limit": args.limit,
        },
    }
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    table_text = build_single_model_table(summary)
    if args.table_path:
        ensure_parent_dir(args.table_path)
        with open(args.table_path, "w", encoding="utf-8") as f:
            f.write(table_text + "\n")

    print()
    print("ToolAlpaca evaluation")
    print(table_text)
    print(f"  summary json:           {args.output_path}")
    print(f"  predictions jsonl:      {predictions_path}")
    if args.table_path:
        print(f"  table markdown:         {args.table_path}")


if __name__ == "__main__":
    main()

