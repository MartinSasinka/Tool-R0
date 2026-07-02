#!/usr/bin/env python3
"""
Unified eval runner for Tool-R0 benchmarks.

Usage:
    python -m eval.run_eval --benchmark bfcl --config eval/configs/baseline.yaml
    python -m eval.run_eval --benchmark toolalpaca --config eval/configs/baseline.yaml
    python -m eval.run_eval --benchmark apibank --config eval/configs/baseline.yaml
    python -m eval.run_eval --benchmark bfcl --config eval/configs/finetuned.yaml --category simple
    python -m eval.run_eval --benchmark toolalpaca --config eval/configs/finetuned.yaml --dry-run

See eval/README.md for full documentation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON config file."""
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:
            print("ERROR: PyYAML required for YAML configs. Install: pip install pyyaml")
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def merge_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override config values with CLI arguments when provided."""
    if args.model_path:
        cfg.setdefault("model", {})["model_path"] = args.model_path
        cfg["model"]["model_name"] = args.model_path
    if args.backend:
        cfg.setdefault("model", {})["backend"] = args.backend
    if args.max_tasks is not None:
        cfg["max_tasks"] = args.max_tasks
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.dry_run:
        cfg["dry_run"] = True
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified eval runner for Tool-R0 benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test with dummy backend (no GPU needed)
  python -m eval.run_eval --benchmark bfcl --config eval/configs/smoke_test.yaml --dry-run

  # Run BFCL with baseline model (all default categories)
  python -m eval.run_eval --benchmark bfcl --config eval/configs/baseline.yaml --profile-name baseline

  # Run BFCL simple category only
  python -m eval.run_eval --benchmark bfcl --config eval/configs/baseline.yaml --category simple

  # Run NESTFUL (nested API sequences, 1861 tasks)
  python -m eval.run_eval --benchmark nestful --config eval/configs/baseline.yaml --profile-name baseline

  # Run ToolAlpaca with fine-tuned model
  python -m eval.run_eval --benchmark toolalpaca --config eval/configs/finetuned.yaml --profile-name finetuned

  # Run ToolAlpaca with custom dataset path
  python -m eval.run_eval --benchmark toolalpaca --config eval/configs/baseline.yaml --dataset-path data/toolalpaca/eval_simulated.json

  # Run API-Bank Level 1 (real API benchmark, 73 APIs)
  python -m eval.run_eval --benchmark apibank --config eval/configs/baseline.yaml --profile-name baseline
""",
    )
    p.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["bfcl", "toolalpaca", "apibank", "tooltalk", "nestful", "appworld"],
        help="Which benchmark to run: bfcl, toolalpaca, apibank, tooltalk, nestful, or appworld (real API execution).",
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML/JSON config file with model and eval settings.",
    )
    p.add_argument("--model-path", type=str, default=None, help="Override model path from config.")
    p.add_argument("--backend", type=str, default=None, choices=["vllm", "openai", "dummy"], help="Override backend.")
    p.add_argument("--max-tasks", type=int, default=None, help="Override max number of tasks (per category for BFCL).")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    p.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    p.add_argument("--dry-run", action="store_true", help="Run with minimal tasks for quick validation.")
    p.add_argument("--profile-name", type=str, default=None, help="Override model profile name for output files.")

    p.add_argument(
        "--category",
        type=str,
        default=None,
        help="BFCL: comma-separated categories (default: simple,multiple,parallel,irrelevance).",
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="ToolAlpaca: path to eval JSON (default: auto-detect or download).",
    )

    p.add_argument("--appworld-root", type=str, default=None, help="AppWorld: APPWORLD_ROOT path.")
    p.add_argument("--appworld-dataset", type=str, default="train", help="AppWorld: dataset split (train/dev).")
    p.add_argument("--appworld-max-difficulty", type=int, default=None, help="AppWorld: filter by max difficulty.")
    p.add_argument("--appworld-max-apis", type=int, default=None, help="AppWorld: filter by max number of APIs.")

    p.add_argument(
        "--nestful-mode",
        type=str,
        default=None,
        help="NESTFUL evaluation mode (default: structural). May be a single mode "
             "(structural | execute | multiturn) or a comma-separated list "
             "(e.g. 'structural,execute,multiturn'). When multiple modes are passed "
             "they all run inside ONE process and share a single vLLM engine — this "
             "saves tens of seconds of model loading per extra mode. 'execute' runs "
             "the predicted call sequence locally (math primitives + IBM/NESTFUL "
             "Python functions); 'multiturn' runs an interactive agent loop. Both "
             "write to suffixed output files so they don't clobber the structural "
             "baseline.",
    )
    p.add_argument(
        "--nestful-max-steps",
        type=int,
        default=None,
        help="NESTFUL multiturn: max turns per task (default: 10).",
    )
    p.add_argument(
        "--nestful-use-judge",
        action="store_true",
        help="NESTFUL execute/multiturn: opt in to the LLM judge fallback for "
             "tasks the executor (primitives + IBM funcs) cannot decide. OFF by "
             "default — recommended path is to clone the IBM repo via "
             "scripts/setup_nestful_funcs.sh and rely on real Python execution. "
             "Only enable for debugging or when the IBM repo is unavailable.",
    )
    p.add_argument(
        "--nestful-no-judge",
        action="store_true",
        help="DEPRECATED no-op: the LLM judge is now OFF by default. Kept for "
             "backwards compatibility with older shell scripts. Use "
             "--nestful-use-judge to opt back in.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    print(f"{'=' * 60}")
    print(f"  Tool-R0 Eval Runner")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Config:    {args.config}")
    print(f"  Dry-run:   {args.dry_run}")
    print(f"{'=' * 60}")

    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)

    model_cfg = cfg.get("model", {})
    profile_name = args.profile_name or cfg.get("profile_name", "default")
    max_tasks = cfg.get("max_tasks")
    batch_size = cfg.get("batch_size", 8)
    dry_run = cfg.get("dry_run", False)
    cache_dir = cfg.get("cache_dir")

    if args.benchmark == "bfcl":
        from eval.benchmarks.bfcl.runner import run as run_bfcl

        if args.category:
            categories = [c.strip() for c in args.category.split(",")]
        else:
            categories = cfg.get("bfcl_categories", ["simple", "multiple", "parallel", "irrelevance"])

        output_dir = cfg.get("output_dir", "eval/results/bfcl")

        summary = run_bfcl(
            model_cfg=model_cfg,
            categories=categories,
            max_tasks=max_tasks,
            output_dir=output_dir,
            model_profile=profile_name,
            dry_run=dry_run,
            cache_dir=cache_dir,
            batch_size=batch_size,
        )

    elif args.benchmark == "toolalpaca":
        from eval.benchmarks.toolalpaca.runner import run as run_toolalpaca

        dataset_path = args.dataset_path or cfg.get("toolalpaca_dataset_path")
        output_dir = cfg.get("output_dir", "eval/results/toolalpaca")

        summary = run_toolalpaca(
            model_cfg=model_cfg,
            dataset_path=dataset_path,
            max_tasks=max_tasks,
            output_dir=output_dir,
            model_profile=profile_name,
            dry_run=dry_run,
            batch_size=batch_size,
        )

    elif args.benchmark == "apibank":
        from eval.benchmarks.apibank.runner import run as run_apibank

        output_dir = cfg.get("output_dir", "eval/results/apibank")

        summary = run_apibank(
            model_cfg=model_cfg,
            max_tasks=max_tasks,
            output_dir=output_dir,
            model_profile=profile_name,
            dry_run=dry_run,
            cache_dir=cache_dir,
            batch_size=batch_size,
        )

    elif args.benchmark == "tooltalk":
        from eval.benchmarks.tooltalk.runner import run as run_tooltalk

        output_dir = cfg.get("output_dir", "eval/results/tooltalk")

        summary = run_tooltalk(
            model_cfg=model_cfg,
            max_tasks=max_tasks,
            output_dir=output_dir,
            model_profile=profile_name,
            dry_run=dry_run,
            cache_dir=cache_dir,
            batch_size=batch_size,
        )

    elif args.benchmark == "nestful":
        from eval.benchmarks.nestful.runner import run as run_nestful

        output_dir = cfg.get("output_dir", "eval/results/nestful")
        raw_modes = args.nestful_mode or cfg.get("nestful_mode", "structural")
        nestful_modes = [m.strip() for m in str(raw_modes).split(",") if m.strip()]
        nestful_max_steps = (
            args.nestful_max_steps
            if args.nestful_max_steps is not None
            else int(cfg.get("nestful_max_steps", 10))
        )
        if args.nestful_no_judge:
            print(
                "[run_eval] --nestful-no-judge is deprecated and a no-op; "
                "the LLM judge is OFF by default. Use --nestful-use-judge to opt in."
            )
        # Default: judge OFF. Opt in via the new flag, or via the legacy
        # config key `nestful_use_judge_fallback` (or the older
        # `nestful_judge_enabled`, which now also defaults to False).
        use_judge_fallback = bool(
            args.nestful_use_judge
            or cfg.get("nestful_use_judge_fallback", False)
            or cfg.get("nestful_judge_enabled", False)
        )

        # Run every requested mode INSIDE THE SAME PROCESS so vLLM is loaded once
        # (a single 4B model load is ~30-60 s + warmup; running structural+execute+
        # multiturn in three subprocesses would pay this 3 times for no reason).
        # The runner caches the engine in eval.model_adapter; reset() is called only
        # at the very end.
        summary = None
        mode_summaries = {}
        for i, mode in enumerate(nestful_modes, start=1):
            print(
                f"\n{'#' * 60}\n"
                f"#  NESTFUL mode {i}/{len(nestful_modes)}: {mode}\n"
                f"#  (vLLM shared with other modes in this process)\n"
                f"{'#' * 60}\n"
            )
            try:
                s = run_nestful(
                    model_cfg=model_cfg,
                    max_tasks=max_tasks,
                    output_dir=output_dir,
                    model_profile=profile_name,
                    dry_run=dry_run,
                    cache_dir=cache_dir,
                    batch_size=batch_size,
                    mode=mode,
                    max_steps=nestful_max_steps,
                    use_judge_fallback=use_judge_fallback,
                )
            except Exception as exc:
                print(f"\n[run_eval] mode={mode} FAILED: {type(exc).__name__}: {exc}")
                mode_summaries[mode] = {"mode": mode, "error": str(exc)}
                continue
            mode_summaries[mode] = s
            summary = s

        if len(nestful_modes) > 1:
            print(f"\n{'=' * 60}\n  NESTFUL multi-mode run complete\n{'=' * 60}")
            for m in nestful_modes:
                ms = mode_summaries.get(m, {})
                if "error" in ms:
                    print(f"  {m:<12} FAILED ({ms['error']})")
                else:
                    score_key = (
                        "partial_match_accuracy_percent"
                        if m == "structural"
                        else "final_answer_accuracy_percent"
                    )
                    print(f"  {m:<12} {ms.get(score_key, '?')}% ({score_key})")

        if summary is None:
            summary = {"benchmark": "nestful", "error": "all_modes_failed", "modes": list(mode_summaries.keys())}

    elif args.benchmark == "appworld":
        from eval.benchmarks.appworld.runner import run as run_appworld

        output_dir = cfg.get("output_dir", "eval/results/appworld")
        aw_root = args.appworld_root or cfg.get("appworld_root")
        aw_dataset = args.appworld_dataset or cfg.get("appworld_dataset", "train")
        aw_max_diff = args.appworld_max_difficulty
        aw_max_apis = args.appworld_max_apis

        summary = run_appworld(
            model_cfg=model_cfg,
            max_tasks=max_tasks,
            output_dir=output_dir,
            model_profile=profile_name,
            dry_run=dry_run,
            cache_dir=cache_dir,
            batch_size=batch_size,
            appworld_root=aw_root,
            dataset_name=aw_dataset,
            max_difficulty=aw_max_diff,
            max_apis=aw_max_apis,
        )

    print(f"\nDone. Results in: {cfg.get('output_dir', 'eval/results/')}")

    from eval.model_adapter import reset as _reset_model
    _reset_model()

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
