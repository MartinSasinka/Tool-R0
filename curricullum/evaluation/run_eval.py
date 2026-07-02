#!/usr/bin/env python3
"""
Run NESTFUL multi-turn evaluation (nestful_evaluation/run.py) for curriculum
baseline + LoRA checkpoints. Outputs match eval_viewer.html expectations:
  <profile>_multiturn_predictions.jsonl
  <profile>_multiturn_summary.json
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
NESTFUL_RUN = os.path.join(REPO_ROOT, "nestful_evaluation", "run.py")

DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_CKPT_ROOT = os.environ.get(
    "CURRICULUM_CKPT_ROOT",
    os.path.join("curricullum", "checkpoints", "qwen3_4b_lora_grpo"),
)
DEFAULT_OUTPUT_DIR = os.path.join("curricullum", "evaluation", "results")
DEFAULT_PREPARED_ROOT = os.path.join("curricullum", "evaluation", "prepared")
BASELINE_PROFILE = "curriculum_baseline"

_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def _log(msg: str) -> None:
    print(f"[curriculum_eval] {msg}", flush=True)


def _is_full_checkpoint(path: str) -> bool:
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return False
    return bool(glob.glob(os.path.join(path, "model*.safetensors")))


def _has_adapter(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def discover_checkpoints(ckpt_root: str) -> List[Tuple[str, str]]:
    """Return sorted (dirname, abs_path) pairs under ckpt_root."""
    if not os.path.isdir(ckpt_root):
        return []
    found: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(ckpt_root)):
        path = os.path.join(ckpt_root, name)
        if not os.path.isdir(path):
            continue
        if _has_adapter(path) or _is_full_checkpoint(path):
            found.append((name, os.path.abspath(path)))
    return found


def profile_slug_for_checkpoint(dirname: str) -> str:
    return f"curriculum_{dirname}"


def prepare_model_path(
    *,
    source_path: str,
    base_model: str,
    prepared_root: str,
    profile: str,
) -> str:
    """Return a vLLM-loadable model directory without modifying training ckpts."""
    if _is_full_checkpoint(source_path):
        return source_path

    if not _has_adapter(source_path):
        raise FileNotFoundError(
            f"No adapter or merged weights in checkpoint: {source_path}"
        )

    prepared_dir = os.path.join(prepared_root, profile)
    marker = os.path.join(prepared_dir, ".merge_done")
    if os.path.isfile(marker) and _is_full_checkpoint(prepared_dir):
        return prepared_dir

    os.makedirs(prepared_dir, exist_ok=True)
    for fname in _ADAPTER_FILES:
        src = os.path.join(source_path, fname)
        if os.path.isfile(src):
            dst = os.path.join(prepared_dir, fname)
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)

    for fname in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        src = os.path.join(source_path, fname)
        if os.path.isfile(src):
            dst = os.path.join(prepared_dir, fname)
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)

    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from grpo_processing import fix_checkpoint_for_vllm

    _log(f"merging LoRA for {profile} -> {prepared_dir}")
    fix_checkpoint_for_vllm(prepared_dir, base_model)

    if not _is_full_checkpoint(prepared_dir):
        raise RuntimeError(f"Merge did not produce full weights in {prepared_dir}")

    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
    return prepared_dir


def build_profiles(
    *,
    base_model: str,
    ckpt_root: str,
    only: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    profiles: List[Dict[str, str]] = [
        {
            "key": "baseline",
            "profile": BASELINE_PROFILE,
            "model_path": base_model,
            "source_path": base_model,
            "kind": "baseline",
        }
    ]
    for dirname, path in discover_checkpoints(ckpt_root):
        profiles.append(
            {
                "key": dirname,
                "profile": profile_slug_for_checkpoint(dirname),
                "model_path": path,
                "source_path": path,
                "kind": "checkpoint",
            }
        )

    if only:
        wanted = {x.strip() for x in only if x.strip()}
        profiles = [p for p in profiles if p["key"] in wanted]
    return profiles


def output_paths(output_dir: str, profile: str) -> Tuple[str, str]:
    pred = os.path.join(output_dir, f"{profile}_multiturn_predictions.jsonl")
    summary = os.path.join(output_dir, f"{profile}_multiturn_summary.json")
    return pred, summary


def run_nestful_eval(
    *,
    model: str,
    profile: str,
    output_dir: str,
    nestful_args: List[str],
) -> None:
    pred_path, summary_path = output_paths(output_dir, profile)
    cmd = [
        sys.executable,
        NESTFUL_RUN,
        "--model",
        model,
        "--model-profile",
        profile,
        "--output-dir",
        output_dir,
        *nestful_args,
    ]
    _log(f"running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    if not os.path.isfile(pred_path) or not os.path.isfile(summary_path):
        raise RuntimeError(
            f"Expected outputs missing after eval:\n  {pred_path}\n  {summary_path}"
        )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NESTFUL multi-turn eval for curriculum baseline + checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--ckpt-root", default=DEFAULT_CKPT_ROOT)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--prepared-root", default=DEFAULT_PREPARED_ROOT)
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run subset only, e.g. baseline stage1_1call stage3_3call",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip profiles whose summary JSON already exists.",
    )
    p.add_argument("--num-rollouts", type=int, default=None)
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--nestful-repo-dir", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--ibm-call-timeout", type=float, default=None)
    p.add_argument("--advance-log-every", type=int, default=None)
    return p


def _optional_flag(name: str, value: Optional[object]) -> List[str]:
    if value is None:
        return []
    return [name, str(value)]


def main() -> int:
    args = build_argparser().parse_args()

    ckpt_root = os.path.join(REPO_ROOT, args.ckpt_root)
    output_dir = os.path.join(REPO_ROOT, args.output_dir)
    prepared_root = os.path.join(REPO_ROOT, args.prepared_root)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(prepared_root, exist_ok=True)

    if not os.path.isfile(NESTFUL_RUN):
        _log(f"ERROR: nestful driver missing: {NESTFUL_RUN}")
        return 2

    profiles = build_profiles(
        base_model=args.base_model,
        ckpt_root=ckpt_root,
        only=args.only,
    )
    if not profiles:
        _log("ERROR: no profiles selected (check --only / --ckpt-root)")
        return 2

    nestful_args: List[str] = []
    nestful_args += _optional_flag("--num-rollouts", args.num_rollouts)
    nestful_args += _optional_flag("--max-tasks", args.max_tasks)
    nestful_args += _optional_flag("--max-steps", args.max_steps)
    nestful_args += _optional_flag("--temperature", args.temperature)
    nestful_args += _optional_flag("--top-p", args.top_p)
    nestful_args += _optional_flag("--max-new-tokens", args.max_new_tokens)
    nestful_args += _optional_flag("--max-model-len", args.max_model_len)
    nestful_args += _optional_flag("--tensor-parallel-size", args.tensor_parallel_size)
    nestful_args += _optional_flag(
        "--gpu-memory-utilization", args.gpu_memory_utilization
    )
    nestful_args += _optional_flag("--seed", args.seed)
    nestful_args += _optional_flag("--nestful-repo-dir", args.nestful_repo_dir)
    nestful_args += _optional_flag("--cache-dir", args.cache_dir)
    nestful_args += _optional_flag("--ibm-call-timeout", args.ibm_call_timeout)
    nestful_args += _optional_flag("--advance-log-every", args.advance_log_every)

    _log(f"profiles={len(profiles)} output_dir={output_dir}")
    failures: List[str] = []

    for item in profiles:
        profile = item["profile"]
        _, summary_path = output_paths(output_dir, profile)
        if args.skip_existing and os.path.isfile(summary_path):
            _log(f"skip existing: {profile}")
            continue

        _log(f"=== {item['key']} ({profile}) ===")
        try:
            if item["kind"] == "baseline":
                model_path = args.base_model
            else:
                model_path = prepare_model_path(
                    source_path=item["source_path"],
                    base_model=args.base_model,
                    prepared_root=prepared_root,
                    profile=profile,
                )
            run_nestful_eval(
                model=model_path,
                profile=profile,
                output_dir=output_dir,
                nestful_args=nestful_args,
            )
        except subprocess.CalledProcessError as exc:
            _log(f"FAILED {profile}: exit code {exc.returncode}")
            failures.append(profile)
        except Exception as exc:
            _log(f"FAILED {profile}: {exc}")
            failures.append(profile)

    if failures:
        _log(f"finished with failures: {', '.join(failures)}")
        return 1

    _log("all profiles completed")
    _log(f"Open eval_viewer.html and load JSONL from: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
