#!/usr/bin/env python3
"""NESTFUL MT-GRPO PARTIAL — entry point.

Thin driver that REUSES the validated infrastructure of the sibling
``../nestful_mtgrpo_minimal`` experiment (data loading, model loading, rollout,
executor, GRPO trainer, official scorer, W&B logging) and changes exactly ONE
thing: the *training* reward is the PARTIAL (graded) gold-trace reward defined
in ``partial_reward.py`` instead of the strict binary one.

Design (zero edits to the strict artifact)
-------------------------------------------
* Everything heavy is imported from the sibling folder (added to sys.path).
* For ``--mode train`` we monkeypatch ``grpo_train.episode_turn_reward_seq`` with
  the partial version BEFORE calling the sibling trainer. The trainer resolves
  that name from its own module globals at call time, so the swap is clean and
  fully localized — the strict experiment on disk is untouched.
* EVALUATION modes (smoke / rollout_eval / final_eval) are delegated to the
  sibling UNCHANGED. They keep computing the strict ``strict_gold_trace_pass``
  and the official NESTFUL metrics, so partial-reward checkpoints are directly
  comparable to strict ones (this is what makes RQ2 answerable).

Path resolution
---------------
INPUT data paths resolve against the sibling (dataset lives there); OUTPUT paths
resolve against THIS folder. See ``_setup_config``.

Usage:
    python run.py --mode smoke        --config config.yaml
    python run.py --mode train        --config config.yaml [--checkpoint PATH]
    python run.py --mode rollout_eval --config config.yaml [--checkpoint PATH]
    python run.py --mode final_eval   --config config.yaml --checkpoint PATH
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIBLING = os.path.join(os.path.dirname(_HERE), "nestful_mtgrpo_minimal")

if not os.path.isdir(_SIBLING):
    raise SystemExit(
        f"[partial] sibling experiment not found: {_SIBLING}\n"
        "This experiment reuses ../nestful_mtgrpo_minimal — keep both folders "
        "side by side under experiments/."
    )

# Sibling first so its modules (grpo_train, reward, rollout, ...) import; then
# this folder so partial_reward imports. (Both folders contain a run.py, so the
# sibling run.py is loaded by explicit path below to avoid the name clash.)
sys.path.insert(0, _SIBLING)
sys.path.insert(0, _HERE)


def _import_by_path(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Sibling run.py under a unique name (it shares the bare name 'run' with this file).
base = _import_by_path(os.path.join(_SIBLING, "run.py"), "mtgrpo_base_run")

import grpo_train             # noqa: E402  sibling trainer (reward injected below)
import partial_reward         # noqa: E402  this folder
import execution_reward       # noqa: E402  this folder (NEW execution-aware reward)
import execution_reward_v2    # noqa: E402  this folder (v2 primary training reward)
import partial_reward_v2      # noqa: E402  this folder (v2 fixed partial baseline)

# Make the shared core importable as a package for v2 runs.
_EXPERIMENTS_ROOT = os.path.dirname(_HERE)
if _EXPERIMENTS_ROOT not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_ROOT)


def _resolve_input(path: str) -> str:
    """Resolve an INPUT data path against the sibling artifact root."""
    if not path or os.path.isabs(path):
        return path
    p = path.replace("\\", "/")
    for legacy in ("experiments/nestful_mtgrpo_partial/",
                   "experiments/nestful_mtgrpo_minimal/"):
        if p.startswith(legacy):
            p = p[len(legacy):]
    return os.path.normpath(os.path.join(_SIBLING, p))


def _resolve_output(path: str) -> str:
    """Resolve an OUTPUT path against THIS folder."""
    if not path or os.path.isabs(path):
        return path
    p = path.replace("\\", "/")
    legacy = "experiments/nestful_mtgrpo_partial/"
    if p.startswith(legacy):
        p = p[len(legacy):]
    return os.path.normpath(os.path.join(_HERE, p))


def _setup_config(config: dict) -> None:
    """Resolve config paths: inputs against sibling, outputs against this folder."""
    paths = config.get("paths", {})
    for key in list(paths.keys()):
        val = paths.get(key)
        if isinstance(val, str) and val:
            paths[key] = _resolve_input(val)

    mcfg = config.get("model", {})
    # lora_adapter is an INPUT checkpoint; output_adapter_dir is an OUTPUT.
    if isinstance(mcfg.get("lora_adapter"), str) and mcfg["lora_adapter"]:
        mcfg["lora_adapter"] = _resolve_input(mcfg["lora_adapter"])
    if isinstance(mcfg.get("output_adapter_dir"), str) and mcfg["output_adapter_dir"]:
        mcfg["output_adapter_dir"] = _resolve_output(mcfg["output_adapter_dir"])

    exp = config.get("experiment", {})
    if isinstance(exp.get("output_dir"), str):
        exp["output_dir"] = _resolve_output(exp["output_dir"])


def _enable_partial_training_reward(config: dict) -> None:
    """Configure weights and swap the trainer's reward to the partial one.

    Affects TRAINING ONLY. Eval modes keep using the strict reward via the
    sibling, so metrics stay comparable across experiments.
    """
    partial_reward.set_weights_from_config(config)
    grpo_train.episode_turn_reward_seq = partial_reward.episode_turn_reward_seq
    # Data-parallel rollout workers run in SEPARATE processes and never see the
    # monkeypatch above; they pick the reward from config['reward']['train_policy'].
    # Set it here so both code paths (in-process and pooled) use partial credit.
    config.setdefault("reward", {})["train_policy"] = "partial_gold_trace"
    print("[partial] training reward = partial_gold_trace "
          "(eval metrics remain strict + official)", flush=True)


def _enable_execution_aware_reward(config: dict) -> None:
    """Configure weights and swap the trainer's reward to the execution-aware one.

    Affects TRAINING ONLY. Eval modes keep using the strict reward via the
    sibling, so metrics stay comparable across all three experiments
    (strict / partial / execution-aware).
    """
    execution_reward.set_weights_from_config(config)
    grpo_train.episode_turn_reward_seq = execution_reward.episode_turn_reward_seq
    # Pooled workers (separate processes) resolve the reward from this key.
    config.setdefault("reward", {})["train_policy"] = "execution_aware"
    print("[partial] training reward = execution_aware "
          "(eval metrics remain strict + official)", flush=True)


def _enable_execution_aware_v2_reward(config: dict) -> None:
    """Swap the trainer reward to execution_aware_v2 (v2 primary training reward)."""
    execution_reward_v2.set_weights_from_config(config)
    grpo_train.episode_turn_reward_seq = execution_reward_v2.episode_turn_reward_seq
    config.setdefault("reward", {})["train_policy"] = "execution_aware_v2"
    print("[partial] training reward = execution_aware_v2 "
          "(eval metrics remain strict + official)", flush=True)


def _enable_partial_v2_reward(config: dict) -> None:
    """Swap the trainer reward to partial_gold_trace_v2 (fixed graded baseline)."""
    partial_reward_v2.set_weights_from_config(config)
    grpo_train.episode_turn_reward_seq = partial_reward_v2.episode_turn_reward_seq
    config.setdefault("reward", {})["train_policy"] = "partial_gold_trace_v2"
    print("[partial] training reward = partial_gold_trace_v2 "
          "(eval metrics remain strict + official)", flush=True)


def _select_train_reward(config: dict) -> None:
    """Pick the TRAINING reward based on config['reward']['train_policy'].

    Default stays ``partial_gold_trace`` (unchanged behavior). The v2 pipeline
    sets ``reward.train_policy: execution_aware_v2`` (primary) or
    ``partial_gold_trace_v2`` (fixed baseline). Legacy policies are unchanged.
    """
    policy = str((config.get("reward", {}) or {}).get("train_policy", "")).lower()
    if policy in ("execution_aware_v2", "execution_v2"):
        _enable_execution_aware_v2_reward(config)
    elif policy in ("partial_gold_trace_v2", "partial_v2"):
        _enable_partial_v2_reward(config)
    elif policy in ("execution_aware", "execution"):
        _enable_execution_aware_reward(config)
    else:
        _enable_partial_training_reward(config)


def main() -> int:
    ap = argparse.ArgumentParser(description="NESTFUL MT-GRPO Partial")
    ap.add_argument("--mode", required=True,
                    choices=["smoke", "rollout_eval", "train", "final_eval", "val_eval"])
    ap.add_argument("--config", default=os.path.join(_HERE, "config.yaml"))
    ap.add_argument("--checkpoint", default=None,
                    help="LoRA adapter / model path (eval modes, or train resume)")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                    help="Override a config value using dot notation.")
    args = ap.parse_args()

    config = base.load_config(args.config)
    base._apply_overrides(config, args.override)
    _setup_config(config)
    # Same observation-truncation guard as the sibling (shared prompt module).
    from prompt import set_observation_limits
    set_observation_limits(config)
    base.print_versions()
    base._ensure_local_data(config)

    # Resolve a relative --checkpoint against the sibling data root for eval/resume.
    checkpoint = args.checkpoint
    if checkpoint and not os.path.isabs(checkpoint) and not os.path.exists(checkpoint):
        cand = _resolve_output(checkpoint)
        checkpoint = cand if os.path.exists(cand) else _resolve_input(checkpoint)

    if args.mode == "smoke":
        return base.mode_smoke(config)
    if args.mode == "rollout_eval":
        return base.mode_rollout_eval(config, checkpoint)
    if args.mode == "final_eval":
        return base.mode_final_eval(config, checkpoint)
    if args.mode == "val_eval":
        # Validation Win stays strict + official (same as the other eval modes),
        # so partial-reward checkpoints remain comparable to strict ones.
        return base.mode_val_eval(config, checkpoint)
    if args.mode == "train":
        _select_train_reward(config)
        return base.mode_train(config, checkpoint)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
