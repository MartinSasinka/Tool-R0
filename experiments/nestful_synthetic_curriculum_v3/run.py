#!/usr/bin/env python3
"""Thin driver for curriculum v3/v3.1 — wires stage-aware reward into partial trainer."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARTIAL = os.path.join(HERE, "..", "nestful_mtgrpo_partial")
MINIMAL = os.path.join(HERE, "..", "nestful_mtgrpo_minimal")
sys.path.insert(0, PARTIAL)
sys.path.insert(0, MINIMAL)
sys.path.insert(0, HERE)

import importlib.util

_partial_run = os.path.join(PARTIAL, "run.py")
_spec = importlib.util.spec_from_file_location("partial_run", _partial_run)
_partial = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_partial)


def _curriculum_version() -> str:
    return os.environ.get("CURRICULUM_VERSION", "v3").lower().replace("-", "_")


def _default_reward_policy() -> str:
    if _curriculum_version() in ("v3_1", "v31"):
        return "execution_aware_v3_1_stepwise"
    return "execution_aware_v2_1_motif"


def _patch_motif_reward():
    import grpo_train
    from lib.reward_motif import execution_aware_v2_1_motif

    def _seq(traj, task, gold_obs=None):
        from nestful_core.rewards import execution_aware_v2_seq  # type: ignore
        res = execution_aware_v2_1_motif(traj, task, gold_obs)
        base = execution_aware_v2_seq(traj, task, gold_obs)
        base["episode_reward"] = res.reward
        base["reward"] = res.reward
        base.update(res.diagnostics)
        return base

    grpo_train.episode_turn_reward_seq = _seq
    print("[v3/run.py] patched grpo_train.episode_turn_reward_seq = execution_aware_v2_1_motif", flush=True)


def _patch_v3_1_reward():
    import grpo_train
    from lib.reward_v3_1 import execution_aware_v3_1_stepwise

    train_stage = int(os.environ.get("TRAIN_STAGE", "0") or "0") or None

    def _seq(traj, task, gold_obs=None):
        from nestful_core.rewards import execution_aware_v2_seq  # type: ignore
        stage_hint = train_stage or task.get("train_stage")
        res = execution_aware_v3_1_stepwise(traj, task, gold_obs, train_stage=stage_hint)
        base = execution_aware_v2_seq(traj, task, gold_obs)
        base["episode_reward"] = res.reward
        base["reward"] = res.reward
        base.update(res.diagnostics)
        return base

    grpo_train.episode_turn_reward_seq = _seq
    print("[v3/run.py] patched grpo_train.episode_turn_reward_seq = execution_aware_v3_1_stepwise", flush=True)


def _hook_select_train_reward():
    orig = _partial._select_train_reward

    def _wrapped(config: dict) -> None:
        explicit = str(os.environ.get("REWARD_NAME", "")).lower()
        default_policy = _default_reward_policy()
        policy = str((config.get("reward", {}) or {}).get("train_policy", default_policy)).lower()

        if explicit == "execution_aware_v2_1_motif" or policy in (
            "execution_aware_v2_1_motif", "motif", "v2_1_motif",
        ):
            _patch_motif_reward()
            config.setdefault("reward", {})["train_policy"] = "execution_aware_v2_1_motif"
            print("[v3/run.py] training reward = execution_aware_v2_1_motif", flush=True)
            return

        if explicit == "execution_aware_v3_1_stepwise" or policy in (
            "execution_aware_v3_1_stepwise", "v3_1_stepwise", "stepwise",
        ) or _curriculum_version() in ("v3_1", "v31"):
            _patch_v3_1_reward()
            config.setdefault("reward", {})["train_policy"] = "execution_aware_v3_1_stepwise"
            print("[v3/run.py] training reward = execution_aware_v3_1_stepwise", flush=True)
            return

        return orig(config)

    _partial._select_train_reward = _wrapped


def main(argv=None):
    _hook_select_train_reward()
    return _partial.main()


if __name__ == "__main__":
    raise SystemExit(main())
