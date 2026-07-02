#!/usr/bin/env python3
"""Thin driver for curriculum v3 — wires execution_aware_v2_1_motif into partial trainer."""
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


def _hook_select_train_reward():
    """Ensure motif policy is applied AFTER partial _select_train_reward would run."""
    orig = _partial._select_train_reward

    def _wrapped(config: dict) -> None:
        policy = str((config.get("reward", {}) or {}).get("train_policy", "")).lower()
        if policy in ("execution_aware_v2_1_motif", "motif", "v2_1_motif"):
            _patch_motif_reward()
            config.setdefault("reward", {})["train_policy"] = "execution_aware_v2_1_motif"
            print("[v3/run.py] training reward = execution_aware_v2_1_motif", flush=True)
            return
        return orig(config)

    _partial._select_train_reward = _wrapped


def main(argv=None):
    _hook_select_train_reward()
    return _partial.main()


if __name__ == "__main__":
    raise SystemExit(main())
