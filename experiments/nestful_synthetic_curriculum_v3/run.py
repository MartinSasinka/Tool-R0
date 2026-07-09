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
    # Use the module-level adapter so r_seq AND episode_reward come from the
    # SAME reward (the old inline _seq mixed execution_aware_v2 r_seq with the
    # motif episode reward). The adapter carries a .reward_policy attribute so
    # grpo_train._verify_reward_dispatch can assert identity.
    import grpo_train
    from lib import reward_motif

    grpo_train.episode_turn_reward_seq = reward_motif.episode_turn_reward_seq
    print("[v3/run.py] patched grpo_train.episode_turn_reward_seq = "
          "lib.reward_motif.episode_turn_reward_seq (execution_aware_v2_1_motif)",
          flush=True)


def _patch_v3_1_reward():
    import grpo_train
    from lib import reward_v3_1

    # TRAIN_STAGE env is read inside the adapter at call time.
    grpo_train.episode_turn_reward_seq = reward_v3_1.episode_turn_reward_seq
    print("[v3/run.py] patched grpo_train.episode_turn_reward_seq = "
          "lib.reward_v3_1.episode_turn_reward_seq (execution_aware_v3_1_stepwise)",
          flush=True)


def _patch_v3_2_reward():
    import grpo_train
    from lib import reward_v3_2_dense

    grpo_train.episode_turn_reward_seq = reward_v3_2_dense.episode_turn_reward_seq
    print("[v3/run.py] patched grpo_train.episode_turn_reward_seq = "
          "lib.reward_v3_2_dense.episode_turn_reward_seq (execution_aware_v3_2_dense)",
          flush=True)


def _hook_select_train_reward():
    orig = _partial._select_train_reward

    def _wrapped(config: dict) -> None:
        explicit = str(os.environ.get("REWARD_NAME", "")
                       or os.environ.get("REWARD_POLICY", "")).lower()
        default_policy = _default_reward_policy()
        policy = str((config.get("reward", {}) or {}).get("train_policy", default_policy)).lower()

        if explicit == "execution_aware_v2_1_motif" or policy in (
            "execution_aware_v2_1_motif", "motif", "v2_1_motif",
        ):
            _patch_motif_reward()
            config.setdefault("reward", {})["train_policy"] = "execution_aware_v2_1_motif"
            print("[v3/run.py] training reward = execution_aware_v2_1_motif", flush=True)
            return

        # v3.2 dense must be matched BEFORE the v3.1 branch: the v3.1 branch
        # also catches on CURRICULUM_VERSION alone and would shadow it.
        if explicit == "execution_aware_v3_2_dense" or policy in (
            "execution_aware_v3_2_dense", "v3_2_dense", "dense",
        ):
            _patch_v3_2_reward()
            config.setdefault("reward", {})["train_policy"] = "execution_aware_v3_2_dense"
            print("[v3/run.py] training reward = execution_aware_v3_2_dense", flush=True)
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
