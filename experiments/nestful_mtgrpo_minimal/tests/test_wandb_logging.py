"""Unit tests for GRPO W&B logging helpers (no network)."""
from __future__ import annotations

import sys
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from grpo_train import (  # noqa: E402
    _compare_rollouts_slotwise,
    _rollout_win_rate,
    _wandb_log_epoch,
    _wandb_log_task,
    _WIN_REWARD_THRESHOLD,
)


def test_rollout_win_rate_threshold():
    assert _WIN_REWARD_THRESHOLD == 0.99
    assert _rollout_win_rate([1.0, 0.5, 0.99]) == 2 / 3
    assert _rollout_win_rate([0.5, 0.5]) == 0.0
    assert _rollout_win_rate([]) == 0.0


def test_compare_rollouts_slotwise():
    prev = [0.2, 0.5, 1.0, 0.24]
    curr = [0.3, 0.4, 1.0, 0.5]
    cmp = _compare_rollouts_slotwise(prev, curr)
    assert cmp["rollouts_compared"] == 4
    assert cmp["rollouts_improved"] == 2   # slots 0 and 3
    assert cmp["rollouts_regressed"] == 1  # slot 1
    assert cmp["rollouts_unchanged"] == 1  # slot 2
    assert cmp["rollout_improve_rate"] == 0.5
    assert cmp["rollout_slot_deltas"] == pytest.approx([0.1, -0.1, 0.0, 0.26])


def test_wandb_helpers_noop_without_run():
    rec = {
        "epoch": 1,
        "task_idx": 3,
        "task_id": "agentic_v4_stage2_000003",
        "mean_reward": 0.55,
        "episode_rewards": [0.5, 1.0, 0.24],
        "n_unique_episode_rewards": 3,
        "win_rate": 1 / 3,
        "max_reward": 1.0,
        "min_reward": 0.24,
        "reward_std_episode": 0.3,
        "n_unique_completion_hashes": 2,
        "predicted_num_calls": [2, 2, 1],
        "gold_num_calls": 2,
        "dead_group": False,
        "contributing_turns": 4,
        "loss": 0.1,
        "rollouts_compared": 8,
        "rollouts_improved": 3,
        "rollouts_regressed": 2,
        "rollouts_unchanged": 3,
        "rollout_improve_rate": 0.375,
        "rollout_slot_deltas": [0.1, -0.05, 0.0, 0.2, -0.1, 0.0, 0.05, -0.02],
    }
    _wandb_log_task(
        None, rec, stage=2, num_tasks=100, task_step=103,
        optimizer_step=5, task_prev_mean=0.4, task_best_mean=0.5,
    )
    _wandb_log_epoch(
        None,
        epoch=1,
        stage=2,
        tasks_seen=100,
        dead_group_rate=0.2,
        mean_unique_rewards=2.5,
        mean_win_rate=0.15,
        mean_reward=0.48,
        tasks_improved=40,
        tasks_regressed=10,
        mean_reward_delta=0.03,
        mean_rollouts_improved_per_task=2.5,
        mean_rollout_improve_rate=0.31,
        total_rollouts_improved=250,
        total_rollouts_regressed=120,
        fallback_used=False,
        optimizer_step=5,
    )
