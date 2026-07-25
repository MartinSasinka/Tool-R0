"""GRPO returns / advantages (matches nestful_mtgrpo_minimal trainer)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

_MINIMAL = Path(__file__).resolve().parents[2].parent / "nestful_mtgrpo_minimal"
if str(_MINIMAL) not in sys.path:
    sys.path.insert(0, str(_MINIMAL))

from grpo_train import _turn_returns  # noqa: E402
from group_stats import compute_group_stats  # noqa: E402

GAMMA = 1.0
LAMBDA = 1.0


def group_returns_and_advantages(
    turn_rewards: List[List[float]],
    episode_rewards: List[float],
) -> Tuple[List[List[float]], "GroupStats"]:
    ep_returns = [
        _turn_returns(seq, float(R), GAMMA, LAMBDA)
        for seq, R in zip(turn_rewards, episode_rewards)
    ]
    gstats = compute_group_stats(ep_returns, [float(x) for x in episode_rewards])
    return ep_returns, gstats


def rollout_scalar_advantage(gstats, rollout_index: int) -> float:
    advs = gstats.advantages[rollout_index]
    return float(advs[-1]) if advs else 0.0


def rollout_advantage_vector(gstats, rollout_index: int) -> List[float]:
    return list(gstats.advantages[rollout_index] or [])
