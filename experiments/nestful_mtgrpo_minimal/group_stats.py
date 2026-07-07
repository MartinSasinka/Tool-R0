"""Torch-free group statistics for GRPO rollout groups.

Fixes the turn-position advantage artifact (audit Bug 3): the previous trainer
computed dead-group detection and advantage normalization over FLATTENED
turn-level returns. For multi-turn episodes the return G_t is mechanically
larger at earlier positions (G_t sums future turn rewards), so a group of
IDENTICAL completions still had nonzero flattened std. Such groups were
counted "alive" and the normalized advantage rewarded turn 1 / penalized
turn 2 — a pure position artifact that can teach premature stopping.

Corrected semantics implemented here:

  * Advantages are computed PER TURN POSITION across completions:
        adv[e][t] = (G[e][t] - mean_pos(t)) / (std_pos(t) + eps)
    where mean/std are taken over the episodes that HAVE a turn t.
    Positions with <2 episodes or zero between-completion std get adv = 0.
  * A group is DEAD iff NO position has nonzero between-completion std.
  * position_artifact_detected = alive under the old flattened logic but dead
    under the corrected between-completion logic.

Pure Python (no torch/numpy) so it is unit-testable anywhere and importable
by both the trainer and the DP rollout workers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

_EPS_STD = 1e-9


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


@dataclass
class GroupStats:
    """All group-level statistics the trainer needs for one rollout group."""
    # Advantages aligned to ep_returns: advantages[e][t] (masked episodes = []).
    advantages: List[List[float]] = field(default_factory=list)
    # Old flattened stats (kept for logging / artifact detection).
    flat_mean: float = 0.0
    flat_std: float = 0.0
    dead_flattened: bool = True
    # Corrected between-completion stats.
    position_means: List[float] = field(default_factory=list)
    position_stds: List[float] = field(default_factory=list)
    between_completion_std_max: float = 0.0
    dead_corrected: bool = True
    position_artifact_detected: bool = False
    # Episode-reward stats.
    episode_reward_std: float = 0.0
    n_alive_positions: int = 0


def compute_group_stats(
    ep_returns: List[List[float]],
    episode_rewards: List[float],
    included: Optional[List[bool]] = None,
) -> GroupStats:
    """Compute corrected group advantages + dead-group / artifact flags.

    Args:
        ep_returns:      per-episode per-turn returns G[e][t] (ragged allowed).
        episode_rewards: per-episode scalar rewards (same length as ep_returns).
        included:        per-episode mask (False = excluded, e.g. clipped).
    """
    n = len(ep_returns)
    if included is None:
        included = [True] * n

    # ── Old flattened logic (for comparison / logging only) ──────────────────
    flat = [g for inc, gs in zip(included, ep_returns) if inc for g in gs]
    flat_mean = _mean(flat)
    if flat:
        flat_std = (sum((g - flat_mean) ** 2 for g in flat) / len(flat)) ** 0.5
    else:
        flat_std = 0.0
    dead_flattened = flat_std <= _EPS_STD

    # ── Corrected per-position (between-completion) logic ────────────────────
    max_len = max((len(gs) for inc, gs in zip(included, ep_returns) if inc),
                  default=0)
    position_means: List[float] = []
    position_stds: List[float] = []
    for t in range(max_len):
        vals = [gs[t] for inc, gs in zip(included, ep_returns)
                if inc and t < len(gs)]
        position_means.append(_mean(vals))
        # <2 completions at this position -> no between-completion contrast.
        position_stds.append(_std(vals) if len(vals) >= 2 else 0.0)

    n_alive = sum(1 for s in position_stds if s > _EPS_STD)
    dead_corrected = n_alive == 0
    between_max = max(position_stds) if position_stds else 0.0
    position_artifact = (not dead_flattened) and dead_corrected

    advantages: List[List[float]] = []
    for inc, gs in zip(included, ep_returns):
        if not inc:
            advantages.append([0.0] * len(gs))
            continue
        adv_row: List[float] = []
        for t, g in enumerate(gs):
            std_t = position_stds[t] if t < len(position_stds) else 0.0
            if std_t > _EPS_STD:
                adv_row.append((g - position_means[t]) / (std_t + 1e-8))
            else:
                adv_row.append(0.0)
        advantages.append(adv_row)

    inc_rewards = [r for inc, r in zip(included, episode_rewards) if inc]
    return GroupStats(
        advantages=advantages,
        flat_mean=flat_mean,
        flat_std=flat_std,
        dead_flattened=dead_flattened,
        position_means=position_means,
        position_stds=position_stds,
        between_completion_std_max=between_max,
        dead_corrected=dead_corrected,
        position_artifact_detected=position_artifact,
        episode_reward_std=_std(inc_rewards),
        n_alive_positions=n_alive,
    )
