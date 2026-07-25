"""Offline Round-1 reward ablation audit (no training, no inference)."""

from __future__ import annotations

__all__ = ["ARMS", "DEFAULT_SEED"]

ARMS = [
    "A0_R0_CURRENT",
    "A1_OUTCOME_ONLY",
    "A2_R3_OUTCOME_FIRST",
    "A3_VERIFIABLE_PROCESS",
    "A4_GATED_VERIFIABLE",
]

DEFAULT_SEED = "20260724"
# Reward-threshold PROXY for the v3_2_dense `fully_correct` class. The
# fully_correct band is [0.90, 1.00] and no other class reaches above 0.80
# (lib/reward_v3_2_dense.BANDS), so `episode_reward >= 0.90` <=> fully_correct
# under execution_aware_v3_2_dense. The old value 0.92 misclassified
# fully_correct rollouts with in-band quality q < 0.2 as failures.
# NOTE: fully_correct is GOLD-TRACE match + final-answer pass — it is NOT the
# path-invariant terminal success (`tool_final_answer_pass`); never report this
# proxy as "terminal success" without that qualifier.
SYNTHETIC_SUCCESS_REWARD = 0.90
WIN_REWARD = 0.99  # internal train "win" proxy (reward >= 0.99); diagnostic only
