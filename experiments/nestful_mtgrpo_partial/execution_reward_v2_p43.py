"""Bare-importable shim for execution_aware_v2_p43 (Variant A).

Trainer + DP-pool workers import reward modules by bare name from ``sys.path``.
Real implementation: ``nestful_core.rewards.execution_aware_v2_p43``.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)

from nestful_core import rewards as _rewards  # noqa: E402

execution_aware_v2_p43 = _rewards.execution_aware_v2_p43
P43_REWARD_VARIANT = _rewards._P43_VARIANT


def set_weights_from_config(config: Dict[str, Any]):
    return _rewards.set_execution_p43_weights_from_config(config)


def get_weights() -> Dict[str, float]:
    return _rewards.get_execution_p43_weights()


def episode_turn_reward_seq(
    trajectory, task: Dict[str, Any], gold_observations: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    return _rewards.execution_aware_v2_p43_seq(trajectory, task, gold_observations)


# Fail-fast identity for dispatch verification (requested == resolved).
episode_turn_reward_seq.reward_policy = "execution_aware_v2_p43"  # type: ignore[attr-defined]
episode_turn_reward_seq.p43_reward_variant = P43_REWARD_VARIANT  # type: ignore[attr-defined]
