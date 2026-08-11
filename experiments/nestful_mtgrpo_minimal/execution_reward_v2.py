"""Bare-importable shim for the execution_aware_v2 reward.

The trainer (parent process) and the DP-pool rollout workers (spawned processes)
import reward modules by BARE name from ``sys.path``. The real implementation
lives in ``nestful_core.rewards``; this shim makes ``execution_aware_v2``
resolvable by bare ``import execution_reward_v2`` while keeping a single source.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

# Put experiments/ on sys.path so ``nestful_core`` is importable as a package.
_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)

from nestful_core import rewards as _rewards  # noqa: E402

execution_aware_v2 = _rewards.execution_aware_v2


def set_weights_from_config(config: Dict[str, Any]):
    return _rewards.set_execution_v2_weights_from_config(config)


def get_weights() -> Dict[str, float]:
    return _rewards.get_execution_v2_weights()


def episode_turn_reward_seq(
    trajectory, task: Dict[str, Any], gold_observations: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    return _rewards.execution_aware_v2_seq(trajectory, task, gold_observations)
