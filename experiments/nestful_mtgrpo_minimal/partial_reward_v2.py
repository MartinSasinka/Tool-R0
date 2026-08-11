"""Bare-importable shim for the partial_gold_trace_v2 reward (see execution_reward_v2)."""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

_EXPERIMENTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)

from nestful_core import rewards as _rewards  # noqa: E402

partial_gold_trace_v2 = _rewards.partial_gold_trace_v2


def set_weights_from_config(config: Dict[str, Any]):
    return _rewards.set_partial_v2_weights_from_config(config)


def episode_turn_reward_seq(
    trajectory, task: Dict[str, Any], gold_observations: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    return _rewards.partial_gold_trace_v2_seq(trajectory, task, gold_observations)
