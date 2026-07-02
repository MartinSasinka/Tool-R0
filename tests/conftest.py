"""Shared fixtures / path setup for the v2 pipeline tests.

Adds ``experiments/`` to ``sys.path`` so ``import nestful_core...`` works, and
exposes helpers to build synthetic trajectories for the reward unit tests.
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXPERIMENTS = os.path.join(_REPO, "experiments")
for _p in (_EXPERIMENTS,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from nestful_core.rollout import Trajectory, Turn  # noqa: E402


GOLD = [
    {"name": "add", "arguments": {"arg_0": 1, "arg_1": 2}, "label": "$var1"},
    {"name": "multiply", "arguments": {"arg_0": "$var1.result$", "arg_1": 3}, "label": "$var2"},
]
GOLD_ANSWER = 9


def make_task(gold=None, ans=GOLD_ANSWER):
    gold = GOLD if gold is None else gold
    return {
        "task_id": "t",
        "gold_calls": gold,
        "gold_answer": ans,
        "num_calls": len(gold),
        "tools": [],
    }


def make_traj(turns_spec, final_obs, stop_reason, *, clipped=False):
    """turns_spec: list of (parsed_call|None, fail_reason|None, observation|None).

    A turn with all three None is treated as a terminal ``[]`` turn.
    """
    tr = Trajectory(task_id="t", stage=2, gold_num_turns=2, executor_mode="full")
    tr.clipped_any = clipped
    tr.stop_reason = stop_reason
    tr.final_observation = final_obs
    for i, (call, fail, obs) in enumerate(turns_spec):
        t = Turn(turn_idx=i, model_text="")
        t.parsed_call = call
        t.fail_reason = fail
        t.observation = obs
        if call is None and fail is None and obs is None:
            t.is_terminal = True
        if fail == "clipped_completion":
            t.clipped_completion = True
        tr.turns.append(t)
    return tr


@pytest.fixture
def task():
    return make_task()
