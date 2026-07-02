"""Unit tests for execution_aware_v2 (the new primary training reward).

Every edge case from request §4 is asserted against the shared case table in
``experiments/comparison/reward_v2_cases.py`` so the test and the audit CSV can
never disagree.
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "experiments"))
sys.path.insert(0, os.path.join(_REPO, "experiments", "comparison"))

import pytest  # noqa: E402

from nestful_core import rewards as R  # noqa: E402
import reward_v2_cases as cases_mod  # noqa: E402

CASES = cases_mod.build_cases()


@pytest.mark.parametrize("name,traj,task,exp", CASES, ids=[c[0] for c in CASES])
def test_execution_aware_v2_bounds(name, traj, task, exp):
    res = R.execution_aware_v2(traj, task)
    r = res.reward
    assert 0.0 <= r <= 1.0
    if "exec_min" in exp:
        assert r >= exp["exec_min"] - 1e-9, f"{name}: {r} < min {exp['exec_min']}"
    if "exec_max" in exp:
        assert r <= exp["exec_max"] + 1e-9, f"{name}: {r} > max {exp['exec_max']}"
    for flag, val in exp.get("flags", {}).items():
        assert res.diagnostics.get(flag) == val, (
            f"{name}: diag[{flag}]={res.diagnostics.get(flag)} != {val}")


def test_breakdown_keys_present():
    name, traj, task, _ = CASES[0]
    diag = R.execution_aware_v2(traj, task).diagnostics
    for key in ("reward", "tool_final_answer_pass", "executable_trajectory",
                "tool_use_completeness", "valid_references",
                "small_gold_trace_progress", "parse_error", "clipped",
                "no_tool_call", "too_few_calls", "invalid_reference",
                "executor_error", "cap_applied"):
        assert key in diag, f"missing breakdown key {key}"


def test_hard_caps_zero():
    for name in ("no_tool_call", "parse_error", "clipped_rollout",
                 "terminal_before_first_successful_tool"):
        n, traj, task, _ = next(c for c in CASES if c[0] == name)
        assert R.execution_aware_v2(traj, task).reward == 0.0, name


def test_floor_for_executable_and_correct():
    n, traj, task, _ = next(c for c in CASES if c[0] == "perfect_gold_trace")
    assert R.execution_aware_v2(traj, task).reward >= 0.85
