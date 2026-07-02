"""Predicate + partial_gold_trace_v2 + registry tests for the v2 reward module."""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "experiments"))
sys.path.insert(0, os.path.join(_REPO, "experiments", "comparison"))

import pytest  # noqa: E402

from nestful_core import rewards as R  # noqa: E402
import reward_v2_cases as cases_mod  # noqa: E402

CASES = {c[0]: c for c in cases_mod.build_cases()}


def _c(name):
    _, traj, task, exp = CASES[name]
    return traj, task, exp


# ---- predicates -----------------------------------------------------------

def test_predicate_parse_error():
    traj, task, _ = _c("parse_error")
    assert R.has_parse_error(traj) is True
    assert R.has_no_tool_call(traj) is True  # parse fail emitted no parsed call


def test_predicate_no_tool_call():
    traj, task, _ = _c("no_tool_call")
    assert R.has_no_tool_call(traj) is True
    assert R.num_successful_calls(traj) == 0


def test_predicate_terminal_before_first_tool():
    traj, task, _ = _c("terminal_before_first_successful_tool")
    assert R.terminal_before_first_successful_tool(traj) is True


def test_predicate_invalid_reference():
    traj, task, _ = _c("invalid_reference")
    assert R.has_invalid_reference(traj) is True
    assert R.has_executor_error(traj) is True


def test_predicate_executor_error():
    traj, task, _ = _c("executor_error")
    assert R.has_executor_error(traj) is True
    assert R.is_executable_trajectory(traj) is False


def test_predicate_executable_and_completeness():
    traj, task, _ = _c("perfect_gold_trace")
    assert R.is_executable_trajectory(traj) is True
    assert R.num_successful_calls(traj) == 2
    assert R.tool_use_completeness(traj, task) == 1.0
    assert R.tool_final_answer_pass(traj, task) is True


def test_predicate_extra_calls():
    traj, task, _ = _c("extra_calls_correct_answer")
    assert R.num_extra_calls(traj, task) == 1


# ---- partial_gold_trace_v2 caps ------------------------------------------

@pytest.mark.parametrize("name", ["parse_error", "clipped_rollout", "no_tool_call",
                                  "terminal_before_first_successful_tool"])
def test_partial_v2_hard_zero(name):
    traj, task, _ = _c(name)
    assert R.partial_gold_trace_v2(traj, task).reward == 0.0


def test_partial_v2_wrong_answer_capped():
    traj, task, _ = _c("valid_executable_trajectory_wrong_answer")
    r = R.partial_gold_trace_v2(traj, task).reward
    assert r <= 0.60 + 1e-9


def test_partial_v2_perfect_high():
    traj, task, _ = _c("perfect_gold_trace")
    assert R.partial_gold_trace_v2(traj, task).reward >= 0.95


# ---- registry + legacy reproducibility -----------------------------------

def test_registry_resolves_all():
    for pol in R.available_policies():
        assert callable(R.get_episode_reward(pol))
        assert callable(R.get_episode_reward_seq(pol))


def test_legacy_matches_frozen_modules():
    from reward import strict_gold_trace_reward
    from partial_reward import partial_gold_trace_reward
    traj, task, _ = _c("perfect_gold_trace")
    assert R.strict_gold_trace_legacy(traj, task).reward == \
        strict_gold_trace_reward(traj, task).reward
    assert R.partial_gold_trace_legacy(traj, task).reward == \
        partial_gold_trace_reward(traj, task).reward


def test_seq_contract_keys():
    traj, task, _ = _c("perfect_gold_trace")
    out = R.execution_aware_v2_seq(traj, task)
    assert set(out) == {"r_seq", "episode_reward", "diagnostics"}
    assert len(out["r_seq"]) == len(traj.turns)
