"""Correctness tests for the NEW execution-aware reward (execution_reward.py).

Run from this folder:  python -m pytest tests/test_execution_reward.py -q
The conftest adds both this folder and the sibling experiment to sys.path.
"""
import math

import execution_reward as er
from _helpers import make_trajectory


def _task():
    return {
        "task_id": "t1",
        "question": "q",
        "tools": [
            {"name": "add", "parameters": {"properties": {"arg_0": {}, "arg_1": {}}}},
            {"name": "multiply", "parameters": {"properties": {"arg_0": {}, "arg_1": {}}}},
        ],
        "gold_calls": [
            {"name": "add", "arguments": {"arg_0": 1, "arg_1": 2}, "label": "$var_1"},
            {"name": "multiply", "arguments": {"arg_0": "$var_1.result$", "arg_1": 10}, "label": "$var_2"},
        ],
        "gold_answer": 30,
        "num_calls": 2,
    }


def setup_function(_):
    er.set_weights_from_config({"execution_reward": dict(er._DEFAULT_WEIGHTS)})


# ---------------------------------------------------------------------------
#  Perfect execution -> 1.0
# ---------------------------------------------------------------------------

def test_perfect_episode_scores_one():
    task = _task()
    gold_obs = [3, 30]
    traj = make_trajectory("t1", task["gold_calls"], [3, 30],
                           gold_num_turns=2, final_observation=30)
    rr = er.execution_aware_reward(traj, task, gold_obs)
    assert math.isclose(rr.reward, 1.0, abs_tol=1e-9)
    d = rr.diagnostics
    assert d["tool_final_answer_pass"] == 1.0
    assert d["executable_trajectory"] == 1.0
    assert d["tool_use_completeness"] == 1.0
    assert d["valid_references"] == 1.0
    assert d["cap_applied"] is None


# ---------------------------------------------------------------------------
#  Hard caps -> R = 0
# ---------------------------------------------------------------------------

def test_no_tool_call_zeroed():
    task = _task()
    # Single terminal turn, no parsed call at all.
    traj = make_trajectory("t1", [None], [None],
                           gold_num_turns=2, final_observation=None,
                           stop_reason="terminal")
    rr = er.execution_aware_reward(traj, task, [3, 30])
    assert rr.reward == 0.0
    assert rr.diagnostics["cap_applied"] == "no_tool_call"


def test_parse_fail_zeroed():
    task = _task()
    traj = make_trajectory("t1", task["gold_calls"], [3, 30],
                           gold_num_turns=2, final_observation=30,
                           fail_reasons=[None, "parse:no_tag"])
    rr = er.execution_aware_reward(traj, task, [3, 30])
    assert rr.reward == 0.0
    assert rr.diagnostics["cap_applied"] == "parse_error"


def test_clipped_zeroed():
    task = _task()
    traj = make_trajectory("t1", task["gold_calls"], [3, 30],
                           gold_num_turns=2, final_observation=30)
    traj.clipped_any = True
    rr = er.execution_aware_reward(traj, task, [3, 30])
    assert rr.reward == 0.0
    assert rr.diagnostics["cap_applied"] == "clipped"


def test_terminal_before_first_successful_tool_zeroed():
    task = _task()
    # One emitted call that exec-failed, then terminated.
    traj = make_trajectory("t1", [task["gold_calls"][0]], [None],
                           gold_num_turns=2, final_observation=None,
                           stop_reason="terminal", fail_reasons=["exec:boom"])
    rr = er.execution_aware_reward(traj, task, [3, 30])
    assert rr.reward == 0.0
    assert rr.diagnostics["cap_applied"] == "terminal_before_first_tool"


# ---------------------------------------------------------------------------
#  Soft caps
# ---------------------------------------------------------------------------

def test_not_executable_bounded():
    task = _task()
    # Emitted a (correct-looking) call that errored; stopped on executor_error
    # (NOT terminal), so it isn't caught by the terminal cap.
    traj = make_trajectory("t1", [task["gold_calls"][0]], [None],
                           gold_num_turns=2, final_observation=None,
                           stop_reason="executor_error", fail_reasons=["exec:boom"])
    rr = er.execution_aware_reward(traj, task, [3, 30])
    assert rr.diagnostics["executable_trajectory"] == 0.0
    assert rr.reward <= er._DEFAULT_WEIGHTS["cap_not_executable"] + 1e-9
    assert rr.diagnostics["cap_applied"] == "not_executable"


def test_too_few_calls_wrong_answer_bounded():
    task = _task()
    # Only the first (correct) call, no final answer -> incomplete + wrong.
    traj = make_trajectory("t1", [task["gold_calls"][0]], [3],
                           gold_num_turns=2, final_observation=3,
                           stop_reason="terminal")
    rr = er.execution_aware_reward(traj, task, [3, 30])
    assert rr.diagnostics["too_few_calls"] is True
    assert rr.reward <= er._DEFAULT_WEIGHTS["cap_incomplete_wrong"] + 1e-9


# ---------------------------------------------------------------------------
#  References
# ---------------------------------------------------------------------------

def test_invalid_reference_lowers_component():
    task = _task()
    calls = [
        task["gold_calls"][0],
        {"name": "multiply", "arguments": {"arg_0": "$var_9.result$", "arg_1": 10}, "label": "$var_2"},
    ]
    traj = make_trajectory("t1", calls, [3, 30],
                           gold_num_turns=2, final_observation=30)
    rr = er.execution_aware_reward(traj, task, [3, 30])
    # One reference, pointing to a non-existent label -> 0.0 validity.
    assert rr.diagnostics["valid_references_present"] is True
    assert rr.diagnostics["valid_references"] == 0.0


def test_no_references_gets_full_credit():
    task = {
        "task_id": "t2", "question": "q",
        "tools": [{"name": "add", "parameters": {"properties": {"arg_0": {}, "arg_1": {}}}}],
        "gold_calls": [{"name": "add", "arguments": {"arg_0": 1, "arg_1": 2}, "label": "$var_1"}],
        "gold_answer": 3, "num_calls": 1,
    }
    traj = make_trajectory("t2", task["gold_calls"], [3],
                           gold_num_turns=1, final_observation=3)
    rr = er.execution_aware_reward(traj, task, [3])
    assert rr.diagnostics["valid_references_present"] is False
    assert rr.diagnostics["valid_references"] == 1.0
    assert math.isclose(rr.reward, 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
#  Execution success beats gold-trace fidelity (the whole point)
# ---------------------------------------------------------------------------

def test_correct_answer_executable_beats_trace_only():
    task = _task()
    gold_obs = [3, 30]
    # Wrong tool names but executes and reaches gold answer (right answer, wrong path).
    right_answer = make_trajectory(
        "t1",
        [{"name": "multiply", "arguments": {"arg_0": 5, "arg_1": 6}, "label": "$var_1"},
         {"name": "multiply", "arguments": {"arg_0": "$var_1.result$", "arg_1": 1}, "label": "$var_2"}],
        [30, 30], gold_num_turns=2, final_observation=30)
    # Perfect gold trace but no final answer reached.
    trace_only = make_trajectory(
        "t1", task["gold_calls"], [3, None],
        gold_num_turns=2, final_observation=None, stop_reason="terminal")
    r_answer = er.execution_aware_reward(right_answer, task, gold_obs).reward
    r_trace = er.execution_aware_reward(trace_only, task, gold_obs).reward
    # The execution-aware reward should value reaching the answer.
    assert r_answer > r_trace


# ---------------------------------------------------------------------------
#  MT-GRPO turn-reward sequence contract
# ---------------------------------------------------------------------------

def test_episode_turn_reward_seq_maps_to_generated_turns():
    task = _task()
    gold_obs = [3, 30]
    traj = make_trajectory(
        "t1",
        [task["gold_calls"][0], task["gold_calls"][1], None],
        [3, 30, None],
        gold_num_turns=2, final_observation=30)
    info = er.episode_turn_reward_seq(traj, task, gold_obs)
    assert len(info["r_seq"]) == 3
    assert info["r_seq"][2] == 0.0                       # terminal turn
    assert all(0.0 <= r <= 1.0 for r in info["r_seq"])
    assert math.isclose(info["episode_reward"], 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
#  Config-driven weights
# ---------------------------------------------------------------------------

def test_weights_from_config_change_reward():
    task = _task()
    gold_obs = [3, 30]
    # Put all component weight on the final answer.
    er.set_weights_from_config({"execution_reward": {
        "w_final": 1.0, "w_executable": 0.0, "w_completeness": 0.0,
        "w_references": 0.0, "w_gold_trace": 0.0,
    }})
    # Right answer via executed calls, wrong path.
    traj = make_trajectory(
        "t1",
        [{"name": "multiply", "arguments": {"arg_0": 5, "arg_1": 6}, "label": "$var_1"},
         {"name": "multiply", "arguments": {"arg_0": "$var_1.result$", "arg_1": 1}, "label": "$var_2"}],
        [30, 30], gold_num_turns=2, final_observation=30)
    rr = er.execution_aware_reward(traj, task, gold_obs)
    assert math.isclose(rr.reward, 1.0, abs_tol=1e-9)
