"""Correctness tests for the PARTIAL (graded) gold-trace reward.

Run from this folder:  python -m pytest tests/ -q
The conftest adds both this folder and the sibling experiment to sys.path.
"""
import math

import partial_reward as pr
from reward import strict_gold_trace_reward
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
    # Reset to documented defaults before each test (0.4/0.3/0.3, 0.7/0.3).
    pr.set_weights_from_config({"partial_reward": dict(pr._DEFAULT_WEIGHTS)})


# ---------------------------------------------------------------------------
#  Perfect episode == strict (both on the [0,1] scale, both 1.0)
# ---------------------------------------------------------------------------

def test_perfect_trace_scores_one():
    task = _task()
    gold_obs = [3, 30]
    traj = make_trajectory("t1", task["gold_calls"], [3, 30],
                           gold_num_turns=2, final_observation=30)
    rr = pr.partial_gold_trace_reward(traj, task, gold_obs)
    assert math.isclose(rr.reward, 1.0, abs_tol=1e-9)
    assert rr.diagnostics["strict_gold_trace_pass"] is True
    assert rr.diagnostics["turn_rewards"] == [1.0, 1.0]
    # On a perfect episode strict and partial agree.
    assert strict_gold_trace_reward(traj, task, gold_obs).reward == 1.0


# ---------------------------------------------------------------------------
#  Partial credit where strict gives 0
# ---------------------------------------------------------------------------

def test_first_turn_correct_second_wrong_gets_partial_credit():
    task = _task()
    gold_obs = [3, 30]
    calls = [
        task["gold_calls"][0],                                  # correct
        {"name": "subtract", "arguments": {"arg_0": 3, "arg_1": 1}},  # wrong name
    ]
    traj = make_trajectory("t1", calls, [3, 2],
                           gold_num_turns=2, final_observation=2)
    strict = strict_gold_trace_reward(traj, task, gold_obs).reward
    partial = pr.partial_gold_trace_reward(traj, task, gold_obs).reward
    assert strict == 0.0
    # turn0 fully correct (1.0), turn1 zero, no final -> 0.7 * (1.0/2) = 0.35
    assert math.isclose(partial, 0.35, abs_tol=1e-9)
    assert partial > strict


def test_correct_answer_wrong_path_gets_final_credit_not_full():
    task = _task()
    gold_obs = [3, 30]
    # Wrong first tool, but final answer is the gold answer (wrong path).
    calls = [
        {"name": "multiply", "arguments": {"arg_0": 1, "arg_1": 2}},
        {"name": "multiply", "arguments": {"arg_0": "$var_1.result$", "arg_1": 10}},
    ]
    traj = make_trajectory("t1", calls, [2, 30],
                           gold_num_turns=2, final_observation=30)
    rr = pr.partial_gold_trace_reward(traj, task, gold_obs)
    # turn0 wrong name -> 0; turn1 correct name+keys+exec(30==30) -> 1.0
    # trace_score = 0.5 -> 0.7*0.5 + 0.3*1(final) = 0.65
    assert math.isclose(rr.reward, 0.65, abs_tol=1e-9)
    assert rr.diagnostics["final_answer_pass"] is True
    assert rr.diagnostics["strict_gold_trace_pass"] is False
    assert rr.diagnostics["answer_correct_wrong_path"] is True
    assert strict_gold_trace_reward(traj, task, gold_obs).reward == 0.0


def test_name_only_match_scores_w_name():
    task = _task()
    gold_obs = [3, 30]
    # Correct names but wrong argument keys on BOTH turns, wrong final.
    calls = [
        {"name": "add", "arguments": {"x": 1, "y": 2}},
        {"name": "multiply", "arguments": {"x": 3, "y": 10}},
    ]
    traj = make_trajectory("t1", calls, [None, None],
                           gold_num_turns=2, final_observation=None)
    rr = pr.partial_gold_trace_reward(traj, task, gold_obs)
    # each turn: name only -> 0.4; trace_score = 0.4; no final -> 0.7*0.4 = 0.28
    assert math.isclose(rr.reward, 0.28, abs_tol=1e-9)
    assert rr.diagnostics["turn_rewards"] == [0.4, 0.4]


# ---------------------------------------------------------------------------
#  Monotonicity: more correct steps => higher reward
# ---------------------------------------------------------------------------

def test_monotonic_in_number_of_correct_steps():
    task = _task()
    gold_obs = [3, 30]
    none = make_trajectory(
        "t1",
        [{"name": "x", "arguments": {}}, {"name": "y", "arguments": {}}],
        [None, None], gold_num_turns=2, final_observation=None)
    one = make_trajectory(
        "t1",
        [task["gold_calls"][0], {"name": "y", "arguments": {}}],
        [3, None], gold_num_turns=2, final_observation=None)
    two = make_trajectory("t1", task["gold_calls"], [3, 30],
                          gold_num_turns=2, final_observation=30)
    r_none = pr.partial_gold_trace_reward(none, task, gold_obs).reward
    r_one = pr.partial_gold_trace_reward(one, task, gold_obs).reward
    r_two = pr.partial_gold_trace_reward(two, task, gold_obs).reward
    assert r_none < r_one < r_two


def test_missing_turns_score_zero_for_that_position():
    task = _task()
    gold_obs = [3, 30]
    # Only one (correct) call emitted; second gold position missing.
    traj = make_trajectory("t1", [task["gold_calls"][0]], [3],
                           gold_num_turns=2, final_observation=3)
    rr = pr.partial_gold_trace_reward(traj, task, gold_obs)
    assert rr.diagnostics["turn_rewards"] == [1.0, 0.0]
    assert math.isclose(rr.reward, 0.35, abs_tol=1e-9)  # 0.7 * 0.5


# ---------------------------------------------------------------------------
#  Clipped episodes are zeroed (same as strict)
# ---------------------------------------------------------------------------

def test_clipped_episode_zero():
    task = _task()
    gold_obs = [3, 30]
    traj = make_trajectory("t1", task["gold_calls"], [3, 30],
                           gold_num_turns=2, final_observation=30)
    traj.clipped_any = True
    rr = pr.partial_gold_trace_reward(traj, task, gold_obs)
    assert rr.reward == 0.0
    assert rr.diagnostics["turn_rewards"] == [0.0, 0.0]


# ---------------------------------------------------------------------------
#  Turn-level reward sequence (MT-GRPO contract)
# ---------------------------------------------------------------------------

def test_episode_turn_reward_seq_maps_to_generated_turns():
    task = _task()
    gold_obs = [3, 30]
    # 3 generated turns: 2 successful calls + 1 terminal (parsed_call=None).
    traj = make_trajectory(
        "t1",
        [task["gold_calls"][0], task["gold_calls"][1], None],
        [3, 30, None],
        gold_num_turns=2, final_observation=30)
    info = pr.episode_turn_reward_seq(traj, task, gold_obs)
    assert len(info["r_seq"]) == 3            # one entry per generated turn
    assert info["r_seq"][:2] == [1.0, 1.0]    # graded gold scores
    assert info["r_seq"][2] == 0.0            # terminal turn
    assert math.isclose(info["episode_reward"], 1.0, abs_tol=1e-9)


def test_parse_fail_turn_gets_zero_in_seq():
    task = _task()
    gold_obs = [3, 30]
    traj = make_trajectory(
        "t1",
        [task["gold_calls"][0], task["gold_calls"][1]],
        [3, 30],
        gold_num_turns=2, final_observation=30,
        fail_reasons=[None, "parse:no_tag"])
    info = pr.episode_turn_reward_seq(traj, task, gold_obs)
    # turn1 has a fail_reason -> excluded from successful mapping -> r=0
    assert info["r_seq"][0] == 1.0
    assert info["r_seq"][1] == 0.0


# ---------------------------------------------------------------------------
#  Config-driven weights
# ---------------------------------------------------------------------------

def test_weights_from_config_change_reward():
    task = _task()
    gold_obs = [3, 30]
    # Put all episode weight on the final answer; trace contributes nothing.
    pr.set_weights_from_config({"partial_reward": {"w_trace": 0.0, "w_final": 1.0}})
    calls = [
        {"name": "multiply", "arguments": {"arg_0": 1, "arg_1": 2}},  # wrong path
        {"name": "multiply", "arguments": {"arg_0": "$var_1.result$", "arg_1": 10}},
    ]
    traj = make_trajectory("t1", calls, [2, 30],
                           gold_num_turns=2, final_observation=30)
    rr = pr.partial_gold_trace_reward(traj, task, gold_obs)
    assert math.isclose(rr.reward, 1.0, abs_tol=1e-9)  # final-only weighting


def test_length_penalty_reduces_reward_for_extra_calls():
    task = _task()
    gold_obs = [3, 30]
    pr.set_weights_from_config({"partial_reward": {"length_penalty": 0.5}})
    # Perfect 2 gold calls + 1 extra call (4 turns, last terminal).
    calls = [
        task["gold_calls"][0], task["gold_calls"][1],
        {"name": "add", "arguments": {"arg_0": 1, "arg_1": 1}},
    ]
    traj = make_trajectory("t1", calls, [3, 30, 2],
                           gold_num_turns=2, final_observation=30)
    rr = pr.partial_gold_trace_reward(traj, task, gold_obs)
    # base 1.0 - 0.5 * (1 extra / 2 gold) = 1.0 - 0.25 = 0.75
    assert math.isclose(rr.reward, 0.75, abs_tol=1e-9)
