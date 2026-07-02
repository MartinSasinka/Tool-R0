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


def test_correct_gold_trace_reward_1():
    task = _task()
    gold_obs = [3, 30]
    traj = make_trajectory(
        "t1",
        calls=task["gold_calls"],
        observations=[3, 30],
        gold_num_turns=2,
        final_observation=30,
    )
    rr = strict_gold_trace_reward(traj, task, gold_obs)
    assert rr.reward == 1.0
    assert rr.diagnostics["final_answer_pass"] is True


def test_wrong_tool_name_reward_0():
    task = _task()
    gold_obs = [3, 30]
    bad = [
        {"name": "multiply", "arguments": {"arg_0": 1, "arg_1": 2}},  # wrong name
        task["gold_calls"][1],
    ]
    traj = make_trajectory("t1", bad, [3, 30], gold_num_turns=2, final_observation=30)
    rr = strict_gold_trace_reward(traj, task, gold_obs)
    assert rr.reward == 0.0
    assert rr.diagnostics["tool_name_ok"] is False


def test_wrong_argument_key_reward_0():
    task = _task()
    gold_obs = [3, 30]
    bad = [
        {"name": "add", "arguments": {"x": 1, "y": 2}},  # wrong keys
        task["gold_calls"][1],
    ]
    traj = make_trajectory("t1", bad, [3, 30], gold_num_turns=2, final_observation=30)
    rr = strict_gold_trace_reward(traj, task, gold_obs)
    assert rr.reward == 0.0
    assert rr.diagnostics["argument_keys_ok"] is False


def test_correct_answer_wrong_chain_reward_0_and_flagged():
    task = _task()
    gold_obs = [3, 30]
    # Right final answer (30) but first call uses the wrong tool.
    bad = [
        {"name": "multiply", "arguments": {"arg_0": 1, "arg_1": 2}},
        {"name": "multiply", "arguments": {"arg_0": "$var_1.result$", "arg_1": 10}},
    ]
    traj = make_trajectory("t1", bad, [2, 30], gold_num_turns=2, final_observation=30)
    rr = strict_gold_trace_reward(traj, task, gold_obs)
    assert rr.reward == 0.0
    assert rr.diagnostics["final_answer_pass"] is True
    assert rr.diagnostics["answer_correct_wrong_path"] is True


def test_too_few_turns_reward_0():
    task = _task()
    gold_obs = [3, 30]
    traj = make_trajectory(
        "t1", [task["gold_calls"][0]], [3], gold_num_turns=2, final_observation=3
    )
    rr = strict_gold_trace_reward(traj, task, gold_obs)
    assert rr.reward == 0.0


def test_numpy_array_observation_no_ambiguous_bool():
    """IBM functions may return numpy arrays; reward must not crash on bool()."""
    import numpy as np
    from executor import matches_gold

    assert matches_gold(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is True
    assert matches_gold(np.array([1.0, 2.0]), np.array([1.0, 3.0])) is False

    task = _task()
    gold_obs = [np.int64(3), np.int64(30)]
    traj = make_trajectory(
        "t1",
        calls=task["gold_calls"],
        observations=[3, 30],
        gold_num_turns=2,
        final_observation=30,
    )
    rr = strict_gold_trace_reward(traj, task, gold_obs)
    assert isinstance(rr.reward, float)
    assert rr.diagnostics["turn_rewards"] == [1.0, 1.0]
