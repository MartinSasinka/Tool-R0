from metrics import (
    answer_supported_by_observations,
    solution_equivalent_score,
    compute_nestful_official_metrics,
)
from _helpers import make_trajectory


def _task(gold_answer=30):
    return {
        "task_id": "t1",
        "tools": [
            {"name": "add", "parameters": {"properties": {"arg_0": {}, "arg_1": {}}}},
            {"name": "multiply", "parameters": {"properties": {"arg_0": {}, "arg_1": {}}}},
        ],
        "gold_calls": [
            {"name": "add", "arguments": {"arg_0": 1, "arg_1": 2}},
            {"name": "multiply", "arguments": {"arg_0": 3, "arg_1": 10}},
        ],
        "gold_answer": gold_answer,
        "num_calls": 2,
    }


def test_supported_when_value_in_observations():
    assert answer_supported_by_observations(30, [3, 30], 30) is True


def test_unsupported_when_value_absent():
    # Final answer matches gold, but never appears in observations -> unsupported.
    assert answer_supported_by_observations(30, [3, 7], 30) is False


def test_unsupported_when_final_mismatch():
    assert answer_supported_by_observations(31, [3, 30], 30) is False


def test_solution_equivalent_alternative_path_passes_full_mode():
    task = _task(30)
    # Alternative (non-gold) but executable path reaching 30, with evidence.
    alt_calls = [
        {"name": "multiply", "arguments": {"arg_0": 3, "arg_1": 10}},
    ]
    traj = make_trajectory(
        "t1", alt_calls, [30], gold_num_turns=2, final_observation=30,
        executor_mode="full",
    )
    res = solution_equivalent_score(traj, task)
    assert res.passed is True
    assert res.limited is False


def test_solution_equivalent_shortcut_fails():
    task = _task(30)
    # Model "reaches" 30 but observations do not contain it (unsupported shortcut).
    calls = [{"name": "add", "arguments": {"arg_0": 1, "arg_1": 2}}]
    traj = make_trajectory(
        "t1", calls, [3], gold_num_turns=2, final_observation=30,
        executor_mode="full",
    )
    res = solution_equivalent_score(traj, task)
    assert res.passed is False
    assert res.reason == "answer_not_supported_by_own_trace"


def test_solution_equivalent_executor_error_fails():
    task = _task(30)
    calls = [{"name": "add", "arguments": {"arg_0": 1, "arg_1": 2}}]
    traj = make_trajectory(
        "t1", calls, [None], gold_num_turns=2, final_observation=None,
        executor_mode="full", fail_reasons=["exec:runtime_error:X"],
        stop_reason="executor_error",
    )
    res = solution_equivalent_score(traj, task)
    assert res.passed is False
    assert res.reason == "executor_error"


def test_solution_equivalent_limited_in_gold_replay():
    task = _task(30)
    calls = task["gold_calls"]
    traj = make_trajectory(
        "t1", calls, [3, 30], gold_num_turns=2, final_observation=30,
        executor_mode="gold_replay",
    )
    res = solution_equivalent_score(traj, task)
    assert res.limited is True


def test_official_metrics_full_match():
    task = _task(30)
    pred = task["gold_calls"]
    m = compute_nestful_official_metrics(pred, task["gold_calls"])
    assert m["f1_func"] == 1.0
    assert m["f1_param"] == 1.0
    assert m["full_sequence_accuracy"] == 1.0
    assert m["partial_sequence_accuracy"] == 1.0


def test_official_metrics_partial():
    task = _task(30)
    pred = [
        {"name": "add", "arguments": {"arg_0": 1, "arg_1": 2}},   # correct
        {"name": "add", "arguments": {"arg_0": 9, "arg_1": 9}},   # wrong second step
    ]
    m = compute_nestful_official_metrics(pred, task["gold_calls"])
    assert m["full_sequence_accuracy"] == 0.0
    assert m["partial_sequence_accuracy"] == 0.5
