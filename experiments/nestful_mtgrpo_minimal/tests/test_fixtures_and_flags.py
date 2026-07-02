"""Tests against tiny_tasks.jsonl fixture + reportability flags + fallback behavior.

These tests do NOT require a real model or DGX. All checks are pure-Python.
"""
import json
import os

import pytest

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "tiny_tasks.jsonl")


# ── fixture loading ────────────────────────────────────────────────────────────

def load_fixture():
    tasks = []
    with open(FIXTURE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def test_fixture_loads_three_tasks():
    tasks = load_fixture()
    assert len(tasks) == 3


def test_fixture_task_ids_unique():
    tasks = load_fixture()
    ids = [t["task_id"] for t in tasks]
    assert len(ids) == len(set(ids))


def test_fixture_fields_present():
    for task in load_fixture():
        for field in ("task_id", "question", "tools", "gold_calls", "gold_answer", "num_calls"):
            assert field in task, f"Missing '{field}' in {task['task_id']}"


def test_fixture_num_calls_matches_gold_calls():
    for task in load_fixture():
        assert task["num_calls"] == len(task["gold_calls"]), task["task_id"]


# ── parser gate (using fixture tools, no model needed) ────────────────────────

from parser import parse_tool_call


def test_parser_gate_no_tag():
    pr = parse_tool_call("just some text without any tag")
    assert not pr.ok
    assert not pr.is_terminal


def test_parser_gate_two_calls():
    pr = parse_tool_call(
        '<tool_call_answer>[{"name":"a","arguments":{}}]</tool_call_answer>'
        '<tool_call_answer>[{"name":"b","arguments":{}}]</tool_call_answer>'
    )
    assert not pr.ok


def test_parser_gate_invalid_json():
    pr = parse_tool_call("<tool_call_answer>{not json}</tool_call_answer>")
    assert not pr.ok


def test_parser_gate_missing_name():
    pr = parse_tool_call('<tool_call_answer>[{"arguments":{}}]</tool_call_answer>')
    assert not pr.ok


def test_parser_gate_arguments_not_dict():
    pr = parse_tool_call('<tool_call_answer>[{"name":"f","arguments":"bad"}]</tool_call_answer>')
    assert not pr.ok


def test_parser_terminal():
    pr = parse_tool_call("<tool_call_answer>[]</tool_call_answer>")
    assert pr.is_terminal
    assert pr.ok


def test_parser_valid_call():
    pr = parse_tool_call('<tool_call_answer>[{"name":"add","arguments":{"arg_0":1,"arg_1":2}}]</tool_call_answer>')
    assert pr.ok
    assert pr.call["name"] == "add"


# ── strict reward with fixture tasks (gold_replay fallback, no IBM) ────────────

from _helpers import make_trajectory
from reward import (
    strict_gold_trace_reward,
    strict_gold_trace_episode_reward,
    strict_gold_turn_rewards,
    episode_turn_reward_seq,
)


def test_episode_reward_is_alias():
    assert strict_gold_trace_episode_reward is strict_gold_trace_reward


def test_strict_reward_gold_path_tiny001():
    task = load_fixture()[0]  # 1-call: add(3,5)=8
    calls = task["gold_calls"]
    traj = make_trajectory(task["task_id"], calls, [8], gold_num_turns=1, final_observation=8)
    rr = strict_gold_trace_reward(traj, task, gold_observations=None)
    assert rr.reward == 1.0


def test_strict_reward_wrong_name():
    task = load_fixture()[0]
    wrong = [{"name": "subtract", "arguments": {"arg_0": 3, "arg_1": 5}}]
    traj = make_trajectory(task["task_id"], wrong, [8], gold_num_turns=1, final_observation=8)
    rr = strict_gold_trace_reward(traj, task, gold_observations=None)
    assert rr.reward == 0.0


def test_strict_reward_too_few_calls():
    task = load_fixture()[1]  # 2-call: multiply then divide
    calls = task["gold_calls"][:1]  # only first call
    traj = make_trajectory(task["task_id"], calls, [12], gold_num_turns=2, final_observation=12)
    rr = strict_gold_trace_reward(traj, task, gold_observations=None)
    assert rr.reward == 0.0


def test_turn_rewards_shape_matches_gold():
    task = load_fixture()[1]  # 2 gold calls
    calls = task["gold_calls"]
    traj = make_trajectory(task["task_id"], calls, [12, 6], gold_num_turns=2, final_observation=6)
    r = strict_gold_turn_rewards(traj, task, gold_observations=None)
    assert len(r) == 2


def test_solution_equivalent_is_not_training_reward():
    """solution_equivalent must NOT appear anywhere in reward.py as a value source."""
    reward_src = open(
        os.path.join(os.path.dirname(__file__), "..", "reward.py"), encoding="utf-8"
    ).read()
    # Only docstring / comment references are allowed; no actual usage as reward value.
    # The word should appear only in docstrings/comments, not in return statements.
    import ast
    tree = ast.parse(reward_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            src_fragment = ast.unparse(node)
            assert "solution_equivalent" not in src_fragment, (
                "solution_equivalent found in a return statement in reward.py — "
                "it must never be used as a training reward."
            )


# ── fallback_used behavior ────────────────────────────────────────────────────

from grpo_train import _turn_returns


def test_turn_returns_all_zero_no_episode_reward():
    G = _turn_returns([0.0, 0.0], episode_reward=0.0, gamma=1.0, lambda_episode=1.0)
    assert G == [0.0, 0.0]


def test_turn_returns_formula_single_turn():
    # T=0, t=0: G_0 = sum_{k=0}^{0} gamma^(0) r_0  +  lambda * gamma^(T-t+1) * R
    #                = 1*1 + 1*gamma^(0-0+1)*1 = 1 + 1 = 2
    G = _turn_returns([1.0], episode_reward=1.0, gamma=1.0, lambda_episode=1.0)
    assert abs(G[0] - 2.0) < 1e-9


def test_example_turn_level_summary_has_correct_keys():
    path = os.path.join(os.path.dirname(__file__), "..", "examples",
                        "example_train_summary_turn_level.json")
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    assert s["mt_grpo_mode"] == "turn_level_minimal"
    assert s["fallback_used"] is False
    assert "gamma" in s and "lambda_episode" in s


def test_example_fallback_summary_has_correct_keys():
    path = os.path.join(os.path.dirname(__file__), "..", "examples",
                        "example_train_summary_fallback.json")
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    assert s["fallback_used"] is True
    assert s["mt_grpo_mode"] == "episode_level"


def test_example_gold_replay_reportability():
    path = os.path.join(os.path.dirname(__file__), "..", "examples",
                        "example_metrics_gold_replay.json")
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    assert m["executor_mode"] == "gold_replay"
    assert m["solution_equivalent_reportable"] is False
    assert m["win_rate_reportable"] is False
    assert "warning" in m


# ── token budget stage defaults ───────────────────────────────────────────────

from rollout import get_stage_token_budget


def test_stage_budget_all_six_stages():
    config = {
        "token_budget": {"stage_defaults": {
            "1": {"max_prompt_tokens": 1280, "max_new_tokens": 1280, "vllm_max_model_length": 2560},
            "2": {"max_prompt_tokens": 2048, "max_new_tokens": 2048, "vllm_max_model_length": 4608},
            "3": {"max_prompt_tokens": 4096, "max_new_tokens": 2048, "vllm_max_model_length": 6144},
            "4": {"max_prompt_tokens": 4096, "max_new_tokens": 2560, "vllm_max_model_length": 6144},
            "5": {"max_prompt_tokens": 4096, "max_new_tokens": 3072, "vllm_max_model_length": 7168},
            "6": {"max_prompt_tokens": 4096, "max_new_tokens": 3072, "vllm_max_model_length": 7168},
        }},
        "generation": {},
    }
    for stage, expected_new in [(1, 1280), (2, 2048), (3, 2048), (4, 2560), (5, 3072), (6, 3072)]:
        b = get_stage_token_budget(config, stage, "eval")
        assert b["max_new_tokens"] == expected_new, f"stage {stage}"


def test_stage_budget_smoke_overrides():
    config = {
        "token_budget": {"stage_defaults": {
            "3": {"max_prompt_tokens": 4096, "max_new_tokens": 2048, "vllm_max_model_length": 6144},
        }},
        "generation": {"max_new_tokens_smoke": 1024},
    }
    b = get_stage_token_budget(config, 3, "smoke")
    assert b["max_new_tokens"] == 1024
