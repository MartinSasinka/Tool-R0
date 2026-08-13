"""P43 final-answer turn + success-metrics tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from interaction_loop import (  # noqa: E402
    derive_interaction_budget, run_tool_agent_loop, assert_p43_interaction_config,
)
from rollout import Trajectory  # noqa: E402
from success_metrics import compute_rollout_success_flags  # noqa: E402


class _FakeExec:
    mode = "synthetic"

    def __init__(self):
        self.n = 0

    def execute(self, call):
        self.n += 1
        from types import SimpleNamespace
        return SimpleNamespace(observation=f"obs_{self.n}", error=None)


def _task(n_calls: int, gold_answer: Any = "FINAL") -> Dict[str, Any]:
    calls = [{"name": f"tool_{i}", "label": f"$var_{i}",
              "arguments": {"x": i}} for i in range(1, n_calls + 1)]
    return {
        "task_id": f"t{n_calls}",
        "num_calls": n_calls,
        "gold_calls": calls,
        "gold_answer": gold_answer,
        "question": "q",
        "tools": [],
    }


def _scripted_gen(scripts: List[str]):
    state = {"i": 0}

    def gen(messages, max_new_tokens):
        i = state["i"]
        state["i"] += 1
        text = scripts[i] if i < len(scripts) else "<tool_call_answer>[]</tool_call_answer>"
        return {"text": text, "clipped": False, "prompt_overflow": False,
                "prompt_tokens": 10, "completion_tokens": 5}

    return gen, state


def _run(n_calls: int, scripts: List[str], config=None):
    config = config or {
        "interaction": {"reserve_final_answer_turn": True},
        "generation": {"max_extra_turns_eval": 1},
        "train": {"max_extra_turns_train": 0},
    }
    task = _task(n_calls)
    budget = derive_interaction_budget(n_calls, config, mode="train")
    traj = Trajectory(task["task_id"], n_calls, n_calls, executor_mode="synthetic")
    history: List[Dict[str, str]] = []
    gen, _ = _scripted_gen(scripts)
    meta = run_tool_agent_loop(
        task=task, config=config, executor=_FakeExec(), traj=traj,
        history=history, generate_fn=gen, max_new_tokens=64, budget=budget,
    )
    return traj, meta, budget


def test_budget_2_to_10_reserves_final():
    cfg = {"interaction": {"reserve_final_answer_turn": True},
           "train": {"max_extra_turns_train": 0}}
    for g in range(2, 11):
        b = derive_interaction_budget(g, cfg, mode="train")
        assert b.max_tool_calls == g
        assert b.max_assistant_turns == g + 1
        assert b.max_assistant_turns > b.max_tool_calls


def test_train_budget_tool_slack_one():
    cfg = {
        "interaction": {
            "reserve_final_answer_turn": True,
            "tool_call_slack": 1,
            "tool_call_slack_cap": 10,
        },
    }
    for g in range(2, 11):
        b = derive_interaction_budget(g, cfg, mode="train")
        assert b.max_tool_calls == g + 1
        assert b.max_assistant_turns == g + 2  # tools + final


def test_fail_fast_reserve_required():
    with pytest.raises(RuntimeError):
        assert_p43_interaction_config(
            {"interaction": {"reserve_final_answer_turn": False}})


def test_2_call_success_final_answer():
    scripts = [
        '<tool_call_answer>[{"name":"tool_1","arguments":{"x":1}}]</tool_call_answer>',
        '<tool_call_answer>[{"name":"tool_2","arguments":{"x":2}}]</tool_call_answer>',
        '<tool_call_answer>[]</tool_call_answer>',
    ]
    traj, meta, budget = _run(2, scripts)
    assert budget.max_tool_calls == 2
    assert budget.max_assistant_turns == 3
    assert meta.tool_calls_executed == 2
    assert meta.assistant_turns == 3
    assert meta.final_response_turn_attempted is True or meta.final_answer_present
    assert meta.final_answer_present is True
    assert traj.stop_reason == "terminal_answer"
    assert traj.final_observation is not None


def test_5_call_success():
    scripts = [
        f'<tool_call_answer>[{{"name":"tool_{i}","arguments":{{"x":{i}}}}}]</tool_call_answer>'
        for i in range(1, 6)
    ] + ['<tool_call_answer>[]</tool_call_answer>']
    traj, meta, _ = _run(5, scripts)
    assert meta.tool_calls_executed == 5
    assert meta.assistant_turns == 6
    assert meta.final_answer_present
    assert traj.stop_reason == "terminal_answer"


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_parametrized_call_lengths_final_opportunity(n):
    scripts = [
        f'<tool_call_answer>[{{"name":"tool_{i}","arguments":{{"x":{i}}}}}]</tool_call_answer>'
        for i in range(1, n + 1)
    ] + ['<tool_call_answer>[]</tool_call_answer>']
    traj, meta, budget = _run(n, scripts)
    assert budget.max_assistant_turns == n + 1
    assert meta.tool_calls_executed == n
    assert meta.final_answer_present
    assert traj.stop_reason == "terminal_answer"


def test_tool_budget_exhaustion_tool_attempt():
    scripts = [
        '<tool_call_answer>[{"name":"tool_1","arguments":{"x":1}}]</tool_call_answer>',
        '<tool_call_answer>[{"name":"tool_2","arguments":{"x":2}}]</tool_call_answer>',
        '<tool_call_answer>[{"name":"tool_3","arguments":{"x":3}}]</tool_call_answer>',
    ]
    traj, meta, _ = _run(2, scripts)
    assert meta.tool_calls_executed == 2
    assert meta.final_response_turn_attempted is True
    assert meta.final_turn_tool_attempt is True
    assert traj.stop_reason == "final_turn_tool_attempt"
    # Third tool must NOT execute
    assert meta.tool_calls_emitted == 3
    assert meta.tool_calls_executed == 2


def test_early_final():
    scripts = [
        '<tool_call_answer>[{"name":"tool_1","arguments":{"x":1}}]</tool_call_answer>',
        '<tool_call_answer>[]</tool_call_answer>',
    ]
    traj, meta, _ = _run(2, scripts)
    assert meta.tool_calls_executed == 1
    assert meta.final_answer_present
    assert traj.stop_reason == "terminal_answer"


def test_wrong_final_after_perfect_trace_flags():
    # Build traj manually: 2 tools + terminal, but obs != gold
    task = _task(2, gold_answer="GOLD")
    traj = Trajectory("t", 2, 2, executor_mode="synthetic")
    from rollout import Turn
    traj.turns = [
        Turn(0, "a", parsed_call={"name": "tool_1", "arguments": {}}, observation="obs_1"),
        Turn(1, "b", parsed_call={"name": "tool_2", "arguments": {}}, observation="WRONG"),
        Turn(2, "[]", is_terminal=True),
    ]
    traj.final_observation = "WRONG"
    traj.stop_reason = "terminal_answer"
    traj.interaction_meta = {
        "final_answer_present": True,
        "final_response_turn_attempted": True,
        "tool_calls_executed": 2,
        "assistant_turns": 3,
        "tool_budget": 2,
        "assistant_turn_budget": 3,
    }
    # Pretend full sequence match via diag
    flags = compute_rollout_success_flags(
        traj, task,
        {"full_sequence_match": True, "strict_gold_trace_success": True,
         "required_subgoals_completed": 2, "required_subgoals_total": 2})
    assert flags["tool_trace_success"] is True
    assert flags["full_sequence_match"] is True
    assert flags["final_answer_correct"] is False
    assert flags["official_win"] is False


def test_reward_not_win():
    task = _task(2, gold_answer="GOLD")
    traj = Trajectory("t", 2, 2, executor_mode="synthetic")
    from rollout import Turn
    traj.turns = [
        Turn(0, "a", parsed_call={"name": "tool_1", "arguments": {}}, observation="x"),
    ]
    traj.final_observation = "x"
    traj.stop_reason = "max_assistant_turns"
    flags = compute_rollout_success_flags(traj, task, {})
    # High partial reward must not imply official_win
    episode_reward = 0.9
    assert flags["official_win"] is False
    assert episode_reward >= 0.9
    # Perfect match case
    traj2 = Trajectory("t2", 2, 2, executor_mode="synthetic")
    traj2.turns = [
        Turn(0, "a", parsed_call={"name": "tool_1", "arguments": {}}, observation="GOLD"),
        Turn(1, "[]", is_terminal=True),
    ]
    traj2.final_observation = "GOLD"
    traj2.stop_reason = "terminal_answer"
    traj2.interaction_meta = {"final_answer_present": True}
    flags2 = compute_rollout_success_flags(traj2, task, {})
    assert flags2["official_win"] is True


def test_eval_budget_matches_historical_assistant_count():
    cfg = {"interaction": {"reserve_final_answer_turn": True},
           "generation": {"max_extra_turns_eval": 1}}
    b = derive_interaction_budget(2, cfg, mode="eval")
    assert b.max_tool_calls == 2
    assert b.max_assistant_turns == 3  # gold + final (historical max_turns)


def test_eval_budget_tool_slack_ten():
    cfg = {
        "interaction": {
            "reserve_final_answer_turn": True,
            "tool_call_slack": 10,
            "tool_call_slack_cap": 10,
        },
        "generation": {"max_extra_turns_eval": 1},
    }
    b = derive_interaction_budget(6, cfg, mode="eval")
    assert b.max_tool_calls == 16  # gold + 10
    assert b.max_assistant_turns == 17  # tools + final



def test_strict_pass_not_continuous_reward():
    task = _task(2)
    traj = Trajectory("t", 2, 2)
    flags = compute_rollout_success_flags(
        traj, task, {"strict_gold_trace_pass": 0.247})
    assert flags["strict_gold_trace_pass"] is False
