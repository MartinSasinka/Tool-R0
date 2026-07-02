"""Tests for graceful prompt-overflow handling in the rollout loop.

When generation reports prompt_overflow (the multi-turn history grew past the
context window), the episode must END cleanly with stop_reason="prompt_overflow"
instead of looping on empty turns or crashing.
"""
from __future__ import annotations

from rollout import run_episode


_TASK = {
    "task_id": "ovf_1",
    "question": "What is 6*7?",
    # Argument keys match what the model emits (arg_0/arg_1) so gold_replay can
    # "execute" the first call successfully and the episode reaches turn 2.
    "gold_calls": [{"name": "multiply", "arguments": {"arg_0": 6, "arg_1": 7}}],
    "gold_answer": "42",
    "num_calls": 1,
    # The executor needs the tool to exist in the schema, else it returns
    # unknown_tool before gold_replay can run.
    "available_tools": [{"name": "multiply", "parameters": {"properties": {"arg_0": {}, "arg_1": {}}}}],
    "tools": [{"name": "multiply", "parameters": {"properties": {"arg_0": {}, "arg_1": {}}}}],
}
_CONFIG = {
    "generation": {"temperature": 0.0, "top_p": 1.0, "max_extra_turns_eval": 1},
    "executor": {"mode": "gold_replay"},
    "token_budget": {},
}


def test_overflow_ends_episode_immediately():
    calls = {"n": 0}

    def overflow_gen(messages, max_new_tokens):
        calls["n"] += 1
        return {
            "text": "",
            "prompt_tokens": 99999,
            "completion_tokens": 0,
            "clipped": False,
            "prompt_overflow": True,
        }

    traj = run_episode(None, None, _TASK, _CONFIG, mode="eval",
                       generate_fn=overflow_gen)

    assert traj.stop_reason == "prompt_overflow"
    assert traj.prompt_overflow is True
    # Must not loop: a single generation attempt, then stop.
    assert calls["n"] == 1
    # The overflow turn is recorded with the right fail reason.
    assert traj.turns and traj.turns[-1].fail_reason == "prompt_overflow"


def test_overflow_on_second_turn():
    """First turn produces a tool call; second turn overflows -> clean stop."""
    state = {"n": 0}

    def gen(messages, max_new_tokens):
        state["n"] += 1
        if state["n"] == 1:
            return {
                "text": '<tool_call_answer>[{"name": "multiply", '
                        '"arguments": {"arg_0": 6, "arg_1": 7}}]</tool_call_answer>',
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "clipped": False,
                "prompt_overflow": False,
            }
        return {
            "text": "",
            "prompt_tokens": 99999,
            "completion_tokens": 0,
            "clipped": False,
            "prompt_overflow": True,
        }

    traj = run_episode(None, None, _TASK, _CONFIG, mode="eval", generate_fn=gen)
    assert traj.stop_reason == "prompt_overflow"
    assert traj.prompt_overflow is True
    assert state["n"] == 2
