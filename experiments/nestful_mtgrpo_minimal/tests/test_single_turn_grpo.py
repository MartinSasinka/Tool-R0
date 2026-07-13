"""Tests for the single-turn (Direct-prompting) GRPO ablation.

Covers grpo_train._rollout_episode_single_turn_for_train via the vllm_gen_fn
branch (no HF model / GPU required):
  * a correct full plan -> per-call turns, gold answer surfaced, strict R=1,
    exactly ONE TurnTokens entry (the whole plan is one completion);
  * parse failure -> one parse-fail turn, strict R=0;
  * a wrong tool in the plan -> executor_error in gold_replay, strict R=0;
  * clipped completion -> clipped_any + masked-style episode;
  * train() rejects single_turn + rollout_pool.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from grpo_train import _rollout_episode_single_turn_for_train, train  # noqa: E402
from reward import episode_turn_reward_seq  # noqa: E402


# ── fixtures ─────────────────────────────────────────────────────────────────

class _FakeTokTorch:
    """Deterministic tokenizer returning torch tensors (for _retokenize_for_logprob)."""
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            return_tensors=None, **kw):
        n = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
        ids = torch.ones(1, n, dtype=torch.long)
        if return_tensors == "pt":
            return ids
        return ids.tolist()[0]

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        n = max(1, len(text) // 4)
        if return_tensors == "pt":
            return torch.ones(1, n, dtype=torch.long)
        return [1] * n


_TASK_2CALL = {
    "task_id": "st_2call",
    "question": "What is (2*3)+4?",
    "gold_calls": [
        {"name": "multiply", "arguments": {"arg_0": 2, "arg_1": 3}, "label": "$var1"},
        {"name": "add", "arguments": {"arg_0": "$var1.result$", "arg_1": 4}, "label": "$var2"},
    ],
    "gold_answer": "10",
    "num_calls": 2,
    "tools": [
        {"name": "multiply", "parameters": {"properties": {"arg_0": {}, "arg_1": {}}}},
        {"name": "add", "parameters": {"properties": {"arg_0": {}, "arg_1": {}}}},
    ],
}

_GOOD_PLAN = (
    '<tool_call_answer>['
    '{"name": "multiply", "arguments": {"arg_0": 2, "arg_1": 3}, "label": "$var1"}, '
    '{"name": "add", "arguments": {"arg_0": "$var1.result$", "arg_1": 4}, "label": "$var2"}'
    ']</tool_call_answer>'
)

_WRONG_TOOL_PLAN = (
    '<tool_call_answer>['
    '{"name": "add", "arguments": {"arg_0": 2, "arg_1": 3}, "label": "$var1"}, '
    '{"name": "add", "arguments": {"arg_0": "$var1.result$", "arg_1": 4}, "label": "$var2"}'
    ']</tool_call_answer>'
)

_CONFIG = {
    "generation": {"temperature": 0.0, "top_p": 1.0},
    "executor": {"mode": "gold_replay"},
    "token_budget": {},
    "training": {"single_turn": True},
    "single_turn": {"num_icl": 0},
}


def _gen_fn(text, clipped=False, overflow=False):
    def fn(messages, max_new_tokens):
        return {"text": text, "prompt_tokens": 10,
                "completion_tokens": max(1, len(text) // 4),
                "clipped": clipped, "prompt_overflow": overflow}
    return fn


def _rollout(text, task=_TASK_2CALL, **gen_kw):
    return _rollout_episode_single_turn_for_train(
        None, _FakeTokTorch(), dict(task), _CONFIG, None,
        vllm_gen_fn=_gen_fn(text, **gen_kw))


# ── correct plan ─────────────────────────────────────────────────────────────

def test_single_turn_correct_plan_scores_full_reward():
    ep = _rollout(_GOOD_PLAN)
    traj = ep.trajectory

    # One completion -> exactly one TurnTokens entry (gradient on ONE segment).
    assert len(ep.turn_tokens) == 1
    # But the plan expands into per-call turns so reward functions see the trace.
    assert traj.num_tool_calls == 2
    assert traj.stop_reason == "single_turn_plan"
    assert [t.parsed_call["name"] for t in traj.turns] == ["multiply", "add"]
    # gold_replay surfaces the gold answer on the final call.
    assert traj.final_observation == "10"

    rinfo = episode_turn_reward_seq(traj, _TASK_2CALL, None)
    assert rinfo["episode_reward"] == 1.0


def test_single_turn_token_accounting_only_on_first_turn():
    ep = _rollout(_GOOD_PLAN)
    turns = ep.trajectory.turns
    assert turns[0].completion_tokens > 0
    assert all(t.completion_tokens == 0 for t in turns[1:])
    # Raw completion text is attached once (not duplicated per call).
    assert turns[0].model_text.startswith("<tool_call_answer>")
    assert all(t.model_text == "" for t in turns[1:])


# ── failure modes ────────────────────────────────────────────────────────────

def test_single_turn_parse_failure_zero_reward():
    ep = _rollout("I think the answer is 10.")
    traj = ep.trajectory
    assert len(ep.turn_tokens) == 1
    assert traj.stop_reason == "parse_fail"
    assert traj.num_tool_calls == 0
    assert traj.turns[0].fail_reason.startswith("parse:")
    assert episode_turn_reward_seq(traj, _TASK_2CALL, None)["episode_reward"] == 0.0


def test_single_turn_wrong_tool_fails_in_gold_replay():
    ep = _rollout(_WRONG_TOOL_PLAN)
    traj = ep.trajectory
    assert traj.stop_reason == "executor_error"
    assert traj.turns[0].fail_reason.startswith("exec:")
    assert episode_turn_reward_seq(traj, _TASK_2CALL, None)["episode_reward"] == 0.0


def test_single_turn_clipped_completion_masked():
    ep = _rollout(_GOOD_PLAN, clipped=True)
    traj = ep.trajectory
    assert traj.clipped_any is True
    assert traj.stop_reason == "clipped"
    assert traj.num_tool_calls == 0
    assert episode_turn_reward_seq(traj, _TASK_2CALL, None)["episode_reward"] == 0.0


def test_single_turn_prompt_overflow_has_no_turn_tokens():
    ep = _rollout(_GOOD_PLAN, overflow=True)
    assert ep.turn_tokens == []
    assert ep.trajectory.prompt_overflow is True
    assert ep.trajectory.clipped_any is True


def test_single_turn_plan_capped_at_gold_plus_4():
    calls = ", ".join(
        '{"name": "multiply", "arguments": {"arg_0": 1, "arg_1": 1}}'
        for _ in range(20))
    ep = _rollout(f"<tool_call_answer>[{calls}]</tool_call_answer>")
    # gold_n=2 -> cap 6; the first call already errors in gold_replay (wrong
    # args vs gold), so at most the cap ever executes.
    assert len(ep.trajectory.turns) <= 6


# ── train() guard ────────────────────────────────────────────────────────────

def test_train_rejects_single_turn_with_rollout_pool(tmp_path):
    config = {
        "training": {"single_turn": True},
        "generation": {},
        "mt_grpo": {"enabled": True},
        "reward": {"train_policy": "strict"},
        "data": {},
    }
    with pytest.raises(ValueError, match="single_turn"):
        train(config, None, None, None, [], str(tmp_path / "log.jsonl"),
              rollout_pool=object())
