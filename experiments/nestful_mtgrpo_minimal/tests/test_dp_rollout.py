"""CPU-only tests for the data-parallel rollout machinery (vllm_dp_pool).

These cover the parts that carry correctness risk and can run WITHOUT CUDA, a
GPU, vLLM, or spawned processes:

  * run_episode_collect — the per-episode rollout+reward that each worker runs,
    exercised with a fake generate_fn and the gold_replay executor;
  * reward-policy resolution (strict vs. partial selection from config);
  * token-id encoding into plain int lists (safe to ship across processes);
  * the parent-side RolloutResult -> Episode conversion (token tensors + reward).

The multiprocessing/vLLM plumbing in DataParallelRolloutPool requires real GPUs
and is intentionally not unit-tested here; run.py degrades to the single-engine
path if the pool fails to start, and the integration is exercised on the pod.
"""
from __future__ import annotations

import pytest

import vllm_dp_pool as dp
from vllm_dp_pool import RolloutResult, run_episode_collect, _resolve_reward_fn, _encode_for_logprob


# ── fakes ────────────────────────────────────────────────────────────────────

class _FakeTok:
    """Deterministic tokenizer: char-based ids, no model needed."""
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, add_generation_prompt=False, **kw):
        n = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
        return list(range(n))

    def encode(self, text, add_special_tokens=False):
        return list(range(max(1, len(text) // 4)))


_TASK = {
    "task_id": "dp_1",
    "question": "What is 6*7?",
    "gold_calls": [{"name": "multiply", "arguments": {"arg_0": 6, "arg_1": 7}}],
    "gold_answer": "42",
    "num_calls": 1,
    "available_tools": [{"name": "multiply",
                         "parameters": {"properties": {"arg_0": {}, "arg_1": {}}}}],
    "tools": [{"name": "multiply",
               "parameters": {"properties": {"arg_0": {}, "arg_1": {}}}}],
}
_CONFIG = {
    "generation": {"temperature": 0.0, "top_p": 1.0},
    "executor": {"mode": "gold_replay"},
    "token_budget": {},
    "reward": {"train_policy": "strict_gold_trace"},
}

_TOOL_CALL_TEXT = ('<tool_call_answer>[{"name": "multiply", '
                   '"arguments": {"arg_0": 6, "arg_1": 7}}]</tool_call_answer>')


def _strict_seq():
    from reward import episode_turn_reward_seq
    return episode_turn_reward_seq


# ── run_episode_collect ────────────────────────────────────────────────────────

def test_run_episode_collect_basic_shape():
    def gen(messages, max_new_tokens):
        return {"text": _TOOL_CALL_TEXT, "prompt_tokens": 50,
                "completion_tokens": 20, "clipped": False, "prompt_overflow": False}

    res = run_episode_collect(
        tokenizer=_FakeTok(), task=_TASK, config=_CONFIG, registry=None,
        generate_fn=gen, reward_fn=_strict_seq(), gold_obs=None,
    )
    assert isinstance(res, RolloutResult)
    assert res.error is None
    # one gold turn -> one generated turn -> one (prompt_ids, completion_ids) pair
    assert len(res.turn_token_ids) == 1
    p_ids, c_ids = res.turn_token_ids[0]
    assert isinstance(p_ids, list) and all(isinstance(x, int) for x in p_ids)
    assert isinstance(c_ids, list) and all(isinstance(x, int) for x in c_ids)
    assert len(c_ids) > 0
    assert res.num_tool_calls == 1
    assert isinstance(res.episode_reward, float)
    assert 0.0 <= res.episode_reward <= 1.0
    # r_seq has one entry per generated turn.
    assert len(res.r_seq) == 1


def test_run_episode_collect_prompt_overflow():
    def gen(messages, max_new_tokens):
        return {"text": "", "prompt_tokens": 99999, "completion_tokens": 0,
                "clipped": False, "prompt_overflow": True}

    res = run_episode_collect(
        tokenizer=_FakeTok(), task=_TASK, config=_CONFIG, registry=None,
        generate_fn=gen, reward_fn=_strict_seq(), gold_obs=None,
    )
    assert res.prompt_overflow is True
    assert res.clipped_any is True
    assert res.stop_reason == "prompt_overflow"
    assert res.turn_token_ids == []          # nothing usable for the update
    assert res.episode_reward == 0.0          # clipped/overflow -> 0


def test_run_episode_collect_parse_fail():
    def gen(messages, max_new_tokens):
        return {"text": "I cannot help with that.", "prompt_tokens": 30,
                "completion_tokens": 8, "clipped": False, "prompt_overflow": False}

    res = run_episode_collect(
        tokenizer=_FakeTok(), task=_TASK, config=_CONFIG, registry=None,
        generate_fn=gen, reward_fn=_strict_seq(), gold_obs=None,
    )
    # The turn is still tokenised (it contributes to the update with reward 0),
    # but no valid tool call was parsed.
    assert res.num_tool_calls == 0
    assert res.zero_tool_calls is True
    assert res.episode_reward == 0.0
    assert res.stop_reason in ("parse_fail", "terminal")


# ── reward-policy resolution ────────────────────────────────────────────────────

def test_resolve_reward_fn_strict(monkeypatch):
    from reward import episode_turn_reward_seq as strict_seq
    assert _resolve_reward_fn({"reward": {"train_policy": "strict_gold_trace"}}) is strict_seq
    # Missing policy defaults to strict (explicit default, not a fallback).
    assert _resolve_reward_fn({}) is strict_seq
    # Unknown policy must HARD-FAIL (audit Bug 1: the old silent fallback to
    # the strict binary reward invalidated the v3/v3.1 pilots) ...
    monkeypatch.delenv("ALLOW_STRICT_REWARD_FALLBACK", raising=False)
    with pytest.raises(ValueError):
        _resolve_reward_fn({"reward": {"train_policy": "nonsense"}})
    # ... unless the fallback is explicitly allowed via the env escape hatch.
    monkeypatch.setenv("ALLOW_STRICT_REWARD_FALLBACK", "1")
    assert _resolve_reward_fn({"reward": {"train_policy": "nonsense"}}) is strict_seq


def test_resolve_reward_fn_partial_if_available():
    """When partial_reward is importable (sibling on path), partial is selected."""
    pr = pytest.importorskip("partial_reward")
    fn = _resolve_reward_fn({"reward": {"train_policy": "partial_gold_trace"},
                             "partial_reward": {}})
    assert fn is pr.episode_turn_reward_seq


# ── token encoding ──────────────────────────────────────────────────────────────

def test_encode_for_logprob_returns_int_lists():
    msgs = [{"role": "user", "content": "hello world this is a prompt"}]
    p_ids, c_ids = _encode_for_logprob(_FakeTok(), msgs, "some completion text")
    assert isinstance(p_ids, list) and all(isinstance(x, int) for x in p_ids)
    assert isinstance(c_ids, list) and all(isinstance(x, int) for x in c_ids)
    assert len(p_ids) > 0 and len(c_ids) > 0


# ── parent-side conversion ──────────────────────────────────────────────────────

def test_episode_from_pool_result_builds_tensors():
    torch = pytest.importorskip("torch")
    from grpo_train import _episode_from_pool_result, TurnTokens

    res = RolloutResult(
        turn_token_ids=[([1, 2, 3], [4, 5]), ([6, 7], [8, 9, 10])],
        episode_reward=0.5,
        r_seq=[1.0, 0.0],
        clipped_any=False,
        zero_tool_calls=False,
        num_tool_calls=2,
        stop_reason="max_turns",
        first_error_turn=1,
    )
    ep = _episode_from_pool_result(res)
    assert ep.reward == 0.5
    assert ep.trajectory.clipped_any is False
    assert ep.trajectory.zero_tool_calls is False
    assert len(ep.turn_tokens) == 2
    assert all(isinstance(tt, TurnTokens) for tt in ep.turn_tokens)
    tt0 = ep.turn_tokens[0]
    assert tt0.prompt_ids.dtype == torch.long
    assert tt0.prompt_ids.tolist() == [1, 2, 3]
    assert tt0.completion_ids.tolist() == [4, 5]
