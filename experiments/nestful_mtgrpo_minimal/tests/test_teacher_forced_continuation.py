"""Tests for teacher-forced continuation training (Stage2b-style curriculum).

Covers:
  * rollout.resolve_teacher_forced_prefix_n — safety gating (enough calls left,
    full-mode requires a clean gold_obs replay, gold_replay is always safe).
  * rollout.build_teacher_forced_prefix — executor scope/turn-counter advance.
  * vllm_dp_pool.run_episode_collect — the DP-worker path: forced turns get NO
    entry in turn_token_ids, and r_seq stays aligned 1:1 with turn_token_ids
    (the critical invariant: no gradient on non-generated tokens).
  * grpo_train._rollout_episode_for_train — the single-engine (non-pool) path,
    exercised via the vllm_gen_fn branch (no HF model / GPU required).
"""
from __future__ import annotations

import pytest

from executor import ToolExecutor
from rollout import build_teacher_forced_prefix, resolve_teacher_forced_prefix_n
from vllm_dp_pool import run_episode_collect


# ── fixtures ─────────────────────────────────────────────────────────────────

class _FakeTok:
    """Deterministic tokenizer: char-based ids, no model needed."""
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, messages, add_generation_prompt=False, **kw):
        n = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
        return list(range(n))

    def encode(self, text, add_special_tokens=False):
        return list(range(max(1, len(text) // 4)))


_TASK_2CALL = {
    "task_id": "tf_2call",
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

_CONTINUATION_TEXT = (
    '<tool_call_answer>[{"name": "add", '
    '"arguments": {"arg_0": "$var1.result$", "arg_1": 4}, "label": "$var2"}]'
    '</tool_call_answer>'
)


def _strict_seq():
    from reward import episode_turn_reward_seq
    return episode_turn_reward_seq


# ── resolve_teacher_forced_prefix_n ─────────────────────────────────────────

def test_resolve_teacher_forced_prefix_n_disabled_by_default():
    task = {"gold_calls": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
    assert resolve_teacher_forced_prefix_n(task, 0, "gold_replay", None) == 0


def test_resolve_teacher_forced_prefix_n_caps_to_leave_one_call_generated():
    task = {"gold_calls": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
    assert resolve_teacher_forced_prefix_n(task, 1, "gold_replay", None) == 1
    # Requesting more than gold_n - 1 must be capped (never force EVERY call).
    assert resolve_teacher_forced_prefix_n(task, 5, "gold_replay", None) == 2


def test_resolve_teacher_forced_prefix_n_single_call_task_never_forces():
    task = {"gold_calls": [{"name": "a"}]}
    assert resolve_teacher_forced_prefix_n(task, 1, "gold_replay", None) == 0


def test_resolve_teacher_forced_prefix_n_full_mode_requires_gold_obs():
    task = {"gold_calls": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
    # No gold_obs available -> refuse to force (can't verify the prefix is safe).
    assert resolve_teacher_forced_prefix_n(task, 1, "full", None) == 0
    # Partial gold_obs (fewer than gold_n) -> still refuse.
    assert resolve_teacher_forced_prefix_n(task, 1, "full", [1, 2]) == 0
    # Full gold_obs coverage -> safe to force.
    assert resolve_teacher_forced_prefix_n(task, 1, "full", [1, 2, 3]) == 1


def test_resolve_teacher_forced_prefix_n_gold_replay_always_safe():
    # gold_replay trivially matches the literal gold call at each position, so
    # it never needs a gold_obs precheck.
    task = {"gold_calls": [{"name": "a"}, {"name": "b"}]}
    assert resolve_teacher_forced_prefix_n(task, 1, "gold_replay", None) == 1


# ── build_teacher_forced_prefix ─────────────────────────────────────────────

def test_build_teacher_forced_prefix_advances_executor_scope():
    task = dict(_TASK_2CALL)
    ex = ToolExecutor(task, registry=None, mode="gold_replay")
    forced_turns, history = build_teacher_forced_prefix(task, ex, 1)

    assert len(forced_turns) == 1
    ft = forced_turns[0]
    assert ft.teacher_forced is True
    assert ft.parsed_call == task["gold_calls"][0]
    assert ft.fail_reason is None

    # History = one assistant turn (the forced call) + one user turn (its
    # real observation), exactly as a normal generated turn would look.
    assert len(history) == 2
    assert history[0]["role"] == "assistant"
    assert "multiply" in history[0]["content"]
    assert history[1]["role"] == "user"

    # Executor scope/turn-counter advanced -> the SECOND gold call now
    # executes cleanly at the correct position (idx=1).
    assert "$var1" in ex.by_label
    res2 = ex.execute(task["gold_calls"][1])
    assert res2.error is None


def test_build_teacher_forced_prefix_zero_is_noop():
    task = dict(_TASK_2CALL)
    ex = ToolExecutor(task, registry=None, mode="gold_replay")
    forced_turns, history = build_teacher_forced_prefix(task, ex, 0)
    assert forced_turns == []
    assert history == []


# ── run_episode_collect (DP worker path) ────────────────────────────────────

def test_run_episode_collect_teacher_forced_prefix_basic():
    config = {
        "generation": {"temperature": 0.0, "top_p": 1.0},
        "executor": {"mode": "gold_replay"},
        "token_budget": {},
        "reward": {"train_policy": "strict_gold_trace"},
        "train": {"teacher_forced_prefix_calls": 1},
    }

    def gen(messages, max_new_tokens):
        return {"text": _CONTINUATION_TEXT, "prompt_tokens": 60,
                "completion_tokens": 20, "clipped": False, "prompt_overflow": False}

    res = run_episode_collect(
        tokenizer=_FakeTok(), task=_TASK_2CALL, config=config, registry=None,
        generate_fn=gen, reward_fn=_strict_seq(), gold_obs=None,
    )
    assert res.error is None
    # Only the GENERATED continuation turn gets a token pair — the forced
    # first call must NEVER appear here (no gradient on non-generated text).
    assert len(res.turn_token_ids) == 1
    # r_seq is aligned 1:1 with turn_token_ids (forced-prefix entries dropped).
    assert len(res.r_seq) == 1
    # Both calls (1 forced + 1 generated) count toward the full trajectory.
    assert res.num_tool_calls == 2
    assert res.reward_diag.get("teacher_forced_prefix_calls") == 1
    # Forced gold call1 + correct generated call2 + matching final answer ->
    # a fully correct trace under the strict reward.
    assert res.episode_reward == 1.0


def test_run_episode_collect_teacher_forced_prefix_wrong_continuation():
    """A wrong continuation must still yield a clean (non-crashing) episode
    with r_seq/turn_token_ids alignment preserved, and a reward < 1.0."""
    config = {
        "generation": {"temperature": 0.0, "top_p": 1.0},
        "executor": {"mode": "gold_replay"},
        "token_budget": {},
        "reward": {"train_policy": "strict_gold_trace"},
        "train": {"teacher_forced_prefix_calls": 1},
    }
    wrong_text = ('<tool_call_answer>[{"name": "subtract", '
                  '"arguments": {"arg_0": 1, "arg_1": 2}, "label": "$var2"}]'
                  '</tool_call_answer>')

    def gen(messages, max_new_tokens):
        return {"text": wrong_text, "prompt_tokens": 60,
                "completion_tokens": 20, "clipped": False, "prompt_overflow": False}

    res = run_episode_collect(
        tokenizer=_FakeTok(), task=_TASK_2CALL, config=config, registry=None,
        generate_fn=gen, reward_fn=_strict_seq(), gold_obs=None,
    )
    assert res.error is None
    assert len(res.turn_token_ids) == 1
    assert len(res.r_seq) == 1
    assert res.episode_reward < 1.0
    assert res.reward_diag.get("teacher_forced_prefix_calls") == 1


def test_run_episode_collect_teacher_forced_disabled_when_config_key_absent():
    """Regression guard: omitting train.teacher_forced_prefix_calls must
    reproduce the exact pre-existing (unforced) behavior."""
    config = {
        "generation": {"temperature": 0.0, "top_p": 1.0},
        "executor": {"mode": "gold_replay"},
        "token_budget": {},
        "reward": {"train_policy": "strict_gold_trace"},
    }

    calls_seen = []

    def gen(messages, max_new_tokens):
        calls_seen.append(messages)
        idx = len(calls_seen)
        if idx == 1:
            return {"text": ('<tool_call_answer>[{"name": "multiply", '
                              '"arguments": {"arg_0": 2, "arg_1": 3}, "label": "$var1"}]'
                              '</tool_call_answer>'),
                    "prompt_tokens": 40, "completion_tokens": 15,
                    "clipped": False, "prompt_overflow": False}
        return {"text": _CONTINUATION_TEXT, "prompt_tokens": 60,
                "completion_tokens": 20, "clipped": False, "prompt_overflow": False}

    res = run_episode_collect(
        tokenizer=_FakeTok(), task=_TASK_2CALL, config=config, registry=None,
        generate_fn=gen, reward_fn=_strict_seq(), gold_obs=None,
    )
    assert res.error is None
    # Nothing forced -> BOTH calls generated -> two token pairs.
    assert len(res.turn_token_ids) == 2
    assert len(res.r_seq) == 2
    assert res.reward_diag.get("teacher_forced_prefix_calls") == 0
    assert res.episode_reward == 1.0


def test_run_episode_collect_teacher_forced_prefix_single_call_task_noop():
    """A 1-call task cannot leave a call to generate -> forcing is skipped."""
    task = {
        "task_id": "tf_1call",
        "question": "What is 6*7?",
        "gold_calls": [{"name": "multiply", "arguments": {"arg_0": 6, "arg_1": 7}}],
        "gold_answer": "42",
        "num_calls": 1,
        "tools": [{"name": "multiply",
                   "parameters": {"properties": {"arg_0": {}, "arg_1": {}}}}],
    }
    config = {
        "generation": {"temperature": 0.0, "top_p": 1.0},
        "executor": {"mode": "gold_replay"},
        "token_budget": {},
        "reward": {"train_policy": "strict_gold_trace"},
        "train": {"teacher_forced_prefix_calls": 1},
    }
    text = ('<tool_call_answer>[{"name": "multiply", '
            '"arguments": {"arg_0": 6, "arg_1": 7}}]</tool_call_answer>')

    def gen(messages, max_new_tokens):
        return {"text": text, "prompt_tokens": 50, "completion_tokens": 20,
                "clipped": False, "prompt_overflow": False}

    res = run_episode_collect(
        tokenizer=_FakeTok(), task=task, config=config, registry=None,
        generate_fn=gen, reward_fn=_strict_seq(), gold_obs=None,
    )
    assert res.error is None
    assert len(res.turn_token_ids) == 1
    assert res.reward_diag.get("teacher_forced_prefix_calls") == 0


# ── grpo_train._rollout_episode_for_train (single-engine / non-pool path) ──

def _fake_tokenizer_torch():
    """Tokenizer mock returning real 1-D LongTensors (needs torch)."""
    from unittest.mock import MagicMock
    import torch

    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.eos_token = "<eos>"

    def _apply_chat_template(messages, *, add_generation_prompt=True,
                              tokenize=True, return_tensors=None):
        n_tokens = 20 + len(messages) * 5
        ids = torch.ones(1, n_tokens, dtype=torch.long)
        if return_tensors == "pt":
            return ids
        return ids.tolist()[0]

    def _encode(text, *, add_special_tokens=True, return_tensors=None):
        n = max(3, len(text) // 5)
        if return_tensors == "pt":
            return torch.ones(n, dtype=torch.long).unsqueeze(0)
        return [1] * n

    tok.apply_chat_template = _apply_chat_template
    tok.encode = _encode
    return tok


def test_rollout_episode_for_train_teacher_forced_prefix():
    torch = pytest.importorskip("torch")
    from grpo_train import _rollout_episode_for_train

    tok = _fake_tokenizer_torch()
    config = {
        "generation": {"temperature": 0.0, "top_p": 1.0},
        "executor": {"mode": "gold_replay"},
        "token_budget": {},
        "train": {"teacher_forced_prefix_calls": 1},
    }

    def vllm_gen_fn(messages, max_new_tokens):
        return {"text": _CONTINUATION_TEXT, "completion_tokens": 20,
                "clipped": False, "prompt_overflow": False}

    from reward import compute_gold_observations
    gold_obs = compute_gold_observations(_TASK_2CALL, registry=None)  # None (gold_replay)

    ep = _rollout_episode_for_train(
        model=None, tokenizer=tok, task=_TASK_2CALL, config=config, registry=None,
        max_turns=2, vllm_gen_fn=vllm_gen_fn, gold_obs=gold_obs,
    )
    assert ep.n_forced_turns == 1
    # Only the generated continuation produced a TurnTokens entry.
    assert len(ep.turn_tokens) == 1
    # Trajectory carries BOTH turns (forced + generated) for reward scoring.
    assert len(ep.trajectory.turns) == 2
    assert ep.trajectory.turns[0].teacher_forced is True
    assert ep.trajectory.turns[1].teacher_forced is False


def test_rollout_episode_for_train_no_forcing_when_key_absent():
    torch = pytest.importorskip("torch")
    from grpo_train import _rollout_episode_for_train

    tok = _fake_tokenizer_torch()
    config = {
        "generation": {"temperature": 0.0, "top_p": 1.0},
        "executor": {"mode": "gold_replay"},
        "token_budget": {},
    }
    calls = []

    def vllm_gen_fn(messages, max_new_tokens):
        calls.append(1)
        if len(calls) == 1:
            return {"text": ('<tool_call_answer>[{"name": "multiply", '
                              '"arguments": {"arg_0": 2, "arg_1": 3}, "label": "$var1"}]'
                              '</tool_call_answer>'),
                    "completion_tokens": 15, "clipped": False, "prompt_overflow": False}
        return {"text": _CONTINUATION_TEXT, "completion_tokens": 20,
                "clipped": False, "prompt_overflow": False}

    ep = _rollout_episode_for_train(
        model=None, tokenizer=tok, task=_TASK_2CALL, config=config, registry=None,
        max_turns=2, vllm_gen_fn=vllm_gen_fn, gold_obs=None,
    )
    assert ep.n_forced_turns == 0
    assert len(ep.turn_tokens) == 2
    assert all(not t.teacher_forced for t in ep.trajectory.turns)
