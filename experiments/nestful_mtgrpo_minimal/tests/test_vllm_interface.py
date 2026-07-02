"""Unit tests for the vLLM generation backend (vllm_generate.py).

These tests verify the interface contract WITHOUT instantiating a real vLLM
LLM engine (which requires GPU + 8+ GB VRAM). They use lightweight mocks to
ensure:
  1. VLLMGenerator.generate_fn() returns a dict compatible with run_episode().
  2. VLLMGenerator.sync_adapter() updates the adapter path correctly.
  3. VLLMGenerator.tokenize_for_logprob() returns 1-D LongTensors compatible
     with grpo_train._sequence_logprob().
  4. build_vllm_generator() computes correct max_model_len from token_budget.
  5. grpo_train._retokenize_for_logprob() returns correct shapes.
  6. run_episode() correctly calls generate_fn when provided (no model needed).
  7. VLLMGenerator is importable without vllm installed (lazy import guard).
"""
from __future__ import annotations

import sys
import types
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fake_tokenizer(vocab_size: int = 100):
    """Minimal tokenizer mock with apply_chat_template and encode."""
    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.eos_token = "<eos>"

    def _apply_chat_template(messages, *, add_generation_prompt=True,
                              tokenize=True, return_tensors=None):
        # return a list of fake token ids or a string
        n_tokens = 20 + len(messages) * 5
        if not tokenize:
            return "[PROMPT]" + " ".join(m["content"] for m in messages)
        import torch
        ids = torch.ones(1, n_tokens, dtype=torch.long)
        if return_tensors == "pt":
            return ids
        return ids.tolist()[0]

    def _encode(text, *, add_special_tokens=True, return_tensors=None):
        n = max(3, len(text) // 5)
        if return_tensors == "pt":
            import torch
            return torch.ones(n, dtype=torch.long).unsqueeze(0)
        # Plain-list path must NOT import torch: generate_fn's prompt-length
        # pre-check calls encode() on every happy-path call, and repeatedly
        # re-importing torch on this CPU build trips a docstring RuntimeError.
        return [1] * n

    tok.apply_chat_template = _apply_chat_template
    tok.encode = _encode
    return tok


def _fake_vllm_output(text: str, n_tokens: int):
    """Minimal vllm output structure mock."""
    token_ids = list(range(n_tokens))
    out = MagicMock()
    out.text = text
    out.token_ids = token_ids

    req_out = MagicMock()
    req_out.outputs = [out]
    req_out.prompt_token_ids = list(range(20))  # fake prompt token ids
    return req_out


# ─────────────────────────────────────────────────────────────────────────────
#  1. generate_fn() interface contract
# ─────────────────────────────────────────────────────────────────────────────

class TestVLLMGeneratorInterface:

    def _make_generator(self, adapter_path=None):
        """Create a VLLMGenerator with a mocked vllm.LLM."""
        from vllm_generate import VLLMGenerator

        fake_llm = MagicMock()
        fake_llm.generate.return_value = [_fake_vllm_output("result_text", n_tokens=8)]

        tok = _fake_tokenizer()

        with patch("vllm_generate.VLLMGenerator.__init__", lambda self_, *a, **kw: None):
            gen = VLLMGenerator.__new__(VLLMGenerator)
        gen._llm = fake_llm
        gen._tokenizer = tok
        gen._temperature = 0.7
        gen._top_p = 0.95
        gen._max_model_len = 8192
        gen._adapter_path = adapter_path
        gen._enable_lora = adapter_path is not None
        return gen, fake_llm

    def test_generate_fn_returns_required_keys(self):
        gen, _ = self._make_generator()
        messages = [{"role": "user", "content": "hello"}]
        # SamplingParams is imported lazily inside generate_fn via `from vllm import`.
        # Patch vllm in sys.modules so the local import resolves.
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = MagicMock(return_value=MagicMock())
        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            result = gen.generate_fn(messages, max_new_tokens=64)

        assert "text" in result
        assert "prompt_tokens" in result
        assert "completion_tokens" in result
        assert "clipped" in result
        assert "prompt_overflow" in result

    def _make_fake_vllm(self):
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = MagicMock(return_value=MagicMock())
        return fake_vllm

    def _make_bare_gen(self, n_tokens):
        from vllm_generate import VLLMGenerator
        fake_llm = MagicMock()
        fake_llm.generate.return_value = [_fake_vllm_output("text", n_tokens=n_tokens)]
        tok = _fake_tokenizer()
        with patch("vllm_generate.VLLMGenerator.__init__", lambda self_, *a, **kw: None):
            gen = VLLMGenerator.__new__(VLLMGenerator)
        gen._llm = fake_llm
        gen._tokenizer = tok
        gen._temperature = 0.7
        gen._top_p = 0.95
        gen._max_model_len = 8192
        gen._adapter_path = None
        gen._enable_lora = False
        return gen

    def test_generate_fn_clipped_when_max_tokens_reached(self):
        n_tokens = 16
        gen = self._make_bare_gen(n_tokens)
        with patch.dict(sys.modules, {"vllm": self._make_fake_vllm()}):
            result = gen.generate_fn([{"role": "user", "content": "x"}],
                                     max_new_tokens=n_tokens)
        assert result["clipped"] is True

    def test_generate_fn_not_clipped_when_below_limit(self):
        n_tokens = 8
        gen = self._make_bare_gen(n_tokens)
        with patch.dict(sys.modules, {"vllm": self._make_fake_vllm()}):
            result = gen.generate_fn([{"role": "user", "content": "x"}],
                                     max_new_tokens=100)
        assert result["clipped"] is False

    def test_generate_fn_prompt_overflow_false_for_short_prompt(self):
        """A short prompt is well within the context window -> no overflow."""
        gen = self._make_bare_gen(5)
        with patch.dict(sys.modules, {"vllm": self._make_fake_vllm()}):
            result = gen.generate_fn([{"role": "user", "content": "x"}], 100)
        assert result["prompt_overflow"] is False

    def test_generate_fn_prompt_overflow_precheck(self):
        """When the rendered prompt exceeds (max_model_len - max_new_tokens) the
        engine is NEVER called; a graceful overflow dict is returned instead."""
        gen = self._make_bare_gen(5)
        gen._max_model_len = 64  # tiny window forces the pre-check to trip
        with patch.dict(sys.modules, {"vllm": self._make_fake_vllm()}):
            # Long content -> fake tokenizer yields many tokens > budget.
            result = gen.generate_fn(
                [{"role": "user", "content": "x" * 5000}], max_new_tokens=32
            )
        assert result["prompt_overflow"] is True
        assert result["text"] == ""
        assert result["completion_tokens"] == 0
        gen._llm.generate.assert_not_called()

    def test_generate_fn_overflow_engine_valueerror(self):
        """A ValueError from the engine mentioning the model length is caught and
        converted to a graceful overflow dict (this was the crash being fixed)."""
        gen = self._make_bare_gen(5)
        gen._llm.generate.side_effect = ValueError(
            "The decoder prompt (length 57136) is longer than the maximum model "
            "length of 8192."
        )
        with patch.dict(sys.modules, {"vllm": self._make_fake_vllm()}):
            result = gen.generate_fn([{"role": "user", "content": "x"}], 100)
        assert result["prompt_overflow"] is True
        assert result["text"] == ""


# ─────────────────────────────────────────────────────────────────────────────
#  2. sync_adapter() updates path
# ─────────────────────────────────────────────────────────────────────────────

def test_sync_adapter_updates_path():
    from vllm_generate import VLLMGenerator
    with patch("vllm_generate.VLLMGenerator.__init__", lambda self_, *a, **kw: None):
        gen = VLLMGenerator.__new__(VLLMGenerator)
    gen._adapter_path = None
    gen._enable_lora = False
    gen._lora_id = 1

    gen.sync_adapter("/tmp/adapter_epoch_1")
    assert gen._adapter_path == "/tmp/adapter_epoch_1"
    assert gen._enable_lora is True

    # Clearing the adapter removes the ACTIVE adapter: _adapter_path is None, so
    # _make_lora_request() (which returns None when not self._adapter_path) serves
    # no LoRA. The engine stays LoRA-capable — _enable_lora cannot be turned off
    # after the engine was built with enable_lora=True (documented contract).
    gen.sync_adapter(None)
    assert gen._adapter_path is None


# ─────────────────────────────────────────────────────────────────────────────
#  3. tokenize_for_logprob() returns 1-D LongTensors
# ─────────────────────────────────────────────────────────────────────────────

def test_tokenize_for_logprob_shapes():
    import torch
    from vllm_generate import VLLMGenerator

    tok = _fake_tokenizer()
    with patch("vllm_generate.VLLMGenerator.__init__", lambda self_, *a, **kw: None):
        gen = VLLMGenerator.__new__(VLLMGenerator)
    gen._tokenizer = tok

    messages = [{"role": "user", "content": "hello world"}]
    p_ids, c_ids = gen.tokenize_for_logprob(messages, "completion text here")

    assert isinstance(p_ids, torch.Tensor), "prompt_ids must be a Tensor"
    assert isinstance(c_ids, torch.Tensor), "completion_ids must be a Tensor"
    assert p_ids.dim() == 1, "prompt_ids must be 1-D"
    assert c_ids.dim() == 1, "completion_ids must be 1-D"
    assert p_ids.shape[0] > 0
    assert c_ids.shape[0] > 0


# ─────────────────────────────────────────────────────────────────────────────
#  4. build_vllm_generator() max_model_len from token_budget
# ─────────────────────────────────────────────────────────────────────────────

def test_build_vllm_generator_max_model_len():
    """build_vllm_generator should pick the largest vllm_max_model_length from
    token_budget.stage_defaults, not the generation.max_model_length fallback."""
    config = {
        "model": {"base_model": "dummy", "lora_adapter": None},
        "hardware": {
            "use_vllm": True,
            "vllm_tensor_parallel_size": 1,
            "vllm_gpu_memory_utilization": 0.85,
            "vllm_enforce_eager": False,
            "bf16": True,
        },
        "generation": {"temperature": 0.7, "top_p": 0.95, "max_model_length": 4096},
        "finetuning": {"lora_r": 16},
        "token_budget": {
            "stage_defaults": {
                "1": {"vllm_max_model_length": 2560},
                "3": {"vllm_max_model_length": 6144},
                "6": {"vllm_max_model_length": 7168},
            }
        },
    }
    tok = _fake_tokenizer()

    # Patch VLLMGenerator.__init__ to avoid GPU allocation.
    with patch("vllm_generate.VLLMGenerator.__init__", return_value=None) as mock_init:
        from vllm_generate import build_vllm_generator
        build_vllm_generator(config, tok, mode="eval")

    _, kwargs = mock_init.call_args
    assert kwargs["max_model_len"] == 7168, \
        f"Expected 7168 (max across stages) but got {kwargs['max_model_len']}"


def test_build_vllm_generator_gpu_util_train_vs_eval():
    """Train mode uses lower gpu_memory_utilization than eval to leave room for HF."""
    config = {
        "model": {"base_model": "dummy", "lora_adapter": None},
        "hardware": {"bf16": True, "vllm_tensor_parallel_size": 1},
        "generation": {"temperature": 0.7, "top_p": 0.95},
        "finetuning": {"lora_r": 16},
        "token_budget": {"stage_defaults": {"1": {"vllm_max_model_length": 4096}}},
    }
    tok = _fake_tokenizer()

    with patch("vllm_generate.VLLMGenerator.__init__", return_value=None) as mock_init:
        from vllm_generate import build_vllm_generator
        build_vllm_generator(config, tok, mode="train")
    _, kwargs_train = mock_init.call_args
    train_util = kwargs_train["gpu_memory_utilization"]

    with patch("vllm_generate.VLLMGenerator.__init__", return_value=None) as mock_init:
        build_vllm_generator(config, tok, mode="eval")
    _, kwargs_eval = mock_init.call_args
    eval_util = kwargs_eval["gpu_memory_utilization"]

    assert train_util < eval_util, \
        f"Train gpu_util ({train_util}) should be < eval gpu_util ({eval_util})"


# ─────────────────────────────────────────────────────────────────────────────
#  5. grpo_train._retokenize_for_logprob() returns correct shapes
# ─────────────────────────────────────────────────────────────────────────────

def test_retokenize_for_logprob_returns_1d_tensors():
    import torch
    from grpo_train import _retokenize_for_logprob

    tok = _fake_tokenizer()
    messages = [
        {"role": "system", "content": "You are a tool-use agent."},
        {"role": "user", "content": "Call get_weather(location='Paris')"},
    ]
    completion = "I'll call get_weather now."
    p_ids, c_ids = _retokenize_for_logprob(tok, messages, completion)

    assert isinstance(p_ids, torch.Tensor)
    assert isinstance(c_ids, torch.Tensor)
    assert p_ids.dim() == 1
    assert c_ids.dim() == 1
    assert p_ids.shape[0] > 0
    assert c_ids.shape[0] > 0


# ─────────────────────────────────────────────────────────────────────────────
#  6. run_episode() correctly calls generate_fn (no model needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_run_episode_uses_generate_fn():
    """When generate_fn is provided, run_episode() must not call model at all."""
    from rollout import run_episode

    call_count = {"n": 0}

    def fake_generate_fn(messages, max_new_tokens):
        call_count["n"] += 1
        # Return a terminal response (empty list inside tags is the stop signal).
        return {
            "text": "<tool_call_answer>[]</tool_call_answer>",
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "clipped": False,
            "prompt_overflow": False,
        }

    task = {
        "task_id": "test_vllm_1",
        "question": "What is 6*7?",
        "gold_calls": [{"name": "multiply", "arguments": {"a": 6, "b": 7}}],
        "gold_answer": "42",
        "num_calls": 1,
        "available_tools": [],
        "tools": [],
    }
    config = {
        "generation": {"temperature": 0.0, "top_p": 1.0, "max_extra_turns_eval": 1},
        "executor": {"mode": "gold_replay"},
        "token_budget": {},
    }

    # model=None: must not be called when generate_fn is provided
    traj = run_episode(None, None, task, config, mode="eval",
                       generate_fn=fake_generate_fn)

    assert call_count["n"] >= 1, "generate_fn was never called"
    assert traj.stop_reason == "terminal"


# ─────────────────────────────────────────────────────────────────────────────
#  6b. tensor-parallel size is clamped to a valid divisor of the KV heads
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_tensor_parallel_size_clamps(monkeypatch):
    import vllm_generate
    # Pretend 4 GPUs are visible.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    # TP=3 is invalid for an 8-KV-head model -> nearest valid lower is 2.
    assert vllm_generate._resolve_tensor_parallel_size(3) == 2
    # TP=4 is valid and within the GPU count.
    assert vllm_generate._resolve_tensor_parallel_size(4) == 4
    # TP beyond the GPU count is capped to the visible count (4 -> valid).
    assert vllm_generate._resolve_tensor_parallel_size(8) == 4
    # TP=1 always valid.
    assert vllm_generate._resolve_tensor_parallel_size(1) == 1


def test_resolve_tensor_parallel_size_single_gpu(monkeypatch):
    import vllm_generate
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert vllm_generate._resolve_tensor_parallel_size(4) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  7. vllm_generate module is importable without vllm installed
# ─────────────────────────────────────────────────────────────────────────────

def test_vllm_generate_importable_without_vllm():
    """The module-level import of vllm_generate must succeed even when vllm is
    not installed — imports are lazy (inside __init__ and generate_fn)."""
    # Remove vllm from sys.modules to simulate absence.
    vllm_backup = sys.modules.pop("vllm", None)
    try:
        # Re-import vllm_generate; should NOT raise ImportError at module level.
        if "vllm_generate" in sys.modules:
            del sys.modules["vllm_generate"]
        import vllm_generate  # must not raise
        assert hasattr(vllm_generate, "VLLMGenerator")
        assert hasattr(vllm_generate, "build_vllm_generator")
    finally:
        if vllm_backup is not None:
            sys.modules["vllm"] = vllm_backup
