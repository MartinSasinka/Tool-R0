"""Local HF inference for the weak solver (training-target model).

Used when WEAK_SOLVER_BACKEND=local so the weak solver is the exact same
Qwen3-4B-Instruct checkpoint used in GRPO/SFT — not an API proxy.

Default: 4-bit load (fits ~6 GB laptop GPUs). Override via LOCAL_WEAK_4BIT=0
for full bf16 on larger GPUs.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

_DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
_INSTANCE: Optional["LocalWeakSolver"] = None


class LocalWeakSolver:
    """Lazy-loaded causal LM for weak-solver role only."""

    def __init__(self) -> None:
        env = os.environ
        self.model_id = env.get("LOCAL_WEAK_MODEL", _DEFAULT_MODEL)
        self.device = env.get("LOCAL_WEAK_DEVICE", "cuda:0")
        self.load_4bit = env.get("LOCAL_WEAK_4BIT", "1") == "1"
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[local_weak] loading {self.model_id} "
              f"({'4-bit' if self.load_4bit else 'bf16'}) on {self.device}",
              flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        kwargs: Dict[str, Any] = {"trust_remote_code": True,
                                  "device_map": self.device}
        if self.load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, **kwargs)
        self._model.eval()
        print("[local_weak] ready", flush=True)

    def generate(self, messages: List[Dict[str, str]], *,
                 temperature: float, max_tokens: int,
                 seed: Optional[int] = None) -> str:
        self._ensure_loaded()
        import torch

        if seed is not None:
            torch.manual_seed(int(seed))

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kwargs)
        new = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new, skip_special_tokens=True).strip()

    def generate_n(self, messages: List[Dict[str, str]], *, temperature: float,
                   max_tokens: int, n: int, seed: Optional[int] = None
                   ) -> List[str]:
        """Sample `n` independent completions for the SAME prompt in one
        batched forward pass (shared prefill) — used by the multi-rollout
        GRPO-signal probe so 8 rollouts cost roughly one generation, not
        eight sequential ones."""
        self._ensure_loaded()
        import torch

        if seed is not None:
            torch.manual_seed(int(seed))

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "do_sample": temperature > 0,
            "num_return_sequences": max(1, n),
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kwargs)
        in_len = inputs["input_ids"].shape[1]
        return [self._tokenizer.decode(out[i][in_len:], skip_special_tokens=True).strip()
               for i in range(out.shape[0])]


def get_local_weak_solver() -> LocalWeakSolver:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = LocalWeakSolver()
    return _INSTANCE


def reset_local_weak_solver() -> None:
    """Test helper — drop cached model."""
    global _INSTANCE
    _INSTANCE = None
