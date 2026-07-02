"""
Unified model adapter for eval benchmarks.

Supports three backends:
  1. vllm  – local HuggingFace model via vLLM (matches existing Tool-R0 eval)
  2. openai – any OpenAI-compatible API endpoint (vLLM serve, TGI, etc.)
  3. dummy  – returns canned responses for smoke-testing without GPU
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# vLLM backend (lazy import – only loaded when selected)
# ---------------------------------------------------------------------------

_vllm_llm = None
_vllm_tokenizer = None


def _init_vllm(cfg: Dict[str, Any]):
    global _vllm_llm, _vllm_tokenizer
    if _vllm_llm is not None:
        return
    from vllm import LLM
    from transformers import AutoTokenizer

    model_path = cfg["model_path"]
    tokenizer_path = cfg.get("tokenizer_path", model_path)
    # GRPO trainer checkpoints can land in three states that all break a vanilla
    # vLLM load:
    #   (a) missing preprocessor / tokenizer JSON (mostly multimodal Qwen3.5-VL);
    #   (b) raw PEFT adapter on disk (`base_model.model.layers.*` keys) — vLLM
    #       does not know that prefix and bails out with
    #       `There is no module or parameter named 'base_model'`;
    #   (c) keys with `language_model.` prefix from VL trainer that need remap.
    # `fix_checkpoint_for_vllm` is the single idempotent entry point that handles
    # all three (merge LoRA -> copy JSONs -> patch tokenizer -> remap keys), so
    # we route every loaded checkpoint through it. Repeated calls are no-ops.
    base = cfg.get("base_model_for_configs") or os.environ.get(
        "TOOL_R0_BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507"
    )
    try:
        from grpo_processing import fix_checkpoint_for_vllm

        adapter_present = os.path.isfile(os.path.join(model_path, "adapter_config.json"))
        missing_preproc = not os.path.isfile(os.path.join(model_path, "preprocessor_config.json"))
        if adapter_present:
            print(f"[model] Detected PEFT adapter in {model_path}; merging into base ({base})...")
        elif missing_preproc:
            print(f"[model] Missing preprocessor configs in {model_path}; copying from {base}...")
        fix_checkpoint_for_vllm(model_path, base)
    except Exception as exc:
        print(
            f"[model] ERROR while preparing checkpoint for vLLM (run manually: "
            f"python scripts/fix_checkpoint_for_vllm.py {model_path} --base-model {base}): {exc}"
        )
        raise
    _vllm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    _vllm_llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.90),
        max_model_len=cfg.get("max_model_len", 4096),
        enforce_eager=True,
    )


def _apply_chat_template(
    messages: List[Dict[str, str]],
    cfg: Dict[str, Any],
) -> str:
    """Apply the tokenizer's chat template to a list of messages.

    This is critical for instruct/chat models — sending raw text
    produces garbage, the model needs the proper special tokens.
    """
    _init_vllm(cfg)
    return _vllm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def _generate_vllm(
    prompts: List[str],
    cfg: Dict[str, Any],
) -> List[str]:
    from vllm import SamplingParams

    _init_vllm(cfg)
    stop_tokens = cfg.get("stop_tokens", [])
    sp = SamplingParams(
        temperature=cfg.get("temperature", 0.0),
        top_p=cfg.get("top_p", 1.0),
        max_tokens=cfg.get("max_new_tokens", 512),
        n=1,
        stop=stop_tokens if stop_tokens else None,
        include_stop_str_in_output=True,
    )
    outputs = _vllm_llm.generate(prompts, sp)
    return [o.outputs[0].text if o.outputs else "" for o in outputs]


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------

def _generate_openai(
    prompts: List[str],
    cfg: Dict[str, Any],
) -> List[str]:
    import openai

    api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "EMPTY")
    base_url = cfg.get("api_base") or os.environ.get("OPENAI_API_BASE")
    model_name = cfg.get("model_name", cfg.get("model_path", "gpt-4"))

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    results: List[str] = []
    for prompt in prompts:
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=cfg.get("temperature", 0.0),
                    max_tokens=cfg.get("max_new_tokens", 512),
                )
                results.append(resp.choices[0].message.content or "")
                break
            except Exception as e:
                if attempt == 2:
                    results.append(f"[ERROR] {e}")
                else:
                    time.sleep(2 ** attempt)
    return results


# ---------------------------------------------------------------------------
# Dummy backend for smoke-testing
# ---------------------------------------------------------------------------

def _generate_dummy(
    prompts: List[str],
    cfg: Dict[str, Any],
) -> List[str]:
    dummy_response = cfg.get("dummy_response", '<think>\nSmoke test reasoning.\n</think>\n<tool_call_answer>[{"name": "dummy_tool", "arguments": {"key": "value"}}]</tool_call_answer>')
    return [dummy_response] * len(prompts)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

BACKENDS = {
    "vllm": _generate_vllm,
    "openai": _generate_openai,
    "dummy": _generate_dummy,
}


TOOL_R0_SYSTEM_PROMPT = (
    "A conversation between user and tool-calling assistant. The user asks a question, "
    "and the assistant uses tools to solve it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> and "
    '<tool_call_answer></tool_call_answer> tags, i.e., <think>\nThis is my '
    'reasoning.\n</think>\n<tool_call_answer>[{"name": "<tool_name>", '
    '"arguments": {"arg1": "value", "arg2": "value2", ...}}, ...]</tool_call_answer>. '
)


def build_chat_prompt(
    user_content: str,
    cfg: Dict[str, Any],
    system_prompt: Optional[str] = None,
) -> str:
    """Build a properly templated chat prompt for the vLLM backend.

    Uses the tokenizer's chat template so the model sees the correct
    special tokens (e.g. <|im_start|> for Qwen).
    """
    if system_prompt is None:
        system_prompt = cfg.get("system_prompt", TOOL_R0_SYSTEM_PROMPT)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    backend = cfg.get("backend", "vllm")
    if backend == "vllm":
        return _apply_chat_template(messages, cfg)
    elif backend == "openai":
        return user_content
    else:
        return user_content


def generate(
    prompts: List[str],
    cfg: Dict[str, Any],
    batch_size: int = 8,
) -> List[str]:
    """Generate completions for a list of prompts using the configured backend.

    Args:
        prompts: list of formatted prompt strings (already chat-templated for vLLM).
        cfg: model config dict with at least {"backend": "vllm"|"openai"|"dummy", ...}.
        batch_size: how many prompts to send per vLLM batch (ignored for openai).

    Returns:
        list of raw completion strings, same length as prompts.
    """
    backend_name = cfg.get("backend", "vllm")
    fn = BACKENDS.get(backend_name)
    if fn is None:
        raise ValueError(f"Unknown backend '{backend_name}'. Choose from: {list(BACKENDS)}")

    if backend_name == "vllm":
        all_results: List[str] = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            print(f"  [model] generating {start}..{start+len(batch)} / {len(prompts)}")
            all_results.extend(fn(batch, cfg))
        return all_results

    return fn(prompts, cfg)


def reset():
    """Reset cached vLLM engine, kill child processes, and free GPU memory."""
    global _vllm_llm, _vllm_tokenizer

    import gc, multiprocessing

    for child in multiprocessing.active_children():
        try:
            child.terminate()
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
        except Exception:
            pass

    if _vllm_llm is not None:
        try:
            del _vllm_llm
        except Exception:
            pass
    _vllm_llm = None
    _vllm_tokenizer = None

    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
