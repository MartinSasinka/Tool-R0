"""GRPO text-only training helpers for multimodal checkpoints (e.g. Qwen3.5).

TRL's GRPOTrainer may load an AutoProcessor for VL models; its apply_chat_template
expects message['content'] as a list of typed parts, while our pipeline uses plain
strings. Force the text tokenizer as processing_class for chat templating.

Additionally, GRPOTrainer saves weight keys with an extra ``model.`` wrapper
(e.g. ``model.language_model.layers.0…``) compared to the HuggingFace Hub
format that vLLM expects (``language_model.model.layers.0…``).
``fix_checkpoint_for_vllm`` remaps them automatically.
"""

from __future__ import annotations

import glob as _glob
import json
import os
import shutil
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase

_HUB_MODEL_ID = os.environ.get("TOOL_R0_BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")


def load_grpo_tokenizer(
    model_name_or_path: str,
    revision: Optional[str] = None,
) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=revision,
        trust_remote_code=True,
    )
    tok.padding_side = "left"
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token
    return tok


def _resolve_hub_model(base_model: str) -> str:
    """Return a HuggingFace Hub model ID, even when *base_model* is a local
    checkpoint path.  Falls back to ``TOOL_R0_BASE_MODEL`` env var or the
    hard-coded default."""
    if not os.path.isdir(base_model):
        return base_model
    return _HUB_MODEL_ID


def _resolve_config_source(base_model: str) -> str:
    """Return a directory from which JSON configs can be copied.

    * If *base_model* is a local directory that contains the needed files,
      use it directly (fast, no download).
    * Otherwise download from the Hub.
    """
    if os.path.isdir(base_model):
        if os.path.isfile(os.path.join(base_model, "preprocessor_config.json")):
            return base_model
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(
            _resolve_hub_model(base_model),
            allow_patterns=["*.json"],
            ignore_patterns=["*.safetensors", "*.bin", "*.pt"],
        )
    except Exception:
        if os.path.isdir(base_model):
            return base_model
        raise


def fix_checkpoint_for_vllm(output_dir: str, base_model: str) -> None:
    """Make a GRPOTrainer checkpoint loadable by vLLM.

    Fixes four issues that arise with GRPO + (optional) PEFT checkpoints:
    0. PEFT adapter present → merge into base weights and save full checkpoint
       (vLLM #34186 silent-zero LoRA bug avoided by never serving raw adapters).
    1. tokenizer_class "TokenizersBackend" → "PreTrainedTokenizerFast"
    2. Missing processor JSON configs → copied from the base model / Hub
    3. Weight key prefix mismatch (model.language_model.* vs
       language_model.model.*) → remapped to match the Hub format

    *base_model* can be a HuggingFace Hub ID **or** a local checkpoint path.
    When it is a local path the Hub ID is resolved via the ``TOOL_R0_BASE_MODEL``
    environment variable (default ``Qwen/Qwen3-4B-Instruct-2507``).
    """
    _maybe_merge_lora_adapter(output_dir, base_model)
    _fix_tokenizer_class(output_dir)
    _copy_missing_configs(output_dir, base_model)
    _remap_weight_keys(output_dir, base_model)


# ---------------------------------------------------------------------------
# 0. GRPO + PEFT workarounds (mutate GRPOConfig in place before trainer build)
# ---------------------------------------------------------------------------

def apply_grpo_peft_workarounds(training_args) -> None:
    """Patch a GRPOConfig in place to dodge known TRL/PEFT incompatibilities.

    Call this just before constructing ``GRPOTrainer(...)`` whenever LoRA is in
    play.  Idempotent and safe to call when LoRA is disabled.

    Mitigates:
    - TRL #3089: ``gradient_checkpointing=True`` + PEFT raises
      ``element 0 of tensors does not require grad`` unless reentrant
      checkpointing is used. We force ``use_reentrant=True``.
    - TRL #3108: ``sync_ref_model`` is silently a no-op for PEFT models.
      We force it off so users don't believe ref sync is happening.
    """
    if getattr(training_args, "gradient_checkpointing", False):
        kwargs = dict(getattr(training_args, "gradient_checkpointing_kwargs", None) or {})
        kwargs.setdefault("use_reentrant", True)
        training_args.gradient_checkpointing_kwargs = kwargs
    if getattr(training_args, "sync_ref_model", False):
        training_args.sync_ref_model = False


# ---------------------------------------------------------------------------
# 0b. Merge LoRA adapter -> full HF checkpoint (run BEFORE remap)
# ---------------------------------------------------------------------------

_LORA_BACKUP_DIR = "_lora_adapter"
_LORA_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "README.md",
)


def _ensure_torchao_for_peft() -> None:
    """PEFT/transformers may require torchao>=0.16; Colab often ships 0.10."""
    try:
        import torchao
        from packaging import version

        ver = getattr(torchao, "__version__", "0.0.0")
        if version.parse(ver) >= version.parse("0.16.0"):
            return
        import subprocess
        import sys

        print(f"[fix_checkpoint] Upgrading torchao {ver} -> >=0.16.0 for PEFT merge...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-U", "torchao>=0.16.0"],
        )
    except ImportError:
        pass


def _maybe_merge_lora_adapter(output_dir: str, base_model: str) -> bool:
    """If *output_dir* contains a PEFT adapter, merge it into the base model
    weights and overwrite the directory with a full HF checkpoint.

    On success the original adapter files are moved into
    ``<output_dir>/_lora_adapter/`` so the merge can be reproduced and so a
    second invocation is a no-op (idempotent).

    Returns ``True`` if a merge happened, ``False`` if no adapter was found.
    """
    adapter_cfg = os.path.join(output_dir, "adapter_config.json")
    if not os.path.isfile(adapter_cfg):
        return False

    print(f"[fix_checkpoint] Detected PEFT adapter at {output_dir}; merging into base.")
    try:
        import torch  # local import: only needed when LoRA is in use
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(
            f"[fix_checkpoint] WARNING: cannot merge LoRA — missing dependency: {exc}. "
            "Install peft+transformers or merge manually."
        )
        return False

    hub_id = _resolve_hub_model(base_model)
    print(f"[fix_checkpoint]   base_model={hub_id}")

    _ensure_torchao_for_peft()

    try:
        base = AutoModelForCausalLM.from_pretrained(
            hub_id,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        peft_model = PeftModel.from_pretrained(base, output_dir, is_trainable=False)
        merged = peft_model.merge_and_unload()
    except Exception as exc:
        print(f"[fix_checkpoint] ERROR: merge_and_unload failed: {exc}")
        raise

    # Backup adapter side-files BEFORE writing merged weights so a re-run is idempotent.
    backup_dir = os.path.join(output_dir, _LORA_BACKUP_DIR)
    os.makedirs(backup_dir, exist_ok=True)
    for fname in _LORA_FILES:
        src = os.path.join(output_dir, fname)
        if os.path.isfile(src):
            shutil.move(src, os.path.join(backup_dir, fname))

    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="9GB")
    AutoTokenizer.from_pretrained(hub_id, trust_remote_code=True).save_pretrained(output_dir)

    # Free GPU/CPU memory promptly — important on shared training nodes.
    del base, peft_model, merged
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"[fix_checkpoint] Merged adapter into full checkpoint at {output_dir}.")
    return True


# ---------------------------------------------------------------------------
# 1. Fix tokenizer_class
# ---------------------------------------------------------------------------

def _fix_tokenizer_class(output_dir: str) -> None:
    path = os.path.join(output_dir, "tokenizer_config.json")
    if not os.path.isfile(path):
        return
    with open(path, "r") as f:
        cfg = json.load(f)
    if cfg.get("tokenizer_class") == "TokenizersBackend":
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[fix_checkpoint] tokenizer_class patched in {path}")


# ---------------------------------------------------------------------------
# 2. Copy missing processor configs
# ---------------------------------------------------------------------------

_WEIGHT_INDEX_PATTERNS = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}

def _copy_missing_configs(output_dir: str, base_model: str) -> None:
    """Copy processor/tokenizer JSON configs that are absent in *output_dir*.

    Source is *base_model* (local dir or Hub ID).  Skips weight-index files.
    """
    try:
        src_dir = _resolve_config_source(base_model)
        copied = []
        for src in sorted(_glob.glob(os.path.join(src_dir, "*.json"))):
            name = os.path.basename(src)
            if name in _WEIGHT_INDEX_PATTERNS:
                continue
            dst = os.path.join(output_dir, name)
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)
                copied.append(name)
        if copied:
            print(f"[fix_checkpoint] Copied from {base_model}: {', '.join(copied)}")
    except Exception as exc:
        print(f"[fix_checkpoint] WARNING: could not copy configs from {base_model}: {exc}")


def ensure_checkpoint_configs_for_vllm(output_dir: str, base_model: Optional[str] = None) -> None:
    """Copy missing HuggingFace JSON configs so vLLM can load multimodal checkpoints.

    HF Trainer checkpoints often omit ``preprocessor_config.json``; vLLM still builds
    the vision processor for Qwen3-VL. Safe to call repeatedly — only copies absent files.

    *base_model* defaults to ``TOOL_R0_BASE_MODEL`` (``Qwen/Qwen3-4B-Instruct-2507``).
    """
    bm = base_model or _HUB_MODEL_ID
    _copy_missing_configs(output_dir, bm)


# ---------------------------------------------------------------------------
# 3. Remap safetensors weight keys
# ---------------------------------------------------------------------------

def _detect_prefix(keys, anchor: str = "embed_tokens.weight"):
    for k in keys:
        if k.endswith(anchor):
            return k[: -len(anchor)]
    return None


def _get_reference_keys(base_model: str) -> set[str] | None:
    """Get the canonical weight key set that vLLM expects.

    Works with both Hub model IDs and local checkpoint paths.
    """
    hub_id = _resolve_hub_model(base_model)

    # Try Hub index first (sharded models)
    try:
        from huggingface_hub import hf_hub_download
        idx_path = hf_hub_download(hub_id, "model.safetensors.index.json")
        with open(idx_path) as f:
            return set(json.load(f)["weight_map"].keys())
    except Exception:
        pass

    # Try local index
    if os.path.isdir(base_model):
        idx_local = os.path.join(base_model, "model.safetensors.index.json")
        if os.path.isfile(idx_local):
            with open(idx_local) as f:
                return set(json.load(f)["weight_map"].keys())

    # Single safetensors file (local checkpoint)
    if os.path.isdir(base_model):
        sf_local = os.path.join(base_model, "model.safetensors")
        if os.path.isfile(sf_local):
            from safetensors import safe_open
            with safe_open(sf_local, framework="pt") as f:
                return set(f.keys())

    return None


def _remap_weight_keys(output_dir: str, base_model: str) -> None:
    """Remap weight key prefixes to match the format vLLM expects."""
    sf_path = os.path.join(output_dir, "model.safetensors")
    if not os.path.isfile(sf_path):
        return

    base_keys = _get_reference_keys(base_model)
    if base_keys is None:
        print("[fix_checkpoint] WARNING: could not get reference weight keys, skipping remap")
        return

    from safetensors.torch import load_file, save_file

    weights = load_file(sf_path)
    ckpt_keys = set(weights.keys())

    if ckpt_keys.issubset(base_keys):
        print("[fix_checkpoint] Weight keys already match base model.")
        return

    ckpt_prefix = _detect_prefix(ckpt_keys)
    base_prefix = _detect_prefix(base_keys)
    if ckpt_prefix is None or base_prefix is None:
        print("[fix_checkpoint] WARNING: could not detect key prefixes, skipping remap")
        return

    ckpt_vis = _detect_prefix(ckpt_keys, "visual.patch_embed.proj.weight")
    base_vis = _detect_prefix(base_keys, "visual.patch_embed.proj.weight")

    text_match = (ckpt_prefix == base_prefix)
    vis_match = (ckpt_vis is None or base_vis is None or ckpt_vis == base_vis)

    if text_match and vis_match:
        print("[fix_checkpoint] All prefixes match, no remap needed.")
        return

    # Build ordered prefix mappings (longest match first)
    mappings: list[tuple[str, str]] = []

    if ckpt_vis is not None and base_vis is not None and ckpt_vis != base_vis:
        mappings.append((ckpt_vis + "visual.", base_vis + "visual."))

    if not text_match:
        mappings.append((ckpt_prefix, base_prefix))

    mappings.sort(key=lambda x: -len(x[0]))

    new_weights = {}
    for key, tensor in weights.items():
        new_key = key
        for src, dst in mappings:
            if key.startswith(src):
                new_key = dst + key[len(src):]
                break
        new_weights[new_key] = tensor

    match_count = len(set(new_weights.keys()) & base_keys)
    print(f"[fix_checkpoint] Remapped weight keys: {match_count}/{len(new_weights)} "
          f"match base model (prefix '{ckpt_prefix}' -> '{base_prefix}')")

    save_file(new_weights, sf_path)

    # Remove stale index file if present
    bad_idx = os.path.join(output_dir, "model.safetensors.index.json")
    if os.path.exists(bad_idx):
        os.remove(bad_idx)
        print(f"[fix_checkpoint] Removed stale {bad_idx}")
