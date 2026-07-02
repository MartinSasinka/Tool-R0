#!/usr/bin/env python3
"""Convert a GRPOTrainer checkpoint so vLLM can load it.

GRPOTrainer (HuggingFace) saves Qwen3.5 weights with keys like:
    model.language_model.layers.0.self_attn.q_proj.weight

The HuggingFace Hub version (which vLLM is tested against) uses:
    language_model.model.layers.0.self_attn.q_proj.weight   (or similar)

This script detects the mismatch automatically and remaps.
"""
import argparse
import json
import os
import sys

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file


def get_base_model_keys(base_model: str) -> dict:
    """Download the weight index from HuggingFace and return the weight_map."""
    idx_path = hf_hub_download(base_model, "model.safetensors.index.json")
    with open(idx_path) as f:
        return json.load(f)["weight_map"]


def detect_prefix(keys, anchor="embed_tokens.weight"):
    """Find the prefix before a known anchor key."""
    for k in keys:
        if k.endswith(anchor):
            return k[: -len(anchor)]
    return None


def remap_weights(weights: dict, base_keys: set) -> dict:
    ckpt_keys = set(weights.keys())
    if ckpt_keys == base_keys or ckpt_keys.issubset(base_keys):
        print("Keys already match. No remapping needed.")
        return weights

    ckpt_prefix = detect_prefix(ckpt_keys)
    base_prefix = detect_prefix(base_keys)
    if ckpt_prefix is None or base_prefix is None:
        print("WARNING: Could not detect prefix from embed_tokens.weight")
        return weights

    print(f"  Text prefix:   '{ckpt_prefix}' -> '{base_prefix}'")

    ckpt_vis = detect_prefix(ckpt_keys, "visual.patch_embed.proj.weight")
    base_vis = detect_prefix(base_keys, "visual.patch_embed.proj.weight")

    mappings = []
    if ckpt_vis and base_vis and ckpt_vis != base_vis:
        print(f"  Visual prefix:  '{ckpt_vis}' -> '{base_vis}'")
        mappings.append((ckpt_vis + "visual.", base_vis + "visual."))

    if ckpt_prefix != base_prefix:
        mappings.append((ckpt_prefix, base_prefix))

    mappings.sort(key=lambda x: -len(x[0]))

    new_weights = {}
    for key, tensor in weights.items():
        new_key = key
        for src, dst in mappings:
            if key.startswith(src):
                new_key = dst + key[len(src) :]
                break
        new_weights[new_key] = tensor

    new_keys = set(new_weights.keys())
    match = len(new_keys & base_keys)
    print(f"  {match}/{len(new_keys)} keys match base model after remap")
    return new_weights


def fix_tokenizer_config(ckpt_dir: str):
    path = os.path.join(ckpt_dir, "tokenizer_config.json")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        cfg = json.load(f)
    if cfg.get("tokenizer_class") == "TokenizersBackend":
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print("  Fixed tokenizer_class: TokenizersBackend -> PreTrainedTokenizerFast")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?",
                    default="./qwen3-4b-tool-r0/iter1_generator/checkpoint-50")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ckpt = args.checkpoint
    base = args.base_model
    print(f"Checkpoint: {ckpt}")
    print(f"Base model: {base}")

    # 1. Get base model key format
    print("\n[1/5] Fetching base model weight index...")
    base_weight_map = get_base_model_keys(base)
    base_keys = set(base_weight_map.keys())
    print(f"  Base model has {len(base_keys)} weight keys")

    # 2. Load checkpoint
    sf_path = os.path.join(ckpt, "model.safetensors")
    print(f"\n[2/5] Loading {sf_path}...")
    weights = load_file(sf_path)
    print(f"  Checkpoint has {len(weights)} weight keys")

    # 3. Remap
    print("\n[3/5] Remapping weight keys...")
    new_weights = remap_weights(weights, base_keys)

    # 4. Save
    if not args.dry_run and new_weights is not weights:
        print(f"\n[4/5] Saving remapped weights to {sf_path}...")
        save_file(new_weights, sf_path)
    else:
        print("\n[4/5] Skipped save (dry-run or no remap needed)")

    # 5. Cleanup
    print("\n[5/5] Fixing configs...")
    bad_idx = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.exists(bad_idx):
        if not args.dry_run:
            os.remove(bad_idx)
        print(f"  Removed stale {bad_idx}")

    fix_tokenizer_config(ckpt)

    print(f"\nDone! Checkpoint is ready for vLLM.")


if __name__ == "__main__":
    main()
