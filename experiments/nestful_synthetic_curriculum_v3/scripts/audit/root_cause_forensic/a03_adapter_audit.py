"""A03 — Checkpoint provenance + delta-to-initialization LoRA analysis.

The prior offline audit computed cosine over RAW flattened adapter weights
(lib/offline_audit/adapters.py::_load_adapter_flat) — dominated by the LoRA A
matrices, which share the same seeded random init across arms while lora_B
starts at zero. Because B_init = 0, the TRUE update to the base weights is
exactly DeltaW = (alpha/r) * B_final @ A_final per module, with no init
checkpoint required. This script:
  1. verifies checkpoints/FINAL == train/checkpoints/adapter_epoch_1 (sha256);
  2. verifies FINAL differs across arms (no checkpoint swap);
  3. decomposes cross-arm cosine into raw-all / lora_A-only / lora_B-only;
  4. computes cross-arm cosine over delta-to-init (B@A) — the correct metric;
  5. reports per-arm update norms.
"""
from __future__ import annotations

from typing import Any, Dict, List

from common import ARMS, load_json, run_dir, sha256_file, write_json


def _load_lora(path):
    import torch
    from safetensors import safe_open
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k).double().cpu()
    return tensors


def _module_pairs(tensors):
    """Yield (module_name, A, B) for lora_A/lora_B weight pairs."""
    pairs = {}
    for k, t in tensors.items():
        if ".lora_A." in k:
            pairs.setdefault(k.split(".lora_A.")[0], {})["A"] = t
        elif ".lora_B." in k:
            pairs.setdefault(k.split(".lora_B.")[0], {})["B"] = t
    for name in sorted(pairs):
        d = pairs[name]
        if "A" in d and "B" in d:
            yield name, d["A"], d["B"]


def _flat_cos(x, y):
    import torch
    num = torch.dot(x, y).item()
    den = (x.norm() * y.norm()).item()
    return num / den if den > 0 else None


def main() -> Dict[str, Any]:
    import torch

    provenance = {}
    mods: Dict[str, Dict[str, tuple]] = {}  # arm -> module -> (A, B)
    raw_all = {}
    raw_a = {}
    raw_b = {}
    norms = {}
    for arm in ARMS:
        rd = run_dir(arm)
        final_st = rd / "checkpoints" / "FINAL" / "adapter_model.safetensors"
        epoch_st = rd / "train" / "checkpoints" / "adapter_epoch_1" / "adapter_model.safetensors"
        man_p = rd / "checkpoints" / "FINAL" / "checkpoint_manifest.json"
        provenance[arm] = {
            "final_sha256": sha256_file(final_st),
            "adapter_epoch_1_sha256": sha256_file(epoch_st),
            "final_equals_epoch1": sha256_file(final_st) == sha256_file(epoch_st),
            "checkpoint_manifest": load_json(man_p) if man_p.is_file() else None,
        }
        tensors = _load_lora(final_st)
        a_flat: List = []
        b_flat: List = []
        mods[arm] = {}
        d_norm2 = 0.0
        a_norm2 = 0.0
        b_norm2 = 0.0
        n_delta = 0
        for name, A, B in _module_pairs(tensors):
            mods[arm][name] = (A, B)
            a_flat.append(A.reshape(-1))
            b_flat.append(B.reshape(-1))
            # ||B@A||_F^2 = trace((B^T B)(A A^T)) — never materialize B@A
            d_norm2 += float(torch.trace((B.T @ B) @ (A @ A.T)))
            a_norm2 += float((A ** 2).sum())
            b_norm2 += float((B ** 2).sum())
            n_delta += A.shape[1] * B.shape[0]
        raw_a[arm] = torch.cat(a_flat)
        raw_b[arm] = torch.cat(b_flat)
        raw_all[arm] = torch.cat([raw_a[arm], raw_b[arm]])
        norms[arm] = {
            "delta_BA_fro_norm": d_norm2 ** 0.5,
            "lora_A_fro_norm": a_norm2 ** 0.5,
            "lora_B_fro_norm": b_norm2 ** 0.5,
            "n_delta_params": n_delta,
        }

    def _delta_dot(arm_x: str, arm_y: str) -> float:
        """<Bx@Ax, By@Ay> summed over shared modules via trace identity."""
        tot = 0.0
        for name in mods[arm_x]:
            if name not in mods[arm_y]:
                continue
            Ax, Bx = mods[arm_x][name]
            Ay, By = mods[arm_y][name]
            tot += float(torch.trace((By.T @ Bx) @ (Ax @ Ay.T)))
        return tot

    pairs = []
    for i, a in enumerate(ARMS):
        for b in ARMS[i + 1:]:
            dd = _delta_dot(a, b)
            na = norms[a]["delta_BA_fro_norm"]
            nb = norms[b]["delta_BA_fro_norm"]
            cos_d = dd / (na * nb) if na > 0 and nb > 0 else None
            dist2 = max(0.0, na * na + nb * nb - 2 * dd)
            pairs.append({
                "arm_a": a, "arm_b": b,
                "cosine_raw_all_weights": _flat_cos(raw_all[a], raw_all[b]),
                "cosine_lora_A_only": _flat_cos(raw_a[a], raw_a[b]),
                "cosine_lora_B_only": _flat_cos(raw_b[a], raw_b[b]),
                "cosine_delta_to_init_BA": cos_d,
                "rel_l2_delta": (dist2 ** 0.5) / ((na + nb) / 2) if (na + nb) > 0 else None,
            })

    a0a4 = next(p for p in pairs
                if p["arm_a"] == "A0_R0_CURRENT" and p["arm_b"] == "A4_GATED_VERIFIABLE")
    payload = {
        "provenance": provenance,
        "per_arm_norms": norms,
        "pairwise": pairs,
        "verdict": {
            "final_checkpoints_all_distinct": len({provenance[a]["final_sha256"] for a in ARMS}) == len(ARMS),
            "final_equals_trained_epoch1_everywhere": all(
                provenance[a]["final_equals_epoch1"] for a in ARMS),
            "a0_vs_a4_cosine_raw": a0a4["cosine_raw_all_weights"],
            "a0_vs_a4_cosine_delta_to_init": a0a4["cosine_delta_to_init_BA"],
            "raw_cosine_is_init_artifact": (
                a0a4["cosine_raw_all_weights"] is not None
                and a0a4["cosine_delta_to_init_BA"] is not None
                and a0a4["cosine_raw_all_weights"] - a0a4["cosine_delta_to_init_BA"] > 0.05),
        },
    }
    write_json("a03_adapter_audit.json", payload)
    return payload


if __name__ == "__main__":
    r = main()
    print(r["verdict"])
    for p in r["pairwise"]:
        print(p["arm_a"], p["arm_b"],
              "raw", round(p["cosine_raw_all_weights"], 6),
              "A", round(p["cosine_lora_A_only"], 6),
              "B", round(p["cosine_lora_B_only"], 6),
              "deltaBA", round(p["cosine_delta_to_init_BA"], 6))
