from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from lib.offline_audit import ARMS
from lib.offline_audit.paths import final_adapter_path, sha256_file
from lib.offline_audit.stats_util import cosine


def _load_adapter_flat(path: Path) -> Tuple[Dict[str, Any], List[float], List[str]]:
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise RuntimeError(f"safetensors required: {e}") from e
    tensors: Dict[str, Any] = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k).float().cpu().view(-1)
            tensors[k] = t
    keys = sorted(tensors.keys())
    flat = [float(x) for k in keys for x in tensors[k].tolist()]
    meta = {
        "n_keys": len(keys),
        "n_params": len(flat),
        "l2_norm": math.sqrt(sum(x * x for x in flat)),
        "max_abs": max(abs(x) for x in flat) if flat else 0.0,
        "sha256_file": sha256_file(path),
    }
    for k in keys:
        v = tensors[k]
        meta[f"norm::{k}"] = float(v.pow(2).sum().sqrt())
    return meta, flat, keys


def _load_module_pairs(path: Path) -> Dict[str, Tuple[Any, Any]]:
    """module_name -> (A, B) LoRA factor tensors (float64)."""
    from safetensors import safe_open
    raw: Dict[str, Dict[str, Any]] = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k).double().cpu()
            if ".lora_A." in k:
                raw.setdefault(k.split(".lora_A.")[0], {})["A"] = t
            elif ".lora_B." in k:
                raw.setdefault(k.split(".lora_B.")[0], {})["B"] = t
    return {name: (d["A"], d["B"]) for name, d in raw.items()
            if "A" in d and "B" in d}


def _delta_norm2(mods: Dict[str, Tuple[Any, Any]]) -> float:
    """||B@A||_F^2 summed over modules via trace((B^T B)(A A^T)) — never
    materializes the full (out x in) update matrix."""
    import torch
    return sum(float(torch.trace((B.T @ B) @ (A @ A.T))) for A, B in mods.values())


def _delta_dot(mods_x: Dict[str, Tuple[Any, Any]],
               mods_y: Dict[str, Tuple[Any, Any]]) -> float:
    """<Bx@Ax, By@Ay> summed over shared modules via the trace identity
    trace(Bx Ax Ay^T By^T) = trace((By^T Bx)(Ax Ay^T))."""
    import torch
    tot = 0.0
    for name, (Ax, Bx) in mods_x.items():
        pair = mods_y.get(name)
        if pair is None:
            continue
        Ay, By = pair
        tot += float(torch.trace((By.T @ Bx) @ (Ax @ Ay.T)))
    return tot


def adapter_analysis(
    runs_root: Path, seed: str, reports_dir: Path, *, skip: bool = False
) -> Dict[str, Any]:
    if skip:
        return {"skipped": True}
    norms_rows = []
    flats: Dict[str, List[float]] = {}
    mods_by_arm: Dict[str, Dict[str, Any]] = {}
    delta_norms: Dict[str, float] = {}
    layer_rows = []
    for arm in ARMS:
        p = final_adapter_path(runs_root, arm, seed)
        if not p.is_file():
            continue
        meta, flat, keys = _load_adapter_flat(p)
        flats[arm] = flat
        mods_by_arm[arm] = _load_module_pairs(p)
        delta_norms[arm] = math.sqrt(max(0.0, _delta_norm2(mods_by_arm[arm])))
        norms_rows.append({"arm": arm,
                           "delta_to_init_BA_fro_norm": delta_norms[arm],
                           **{k: v for k, v in meta.items() if not k.startswith("norm::")}})
        for k in keys:
            layer_rows.append({"arm": arm, "layer": k, "norm": meta.get(f"norm::{k}")})

    pair_rows = []
    arms_list = [a for a in ARMS if a in flats]
    for i, a1 in enumerate(arms_list):
        for a2 in arms_list[i + 1 :]:
            c = cosine(flats[a1], flats[a2])
            diff = math.sqrt(sum((x - y) ** 2 for x, y in zip(flats[a1], flats[a2])))
            na = math.sqrt(sum(x * x for x in flats[a1]))
            nb = math.sqrt(sum(x * x for x in flats[a2]))
            rel = diff / ((na + nb) / 2) if (na + nb) > 0 else None
            # PRIMARY metric: cosine over the effective weight update
            # DeltaW = B @ A per module. lora_B is zero-initialized, so B@A IS
            # the delta to initialization — no init checkpoint needed. The raw
            # flat cosine is dominated by the (shared, seeded) lora_A random
            # init and reads ~1.0 for ANY two runs from the same seed; it must
            # never be used as evidence that two training runs are equivalent.
            dd = _delta_dot(mods_by_arm[a1], mods_by_arm[a2])
            dn1, dn2 = delta_norms[a1], delta_norms[a2]
            cos_delta = dd / (dn1 * dn2) if dn1 > 0 and dn2 > 0 else None
            dist2 = max(0.0, dn1 * dn1 + dn2 * dn2 - 2 * dd)
            rel_delta = ((dist2 ** 0.5) / ((dn1 + dn2) / 2)
                         if (dn1 + dn2) > 0 else None)
            pair_rows.append(
                {
                    "arm_a": a1,
                    "arm_b": a2,
                    "cosine_delta_to_init_BA": cos_delta,
                    "normalized_distance_delta_to_init_BA": rel_delta,
                    "cosine_flat": c,
                    "cosine_flat_note": "init-dominated; diagnostic only",
                    "l2_distance": diff,
                    "normalized_distance": rel,
                }
            )

    if norms_rows:
        with open(reports_dir / "adapter_norms.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(norms_rows[0].keys()))
            w.writeheader()
            w.writerows(norms_rows)
    if pair_rows:
        with open(reports_dir / "adapter_pairwise_similarity.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
            w.writeheader()
            w.writerows(pair_rows)
    if layer_rows:
        with open(reports_dir / "adapter_layerwise_similarity.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["arm", "layer", "norm"])
            w.writeheader()
            w.writerows(layer_rows)

    md = [
        "# Adapter analysis (CPU, LoRA only)",
        "",
        "Primary metric: `cosine_delta_to_init_BA` — cosine over the effective",
        "update DeltaW = B@A (lora_B is zero-initialized, so B@A is exactly the",
        "delta to initialization). The raw flat cosine over absolute adapter",
        "weights is dominated by the shared seeded lora_A init and reads ~1.0",
        "for any two same-seed runs; it is kept only as a diagnostic.",
        "",
    ]
    for r in pair_rows:
        if "A0" in r["arm_a"] and "A4" in r["arm_b"]:
            md.append(f"**A0 vs A4**: delta-to-init cosine={r['cosine_delta_to_init_BA']}, "
                      f"raw flat cosine={r['cosine_flat']} (init artifact), "
                      f"rel_dist_delta={r['normalized_distance_delta_to_init_BA']}")
    (reports_dir / "ADAPTER_ANALYSIS.md").write_text("\n".join(md), encoding="utf-8")
    return {"norms": norms_rows, "pairs": pair_rows}
