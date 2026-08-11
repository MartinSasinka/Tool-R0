"""Canonical data loading (re-export of minimal/data.py) + v2 split builder.

Adds ``build_synthetic_splits`` which carves a fixed, stratified-by-num_calls
held-out VALIDATION split from the cleaned synthetic curriculum. The real
NESTFUL ``nestful_data.jsonl`` is NEVER touched by training or validation — it
stays the clean test set (decision: keep test paper-comparable, validate on
synthetic).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

from . import ensure_paths

ensure_paths()

from data import (  # noqa: E402,F401
    load_tasks,
    load_tasks_mixed,
    normalize_task,
)


def _file_md5(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_synthetic_splits(
    clean_dir: str,
    out_dir: str,
    *,
    stages: List[int] = (1, 2, 3, 4),
    val_fraction: float = 0.12,
    seed: int = 1234,
) -> Dict[str, Any]:
    """Carve stratified train/val splits from cleaned per-stage synthetic files.

    For each stage file ``epoch_N_Ncall.jsonl`` we deterministically hold out
    ``val_fraction`` of rows as validation. Stratification is by stage (== num
    gold calls), which is the dominant difficulty axis in the synthetic data.

    Writes, under ``out_dir``:
      synthetic_train_stageN.jsonl / synthetic_val_stageN.jsonl  (per stage)
      synthetic_train.jsonl / synthetic_val.jsonl                (concatenated)
      splits_manifest.json                                       (counts + md5)

    Returns the manifest dict. Pure file IO — safe to call locally.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    manifest: Dict[str, Any] = {
        "seed": seed,
        "val_fraction": val_fraction,
        "clean_dir": clean_dir,
        "stages": list(stages),
        "per_stage": [],
        "note": "Validation is SYNTHETIC and held out from training. "
                "nestful_data.jsonl stays the clean test set.",
    }
    all_train_path = os.path.join(out_dir, "synthetic_train.jsonl")
    all_val_path = os.path.join(out_dir, "synthetic_val.jsonl")
    train_all: List[Dict[str, Any]] = []
    val_all: List[Dict[str, Any]] = []

    for n in stages:
        src = os.path.join(clean_dir, f"epoch_{n}_{n}call.jsonl")
        if not os.path.isfile(src):
            manifest["per_stage"].append({"stage": n, "missing": src})
            continue
        rows = _read_rows(src)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(rows) * val_fraction))) if rows else 0
        val_idx = set(idx[:n_val])
        tr = [rows[i] for i in range(len(rows)) if i not in val_idx]
        va = [rows[i] for i in range(len(rows)) if i in val_idx]
        tp = os.path.join(out_dir, f"synthetic_train_stage{n}.jsonl")
        vp = os.path.join(out_dir, f"synthetic_val_stage{n}.jsonl")
        with open(tp, "w", encoding="utf-8") as f:
            for r in tr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(vp, "w", encoding="utf-8") as f:
            for r in va:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        train_all.extend(tr)
        val_all.extend(va)
        manifest["per_stage"].append({
            "stage": n, "source": src, "total": len(rows),
            "train": len(tr), "val": len(va),
            "train_file": tp, "val_file": vp,
            "train_md5": _file_md5(tp), "val_md5": _file_md5(vp),
        })

    with open(all_train_path, "w", encoding="utf-8") as f:
        for r in train_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(all_val_path, "w", encoding="utf-8") as f:
        for r in val_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest["train_total"] = len(train_all)
    manifest["val_total"] = len(val_all)
    manifest["train_file"] = all_train_path
    manifest["val_file"] = all_val_path
    manifest["train_md5"] = _file_md5(all_train_path)
    manifest["val_md5"] = _file_md5(all_val_path)
    with open(os.path.join(out_dir, "splits_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
