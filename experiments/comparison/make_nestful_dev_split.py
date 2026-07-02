#!/usr/bin/env python3
"""Carve a held-out REAL NESTFUL dev subset that is DISJOINT from the test set.

The repo only ships one real NESTFUL file (`nestful_data.jsonl`, 1861 items) which
was previously used both as the eval/test set AND ignored during training (training
validated on synthetic data). That let the reward drift away from real-task success
with no corrective signal (ROOT_CAUSE_ANALYSIS.md #2).

This script deterministically partitions `nestful_data.jsonl` by `sample_id` into:
  - nestful_dev.jsonl   (small held-out dev, default 200) — for validation/selection
  - nestful_test.jsonl  (the remaining items)             — for final reporting ONLY

The split is stratified by gold call count (len(output)) so dev is representative,
and is seed-deterministic and idempotent. dev and test sample_ids are guaranteed
disjoint. A manifest records the seed, sizes, and the exact id lists.

Usage:
  python experiments/comparison/make_nestful_dev_split.py            # default 200 dev
  python experiments/comparison/make_nestful_dev_split.py --dev-size 250 --seed 7
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_MINIMAL = os.path.join(_HERE, "..", "nestful_mtgrpo_minimal")
_DEFAULT_SRC = os.path.join(
    _MINIMAL, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl"
)
_DEFAULT_OUTDIR = os.path.join(_MINIMAL, "data", "splits")


def _read_rows(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sid(row: dict) -> str:
    return str(row.get("sample_id") or row.get("task_id") or row.get("id"))


def _gold_calls(row: dict) -> int:
    out = row.get("output")
    return len(out) if isinstance(out, list) else 0


def make_split(src: str, outdir: str, dev_size: int, seed: int) -> Dict[str, object]:
    rows = _read_rows(src)
    # Deterministic order independent of file ordering.
    rows.sort(key=_sid)

    # Stratify by gold call count so dev mirrors the corpus difficulty mix.
    buckets: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        buckets[_gold_calls(r)].append(r)

    rng = random.Random(seed)
    total = len(rows)
    dev_size = max(0, min(dev_size, total))
    dev_rows: List[dict] = []
    for k in sorted(buckets):
        b = list(buckets[k])
        rng.shuffle(b)
        take = round(dev_size * len(b) / total)
        dev_rows.extend(b[:take])

    # Adjust to hit dev_size exactly (rounding may over/undershoot).
    dev_ids = {_sid(r) for r in dev_rows}
    if len(dev_rows) > dev_size:
        dev_rows = dev_rows[:dev_size]
        dev_ids = {_sid(r) for r in dev_rows}
    elif len(dev_rows) < dev_size:
        for r in rows:
            if _sid(r) not in dev_ids:
                dev_rows.append(r)
                dev_ids.add(_sid(r))
                if len(dev_rows) >= dev_size:
                    break

    dev_ids = {_sid(r) for r in dev_rows}
    test_rows = [r for r in rows if _sid(r) not in dev_ids]

    # Hard invariant: dev and test must be disjoint and cover everything.
    test_ids = {_sid(r) for r in test_rows}
    assert dev_ids.isdisjoint(test_ids), "dev/test overlap!"
    assert len(dev_ids) + len(test_ids) == total, "split does not cover corpus"

    os.makedirs(outdir, exist_ok=True)
    dev_path = os.path.join(outdir, "nestful_dev.jsonl")
    test_path = os.path.join(outdir, "nestful_test.jsonl")
    with open(dev_path, "w", encoding="utf-8") as fh:
        for r in dev_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(test_path, "w", encoding="utf-8") as fh:
        for r in test_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "source": os.path.abspath(src),
        "seed": seed,
        "total": total,
        "dev_size": len(dev_rows),
        "test_size": len(test_rows),
        "dev_path": os.path.abspath(dev_path),
        "test_path": os.path.abspath(test_path),
        "dev_gold_call_hist": _hist(dev_rows),
        "test_gold_call_hist": _hist(test_rows),
        "dev_sample_ids": sorted(dev_ids),
    }
    with open(os.path.join(outdir, "nestful_devtest_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest


def _hist(rows: List[dict]) -> Dict[str, int]:
    h: Dict[int, int] = defaultdict(int)
    for r in rows:
        h[_gold_calls(r)] += 1
    return {str(k): h[k] for k in sorted(h)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Carve held-out NESTFUL dev split.")
    ap.add_argument("--src", default=_DEFAULT_SRC)
    ap.add_argument("--outdir", default=_DEFAULT_OUTDIR)
    ap.add_argument("--dev-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    m = make_split(args.src, args.outdir, args.dev_size, args.seed)
    print(json.dumps({k: v for k, v in m.items() if k != "dev_sample_ids"},
                     indent=2, ensure_ascii=False))
    print(f"[make_nestful_dev_split] dev -> {m['dev_path']}")
    print(f"[make_nestful_dev_split] test -> {m['test_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
