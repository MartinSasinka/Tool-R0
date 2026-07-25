#!/usr/bin/env python3
"""Prepare the 24-task Stage-3 subset for the dispatch canary.

Takes the first 24 rows of the frozen Round-1 train subset (same order /
same seed provenance). Both canary arms must train on this exact file.

Usage (repo root or v3 dir):
  python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/prepare_canary_subset_24.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
SRC = _V3 / "reports" / "reward_ablation" / "data" / "train_subset_160.jsonl"
OUT_DIR = _V3 / "reports" / "reward_ablation" / "data"
OUT = OUT_DIR / "canary_subset_24.jsonl"
MANIFEST = OUT_DIR / "canary_subset_24_manifest.json"
N = 24


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not SRC.is_file():
        raise SystemExit(f"missing {SRC} — run prepare_train_subset_160.py first")
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(rows) < N:
        raise SystemExit(f"train subset has {len(rows)} rows; need >= {N}")
    selected = rows[:N]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in selected:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": "dispatch_canary_subset_24",
        "n_selected": N,
        "source": str(SRC),
        "source_sha256": _sha256_file(SRC),
        "out": str(OUT),
        "out_sha256": _sha256_file(OUT),
        "sample_ids": [r.get("sample_id") or r.get("task_id") for r in selected],
        "note": "First 24 rows of frozen train_subset_160.jsonl; shared by A1 and A4 canary arms.",
    }
    MANIFEST.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[canary_subset_24] wrote {OUT} ({N} rows)")
    print(f"[canary_subset_24] sha256={man['out_sha256']}")
    print(f"[canary_subset_24] manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
