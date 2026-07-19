#!/usr/bin/env python3
"""Materialize the pure Stage 3 training-ready JSONL (326 rows, no Stage 2).

Source of truth: phase2 mix = 326× stage3 + 140× stage2 replay. This script
extracts only Stage 3 rows into
``data/training_ready_v5/filtered/stage3_train_ready.jsonl`` without rewriting
the original phase2 file.

Usage:
  python scripts/data/materialize_pure_stage3.py
  python scripts/data/materialize_pure_stage3.py --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3 = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PHASE2 = os.path.join(
    _V3, "data", "training_ready_v5", "filtered",
    "phase2_stage3_plus_stage2_replay.jsonl")
_OUT = os.path.join(
    _V3, "data", "training_ready_v5", "filtered", "stage3_train_ready.jsonl")
_MANIFEST = os.path.join(
    _V3, "data", "training_ready_v5", "manifests",
    "stage3_train_ready_manifest.json")

EXPECTED_ROWS = 326


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=_PHASE2)
    ap.add_argument("--out", default=_OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.source):
        raise SystemExit(f"[materialize] missing source: {args.source}")

    if os.path.isfile(args.out) and not args.force:
        n = sum(1 for line in open(args.out, encoding="utf-8") if line.strip())
        print(f"[materialize] exists ({n} rows): {args.out}")
        if n != EXPECTED_ROWS:
            raise SystemExit(
                f"[materialize] ABORT: existing file has {n} rows, "
                f"expected {EXPECTED_ROWS}; use --force")
        print(f"[materialize] sha256={_sha256(args.out)}")
        return 0

    rows = []
    with open(args.source, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stage = str(row.get("stage") or "")
            if "stage3" in stage and "stage2" not in stage:
                rows.append(row)
            elif stage.startswith("stage3"):
                rows.append(row)

    # Prefer explicit stage3_3call marker; exclude anything with stage2.
    cleaned = []
    for row in rows:
        stage = str(row.get("stage") or "")
        if "stage2" in stage:
            continue
        if "stage3" not in stage:
            continue
        n_calls = row.get("num_calls")
        gold = row.get("gold_calls") or []
        if n_calls != 3 or len(gold) != 3:
            raise SystemExit(
                f"[materialize] ABORT: {row.get('sample_id')} "
                f"num_calls={n_calls} len(gold)={len(gold)}")
        cleaned.append(row)

    if len(cleaned) != EXPECTED_ROWS:
        raise SystemExit(
            f"[materialize] ABORT: extracted {len(cleaned)} Stage 3 rows, "
            f"expected {EXPECTED_ROWS}")

    ids = [str(r.get("sample_id")) for r in cleaned]
    if len(ids) != len(set(ids)):
        raise SystemExit("[materialize] ABORT: duplicate sample_ids")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in cleaned:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    digest = _sha256(args.out)
    man = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": os.path.abspath(args.source),
        "source_sha256": _sha256(args.source),
        "out": os.path.abspath(args.out),
        "sha256": digest,
        "rows": len(cleaned),
        "expected_rows": EXPECTED_ROWS,
        "stage_filter": "stage3_* excluding any stage2",
        "note": "Derived from phase2 mix; original phase2 file unchanged.",
    }
    os.makedirs(os.path.dirname(_MANIFEST), exist_ok=True)
    with open(_MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, ensure_ascii=False)

    print(f"[materialize] wrote {len(cleaned)} rows -> {args.out}")
    print(f"[materialize] sha256={digest}")
    print(f"[materialize] manifest -> {_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
