#!/usr/bin/env python3
"""Seed pilot3 candidate + validated pools from frozen pilot2 (no re-V4).

Pilot2 artefacts are only *read*. Pilot3 files are overwritten. Already-validated
pilot2 rows skip the expensive V4 minimal-path search on the next validate
pass: ``run_pilot3.py --from-pilot2`` validates only newly generated rows.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

FACTORY = Path(__file__).resolve().parents[1]
OUT = FACTORY / "outputs"


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="pilot2")
    ap.add_argument("--version", default="pilot3")
    args = ap.parse_args()

    # Prefer the full validated pool (more cells to draw from); fall back to
    # selected if validated is missing.
    src_val = OUT / "validated" / f"validated_{args.baseline}.jsonl"
    src_sel = OUT / "selected" / f"selected_{args.baseline}.jsonl"
    src_cells = OUT / "candidates" / f"cells_{args.baseline}.json"
    if not src_val.is_file():
        raise SystemExit(f"missing {src_val}")
    rows = read_jsonl(src_val)
    if not rows and src_sel.is_file():
        rows = read_jsonl(src_sel)
    if not rows:
        raise SystemExit("no pilot2 rows to seed")

    # Tag provenance; keep original task_ids so leakage/dedup stay consistent.
    for r in rows:
        prov = dict(r.get("provenance") or {})
        prov["seeded_from"] = args.baseline
        r["provenance"] = prov
        r["pilot3_seed"] = True

    dst_cand = OUT / "candidates" / f"candidates_{args.version}.jsonl"
    dst_val = OUT / "validated" / f"validated_{args.version}.jsonl"
    dst_cells = OUT / "candidates" / f"cells_{args.version}.json"
    write_jsonl(dst_cand, rows)
    write_jsonl(dst_val, rows)

    if src_cells.is_file():
        shutil.copyfile(src_cells, dst_cells)

    # Mark validate as DONE for the seeded rows so a later incremental validate
    # can merge; generate will append with start_index offset.
    write_json(OUT / "validated" / f"_validate_{args.version}.DONE.json",
               {"step": f"validate_{args.version}", "n": len(rows),
                "seeded_from": args.baseline, "skip_v4_for_seed": True})
    write_json(OUT / "candidates" / f"_generate_{args.version}.DONE.json",
               {"step": f"generate_{args.version}", "n": len(rows),
                "seeded_from": args.baseline})
    write_json(OUT / "candidates" / f"gen_stats_{args.version}.json", {
        "n_generated": len(rows),
        "cells": {},
        "seeded_from": args.baseline,
        "note": "seed only; expand via generate --append",
    })
    # Clear downstream DONE markers so select/split/export re-run on the
    # expanded pool.
    for marker in (
        OUT / "selected" / f"_select_{args.version}.DONE.json",
        OUT / "selected" / f"_probe_{args.version}.DONE.json",
        OUT / "selected" / f"_export_{args.version}.DONE.json",
        OUT / "splits" / f"_split_{args.version}.DONE.json",
        OUT / "validated" / f"_paraphrase_{args.version}.DONE.json",
    ):
        if marker.is_file():
            marker.unlink()

    ids_path = OUT / "candidates" / f"seed_ids_{args.version}.json"
    write_json(ids_path, {"n": len(rows),
                          "task_ids": [r["task_id"] for r in rows]})
    print(f"[seed] {len(rows)} {args.baseline} rows -> "
          f"candidates/validated_{args.version} (V4 skipped for these)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
