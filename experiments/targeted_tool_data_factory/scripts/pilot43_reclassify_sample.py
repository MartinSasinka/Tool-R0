"""Reclassify stored candidates with the current pattern rules and report drift."""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random

from targeted_tool_data.pilot43.tasks import rebuild


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    ap.add_argument("--file", default="semantic_hard_valid.jsonl")
    ap.add_argument("--sample", type=int, default=400)
    args = ap.parse_args()

    path = pathlib.Path(args.out_dir) / args.file
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    rng = random.Random(7)
    if len(rows) > args.sample:
        rows = rng.sample(rows, args.sample)

    drift = collections.Counter()
    lost = 0
    same = 0
    failed = 0
    for row in rows:
        try:
            inst, _bp, _plan = rebuild(row)
        except Exception:  # noqa: BLE001
            failed += 1
            continue
        was = row.get("actual_primary_pattern", "")
        now = inst.actual_primary_pattern
        if was == now:
            same += 1
        else:
            drift[f"{was} -> {now}"] += 1
        if was and was not in inst.actual_patterns:
            lost += 1

    n = same + sum(drift.values())
    print(f"{same}/{n} kept their primary label ({failed} could not be rebuilt)")
    print(f"{lost}/{n} no longer satisfy the label they were generated under")
    for change, count in drift.most_common(20):
        print(f"  {count:5d}  {change}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
