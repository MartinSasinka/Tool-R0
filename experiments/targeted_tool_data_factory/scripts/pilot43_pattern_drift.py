"""How far the stored candidate pattern labels drifted from a fresh classification.

Selection strata are built from the labels written at generation time; the export
recomputes them from the rebuilt program. Any drift between the two means tasks
were placed in strata under a label the export no longer carries.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    ap.add_argument("--file", default="train_master_5000.jsonl")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    stored: dict[str, str] = {}
    for name in ("semantic_hard_valid.jsonl", "semantic_candidates.jsonl"):
        path = out / name
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                stored.setdefault(rec["task_id"], rec.get("actual_primary_pattern", ""))
        break

    drift = collections.Counter()
    n = matched = 0
    with open(out / args.file, "r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            was = stored.get(rec["task_id"])
            if was is None:
                continue
            n += 1
            now = rec["declared"]["structural_pattern"]
            if was == now:
                matched += 1
            else:
                drift[f"{was} -> {now}"] += 1

    print(f"{matched}/{n} tasks kept their generation-time primary pattern")
    for change, count in drift.most_common(20):
        print(f"  {count:5d}  {change}")

    requested_lost = 0
    with open(out / args.file, "r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            d = rec["declared"]
            if d["requested_structural_skill"] not in d["satisfied_patterns"]:
                requested_lost += 1
    print(f"\n{requested_lost} tasks no longer satisfy their requested skill")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
