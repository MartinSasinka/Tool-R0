"""Build a small stand-in run directory so the downstream stages can be exercised.

Selection, export, gates, audit, the human package and the freeze all read the
same files as the real run. Copying a few hundred verified tasks into a scratch
directory lets every one of them be run for real, quickly, without touching the
run being rendered.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

FULL_COPIES = ("target_profile_v3.json", "workflow_registry_v3.json",
               "primitive_registry_v3.json")
SUBSET = ("verified_candidates.jsonl", "query_render_shortlist.jsonl",
          "per_task_validation_ledger.jsonl", "v4_per_task.jsonl",
          "per_node_necessity.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="outputs/pilot4_3_nestful_final")
    ap.add_argument("--dest", default="outputs/_p43_dryrun")
    ap.add_argument("--tasks", type=int, default=900)
    args = ap.parse_args()

    src, dst = pathlib.Path(args.source), pathlib.Path(args.dest)
    dst.mkdir(parents=True, exist_ok=True)

    keep: set[str] = set()
    with (src / "verified_candidates.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("selectable"):
                keep.add(row["task_id"])
            if len(keep) >= args.tasks:
                break
    print(f"keeping {len(keep)} selectable task ids")

    for name in FULL_COPIES:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)

    for name in SUBSET:
        path = src / name
        if not path.is_file():
            print(f"  {name}: missing at source")
            continue
        written = 0
        with path.open(encoding="utf-8") as fh, \
                (dst / name).open("w", encoding="utf-8", newline="\n") as out:
            for line in fh:
                row = json.loads(line)
                if row.get("task_id") in keep:
                    out.write(line if line.endswith("\n") else line + "\n")
                    written += 1
        print(f"  {name}: {written} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
