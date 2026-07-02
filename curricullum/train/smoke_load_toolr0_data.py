#!/usr/bin/env python3
"""Smoke test: load Tool-R0 training data with variable row counts and no nestful_repo."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Force missing IBM repo path for fallback coverage.
os.environ["NESTFUL_REPO_DIR"] = os.path.join(ROOT, "_smoke_missing_nestful_repo")

from curricullum.train.prepare_dataset_toolr0 import load_toolr0_jsonl  # noqa: E402


def main() -> int:
    data_dir = os.path.join(ROOT, "curricullum", "data", "filtered_toolr0_synthetic")
    totals = {}
    for epoch in (1, 2, 3):
        path = os.path.join(data_dir, f"epoch_{epoch}_{epoch}call.jsonl")
        if not os.path.isfile(path):
            print(f"[skip] missing {path}")
            continue
        with open(path, encoding="utf-8") as f:
            jsonl_lines = sum(1 for line in f if line.strip())
        records, stats = load_toolr0_jsonl(path, default_num_calls=epoch)
        totals[epoch] = (jsonl_lines, len(records), stats.get("ibm_fallback", 0))
        print(
            f"epoch{epoch}: jsonl_tasks={jsonl_lines} train_rows={len(records)} "
            f"ibm_fallback={stats.get('ibm_fallback', 0)}"
        )

    if not totals:
        print("[err] no epoch files found under filtered_toolr0_synthetic/")
        return 1

    e3 = totals.get(3)
    if e3 and e3[0] < 500:
        print(f"[ok] epoch3 has {e3[0]} tasks (<500) — train will use all of them")

    print("[ok] smoke load passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
