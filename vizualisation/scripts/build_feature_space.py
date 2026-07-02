#!/usr/bin/env python3
"""Build numeric feature matrix from trajectory metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vizualisation.scripts.lib.io_utils import FEATURE_COLUMNS, load_jsonl_list, log  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "trajectory_metrics.jsonl"
    if not metrics_path.is_file():
        log("features", f"ERROR: missing {metrics_path}")
        return 2

    rows = load_jsonl_list(metrics_path)
    df = pd.DataFrame(rows)
    meta_cols = ["sample_id", "checkpoint", "rollout_idx", "error_type"]
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    out_cols = meta_cols + FEATURE_COLUMNS
    out = df[out_cols]
    out_path = run_dir / "feature_matrix.csv"
    out.to_csv(out_path, index=False)
    log("features", f"wrote {len(out)} rows x {len(FEATURE_COLUMNS)} features -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
