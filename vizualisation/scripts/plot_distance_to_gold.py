#!/usr/bin/env python3
"""Distance to gold in standardized feature space and canonical edit distance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vizualisation.scripts.lib.io_utils import FEATURE_COLUMNS, checkpoints_order, load_config, log, run_dir_from_config  # noqa: E402
from vizualisation.scripts.lib.plotting_utils import apply_style, save_figure  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run_dir", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else run_dir_from_config(cfg)
    std_path = run_dir / "feature_matrix_standardized.csv"
    metrics_path = run_dir / "trajectory_metrics.jsonl"
    if not std_path.is_file():
        log("plot", f"ERROR: missing {std_path}")
        return 2

    std = pd.read_csv(std_path)
    from vizualisation.scripts.lib.io_utils import load_jsonl_list

    metrics = pd.DataFrame(load_jsonl_list(run_dir / "trajectory_metrics.jsonl"))
    feat_cols = [c for c in FEATURE_COLUMNS if c in std.columns]
    gold = std[std["checkpoint"] == "gold"].set_index("sample_id")
    records = []
    order = [c for c in checkpoints_order(cfg) if c != "gold"]

    for cp in order:
        sub_std = std[std["checkpoint"] == cp]
        sub_met = metrics[metrics["checkpoint"] == cp]
        feat_dists = []
        for _, row in sub_std.iterrows():
            sid = row["sample_id"]
            if sid not in gold.index:
                continue
            gvec = gold.loc[sid, feat_cols].astype(float).values
            pvec = row[feat_cols].astype(float).values
            feat_dists.append(float(np.linalg.norm(pvec - gvec)))

        def _add_bucket(bucket, ng_label):
            edits = bucket["trajectory_edit_distance"].astype(float) if "trajectory_edit_distance" in bucket.columns else pd.Series(dtype=float)
            records.append(
                {
                    "checkpoint": cp,
                    "num_calls_gold": ng_label,
                    "mean_feature_l2_to_gold": float(np.mean(feat_dists)) if feat_dists else np.nan,
                    "median_canonical_edit_distance": float(edits.median()) if len(edits) else np.nan,
                    "mean_canonical_edit_distance": float(edits.mean()) if len(edits) else np.nan,
                    "se_canonical_edit_distance": float(edits.sem()) if len(edits) > 1 else 0.0,
                    "n": len(bucket),
                }
            )

        _add_bucket(sub_met, "all")
        if "num_calls_gold" in sub_met.columns:
            for ng in sorted(sub_met["num_calls_gold"].dropna().unique()):
                _add_bucket(sub_met[sub_met["num_calls_gold"] == ng], int(ng))

    dist_df = pd.DataFrame(records)
    overall = dist_df[dist_df["num_calls_gold"] == "all"] if "all" in dist_df.get("num_calls_gold", pd.Series()).values else dist_df.groupby("checkpoint").first().reset_index()
    dist_path = run_dir / "distance_to_gold.csv"
    dist_df.to_csv(dist_path, index=False)

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    if not overall.empty:
        ax.bar(overall["checkpoint"], overall["mean_feature_l2_to_gold"], alpha=0.8)
    ax.set_ylabel("Mean L2 distance to gold (standardized features)")
    ax.set_xlabel("Checkpoint")
    ax.set_title("Distance to gold trajectories by checkpoint")
    saved = save_figure(fig, run_dir / "figures" / "distance_to_gold_by_checkpoint", cfg)
    log("plot", f"saved {saved} and {dist_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
