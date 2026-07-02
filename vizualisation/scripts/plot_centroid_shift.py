#!/usr/bin/env python3
"""Plot checkpoint centroid shift in feature and 2D space."""

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
from vizualisation.scripts.lib.plotting_utils import apply_style, checkpoint_colors, save_figure  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run_dir", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else run_dir_from_config(cfg)
    embed_path = run_dir / "embedding_2d.csv"
    std_path = run_dir / "feature_matrix_standardized.csv"
    if not embed_path.is_file() or not std_path.is_file():
        log("plot", "ERROR: missing embedding or standardized features")
        return 2

    embed = pd.read_csv(embed_path)
    std = pd.read_csv(std_path)
    order = checkpoints_order(cfg) + ["gold"]
    colors = checkpoint_colors(order)

    centroids_2d = {}
    centroids_feat = {}
    for cp in order:
        e = embed[embed["checkpoint"] == cp]
        s = std[std["checkpoint"] == cp]
        if len(e):
            centroids_2d[cp] = (e["x"].mean(), e["y"].mean())
        if len(s):
            cols = [c for c in FEATURE_COLUMNS if c in s.columns]
            centroids_feat[cp] = s[cols].mean().values

    gold_vec = centroids_feat.get("gold")
    dist_rows = []
    model_cps = [c for c in order if c != "gold"]
    for i, cp in enumerate(model_cps):
        vec = centroids_feat.get(cp)
        if vec is None:
            continue
        if gold_vec is not None:
            dist_rows.append(
                {
                    "checkpoint": cp,
                    "feature_l2_distance_to_gold_centroid": float(np.linalg.norm(vec - gold_vec)),
                }
            )
        if i > 0:
            prev = model_cps[i - 1]
            prev_vec = centroids_feat.get(prev)
            if prev_vec is not None:
                dist_rows.append(
                    {
                        "transition": f"{prev} -> {cp}",
                        "consecutive_feature_l2_distance": float(np.linalg.norm(vec - prev_vec)),
                    }
                )

    pd.DataFrame(dist_rows).to_csv(run_dir / "centroid_distances.csv", index=False)

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    for cp in model_cps:
        if cp not in centroids_2d:
            continue
        x, y = centroids_2d[cp]
        ax.scatter([x], [y], s=120, color=colors.get(cp), label=cp, zorder=3)
        ax.annotate(cp, (x, y), fontsize=8)
    if "gold" in centroids_2d:
        gx, gy = centroids_2d["gold"]
        ax.scatter([gx], [gy], s=150, color="black", marker="*", label="gold", zorder=4)
    pts = [centroids_2d[c] for c in model_cps if c in centroids_2d]
    if len(pts) > 1:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("PC1 (exploratory projection)")
    ax.set_ylabel("PC2 (exploratory projection)")
    ax.set_title("Checkpoint centroid shift (2D exploratory view)")
    ax.legend(fontsize=8)
    saved = save_figure(fig, run_dir / "figures" / "centroid_shift", cfg)
    log("plot", f"saved {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
