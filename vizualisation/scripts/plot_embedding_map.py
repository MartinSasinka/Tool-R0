#!/usr/bin/env python3
"""Exploratory 2D embedding scatter plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vizualisation.scripts.lib.io_utils import checkpoints_order, load_config, log, run_dir_from_config  # noqa: E402
from vizualisation.scripts.lib.plotting_utils import apply_style, checkpoint_colors, save_figure  # noqa: E402


def scatter_by(df, color_col, title, out_stem, cfg, run_dir, categorical=False):
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    if categorical:
        for val in sorted(df[color_col].dropna().unique(), key=str):
            sub = df[df[color_col] == val]
            ax.scatter(sub["x"], sub["y"], s=8, alpha=0.25, label=str(val))
        ax.legend(fontsize=7, markerscale=2)
    else:
        sc = ax.scatter(df["x"], df["y"], c=df[color_col].astype(float), s=8, alpha=0.25, cmap="viridis")
        plt.colorbar(sc, ax=ax, label=color_col)
    ax.set_xlabel("PC1 (exploratory projection)")
    ax.set_ylabel("PC2 (exploratory projection)")
    ax.set_title(title)
    return save_figure(fig, run_dir / "figures" / out_stem, cfg)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run_dir", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else run_dir_from_config(cfg)
    embed_path = run_dir / "embedding_2d.csv"
    if not embed_path.is_file():
        log("plot", f"ERROR: missing {embed_path}")
        return 2

    df = pd.read_csv(embed_path)
    order = checkpoints_order(cfg) + ["gold"]
    colors = checkpoint_colors(order)
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    for cp in order:
        sub = df[df["checkpoint"] == cp]
        if sub.empty:
            continue
        ax.scatter(sub["x"], sub["y"], s=8, alpha=0.25, label=cp, color=colors.get(cp))
    ax.set_xlabel("PC1 (exploratory projection)")
    ax.set_ylabel("PC2 (exploratory projection)")
    ax.set_title("Trajectory representation space by checkpoint (exploratory)")
    ax.legend(fontsize=8, markerscale=2)
    save_figure(fig, run_dir / "figures" / "embedding_by_checkpoint", cfg)

    for col, stem, cat in [
        ("valid_json", "embedding_by_valid_json", False),
        ("reference_score", "embedding_by_reference_score", False),
        ("dependency_depth_score", "embedding_by_dependency_depth", False),
        ("final_answer_score", "embedding_by_final_answer", False),
        ("error_type", "embedding_by_error_type", True),
    ]:
        if col in df.columns:
            scatter_by(
                df,
                col,
                f"Embedding colored by {col} (exploratory)",
                stem,
                cfg,
                run_dir,
                categorical=cat,
            )
    log("plot", "saved embedding figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
