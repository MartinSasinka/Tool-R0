#!/usr/bin/env python3
"""Plot mean component scores by checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vizualisation.scripts.lib.io_utils import COMPONENT_LABELS, checkpoints_order, load_config, log, run_dir_from_config  # noqa: E402
from vizualisation.scripts.lib.plotting_utils import apply_style, checkpoint_colors, save_figure  # noqa: E402

PLOT_COMPONENTS = [
    "format_score",
    "call_count_score",
    "tool_name_score",
    "label_score",
    "argument_key_score",
    "argument_value_score",
    "reference_score",
    "dependency_depth_score",
    "final_answer_score",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run_dir", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else run_dir_from_config(cfg)
    summary_path = run_dir / "checkpoint_summary.csv"
    if not summary_path.is_file():
        log("plot", f"ERROR: missing {summary_path}")
        return 2

    df = pd.read_csv(summary_path)
    primary = df[df.get("summary_policy", pd.Series()) == "best_score"]
    if primary.empty:
        primary = df[df["aggregation_level"] == "sample_level"] if "aggregation_level" in df.columns else df
    if primary.empty:
        primary = df.iloc[:1]

    order = [c for c in checkpoints_order(cfg) if c in primary["checkpoint"].values]
    colors = checkpoint_colors(order)
    apply_style()

    labels = [COMPONENT_LABELS.get(c, c) for c in PLOT_COMPONENTS]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    for cp in order:
        row = primary[primary["checkpoint"] == cp]
        if row.empty:
            continue
        row = row.iloc[0]
        ys = [float(row.get(c, 0)) for c in PLOT_COMPONENTS]
        ax.plot(x, ys, marker="o", label=cp, color=colors.get(cp))

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mean component score")
    ax.set_xlabel("Behavioral component")
    ax.set_title("Skill profile by checkpoint (sample-level, best_score)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    saved = save_figure(fig, run_dir / "figures" / "skill_profile_by_checkpoint", cfg)
    log("plot", f"saved {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
