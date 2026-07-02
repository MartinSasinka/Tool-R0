#!/usr/bin/env python3
"""Plot per-stage component gains."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vizualisation.scripts.lib.io_utils import COMPONENT_LABELS, load_config, log, run_dir_from_config  # noqa: E402
from vizualisation.scripts.lib.plotting_utils import apply_style, save_figure  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run_dir", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else run_dir_from_config(cfg)
    gain_path = run_dir / "gain_summary.csv"
    if not gain_path.is_file():
        log("plot", f"ERROR: missing {gain_path}")
        return 2

    df = pd.read_csv(gain_path)
    if df.empty:
        log("plot", "WARNING: empty gain summary; skipping figure")
        return 0

    transitions = sorted(df["transition"].unique())
    components = sorted(df["component"].unique(), key=lambda c: list(COMPONENT_LABELS.keys()).index(c) if c in COMPONENT_LABELS else 999)
    apply_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / max(len(transitions), 1)
    for i, tr in enumerate(transitions):
        sub = df[df["transition"] == tr]
        ys = [float(sub[sub["component"] == c]["gain"].iloc[0]) if len(sub[sub["component"] == c]) else 0 for c in components]
        xs = [j + i * width for j in range(len(components))]
        ax.bar(xs, ys, width=width, label=tr)

    ax.set_xticks([j + width * (len(transitions) - 1) / 2 for j in range(len(components))])
    ax.set_xticklabels([COMPONENT_LABELS.get(c, c) for c in components], rotation=35, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Gain (delta mean score)")
    ax.set_title("Checkpoint gains by curriculum stage")
    ax.legend(fontsize=8)
    saved = save_figure(fig, run_dir / "figures" / "checkpoint_gains", cfg)
    log("plot", f"saved {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
