#!/usr/bin/env python3
"""Error type distribution by checkpoint."""

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
from vizualisation.scripts.lib.plotting_utils import apply_style, save_figure  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run_dir", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else run_dir_from_config(cfg)
    from vizualisation.scripts.lib.io_utils import load_jsonl_list

    metrics = pd.DataFrame(load_jsonl_list(run_dir / "trajectory_metrics.jsonl"))
    model = metrics[metrics["checkpoint"] != "gold"]
    order = checkpoints_order(cfg)
    pivot = pd.crosstab(model["checkpoint"], model["error_type"])
    pivot = pivot.reindex(order).fillna(0)

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("Count (rollout rows)")
    ax.set_xlabel("Checkpoint")
    ax.set_title("Error type distribution by checkpoint")
    ax.legend(bbox_to_anchor=(1.02, 1), fontsize=7)
    saved = save_figure(fig, run_dir / "figures" / "error_type_distribution_by_checkpoint", cfg)
    log("plot", f"saved {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
