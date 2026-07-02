"""Matplotlib helpers for publication figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def save_figure(fig, path_stem: Path, cfg: Dict[str, Any]) -> List[str]:
    fig_cfg = cfg.get("figures") or {}
    formats = fig_cfg.get("formats") or ["png", "pdf"]
    dpi = fig_cfg.get("dpi", 300)
    saved = []
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = path_stem.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        saved.append(str(out))
    plt.close(fig)
    return saved


def checkpoint_colors(order: List[str]) -> Dict[str, str]:
    palette = [
        "#4C72B0",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#CCB974",
        "#64B5CD",
        "#937860",
        "#DA8BC3",
    ]
    return {cp: palette[i % len(palette)] for i, cp in enumerate(order)}
