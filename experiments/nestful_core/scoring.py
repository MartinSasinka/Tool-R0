"""Canonical official-scorer wrapper (re-export of minimal/nestful_official_score.py).

This is the ONLY sanctioned path to the real NESTFUL scorer
(``data/NESTFUL-main/src/scorer.py``). All per-sample and aggregate official Win
numbers must go through here so they cannot diverge.
"""
from __future__ import annotations

from . import ensure_paths

ensure_paths()

from nestful_official_score import (  # noqa: E402,F401
    add_labels,
    build_item,
    load_raw_dataset,
    rescore_direct_predictions,
    rescore_trajectories,
    score_items,
    score_items_per_sample,
)
