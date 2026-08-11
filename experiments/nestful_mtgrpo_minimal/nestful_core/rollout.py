"""Canonical rollout / trajectory schema (re-export of minimal/rollout.py)."""
from __future__ import annotations

from . import ensure_paths

ensure_paths()

from rollout import (  # noqa: E402,F401
    Trajectory,
    Turn,
    generate_once,
    get_stage_token_budget,
    run_episode,
)
