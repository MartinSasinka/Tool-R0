"""Canonical eval loop (re-export of minimal run_episode) + thin helpers."""
from __future__ import annotations

from typing import Any, Dict, List

from . import ensure_paths

ensure_paths()

from rollout import run_episode  # noqa: E402,F401


def max_turns_for(task: Dict[str, Any], *, train: bool) -> int:
    """v2 turn budget: train and eval both get gold_n + 1, hard-capped at gold_n+4.

    Fixes the legacy mismatch where training used ``max_turns = gold_n`` (forcing
    exactly gold-length behaviour) while eval allowed ``gold_n + 1``.
    """
    gold_n = int(task.get("num_calls") or len(task.get("gold_calls", [])))
    return max(1, min(gold_n + 1, gold_n + 4))
