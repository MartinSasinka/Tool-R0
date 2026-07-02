"""nestful_core — single import surface shared by the minimal and partial experiments.

Why this package exists
-----------------------
``nestful_mtgrpo_minimal`` and ``nestful_mtgrpo_partial`` historically duplicated /
cross-imported parser, executor, rollout, prompt, scorer and reward logic. That
made it possible for the two experiments to silently drift apart. ``nestful_core``
makes the shared logic a single source of truth:

* ``parser`` / ``executor`` / ``rollout`` / ``prompt`` / ``scoring`` / ``data`` /
  ``eval_loop`` re-export the *canonical* implementations that live in
  ``nestful_mtgrpo_minimal`` (the proven, reproducible training path). They are
  thin re-export shims — there is exactly ONE implementation, referenced here.
* ``rewards`` is the only genuinely new module: it hosts the explicit reward
  predicates, the LEGACY reward policies (delegating to the frozen
  ``reward.py`` / ``partial_reward.py`` / ``execution_reward.py`` so old numbers
  reproduce bit-for-bit) AND the new ``partial_gold_trace_v2`` /
  ``execution_aware_v2`` policies.
* ``logging_utils`` provides CSV hygiene + reward-component logging helpers.

Importing this package puts the minimal and partial experiment folders on
``sys.path`` so the canonical bare-name modules (``parser``, ``executor`` …) and
the legacy reward modules resolve, regardless of the caller's CWD.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.dirname(_HERE)
MINIMAL_DIR = os.path.join(_EXPERIMENTS, "nestful_mtgrpo_minimal")
PARTIAL_DIR = os.path.join(_EXPERIMENTS, "nestful_mtgrpo_partial")


def ensure_paths() -> None:
    """Idempotently add the minimal + partial experiment dirs to ``sys.path``.

    The minimal dir is the canonical home of parser/executor/rollout/prompt/
    reward/metrics/nestful_official_score; the partial dir holds the legacy
    partial + execution-aware rewards. Both are imported by bare name elsewhere.
    """
    for d in (MINIMAL_DIR, PARTIAL_DIR):
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)


ensure_paths()

__all__ = ["ensure_paths", "MINIMAL_DIR", "PARTIAL_DIR"]
