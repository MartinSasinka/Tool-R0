"""Shared helpers for the self-contained MT-GRPO training loop.

The package lives inside ``nestful_mtgrpo_minimal`` and never reaches into a
sibling experiment.  The containing directory is added to ``sys.path`` only so
the original bare-module imports used by the trainer remain compatible.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
MINIMAL_DIR = os.path.dirname(_HERE)


def ensure_paths() -> None:
    """Idempotently make the containing training directory importable."""
    if MINIMAL_DIR not in sys.path:
        sys.path.insert(0, MINIMAL_DIR)


ensure_paths()

__all__ = ["ensure_paths", "MINIMAL_DIR"]
