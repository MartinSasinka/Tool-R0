"""Canonical executor (re-export of nestful_mtgrpo_minimal/executor.py)."""
from __future__ import annotations

from . import ensure_paths

ensure_paths()

from executor import (  # noqa: E402,F401
    ExecResult,
    IBMFunctionRegistry,
    MalformedToolCallError,
    ToolExecutor,
    coerce_numeric,
    detect_ibm_functions_dir,
    matches_gold,
    normalize_arguments,
    resolve_variables,
)
