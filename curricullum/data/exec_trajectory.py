#!/usr/bin/env python3
"""Execute NESTFUL gold/predicted tool trajectories via IBM helpers."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nestful_evaluation.run import (  # noqa: E402
    DEFAULT_REPO_URL,
    IBMFunctionRegistry,
    _matches_gold,
    _normalize_call,
    ensure_ibm_repo,
    execute_one,
)

_IBM_WARNED = False


def _warn_once(msg: str) -> None:
    global _IBM_WARNED
    if not _IBM_WARNED:
        print(f"[exec_trajectory] WARNING: {msg}", file=sys.stderr)
        _IBM_WARNED = True


def ensure_nestful_repo(repo_dir: Optional[str] = None) -> str:
    """Best-effort: return repo path; clone IBM/NESTFUL if missing."""
    repo_dir = repo_dir or os.environ.get("NESTFUL_REPO_DIR", "nestful_repo")
    if ensure_ibm_repo(repo_dir):
        return repo_dir

    import subprocess

    parent = os.path.dirname(os.path.abspath(repo_dir)) or "."
    os.makedirs(parent, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", DEFAULT_REPO_URL, repo_dir],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        _warn_once(
            f"nestful_repo unavailable at {repo_dir!r} ({exc}). "
            "Training will use gold_answer fallbacks for exec reward."
        )
        return repo_dir

    if not ensure_ibm_repo(repo_dir):
        _warn_once(
            f"nestful_repo clone at {repo_dir!r} is incomplete. "
            "Training will use gold_answer fallbacks for exec reward."
        )
    return repo_dir


def _coerce_calls(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, dict):
            norm = _normalize_call(item)
            if norm:
                out.append(norm)
            elif item.get("name"):
                out.append(item)
    return out


def get_ibm_registry(repo_dir: Optional[str] = None) -> Optional[IBMFunctionRegistry]:
    repo_dir = ensure_nestful_repo(repo_dir)
    if not ensure_ibm_repo(repo_dir):
        return None
    return IBMFunctionRegistry(repo_dir)


_JSON_SCALARS = (str, int, float, bool, type(None))


def _coerce_result(value: Any) -> Any:
    """Coerce IBM executor output to a JSON-serializable scalar or string.

    IBM helpers can return datetime, numpy scalars, Decimal, custom objects,
    etc. We keep the raw value for in-memory use (reward comparison via
    _matches_gold), but callers that need to json.dumps should use this.
    """
    if isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, list):
        return [_coerce_result(v) for v in value]
    if isinstance(value, dict):
        return {k: _coerce_result(v) for k, v in value.items()}
    # numpy / torch scalars
    try:
        import numpy as np
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    # fallback: str representation
    return str(value)


def execute_trajectory(
    calls: List[Dict[str, Any]],
    *,
    ibm_registry: Optional[IBMFunctionRegistry] = None,
    stop_on_error: bool = True,
) -> Tuple[Optional[Any], List[Dict[str, Any]], Optional[str]]:
    """Run calls in order. Returns (final_value, trace_dicts, error).

    final_value is always JSON-serializable (coerced via _coerce_result).
    """
    if ibm_registry is None:
        ibm_registry = get_ibm_registry()
    if ibm_registry is None:
        return None, [], "ibm_registry_unavailable"

    by_label: Dict[str, Any] = {}
    indexed: List[Any] = []
    traces: List[Dict[str, Any]] = []
    final_value: Any = None

    for i, call in enumerate(calls):
        trace = execute_one(
            call,
            by_label,
            indexed,
            index=i,
            ibm_registry=ibm_registry,
        )
        coerced = _coerce_result(trace.result)
        traces.append({
            "index": trace.index,
            "name": trace.name,
            "error": trace.error,
            "result": coerced,
        })
        if trace.error:
            if stop_on_error:
                return final_value, traces, trace.error
            continue
        # Keep raw result in scope for _matches_gold (uses original type),
        # but store coerced value as final_value so callers can safely serialize.
        by_label[trace.label] = trace.result
        indexed.append(trace.result)
        final_value = coerced

    return final_value, traces, None


def verify_gold_row(
    output: Any,
    gold_answer: Any,
    *,
    ibm_registry: Optional[IBMFunctionRegistry] = None,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    calls = _coerce_calls(output)
    if not calls:
        return False, None, "empty_output"
    final_value, _, err = execute_trajectory(calls, ibm_registry=ibm_registry)
    if err:
        return False, final_value, err
    ok = _matches_gold(final_value, gold_answer)
    return ok, final_value, None if ok else "executor_mismatch"
