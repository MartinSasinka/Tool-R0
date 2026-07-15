"""Bridge from the Agentic Self-Instruct pipeline (lib/agentic_data) to the
versioned v5 executable synthetic tool registry and the TRAINER'S real
executor (``executor.py`` ``ToolExecutor(mode="synthetic")``).

This is the SINGLE integration point between the challenger/weak/strong/
judge loop and the tool registry: every module that used to import
``TOOLS`` / ``execute_call`` / ``tool_schema`` / ``question_hash`` /
``trace_hash`` from ``..nestful_like_generator`` (the legacy ~34-tool
registry) imports from here instead, so the generator and the trainer are
GUARANTEED to agree on tool schemas, semantics and execution behavior — a
wrong predicted argument value executes for real and never falls back to
the gold observation, exactly like GRPO training with ``executor.mode=
synthetic``.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_EXPERIMENTS = os.path.abspath(os.path.join(_V3_ROOT, ".."))
_MINIMAL = os.path.join(_EXPERIMENTS, "nestful_mtgrpo_minimal")
for _p in (_V3_ROOT, _MINIMAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.synthetic_tools import (  # noqa: E402
    ALL_TOOL_NAMES,
    REGISTRY_VERSION,
    TOOLS,
    registry_hash,
    semantics_compatible,
    tool_schema,
)
from lib.synthetic_gen_v5 import question_hash, trace_hash  # noqa: E402
from executor import ToolExecutor  # noqa: E402

REGISTRY_SOURCE = "synthetic_tools_v5"

# Error-message-prefix -> coarse legacy failure category, so callers that
# historically branched on {"wrong_tool", "wrong_args", "invalid_reference",
# "execution_error"} (solver scoring, diversity accounting) keep working
# unchanged against the new executor's more specific error strings.
_ERROR_CATEGORY_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("unknown_tool:", "wrong_tool"),
    ("synthetic:unregistered_tool:", "wrong_tool"),
    ("unresolved_variable:", "invalid_reference"),
    ("unresolved_field:", "invalid_reference"),
    ("invalid_arguments_type", "wrong_args"),
    ("synthetic:unknown_argument:", "wrong_args"),
    ("synthetic:missing_required_argument:", "wrong_args"),
    ("synthetic:argument_type_mismatch:", "wrong_args"),
    ("synthetic:argument_below_min:", "wrong_args"),
    ("synthetic:argument_above_max:", "wrong_args"),
    ("synthetic:array_too_short:", "wrong_args"),
    ("synthetic:array_element_type:", "wrong_args"),
    ("synthetic:division_by_zero:", "execution_error"),
    ("synthetic:runtime_error:", "execution_error"),
)


def categorize_exec_error(err: Optional[str]) -> str:
    """Map a ``ToolExecutor(mode="synthetic")`` error string to one of the
    legacy coarse categories (``wrong_tool`` / ``wrong_args`` /
    ``invalid_reference`` / ``execution_error``)."""
    if not err:
        return "execution_error"
    for prefix, category in _ERROR_CATEGORY_PREFIXES:
        if err.startswith(prefix):
            return category
    return "execution_error"


def _all_tool_schemas() -> List[Dict[str, Any]]:
    return [tool_schema(n) for n in ALL_TOOL_NAMES]


def execute_gold_trace(gold_calls: List[Dict[str, Any]]
                       ) -> Tuple[Optional[List[Any]], Optional[str]]:
    """Execute a candidate GOLD trace through the REAL trainer executor
    (``executor.mode="synthetic"``). Returns ``(observations, error)`` — same
    signature as the legacy ``verifier.execute_gold_trace()`` so callers are
    unaffected by the registry swap. A wrong/missing/extra argument, a bad
    reference, or an out-of-range value is a HARD error here (never silently
    replaced by a gold value)."""
    names = sorted({c.get("name") for c in gold_calls if isinstance(c, dict)})
    unknown = [n for n in names if n not in TOOLS]
    if unknown:
        return None, f"unknown tool '{unknown[0]}'"
    task = {"tools": [tool_schema(n) for n in names], "gold_calls": []}
    ex = ToolExecutor(task, mode="synthetic")
    observations: List[Any] = []
    for i, call in enumerate(gold_calls):
        res = ex.execute(call)
        if res.error is not None:
            return None, f"call {i + 1} ({call.get('name')}): {res.error}"
        observations.append(res.observation)
    return observations, None


def execute_predicted_calls(predicted: List[Dict[str, Any]]
                            ) -> Tuple[List[Any], Optional[str]]:
    """Execute a SOLVER's predicted calls through the real executor against
    the FULL registry (mirrors the legacy behavior of checking predicted
    tool names against the global tool set, not just the offered menu). A
    call with wrong values/refs/types fails, or executes for real and
    returns a genuinely wrong (non-gold) observation — it can never receive
    the gold result. Returns ``(observations, coarse_error_category)``."""
    task = {"tools": _all_tool_schemas(), "gold_calls": []}
    ex = ToolExecutor(task, mode="synthetic")
    observations: List[Any] = []
    for call in predicted:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            return observations, "wrong_tool"
        if not isinstance(call.get("arguments"), dict):
            return observations, "wrong_args"
        res = ex.execute(call)
        if res.error is not None:
            return observations, categorize_exec_error(res.error)
        observations.append(res.observation)
    return observations, None


def replay_task(row: Dict[str, Any]) -> Tuple[bool, Any]:
    """Re-execute a dataset row's gold trace through the REAL trainer executor;
    True iff the final observation equals the row's stored ``gold_answer``
    (defense-in-depth replay before a dataset is written/shipped)."""
    observations, err = execute_gold_trace(row["gold_calls"])
    if err is not None:
        return False, f"replay_error: {err}"
    final = observations[-1] if observations else None
    return final == row["gold_answer"], final


__all__ = [
    "ALL_TOOL_NAMES", "REGISTRY_VERSION", "REGISTRY_SOURCE", "TOOLS",
    "registry_hash", "semantics_compatible", "tool_schema",
    "question_hash", "trace_hash", "categorize_exec_error",
    "execute_gold_trace", "execute_predicted_calls", "replay_task",
]
