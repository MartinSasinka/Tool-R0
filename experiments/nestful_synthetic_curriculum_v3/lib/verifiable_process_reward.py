"""Deterministically verifiable process-reward components for the reward
ablation (A3_VERIFIABLE_PROCESS / A4_GATED_VERIFIABLE).

Hard constraint (see reports/reward_ablation/ABLATION_PLAN.md §5): every
component here MUST be checkable from the trajectory + the tool catalog +
the (real, synthetic-executor) execution outcome alone. None of them may
compare the model's calls/arguments to the GOLD trace — that would just be
process-reward-via-gold-similarity again (already covered by A2/R3).

Verifiable components (all in [0, 1], computed over EMITTED calls only):

  format_valid              — trajectory parsed cleanly, wasn't clipped.
  tool_exists_frac          — fraction of calls naming a tool that is
                               actually offered in this task's tool catalog.
  schema_keys_valid_frac    — fraction of calls whose argument keys are a
                               subset of that tool's declared parameters and
                               whose required parameters are all present.
  type_range_valid_frac     — of the schema-key-valid calls, fraction whose
                               argument values satisfy the declared JSON
                               type (and min/max/min_len, when declared).
  reference_resolvable_frac — fraction of `$varN[.field]$` references used
                               that the executor could actually resolve
                               (`nestful_core.rewards.valid_references_fraction`).
  execution_success_frac    — fraction of calls the REAL synthetic/full
                               executor actually executed successfully
                               (`executable_fraction`).
  execution_integrity_frac  — fraction of emitted calls that produced no
                               executor-side failure (`turn.fail_reason is
                               None`); a call only reaches this state if the
                               state built by all prior real calls/
                               observations was valid AND any reference it
                               used actually grounded in that real prior
                               observation — i.e. this is the composite
                               "valid state transition + observation
                               grounding" signal. It is NOT a gold-trace
                               comparison: whether execution succeeds is a
                               property of the executor + prior state only.

No component here re-derives anything `lib/reward_v3_1.py` / `nestful_core`
does not already compute; this module only recombines existing, audited,
gold-free predicates plus a small amount of new (tool-catalog-only) schema
checking.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from lib.reward_v3_1 import _emitted_calls, _predicates  # type: ignore  # noqa: E402

VERIFIABLE_WEIGHTS: Dict[str, float] = {
    "format_valid": 0.10,
    "tool_exists_frac": 0.15,
    "schema_keys_valid_frac": 0.15,
    "type_range_valid_frac": 0.15,
    "reference_resolvable_frac": 0.15,
    "execution_success_frac": 0.15,
    "execution_integrity_frac": 0.15,
}
assert abs(sum(VERIFIABLE_WEIGHTS.values()) - 1.0) < 1e-9


def _type_ok(declared: str, value: Any) -> bool:
    """Permissive, gold-free JSON-schema type check.

    Unknown/compound declared types (e.g. legacy NESTFUL "int or float")
    can't be verified with certainty either way, so they are NOT counted as
    a violation (a verifiable-only reward must never guess; it can only
    confirm or stay silent).
    """
    d = (declared or "").strip().lower()
    if d in ("number", "float", "double"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if d == "integer":
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        return isinstance(value, float) and float(value).is_integer()
    if d == "boolean":
        return isinstance(value, bool)
    if d == "string":
        return isinstance(value, str)
    if d == "array":
        return isinstance(value, list)
    if d == "object":
        return isinstance(value, dict)
    return True  # unverifiable declared type -> not a violation


def _tool_schema_map(task: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {t.get("name"): t for t in (task.get("tools") or []) if t.get("name")}


def _call_name_args(turn) -> Optional[Dict[str, Any]]:
    call = getattr(turn, "parsed_call", None)
    return call if isinstance(call, dict) else None


def tool_exists_frac(trajectory, task: Dict[str, Any]) -> float:
    schema = _tool_schema_map(task)
    calls = [c for t in _emitted_calls(trajectory) if (c := _call_name_args(t))]
    if not calls:
        return 0.0
    ok = sum(1 for c in calls if (c.get("name") or "") in schema)
    return ok / len(calls)


def schema_keys_valid_frac(trajectory, task: Dict[str, Any]) -> float:
    schema = _tool_schema_map(task)
    calls = [c for t in _emitted_calls(trajectory) if (c := _call_name_args(t))]
    if not calls:
        return 0.0
    ok = 0
    for c in calls:
        tool = schema.get(c.get("name") or "")
        if not tool:
            continue
        params = (tool.get("parameters") or {})
        props = params.get("properties") or {}
        required = set(params.get("required") or [])
        arg_keys = set((c.get("arguments") or {}).keys())
        if not arg_keys.issubset(set(props.keys())):
            continue
        if not required.issubset(arg_keys):
            continue
        ok += 1
    return ok / len(calls)


def type_range_valid_frac(trajectory, task: Dict[str, Any]) -> float:
    schema = _tool_schema_map(task)
    calls = [c for t in _emitted_calls(trajectory) if (c := _call_name_args(t))]
    if not calls:
        return 0.0
    ok = 0
    for c in calls:
        tool = schema.get(c.get("name") or "")
        if not tool:
            continue
        props = (tool.get("parameters") or {}).get("properties") or {}
        args = c.get("arguments") or {}
        valid = True
        for key, value in args.items():
            meta = props.get(key)
            if meta is None:
                valid = False
                break
            declared = meta.get("type", "")
            if not _type_ok(declared, value):
                valid = False
                break
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if "minimum" in meta and float(value) < float(meta["minimum"]):
                    valid = False
                    break
                if "maximum" in meta and float(value) > float(meta["maximum"]):
                    valid = False
                    break
            if isinstance(value, list) and "minItems" in meta and len(value) < int(meta["minItems"]):
                valid = False
                break
        if valid:
            ok += 1
    return ok / len(calls)


def execution_integrity_frac(trajectory) -> float:
    calls = _emitted_calls(trajectory)
    if not calls:
        return 0.0
    ok = sum(1 for t in calls if getattr(t, "fail_reason", None) is None)
    return ok / len(calls)


def verifiable_process_components(trajectory, task: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, float]:
    """All components in [0, 1]; none compare to the gold trace."""
    refs = pred.get("refs")
    reference_resolvable = 1.0 if refs is None else float(refs)
    return {
        "format_valid": 1.0 if (not pred["parse_err"] and not pred["clipped"]) else 0.0,
        "tool_exists_frac": tool_exists_frac(trajectory, task),
        "schema_keys_valid_frac": schema_keys_valid_frac(trajectory, task),
        "type_range_valid_frac": type_range_valid_frac(trajectory, task),
        "reference_resolvable_frac": reference_resolvable,
        "execution_success_frac": float(pred["executable_frac"]),
        "execution_integrity_frac": execution_integrity_frac(trajectory),
    }


def verifiable_process_score(components: Dict[str, float]) -> float:
    return round(sum(VERIFIABLE_WEIGHTS[k] * components[k] for k in VERIFIABLE_WEIGHTS), 6)


def gate_open(pred: Dict[str, Any]) -> bool:
    """A4 gate: full process tie-breaker only for a fully executable
    trajectory (no parse/no-call failure, no fully-failed execution)."""
    if pred["parse_err"] or pred["clipped"] or pred["no_tool"]:
        return False
    if float(pred["executable_frac"]) <= 0.0:
        return False
    return True
