"""Pilot4 V4 wrapped as a strict, cached pre-selection gate."""
from __future__ import annotations

from typing import Any, Dict

from ..pilot4.validate import v4_minimal_path

_CACHE: Dict[str, Dict[str, Any]] = {}


def evaluate_v4(record: Dict[str, Any], *, max_evals: int = 4000) -> Dict[str, Any]:
    key = record["semantic_program_id"]
    if key in _CACHE:
        return dict(_CACHE[key], cached=True)
    gold_len = len(record.get("gold_calls") or [])
    target = record.get("gold_answer")
    constants = [v for n in record.get("semantic_program", {}).get("nodes", [])
                 for v in n.get("inputs", {}).values()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)]
    constant_shortcut = any(abs(float(v) - float(target)) <= 1e-9 for v in constants) \
        if isinstance(target, (int, float)) and not isinstance(target, bool) else False
    errors, meta = v4_minimal_path(record, max_depth=max(0, gold_len - 1),
                                   max_evals=max_evals)
    shortcut_depth = meta.get("shortcut_depth")
    has_shortcut = constant_shortcut or (
        shortcut_depth is not None and int(shortcut_depth) < gold_len)
    unresolved = bool(meta.get("exhausted"))
    result = {"semantic_program_id": key, "passed": not has_shortcut and not unresolved,
              "has_shortcut": has_shortcut, "constant_shortcut": constant_shortcut,
              "unresolved": unresolved, "gold_call_count": gold_len,
              "search": meta, "raw_reasons": errors}
    _CACHE[key] = result
    return result


def filter_v4_safe(records: list[Dict[str, Any]]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    safe, rejected = [], []
    for row in records:
        gate = evaluate_v4(row)
        row["v4_gate"] = gate
        (safe if gate["passed"] else rejected).append(row)
    return safe, rejected
