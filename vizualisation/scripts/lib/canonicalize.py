"""Canonical textual trajectory representations."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from vizualisation.scripts.lib.rewards_bridge import extract_references_from_value


def _normalize_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _format_args(args: Any) -> str:
    if not isinstance(args, dict):
        return "()"
    parts = []
    for k in sorted(args.keys()):
        parts.append(f"{k}={_normalize_scalar(args[k])}")
    return "(" + ",".join(parts) + ")"


def calls_to_canonical(calls: Optional[List[Dict[str, Any]]], answer: Any = None) -> str:
    lines: List[str] = []
    if calls:
        for i, call in enumerate(calls, start=1):
            name = call.get("name", "?")
            args = call.get("arguments", {})
            lines.append(f"CALL_{i} name={name} args={_format_args(args)}")
    if answer is not None and answer != "":
        if not isinstance(answer, str):
            answer = json.dumps(answer, ensure_ascii=False)
        lines.append(f"ANSWER={answer}")
    return "\n".join(lines)


def canonicalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    gold_calls = row.get("gold_output")
    if isinstance(gold_calls, str):
        try:
            gold_calls = json.loads(gold_calls)
        except Exception:
            gold_calls = None

    pred_calls = row.get("prediction_output")
    gold_answer = row.get("gold_answer")
    pred_answer = row.get("prediction_answer")

    out = dict(row)
    out["canonical_gold"] = calls_to_canonical(gold_calls, gold_answer)
    out["canonical_pred"] = calls_to_canonical(pred_calls, pred_answer)
    return out


def levenshtein_ratio(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    dist = prev[lb]
    return dist / max(la, lb)
