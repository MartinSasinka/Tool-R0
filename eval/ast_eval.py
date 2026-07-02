"""
AST matching evaluation — binary accuracy metric matching the paper.

The Tool-R0 paper (and prior work, Patil et al. 2024) evaluates ALL
benchmarks with AST matching: for each sample, either the predicted
tool call is structurally correct (function name + parameter names +
parameter values all match) -> 1, or not -> 0.

This module is self-contained (no wandb dependency) and provides
`ast_match` for paper-comparable binary accuracy across all benchmarks.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

_NUM_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")


def _coerce_number(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        try:
            return float(x)
        except (OverflowError, ValueError):
            return None
    if isinstance(x, str) and _NUM_RE.match(x):
        s = x.strip()
        if len(s.lstrip("+-").replace(".", "")) > 15:
            return None
        try:
            return float(s)
        except (OverflowError, ValueError):
            return None
    return None


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, set):
        return sorted([_to_jsonable(v) for v in x], key=str)
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return x


def _canonical_json(x: Any) -> str:
    return json.dumps(_to_jsonable(x), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def robust_value_match(v1: Any, v2: Any) -> bool:
    """Compare two values with type coercion and whitespace normalization.

    Mirrors rewards_solver.robust_value_match exactly — same logic the
    paper uses for both training rewards and AST evaluation.
    """
    if v1 == v2:
        return True

    if isinstance(v1, (int, str)) and isinstance(v2, (int, str)):
        s1 = str(v1).strip()
        s2 = str(v2).strip()
        if _NUM_RE.match(s1) and _NUM_RE.match(s2):
            return s1.lstrip("+") == s2.lstrip("+")

    n1 = _coerce_number(v1)
    n2 = _coerce_number(v2)
    if n1 is not None and n2 is not None:
        return abs(n1 - n2) < 1e-9

    if isinstance(v1, str) and isinstance(v2, str):
        s1 = " ".join(v1.strip().split())
        s2 = " ".join(v2.strip().split())
        return s1 == s2

    return _canonical_json(v1) == _canonical_json(v2)


def _f1_keys(pred_keys: set, gt_keys: set) -> float:
    if not pred_keys and not gt_keys:
        return 1.0
    if not pred_keys or not gt_keys:
        return 0.0
    inter = len(pred_keys & gt_keys)
    if inter == 0:
        return 0.0
    prec = inter / len(pred_keys)
    rec = inter / len(gt_keys)
    return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


def score_tool_call(
    predicted: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Tuple[float, float, float]:
    """Score a single call: (name_score, key_score, value_score) in [0,1].

    Mirrors rewards_solver.score_tool_call exactly.
    """
    pred_name = (predicted.get("name") or "").strip()
    gt_name = (ground_truth.get("name") or "").strip()
    name_score = 1.0 if pred_name == gt_name and pred_name != "" else 0.0

    pred_args = predicted.get("arguments", {})
    gt_args = ground_truth.get("arguments", {})
    if not isinstance(pred_args, dict):
        pred_args = {}
    if not isinstance(gt_args, dict):
        gt_args = {}

    pred_keys = set(pred_args.keys())
    gt_keys = set(gt_args.keys())
    key_score = _f1_keys(pred_keys, gt_keys)

    inter = pred_keys & gt_keys
    if not inter:
        value_score = 1.0 if (not pred_keys and not gt_keys) else 0.0
    else:
        matches = sum(
            1 for k in inter
            if robust_value_match(pred_args.get(k), gt_args.get(k))
        )
        value_score = matches / len(inter)

    return name_score, key_score, value_score


def ast_match_single(
    predicted: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> bool:
    """Binary AST match for a single tool call.

    Returns True iff function name, ALL parameter keys, and ALL
    parameter values match (with type coercion).
    """
    name_score, key_score, value_score = score_tool_call(predicted, ground_truth)
    return name_score == 1.0 and key_score == 1.0 and value_score == 1.0


def ast_match(
    predicted_calls: Optional[List[Dict[str, Any]]],
    gold_calls: List[Dict[str, Any]],
) -> bool:
    """Binary AST matching for a complete sample (may have multiple calls).

    Uses greedy best-match pairing (same as the paper's Eq. 7).
    A sample passes only if:
      - Number of predicted calls == number of gold calls
      - Every gold call has a perfect match (name + keys + values)
    """
    if not gold_calls:
        return not predicted_calls or len(predicted_calls) == 0

    if not predicted_calls or len(predicted_calls) != len(gold_calls):
        return False

    used = set()
    for gt in gold_calls:
        found = False
        for i, pred in enumerate(predicted_calls):
            if i in used:
                continue
            if ast_match_single(pred, gt):
                used.add(i)
                found = True
                break
        if not found:
            return False

    return True
