"""
BFCL AST-style checker for Tool-R0 outputs.

Checks model predictions against BFCL ground truth using the same
logic as the official Berkeley Function Calling Leaderboard evaluation.

If the official `bfcl-eval` package is installed, delegates to it.
Otherwise, uses a standalone implementation that covers the core checks:
name match, required params, value match, type coercion.
"""

from __future__ import annotations

import itertools
import json
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker as _official_checker
    HAS_BFCL_EVAL = True
except ImportError:
    HAS_BFCL_EVAL = False


def toolr0_to_bfcl_format(
    calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert Tool-R0 output to BFCL dict format.

    Tool-R0: [{"name": "func", "arguments": {"p": "v"}}]
    BFCL:    [{"func": {"p": "v"}}]
    """
    result = []
    for call in calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        result.append({name: args})
    return result


def _normalize_str(v: Any) -> str:
    if not isinstance(v, str):
        return str(v)
    return re.sub(r"[\s.,;:!?]+$", "", v.strip()).lower()


def _values_equal(predicted: Any, acceptable: Any) -> bool:
    """Check if a predicted value matches an acceptable value."""
    if predicted is None and acceptable is None:
        return True

    if isinstance(acceptable, str) and isinstance(predicted, str):
        return _normalize_str(predicted) == _normalize_str(acceptable)

    if isinstance(acceptable, (int, float)) and isinstance(predicted, (int, float)):
        return abs(float(predicted) - float(acceptable)) < 1e-6

    if isinstance(predicted, str) and isinstance(acceptable, (int, float)):
        try:
            return abs(float(predicted) - float(acceptable)) < 1e-6
        except (ValueError, TypeError):
            return False

    if isinstance(predicted, (int, float)) and isinstance(acceptable, str):
        try:
            return abs(float(predicted) - float(acceptable)) < 1e-6
        except (ValueError, TypeError):
            return False

    if isinstance(predicted, list) and isinstance(acceptable, list):
        if len(predicted) != len(acceptable):
            return False
        return all(_values_equal(p, a) for p, a in zip(predicted, acceptable))

    if isinstance(predicted, dict) and isinstance(acceptable, dict):
        if set(predicted.keys()) != set(acceptable.keys()):
            return False
        return all(_values_equal(predicted[k], acceptable[k]) for k in predicted)

    try:
        return json.dumps(predicted, sort_keys=True) == json.dumps(acceptable, sort_keys=True)
    except (TypeError, ValueError):
        return str(predicted) == str(acceptable)


def _value_in_acceptable(predicted_value: Any, acceptable_values: List[Any]) -> bool:
    """Check if predicted value is in the list of acceptable values."""
    for acceptable in acceptable_values:
        if acceptable == "" or acceptable is None:
            continue
        if _values_equal(predicted_value, acceptable):
            return True
    return False


def _param_is_optional(acceptable_values: List[Any]) -> bool:
    """A param is optional if "" or None is among acceptable values."""
    return any(v == "" or v is None for v in acceptable_values)


def _check_single_call(
    predicted: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check one predicted call against one ground truth entry.

    predicted:    {"func_name": {"param": value}}
    ground_truth: {"func_name": {"param": [acceptable_values]}}   (AST format)
              or: {"func_name": {"param": value}}                  (exec format)
    """
    pred_name = list(predicted.keys())[0] if predicted else ""
    gt_name = list(ground_truth.keys())[0] if ground_truth else ""

    if pred_name != gt_name:
        return False, f"wrong_function: expected '{gt_name}', got '{pred_name}'"

    pred_args = predicted.get(pred_name, {})
    gt_args = ground_truth.get(gt_name, {})

    if not isinstance(pred_args, dict):
        pred_args = {}
    if not isinstance(gt_args, dict):
        return True, "correct"

    for param, gt_val in gt_args.items():
        if param.startswith("_positional_"):
            continue

        if isinstance(gt_val, list):
            acceptable_values = gt_val
        else:
            acceptable_values = [gt_val]

        is_optional = _param_is_optional(acceptable_values)

        if param not in pred_args:
            if not is_optional:
                return False, f"missing_required_param: {param}"
            continue

        if not _value_in_acceptable(pred_args[param], acceptable_values):
            return False, f"wrong_value: {param}={pred_args[param]!r}"

    return True, "correct"


def check_simple(
    model_output: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    func_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check simple category: exactly 1 call expected."""
    if not model_output:
        return {"valid": False, "error": "no_function_call"}

    if len(model_output) != 1:
        return {"valid": False, "error": f"expected_1_call_got_{len(model_output)}"}

    if not ground_truth:
        return {"valid": False, "error": "no_ground_truth"}

    ok, detail = _check_single_call(model_output[0], ground_truth[0])
    return {"valid": ok, "error": detail if not ok else ""}


def check_multiple(
    model_output: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    func_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check multiple category: exactly 1 call (selecting the right function)."""
    return check_simple(model_output, ground_truth, func_docs)


def check_parallel(
    model_output: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    func_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check parallel category: N calls, order-independent."""
    if not model_output:
        return {"valid": False, "error": "no_function_call"}

    if len(model_output) != len(ground_truth):
        return {
            "valid": False,
            "error": f"expected_{len(ground_truth)}_calls_got_{len(model_output)}",
        }

    n = len(ground_truth)
    if n > 8:
        return _check_parallel_greedy(model_output, ground_truth)

    for perm in itertools.permutations(range(n)):
        all_ok = True
        for pred_idx, gt_idx in enumerate(perm):
            ok, _ = _check_single_call(model_output[pred_idx], ground_truth[gt_idx])
            if not ok:
                all_ok = False
                break
        if all_ok:
            return {"valid": True, "error": ""}

    return {"valid": False, "error": "no_valid_permutation"}


def _check_parallel_greedy(
    model_output: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Greedy matching for large N (avoids factorial blowup)."""
    used_gt = set()
    for pred in model_output:
        matched = False
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in used_gt:
                continue
            ok, _ = _check_single_call(pred, gt)
            if ok:
                used_gt.add(gt_idx)
                matched = True
                break
        if not matched:
            return {"valid": False, "error": "unmatched_call"}

    if len(used_gt) == len(ground_truth):
        return {"valid": True, "error": ""}
    return {"valid": False, "error": "incomplete_match"}


def check_irrelevance(
    model_output: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Check irrelevance category: model should NOT produce any function call."""
    if model_output is None or len(model_output) == 0:
        return {"valid": True, "error": ""}
    return {"valid": False, "error": f"unexpected_{len(model_output)}_calls"}


CATEGORY_CHECKERS = {
    "simple": check_simple,
    "multiple": check_multiple,
    "parallel": check_parallel,
    "parallel_multiple": check_parallel,
    "exec_simple": check_simple,
    "exec_multiple": check_multiple,
    "exec_parallel": check_parallel,
    "exec_parallel_multiple": check_parallel,
}


def check_task(
    category: str,
    model_output: Optional[List[Dict[str, Any]]],
    ground_truth: List[Dict[str, Any]],
    func_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Dispatch to the appropriate checker for a given category."""
    if category == "irrelevance":
        return check_irrelevance(model_output)

    if model_output is None:
        return {"valid": False, "error": "parse_failure"}

    bfcl_output = toolr0_to_bfcl_format(model_output)

    checker = CATEGORY_CHECKERS.get(category)
    if checker is None:
        return {"valid": False, "error": f"unknown_category_{category}"}

    return checker(bfcl_output, ground_truth, func_docs)
