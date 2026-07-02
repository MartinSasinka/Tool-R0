"""Trajectory component metrics and error classification."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from vizualisation.scripts.lib.canonicalize import levenshtein_ratio
from vizualisation.scripts.lib.rewards_bridge import (
    extract_predicted_answer,
    extract_predicted_calls,
    extract_references_from_value,
    parse_completion,
    score_argument_keys,
    score_argument_values,
    score_call_count,
    score_final_answer,
    score_format,
    score_labels,
    score_references,
    score_tool_names,
    values_match,
)


def compute_dependency_depth(calls: Optional[List[Dict[str, Any]]]) -> int:
    if not calls:
        return 0
    refs_by_call: Dict[int, Set[int]] = defaultdict(set)
    for i, c in enumerate(calls, start=1):
        args = c.get("arguments", {})
        if not isinstance(args, dict):
            continue
        for v in args.values():
            for ref_idx, _ in extract_references_from_value(v):
                if ref_idx < i:
                    refs_by_call[i].add(ref_idx)
    memo: Dict[int, int] = {}

    def depth(i: int) -> int:
        if i in memo:
            return memo[i]
        preds = refs_by_call.get(i, set())
        if not preds:
            memo[i] = 1
            return 1
        memo[i] = 1 + max(depth(p) for p in preds)
        return memo[i]

    return max(depth(i) for i in range(1, len(calls) + 1))


def score_dependency_depth(pred_calls, gold_calls) -> float:
    if not gold_calls:
        return 1.0 if not pred_calls else 0.0
    gd = compute_dependency_depth(gold_calls)
    pd = compute_dependency_depth(pred_calls)
    if gd == 0:
        return 1.0 if pd == 0 else 0.0
    if pd == gd:
        return 1.0
    return max(0.0, 1.0 - abs(pd - gd) / gd)


def reference_diagnostics(
    pred_calls: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    invalid = 0
    forward = False
    self_ref = False
    if not pred_calls:
        return {
            "invalid_reference_count": 0,
            "has_forward_reference": False,
            "has_self_reference": False,
        }
    for i, call in enumerate(pred_calls, start=1):
        args = call.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        for v in args.values():
            for ref_idx, _ in extract_references_from_value(v):
                if ref_idx >= i:
                    forward = True
                    invalid += 1
                if ref_idx == i:
                    self_ref = True
                if ref_idx < 1 or ref_idx > len(pred_calls):
                    invalid += 1
    return {
        "invalid_reference_count": invalid,
        "has_forward_reference": forward,
        "has_self_reference": self_ref,
    }


def classify_error_type(components: Dict[str, Any]) -> str:
    if not components.get("valid_json"):
        return "invalid_json"
    if components.get("num_calls_pred", 0) == 0 and not components.get("final_answer_exact_match"):
        return "missing_output"
    if components.get("call_count_score", 0) < 1.0:
        return "wrong_call_count"
    if components.get("tool_name_score", 0) < 1.0:
        return "wrong_tool"
    if components.get("label_score", 0) < 1.0:
        return "wrong_labels"
    if components.get("argument_key_score", 0) < 1.0:
        return "wrong_argument_keys"
    if components.get("argument_value_score", 0) < 1.0:
        return "wrong_argument_values"
    if components.get("reference_score", 0) < 1.0:
        return "wrong_references"
    if components.get("dependency_depth_score", 0) < 1.0:
        return "wrong_dependency_depth"
    if components.get("final_answer_score", 0) < 1.0:
        return "wrong_final_answer"
    if components.get("exact_trajectory_match"):
        return "exact_match"
    return "partial_match"


def build_pred_obj(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    calls = row.get("prediction_output")
    answer = row.get("prediction_answer")
    if calls is None and not answer and not row.get("prediction_raw"):
        return None
    obj: Dict[str, Any] = {}
    if calls is not None:
        obj["output"] = calls
    if answer is not None:
        obj["answer"] = answer
    if not obj and row.get("prediction_raw"):
        parsed, _ = parse_completion(str(row["prediction_raw"]))
        return parsed
    return obj if obj else None


def compute_row_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    gold_calls = row.get("gold_output") or []
    if not isinstance(gold_calls, list):
        gold_calls = []

    gold_answer = row.get("gold_answer")
    if gold_answer is not None and not isinstance(gold_answer, str):
        import json

        gold_answer = json.dumps(gold_answer, ensure_ascii=False)

    num_gold = len(gold_calls)
    pred_obj = build_pred_obj(row)
    pred_calls = row.get("prediction_output")
    if pred_calls is None and pred_obj:
        pred_calls = extract_predicted_calls(pred_obj)

    pred_answer = row.get("prediction_answer")
    if pred_answer is None and pred_obj:
        pred_answer = extract_predicted_answer(pred_obj)

    valid_json = pred_obj is not None or pred_calls is not None

    ref_diag = reference_diagnostics(pred_calls)
    dep_pred = compute_dependency_depth(pred_calls)
    dep_gold = compute_dependency_depth(gold_calls)
    num_pred = len(pred_calls) if pred_calls else 0

    canonical_pred = row.get("canonical_pred", "")
    canonical_gold = row.get("canonical_gold", "")
    edit_dist = levenshtein_ratio(canonical_pred, canonical_gold)
    exact_traj = canonical_pred == canonical_gold and bool(canonical_gold)
    exact_ans = bool(
        pred_answer is not None
        and gold_answer is not None
        and values_match(pred_answer, gold_answer)
    )

    components = {
        "valid_json": float(valid_json),
        "format_score": score_format(pred_obj, num_gold),
        "call_count_score": score_call_count(pred_calls, num_gold),
        "tool_name_score": score_tool_names(pred_calls, gold_calls),
        "label_score": score_labels(pred_calls, num_gold),
        "argument_key_score": score_argument_keys(pred_calls, gold_calls),
        "argument_value_score": score_argument_values(pred_calls, gold_calls),
        "reference_score": score_references(pred_calls, gold_calls),
        "dependency_depth_score": score_dependency_depth(pred_calls, gold_calls),
        "final_answer_score": score_final_answer(pred_answer, str(gold_answer or "")),
        "exact_trajectory_match": float(exact_traj),
        "final_answer_exact_match": float(exact_ans),
        "dependency_depth_pred": dep_pred,
        "dependency_depth_gold": dep_gold,
        "num_calls_pred": num_pred,
        "num_calls_gold": num_gold,
        "trajectory_edit_distance": edit_dist,
        **ref_diag,
    }
    error_type = classify_error_type(components)

    out = {
        "sample_id": row.get("sample_id"),
        "checkpoint": row.get("checkpoint"),
        "rollout_idx": row.get("rollout_idx", 0),
        "score": row.get("score"),
        "status": row.get("status"),
        "verdict": row.get("verdict"),
        "error_type": error_type,
        **components,
    }
    return out


def gold_feature_row(gold_calls: List[Dict[str, Any]], gold_answer: Any) -> Dict[str, float]:
    n = len(gold_calls)
    dep = compute_dependency_depth(gold_calls)
    return {
        "format_score": 1.0,
        "call_count_score": 1.0,
        "tool_name_score": 1.0,
        "label_score": 1.0,
        "argument_key_score": 1.0,
        "argument_value_score": 1.0,
        "reference_score": 1.0,
        "dependency_depth_score": 1.0,
        "final_answer_score": 1.0,
        "valid_json": 1.0,
        "exact_trajectory_match": 1.0,
        "final_answer_exact_match": 1.0,
        "dependency_depth_pred": float(dep),
        "dependency_depth_gold": float(dep),
        "num_calls_pred": float(n),
        "num_calls_gold": float(n),
        "invalid_reference_count": 0.0,
        "trajectory_edit_distance": 0.0,
    }
