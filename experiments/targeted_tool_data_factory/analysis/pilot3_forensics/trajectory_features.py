"""Paired trajectory fingerprints and divergence classification."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .graph_features import is_reference, parse_reference
from .io import as_bool, short_hash


def _parsed_calls(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls = []
    for t in traj.get("turns") or []:
        pc = t.get("parsed_call")
        if isinstance(pc, dict) and pc.get("name"):
            calls.append(pc)
        elif isinstance(pc, list):
            for item in pc:
                if isinstance(item, dict) and item.get("name"):
                    calls.append(item)
    return calls


def _tool_names(calls: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(c.get("name") or "") for c in calls]


def _arg_key_seq(calls: Sequence[Dict[str, Any]]) -> List[List[str]]:
    out = []
    for c in calls:
        args = c.get("arguments") or {}
        if isinstance(args, dict):
            out.append(sorted(str(k) for k in args.keys()))
        else:
            out.append([])
    return out


def _ref_structure(calls: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    struct = []
    for c in calls:
        refs = []
        args = c.get("arguments") or {}
        if isinstance(args, dict):
            for k, v in sorted(args.items()):
                ref = parse_reference(v)
                if ref:
                    refs.append({"key": k, "label": ref["label_norm"], "out": ref["output_key"]})
        struct.append(refs)
    return struct


def fingerprint(row: Dict[str, Any]) -> Dict[str, Any]:
    traj = row.get("_traj") or {}
    calls = _parsed_calls(traj)
    turns = traj.get("turns") or []
    model_texts = [str(t.get("model_text") or "") for t in turns]
    return {
        "n_model_turns": len(turns),
        "n_parsed_tool_calls": len(calls),
        "tool_name_seq": _tool_names(calls),
        "arg_key_seq": _arg_key_seq(calls),
        "reference_structure": _ref_structure(calls),
        "stop_reason": traj.get("stop_reason"),
        "parse_valid": as_bool(traj.get("parse_valid")),
        "executable": as_bool(traj.get("executable")),
        "execution_error": str(traj.get("execution_error") or ""),
        "execution_error_category": _exec_error_category(traj),
        "terminal_status": _terminal_status(row),
        "predicted_answer": traj.get("pred_answer"),
        "final_answer_pass": bool(as_bool(row.get("final_answer_pass"))),
        "strict_gold_trace_pass": bool(as_bool(row.get("strict_gold_trace_pass"))),
        "solution_equivalent_pass": bool(as_bool(row.get("solution_equivalent_pass"))),
        "alternative_valid_solution_pass": bool(as_bool(row.get("alternative_valid_solution_pass"))),
        "official_win": bool(as_bool(traj.get("official_win"))),
        "model_text_hash": short_hash(model_texts),
        "parsed_calls_hash": short_hash([
            {"name": c.get("name"), "arguments": c.get("arguments"), "label": c.get("label")}
            for c in calls
        ]),
        "calls": calls,
        "model_texts": model_texts,
    }


def _exec_error_category(traj: Dict[str, Any]) -> str:
    err = str(traj.get("execution_error") or "")
    blob = (err + " " + str(traj.get("mismatch_reason") or "")).lower()
    for t in traj.get("turns") or []:
        if t.get("fail_reason"):
            blob += " " + str(t.get("fail_reason")).lower()
    if not blob.strip():
        return "none"
    if "unresolved" in blob or "bad_ref" in blob or "missing_ref" in blob:
        return "reference"
    if "unknown_tool" in blob:
        return "unknown_tool"
    if "type" in blob or "schema" in blob or "arg" in blob:
        return "argument"
    if "timeout" in blob:
        return "timeout"
    if traj.get("executable") is False:
        return "executor_other"
    return "other"


def _terminal_status(row: Dict[str, Any]) -> str:
    traj = row.get("_traj") or {}
    if as_bool(traj.get("official_win")):
        return "official_win"
    if as_bool(row.get("solution_equivalent_pass")):
        return "solution_equivalent"
    if as_bool(traj.get("executable")) and not as_bool(row.get("final_answer_pass")):
        return "executable_wrong_answer"
    if traj.get("parse_valid") is False:
        return "parse_invalid"
    if traj.get("executable") is False:
        return "not_executable"
    return "other_fail"


def first_divergent_turn(fp_a: Dict[str, Any], fp_b: Dict[str, Any]) -> Optional[int]:
    texts_a = fp_a.get("model_texts") or []
    texts_b = fp_b.get("model_texts") or []
    calls_a = fp_a.get("calls") or []
    calls_b = fp_b.get("calls") or []
    n = max(len(texts_a), len(texts_b), len(calls_a), len(calls_b))
    for i in range(n):
        ta = texts_a[i] if i < len(texts_a) else None
        tb = texts_b[i] if i < len(texts_b) else None
        ca = calls_a[i] if i < len(calls_a) else None
        cb = calls_b[i] if i < len(calls_b) else None
        if ta != tb or _call_norm(ca) != _call_norm(cb):
            return i
    return None


def _call_norm(c: Optional[Dict[str, Any]]) -> Any:
    if c is None:
        return None
    return {
        "name": c.get("name"),
        "arguments": c.get("arguments"),
        "label": c.get("label"),
    }


def classify_divergence(fp_c0: Dict[str, Any], fp_d1: Dict[str, Any]) -> str:
    if fp_c0["model_text_hash"] == fp_d1["model_text_hash"] and fp_c0["parsed_calls_hash"] == fp_d1["parsed_calls_hash"]:
        if fp_c0["official_win"] != fp_d1["official_win"] or fp_c0["predicted_answer"] != fp_d1["predicted_answer"]:
            return "LABEL_DIFFERENCE"
        return "IDENTICAL_TEXT"
    if fp_c0["parsed_calls_hash"] == fp_d1["parsed_calls_hash"]:
        if fp_c0["predicted_answer"] != fp_d1["predicted_answer"]:
            return "IDENTICAL_CALLS_DIFFERENT_FINAL_ANSWER"
        return "IDENTICAL_CALLS_DIFFERENT_TEXT"
    if fp_c0["parse_valid"] != fp_d1["parse_valid"]:
        return "PARSE_VALIDITY_DIFFERENCE"
    tools_a, tools_b = fp_c0["tool_name_seq"], fp_d1["tool_name_seq"]
    if tools_a and tools_b and tools_a[0] != tools_b[0]:
        return "DIFFERENT_FIRST_TOOL"
    if fp_c0["n_parsed_tool_calls"] != fp_d1["n_parsed_tool_calls"]:
        return "TOOL_COUNT_DIFFERENCE"
    if tools_a != tools_b:
        return "DIFFERENT_LATER_TOOL" if tools_a and tools_b else "TOOL_COUNT_DIFFERENCE"
    # same tools
    if fp_c0["reference_structure"] != fp_d1["reference_structure"]:
        return "REFERENCE_DIFFERENCE"
    if fp_c0["arg_key_seq"] != fp_d1["arg_key_seq"] or fp_c0["parsed_calls_hash"] != fp_d1["parsed_calls_hash"]:
        return "SAME_TOOLS_DIFFERENT_ARGUMENTS"
    if fp_c0["stop_reason"] != fp_d1["stop_reason"]:
        return "STOPPING_DIFFERENCE"
    if fp_c0["executable"] != fp_d1["executable"] or fp_c0["execution_error_category"] != fp_d1["execution_error_category"]:
        return "EXECUTION_DIFFERENCE"
    if fp_c0["predicted_answer"] != fp_d1["predicted_answer"] and tools_a == tools_b:
        return "ANSWER_ONLY_DIFFERENCE"
    return "OTHER"


def describe_divergence(fp_c0: Dict[str, Any], fp_d1: Dict[str, Any], category: str) -> str:
    return (
        f"{category}; tools C0={fp_c0['tool_name_seq']} D1={fp_d1['tool_name_seq']}; "
        f"stop C0={fp_c0['stop_reason']} D1={fp_d1['stop_reason']}; "
        f"exec C0={fp_c0['executable']}/{fp_c0['execution_error_category']} "
        f"D1={fp_d1['executable']}/{fp_d1['execution_error_category']}; "
        f"ans_pass C0={fp_c0['final_answer_pass']} D1={fp_d1['final_answer_pass']}"
    )


def pair_trajectory_features(
    c0_row: Dict[str, Any],
    d1_row: Dict[str, Any],
    *,
    outcome: str,
) -> Dict[str, Any]:
    fp0 = fingerprint(c0_row)
    fp1 = fingerprint(d1_row)
    cat = classify_divergence(fp0, fp1)
    turn = first_divergent_turn(fp0, fp1)
    return {
        "sample_id": str(c0_row.get("sample_id")),
        "outcome": outcome,
        "first_divergent_turn": turn if turn is not None else -1,
        "divergence_category": cat,
        "c0_tool_seq": json.dumps(fp0["tool_name_seq"]),
        "d1_tool_seq": json.dumps(fp1["tool_name_seq"]),
        "c0_stop_reason": fp0["stop_reason"],
        "d1_stop_reason": fp1["stop_reason"],
        "c0_parse_valid": fp0["parse_valid"],
        "d1_parse_valid": fp1["parse_valid"],
        "c0_executable": fp0["executable"],
        "d1_executable": fp1["executable"],
        "c0_execution_error_category": fp0["execution_error_category"],
        "d1_execution_error_category": fp1["execution_error_category"],
        "c0_final_answer_pass": fp0["final_answer_pass"],
        "d1_final_answer_pass": fp1["final_answer_pass"],
        "c0_official_win": fp0["official_win"],
        "d1_official_win": fp1["official_win"],
        "c0_pred_answer": fp0["predicted_answer"],
        "d1_pred_answer": fp1["predicted_answer"],
        "c0_n_calls": fp0["n_parsed_tool_calls"],
        "d1_n_calls": fp1["n_parsed_tool_calls"],
        "description": describe_divergence(fp0, fp1, cat),
        "_fp_c0": fp0,
        "_fp_d1": fp1,
    }
