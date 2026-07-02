"""Prediction format adapters and rollout handling."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from vizualisation.scripts.lib.rewards_bridge import parse_completion

_TOOL_CALL_ANSWER_RE = re.compile(
    r"<tool_call_answer>\s*(.*?)\s*</tool_call_answer>",
    re.DOTALL | re.IGNORECASE,
)


def _coerce_calls(raw: Any) -> Optional[List[Dict[str, Any]]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if not isinstance(raw, list):
        return None
    calls = []
    for item in raw:
        if isinstance(item, dict) and item.get("name"):
            calls.append(item)
    return calls if calls else None


def _join_raw(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        return "\n---\n".join(str(x) for x in raw)
    return str(raw)


def parse_tool_calls_from_completion(text: str) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None
    all_calls: List[Dict[str, Any]] = []
    for m in _TOOL_CALL_ANSWER_RE.finditer(text):
        payload = m.group(1).strip()
        if not payload or payload == "[]":
            continue
        try:
            parsed = json.loads(payload)
        except Exception:
            continue
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and item.get("name"):
                    all_calls.append(item)
        elif isinstance(parsed, dict) and parsed.get("name"):
            all_calls.append(parsed)
    return all_calls if all_calls else None


def infer_prediction_from_row(row: Dict[str, Any]) -> tuple:
    """Return (prediction_output, prediction_answer, prediction_raw, parse_flags)."""
    flags: List[str] = []

    pred_raw = row.get("prediction_raw")
    if pred_raw is None:
        pred_raw = _join_raw(row.get("raw_completions"))

    pred_out = row.get("prediction_output")
    if pred_out is None:
        pred_out = row.get("output")
    if pred_out is None:
        pred_out = row.get("predicted_calls")
    if pred_out is None:
        pred_out = row.get("execution_trace")

    pred_answer = row.get("prediction_answer")
    if pred_answer is None:
        pred_answer = row.get("answer")
    if pred_answer is None:
        pred_answer = row.get("predicted_final")
    if pred_answer is None:
        pred_answer = row.get("final_answer")

    calls = _coerce_calls(pred_out)
    if calls is None and pred_raw:
        calls = parse_tool_calls_from_completion(str(pred_raw))
        if calls:
            flags.append("recovered_from_tool_call_answer")

    if calls is None and pred_raw:
        obj, _ = parse_completion(str(pred_raw))
        if obj:
            from vizualisation.scripts.lib.rewards_bridge import extract_predicted_calls

            calls = extract_predicted_calls(obj)
            if calls:
                flags.append("recovered_from_json_completion")
            if pred_answer is None:
                from vizualisation.scripts.lib.rewards_bridge import extract_predicted_answer

                pred_answer = extract_predicted_answer(obj)

    if pred_answer is not None and not isinstance(pred_answer, str):
        pred_answer = json.dumps(pred_answer, ensure_ascii=False)

    if calls is None:
        flags.append("missing_prediction_output")

    return calls, pred_answer, pred_raw, flags


def normalize_prediction_row(row: Dict[str, Any], checkpoint: str) -> Dict[str, Any]:
    sample_id = row.get("sample_id") or row.get("task_id")
    if not sample_id:
        return {"_skip": True, "parse_error": "missing_sample_id"}

    calls, pred_answer, pred_raw, flags = infer_prediction_from_row(row)

    gold_out = row.get("gold_output")
    if gold_out is None:
        gold_out = row.get("gold_calls")
    gold_calls = _coerce_calls(gold_out)

    gold_answer = row.get("gold_answer")
    inp = row.get("input") or row.get("question") or ""
    tools = row.get("tools")

    return {
        "sample_id": str(sample_id),
        "checkpoint": row.get("checkpoint") or checkpoint,
        "rollout_idx": row.get("rollout_idx", 0),
        "score": row.get("score"),
        "status": row.get("status"),
        "verdict": row.get("verdict"),
        "input": inp,
        "tools": tools,
        "gold_output": gold_calls,
        "gold_answer": gold_answer,
        "prediction_raw": pred_raw,
        "prediction_output": calls,
        "prediction_answer": pred_answer,
        "parse_flags": flags,
        "source_format": row.get("_source_format", "unknown"),
    }


def expand_rollouts(rows: List[Dict[str, Any]], policy: str) -> List[Dict[str, Any]]:
    if policy == "all_rollouts":
        return rows

    from collections import defaultdict

    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r["sample_id"], r["checkpoint"])
        groups[key].append(r)

    selected: List[Dict[str, Any]] = []
    for key, items in groups.items():
        pick = _select_rollout(items, policy)
        if pick:
            selected.append(pick)
    return selected


def _select_rollout(items: List[Dict[str, Any]], policy: str) -> Optional[Dict[str, Any]]:
    if not items:
        return None
    if policy == "rollout_idx_0":
        for r in sorted(items, key=lambda x: x.get("rollout_idx", 0)):
            if r.get("rollout_idx", 0) == 0:
                return r
        return sorted(items, key=lambda x: x.get("rollout_idx", 0))[0]
    if policy == "pass_first":
        for r in sorted(items, key=lambda x: x.get("rollout_idx", 0)):
            if r.get("status") == "completed" or r.get("verdict") == "pass":
                return r
        return _select_rollout(items, "best_score")
    if policy == "mean_over_rollouts":
        return sorted(items, key=lambda x: x.get("rollout_idx", 0))[0]
    # best_score default
    def score_key(r):
        s = r.get("score")
        try:
            return float(s) if s is not None else -1.0
        except (TypeError, ValueError):
            return -1.0

    return max(items, key=score_key)


def load_gold_index(path) -> Dict[str, Dict[str, Any]]:
    from vizualisation.scripts.lib.io_utils import read_jsonl

    index: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        sid = row.get("sample_id")
        if not sid:
            continue
        output = row.get("output")
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except Exception:
                pass
        tools = row.get("tools")
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except Exception:
                pass
        index[str(sid)] = {
            "sample_id": str(sid),
            "input": row.get("input", ""),
            "tools": tools,
            "gold_output": output,
            "gold_answer": row.get("gold_answer"),
        }
    return index
