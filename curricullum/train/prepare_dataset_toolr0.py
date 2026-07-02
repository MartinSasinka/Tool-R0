#!/usr/bin/env python3
"""Prepare GRPO dataset in Tool-R0 multi-turn format (eval-aligned)."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from curricullum.train.prepare_dataset import (  # noqa: E402
    FINAL_FIELDS,
    coerce_json,
    filter_by_tokenizer_length,
    parse_row,
    records_to_dataset,
)
from nestful_evaluation.run import (  # noqa: E402
    TOOL_R0_SYSTEM_PROMPT,
    _format_tool_response,
    build_user_content,
    execute_one,
)
from curricullum.data.exec_trajectory import execute_trajectory, get_ibm_registry  # noqa: E402

EXTRA_FIELDS = ("turn_idx", "training_format", "expected_result")

_STUB_TOOL_RESULT = "<ibm_unavailable>"


def _coerce_gold_answer(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                import ast

                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return value
    return value


def _resolve_expected_result(
    row: Dict[str, Any],
    calls: List[Dict[str, Any]],
    *,
    ibm_registry,
) -> Tuple[Any, Optional[str]]:
    exp, _, err = execute_trajectory(calls, ibm_registry=ibm_registry)
    if err or exp is None:
        fallback = _coerce_gold_answer(row.get("gold_answer"))
        if fallback is not None:
            return fallback, err or "ibm_fallback_gold_answer"
        return exp, err
    return exp, None


def build_task_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize tools list to eval-style OpenAI function schema."""
    out = []
    for t in tools:
        if "parameters" in t and isinstance(t["parameters"], dict) and "properties" in t["parameters"]:
            out.append(t)
            continue
        properties = {}
        for param_name, param_spec in (t.get("parameters") or {}).items():
            if isinstance(param_spec, dict):
                properties[param_name] = {
                    "type": param_spec.get("type", "string"),
                    "description": param_spec.get("description", ""),
                }
        out.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
            },
        })
    return out


def _gold_completion_thinking(call: Dict[str, Any]) -> str:
    name = call.get("name", "")
    args = call.get("arguments") or {}
    arg_bits = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    return f"Call {name}({arg_bits})"


def build_gold_assistant_turn(call: Dict[str, Any]) -> str:
    payload = json.dumps(
        [{"name": call["name"], "arguments": call.get("arguments", {})}],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    thinking = _gold_completion_thinking(call)
    return (
        f"<think>\n{thinking}\n</think>\n"
        f"<tool_call_answer>{payload}</tool_call_answer>"
    )


def simulate_prefix_messages(
    task: Dict[str, Any],
    calls: List[Dict[str, Any]],
    *,
    ibm_registry,
    through_turn: int,
) -> Tuple[List[Dict[str, str]], List[Any]]:
    """Build chat prefix through assistant turn `through_turn` (0-indexed, exclusive of target)."""
    tools = build_task_tools(task["tools"])
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": TOOL_R0_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content({
            "question": task["input"],
            "tools": tools,
        })},
    ]
    results: List[Any] = []
    by_label: Dict[str, Any] = {}
    indexed: List[Any] = []

    for turn in range(through_turn):
        call = calls[turn]
        messages.append({"role": "assistant", "content": build_gold_assistant_turn(call)})
        if ibm_registry is None:
            stub = _STUB_TOOL_RESULT
            label = (call.get("label") or f"$var_{turn + 1}").strip()
            by_label[label] = stub
            indexed.append(stub)
            results.append(stub)
            messages.append({
                "role": "user",
                "content": _format_tool_response(
                    {"name": call.get("name", ""), "arguments": call.get("arguments") or {}},
                    stub,
                ),
            })
            continue

        trace = execute_one(
            call,
            by_label,
            indexed,
            index=turn,
            ibm_registry=ibm_registry,
        )
        if trace.error:
            stub = _STUB_TOOL_RESULT
            by_label[trace.label] = stub
            indexed.append(stub)
            results.append(stub)
            messages.append({
                "role": "user",
                "content": _format_tool_response(
                    {"name": trace.name, "arguments": trace.arguments_resolved},
                    stub,
                ),
            })
            continue
        by_label[trace.label] = trace.result
        indexed.append(trace.result)
        results.append(trace.result)
        messages.append({
            "role": "user",
            "content": _format_tool_response(
                {"name": trace.name, "arguments": trace.arguments_resolved},
                trace.result,
            ),
        })
    return messages, results


def expand_record(row: Dict[str, Any], ibm_registry) -> List[Dict[str, Any]]:
    calls = row["gold_output"]
    if not calls:
        return []
    num_calls = int(row.get("num_calls") or len(calls))
    turns = num_calls if num_calls <= len(calls) else len(calls)
    expanded: List[Dict[str, Any]] = []

    for turn_idx in range(turns):
        prefix, _ = simulate_prefix_messages(row, calls, ibm_registry=ibm_registry, through_turn=turn_idx)
        gold_call = calls[turn_idx]
        prefix_calls = calls[:turn_idx]
        expected_result, err = _resolve_expected_result(
            row,
            calls[: turn_idx + 1],
            ibm_registry=ibm_registry,
        )
        rec = {
            "sample_id": f"{row['sample_id']}-t{turn_idx}",
            "input": row["input"],
            "tools": row["tools"],
            "gold_output": [gold_call],
            "gold_answer": row["gold_answer"],
            "stage": row.get("stage", f"epoch{num_calls}"),
            "num_calls": 1,
            "turn_idx": turn_idx,
            "task_num_calls": num_calls,
            "prefix_calls": prefix_calls,
            "expected_result": expected_result,
            "exec_error": err,
            "prompt": prefix,
            "prompt_text": json.dumps(prefix, ensure_ascii=False)[:500],
        }
        expanded.append(rec)
    return expanded


def load_toolr0_jsonl(
    path: str,
    default_num_calls: Optional[int] = None,
    max_prompt_tokens: Optional[int] = None,
    skip_over_budget: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    ibm = get_ibm_registry()
    if ibm is None:
        print(
            "[data] WARNING: IBM registry unavailable — using gold_answer fallbacks "
            "and stub tool responses for multi-turn prefixes",
            file=sys.stderr,
        )
    records: List[Dict[str, Any]] = []
    stats = {
        "malformed": 0,
        "skipped_prompt_budget": 0,
        "lines": 0,
        "expanded": 0,
        "ibm_fallback": 0,
    }

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            stats["lines"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed"] += 1
                continue

            parsed = parse_row({k: row[k] for k in FINAL_FIELDS}, default_num_calls=default_num_calls)
            if parsed is None:
                stats["malformed"] += 1
                continue

            num_calls = int(parsed.get("num_calls") or len(parsed["gold_output"]))
            if num_calls == 1 and len(parsed["gold_output"]) == 1:
                # single-turn row
                tools = build_task_tools(parsed["tools"])
                parsed["prompt"] = [
                    {"role": "system", "content": TOOL_R0_SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_content({
                        "question": parsed["input"],
                        "tools": tools,
                    })},
                ]
                parsed["turn_idx"] = int(row.get("turn_idx", 0))
                parsed["task_num_calls"] = 1
                parsed["prefix_calls"] = []
                exp, err = _resolve_expected_result(
                    parsed,
                    parsed["gold_output"],
                    ibm_registry=ibm,
                )
                parsed["expected_result"] = exp
                parsed["exec_error"] = err
                if err:
                    stats["ibm_fallback"] += 1
                records.append(parsed)
                stats["expanded"] += 1
            else:
                for rec in expand_record(parsed, ibm):
                    if rec.get("exec_error"):
                        stats["ibm_fallback"] += 1
                    records.append(rec)
                    stats["expanded"] += 1

    if not records:
        raise SystemExit(f"[data] no valid rows in {path}")

    print(
        f"[data] tool_r0 loaded {len(records)} training rows from {path} "
        f"(jsonl_lines={stats['lines']} malformed={stats['malformed']} "
        f"ibm_fallback={stats['ibm_fallback']})",
        file=sys.stderr,
    )
    return records, stats


def _json_sanitize(obj: Any) -> Any:
    """Recursively coerce *obj* into a JSON-serializable structure.

    Handles the two cases json.dumps rejects outright (and that a custom
    JSONEncoder.default cannot intercept, because it only sees *values*):
    - dict keys that are not str/int/float/bool/None — e.g. a Python ``type``
      returned by an IBM tool execution. Such keys are stringified.
    - arbitrary non-serializable values — stringified as a last resort.
    This guarantees dataset assembly never crashes on one pathological tool
    result (TypeError: keys must be str, int, float, bool or None, not type).
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        out: Dict[Any, Any] = {}
        for k, v in obj.items():
            if k is not None and not isinstance(k, (str, int, float, bool)):
                k = str(k)
            out[k] = _json_sanitize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    return str(obj)


def _safe_json_dumps(value: Any) -> str:
    """Serialize value to JSON string; coerce non-serializable keys/values to str."""
    if value is None:
        return ""
    return json.dumps(_json_sanitize(value), ensure_ascii=False)


def records_to_toolr0_dataset(records: List[Dict[str, Any]]):
    from datasets import Dataset

    rows = []
    skipped = 0
    for rec in records:
        try:
            gold_output = rec["gold_output"]
            if not isinstance(gold_output, str):
                gold_output = _safe_json_dumps(gold_output)
            exp = rec.get("expected_result")
            rows.append({
                "prompt": rec["prompt"],
                "gold_output": gold_output,
                "gold_answer": str(rec["gold_answer"]) if rec.get("gold_answer") is not None else "",
                "sample_id": rec["sample_id"],
                "stage": rec["stage"],
                "num_calls": rec["num_calls"],
                "turn_idx": rec.get("turn_idx", 0),
                "task_num_calls": rec.get("task_num_calls", rec["num_calls"]),
                "expected_result": _safe_json_dumps(exp) if exp is not None else "",
                "prefix_calls": _safe_json_dumps(rec.get("prefix_calls") or []),
            })
        except Exception as exc:  # never let one pathological row abort the whole run
            skipped += 1
            print(
                f"[data] skip {rec.get('sample_id', '?')}: dataset_build_error={exc}",
                file=sys.stderr,
            )
    if skipped:
        print(f"[data] tool_r0 dataset: skipped {skipped} unserializable rows", file=sys.stderr)
    if not rows:
        raise SystemExit("[data] tool_r0 dataset empty after sanitization")
    return Dataset.from_list(rows)
