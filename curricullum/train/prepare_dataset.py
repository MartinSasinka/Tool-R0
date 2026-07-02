#!/usr/bin/env python3
"""
prepare_dataset.py

Load NESTFUL-compatible curriculum JSONL and build GRPO training records.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
if DATA_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(DATA_DIR))

from context_budget import estimate_prompt_tokens  # noqa: E402

FINAL_FIELDS = ("sample_id", "input", "tools", "output", "gold_answer")

PROMPT_RULES = """You are solving a NESTFUL-style nested tool-calling task.

Given a user request and available tools, plan the full multi-step tool trajectory and final answer.

You may write brief reasoning before the JSON. The graded answer must be a single JSON object with keys output and answer (no markdown fences).

Rules:
- output must be a JSON array of ALL tool calls in execution order (one object per call).
- Each call object must include: name, label, arguments.
- Labels must be sequential: $var_1, $var_2, $var_3, ...
- When a later call depends on an earlier result, use references like $var_1.output_0$ in arguments.
- answer is the final user-facing result after executing the trajectory.
- Use only tools from the provided list.
- Do not leave output empty if the task requires tool calls.

Example shape (3-call task):
{"output":[{"name":"divide","label":"$var_1","arguments":{"arg_0":25,"arg_1":100}},{"name":"multiply","label":"$var_2","arguments":{"arg_0":"$var_1.output_0$","arg_1":80}},{"name":"add","label":"$var_3","arguments":{"arg_0":20,"arg_1":"$var_2.output_0$"}}],"answer":"40"}
"""


def coerce_json(val: Any) -> Optional[Any]:
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return None
    return None


def parse_epoch_from_id(sample_id: str) -> Optional[int]:
    m = re.search(r"epoch(\d+)", sample_id or "")
    return int(m.group(1)) if m else None


def parse_row(row: Dict[str, Any], default_num_calls: Optional[int] = None) -> Optional[Dict[str, Any]]:
    keys = set(row.keys())
    if keys != set(FINAL_FIELDS):
        return None
    if not isinstance(row.get("sample_id"), str):
        return None
    if not isinstance(row.get("input"), str) or not row["input"].strip():
        return None
    if not isinstance(row.get("gold_answer"), str):
        return None

    tools = coerce_json(row.get("tools"))
    output = coerce_json(row.get("output"))
    if not isinstance(tools, list) or not isinstance(output, list):
        return None
    if not output:
        return None

    num_calls = default_num_calls
    if num_calls is None:
        num_calls = parse_epoch_from_id(row["sample_id"]) or len(output)

    stage = f"epoch{num_calls}"
    return {
        "sample_id": row["sample_id"],
        "input": row["input"],
        "tools": tools,
        "gold_output": output,
        "gold_answer": row["gold_answer"],
        "stage": stage,
        "num_calls": num_calls,
    }


def build_prompt_text(input_text: str, tools: List[Dict[str, Any]]) -> str:
    tools_json = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{PROMPT_RULES}\n"
        f"User request:\n{input_text}\n\n"
        f"Available tools:\n{tools_json}"
    )


def build_chat_prompt(input_text: str, tools: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    tools_json = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    user_content = f"User request:\n{input_text}\n\nAvailable tools:\n{tools_json}"
    return [
        {"role": "system", "content": PROMPT_RULES.strip()},
        {"role": "user", "content": user_content},
    ]


def estimate_record_prompt_tokens(record: Dict[str, Any]) -> int:
    return estimate_prompt_tokens(record["input"], record["tools"])


def load_nestful_jsonl(
    path: str,
    default_num_calls: Optional[int] = None,
    max_prompt_tokens: Optional[int] = None,
    skip_over_budget: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not os.path.isfile(path):
        print(f"[data] missing file: {path}", file=sys.stderr)
        sys.exit(1)

    records: List[Dict[str, Any]] = []
    stats = {"malformed": 0, "skipped_prompt_budget": 0, "lines": 0}

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            stats["lines"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed"] += 1
                print(f"[data] malformed line {line_no}: json_decode_error", file=sys.stderr)
                continue
            if not isinstance(row, dict):
                stats["malformed"] += 1
                continue

            parsed = parse_row(row, default_num_calls=default_num_calls)
            if parsed is None:
                stats["malformed"] += 1
                print(f"[data] malformed line {line_no}: invalid_fields", file=sys.stderr)
                continue

            if max_prompt_tokens is not None:
                est = estimate_record_prompt_tokens(parsed)
                if est > max_prompt_tokens:
                    stats["skipped_prompt_budget"] += 1
                    if skip_over_budget:
                        print(
                            f"[data] skip line {line_no} ({parsed['sample_id']}): "
                            f"prompt_tokens_est={est} > {max_prompt_tokens}",
                            file=sys.stderr,
                        )
                        continue

            parsed["prompt"] = build_chat_prompt(parsed["input"], parsed["tools"])
            parsed["prompt_text"] = build_prompt_text(parsed["input"], parsed["tools"])
            parsed["prompt_tokens_est"] = estimate_record_prompt_tokens(parsed)
            records.append(parsed)

    if not records:
        print(f"[data] no valid rows in {path}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[data] loaded {len(records)} rows from {path} "
        f"(malformed={stats['malformed']} skipped_budget={stats['skipped_prompt_budget']})",
        file=sys.stderr,
    )
    return records, stats


def records_to_dataset(records: List[Dict[str, Any]]) -> Dataset:
    rows = []
    for rec in records:
        gold_output = rec["gold_output"]
        if not isinstance(gold_output, str):
            gold_output = json.dumps(gold_output, ensure_ascii=False, separators=(",", ":"))
        rows.append(
            {
                "prompt": rec["prompt"],
                "gold_output": gold_output,
                "gold_answer": rec["gold_answer"],
                "sample_id": rec["sample_id"],
                "stage": rec["stage"],
                "num_calls": rec["num_calls"],
            }
        )
    return Dataset.from_list(rows)


def filter_by_tokenizer_length(
    records: List[Dict[str, Any]],
    tokenizer,
    max_prompt_length: int,
) -> Tuple[List[Dict[str, Any]], int]:
    kept: List[Dict[str, Any]] = []
    skipped = 0
    for rec in records:
        try:
            text = tokenizer.apply_chat_template(
                rec["prompt"],
                tokenize=False,
                add_generation_prompt=True,
            )
            n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:
            est = rec.get("prompt_tokens_est")
            if est:
                n_tokens = est * 3
            else:
                # No precomputed estimate (e.g. tool_r0 multi-turn records): approximate
                # from the serialized prompt (~4 chars/token) so an over-budget row is
                # still dropped instead of being silently kept at n_tokens=0.
                raw = rec.get("prompt")
                try:
                    raw_len = len(raw) if isinstance(raw, str) else len(json.dumps(raw, ensure_ascii=False))
                except Exception:
                    raw_len = 0
                n_tokens = raw_len // 4

        if n_tokens > max_prompt_length:
            skipped += 1
            print(
                f"[data] skip {rec['sample_id']}: tokenizer_len={n_tokens} > {max_prompt_length}",
                file=sys.stderr,
            )
            continue
        rec["prompt_token_len"] = n_tokens
        kept.append(rec)

    if not kept:
        print(
            f"[data] all rows exceed max_prompt_length={max_prompt_length}",
            file=sys.stderr,
        )
        sys.exit(1)
    return kept, skipped
