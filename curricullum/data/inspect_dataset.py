#!/usr/bin/env python3
"""
inspect_dataset.py

Print a human-readable summary of a NESTFUL curriculum JSONL file.
Skips malformed rows instead of crashing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from context_budget import (
    DEFAULT_TARGET_MAX_COMPLETION_TOKENS,
    DEFAULT_TARGET_PROMPT_TOKENS,
    estimate_training_context,
)

_VAR_REF_RE = re.compile(r"^\$var_(\d+)(?:\.([A-Za-z_][\w]*))?\$$")
FINAL_FIELDS = ("sample_id", "input", "tools", "output", "gold_answer")


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
    m = re.search(r"epoch(\d+)", sample_id)
    if m:
        return int(m.group(1))
    return None


def extract_references_from_value(value: Any) -> List[Tuple[int, Optional[str]]]:
    refs: List[Tuple[int, Optional[str]]] = []
    if isinstance(value, str):
        m = _VAR_REF_RE.match(value.strip())
        if m:
            refs.append((int(m.group(1)), m.group(2)))
    elif isinstance(value, list):
        for item in value:
            refs.extend(extract_references_from_value(item))
    return refs


def compute_dependency_depth(calls: List[Dict[str, Any]]) -> int:
    n = len(calls)
    if n == 0:
        return 0
    refs_by_call: Dict[int, set] = {}
    for i, call in enumerate(calls, start=1):
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            continue
        for v in args.values():
            for ref_idx, _ in extract_references_from_value(v):
                if ref_idx < i:
                    refs_by_call.setdefault(i, set()).add(ref_idx)
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

    return max(depth(i) for i in range(1, n + 1))


def count_invalid_references(output: List[Dict[str, Any]]) -> int:
    invalid = 0
    for i, call in enumerate(output, start=1):
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            continue
        for v in args.values():
            for ref_idx, _ in extract_references_from_value(v):
                if ref_idx >= i:
                    invalid += 1
    return invalid


def call_name_pattern(calls: List[Dict[str, Any]]) -> str:
    return "->".join(str(c.get("name", "")) for c in calls)


def truncate(s: str, n: int = 300) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def validate_row_fields(row: Dict[str, Any]) -> Optional[str]:
    keys = set(row.keys())
    if keys != set(FINAL_FIELDS):
        extra = keys - set(FINAL_FIELDS)
        missing = set(FINAL_FIELDS) - keys
        parts = []
        if extra:
            parts.append(f"extra={sorted(extra)}")
        if missing:
            parts.append(f"missing={sorted(missing)}")
        return "bad_fields:" + ",".join(parts)
    if not isinstance(row.get("sample_id"), str):
        return "bad_sample_id"
    if not isinstance(row.get("input"), str):
        return "bad_input"
    if not isinstance(row.get("tools"), str):
        return "tools_not_string"
    if not isinstance(row.get("output"), str):
        return "output_not_string"
    if not isinstance(row.get("gold_answer"), str):
        return "gold_answer_not_string"
    if coerce_json(row.get("tools")) is None:
        return "tools_invalid_json"
    if coerce_json(row.get("output")) is None:
        return "output_invalid_json"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect NESTFUL curriculum JSONL")
    ap.add_argument("--path", required=True)
    ap.add_argument("--max_examples", type=int, default=3)
    ap.add_argument("--target_prompt_tokens", type=int, default=DEFAULT_TARGET_PROMPT_TOKENS)
    ap.add_argument(
        "--target_max_completion_tokens",
        type=int,
        default=DEFAULT_TARGET_MAX_COMPLETION_TOKENS,
    )
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    malformed: Counter = Counter()
    line_no = 0

    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            line_no += 1
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed["json_decode_error"] += 1
                print(f"[inspect] malformed line {line_no}: json_decode_error", file=sys.stderr)
                continue
            if not isinstance(row, dict):
                malformed["not_object"] += 1
                continue
            err = validate_row_fields(row)
            if err:
                malformed[err] += 1
                print(f"[inspect] malformed line {line_no}: {err}", file=sys.stderr)
                continue
            rows.append(row)

    epoch_counter: Counter = Counter()
    out_len_counter: Counter = Counter()
    depth_counter: Counter = Counter()
    pattern_counter: Counter = Counter()
    tool_names: set = set()
    invalid_ref_total = 0
    sample_ids: List[str] = []
    prompt_tokens: List[int] = []
    completion_tokens: List[int] = []
    total_tokens: List[int] = []
    over_prompt = 0
    over_completion = 0

    for row in rows:
        sid = row.get("sample_id", "")
        sample_ids.append(sid)
        ep = parse_epoch_from_id(sid)
        if ep is not None:
            epoch_counter[ep] += 1

        output = coerce_json(row.get("output"))
        if isinstance(output, list):
            out_len_counter[len(output)] += 1
            invalid_ref_total += count_invalid_references(output)
            depth_counter[compute_dependency_depth(output)] += 1
            pattern_counter[call_name_pattern(output)] += 1

        tools = coerce_json(row.get("tools"))
        if isinstance(tools, list):
            for t in tools:
                if isinstance(t, dict) and isinstance(t.get("name"), str):
                    tool_names.add(t["name"])

        if isinstance(output, list) and isinstance(tools, list):
            ctx = estimate_training_context(str(row.get("input", "")), tools, output)
            prompt_tokens.append(ctx["prompt_tokens_est"])
            completion_tokens.append(ctx["completion_tokens_est"])
            total_tokens.append(ctx["total_tokens_est"])
            if ctx["prompt_tokens_est"] > args.target_prompt_tokens:
                over_prompt += 1
            if ctx["completion_tokens_est"] > args.target_max_completion_tokens:
                over_completion += 1

    print(f"Total valid examples: {len(rows)}")
    if malformed:
        print(f"Malformed rows skipped: {sum(malformed.values())}")
        for k, v in malformed.most_common():
            print(f"  {k}: {v}")
    print("Examples per epoch:")
    for ep in sorted(epoch_counter.keys()):
        print(f"  epoch {ep}: {epoch_counter[ep]}")
    print("Output length distribution:")
    for length in sorted(out_len_counter.keys()):
        print(f"  len={length}: {out_len_counter[length]}")
    print("Dependency depth distribution:")
    for depth in sorted(depth_counter.keys()):
        print(f"  depth={depth}: {depth_counter[depth]}")
    print(f"Unique tool count: {len(tool_names)}")
    print(f"Invalid reference count: {invalid_ref_total}")
    if prompt_tokens:
        print(
            f"Training context estimate (char//3, targets prompt<={args.target_prompt_tokens}, "
            f"completion<={args.target_max_completion_tokens}):"
        )
        print(
            f"  prompt tokens: min={min(prompt_tokens)} max={max(prompt_tokens)} "
            f"avg={sum(prompt_tokens) // len(prompt_tokens)}"
        )
        print(
            f"  completion tokens: min={min(completion_tokens)} max={max(completion_tokens)} "
            f"avg={sum(completion_tokens) // len(completion_tokens)}"
        )
        print(
            f"  total tokens: min={min(total_tokens)} max={max(total_tokens)} "
            f"avg={sum(total_tokens) // len(total_tokens)}"
        )
        print(f"  over prompt budget: {over_prompt}/{len(rows)}")
        print(f"  over completion budget: {over_completion}/{len(rows)}")
    print("Most common call-name sequences:")
    for pat, cnt in pattern_counter.most_common(10):
        print(f"  {pat}: {cnt}")
    print("First sample IDs:")
    for sid in sample_ids[:10]:
        print(f"  {sid}")

    print("\nReadable examples:")
    for row in rows[: args.max_examples]:
        output = coerce_json(row.get("output")) or []
        print("---")
        print(f"id: {row.get('sample_id')}")
        print(f"input: {truncate(str(row.get('input', '')))}")
        print(f"output_len: {len(output) if isinstance(output, list) else '?'}")
        if isinstance(output, list):
            print(f"dependency_depth: {compute_dependency_depth(output)}")
        print(f"gold_answer: {row.get('gold_answer')}")
        if isinstance(output, list) and output:
            print(f"first_call: {json.dumps(output[0], ensure_ascii=False)[:200]}")


if __name__ == "__main__":
    main()
