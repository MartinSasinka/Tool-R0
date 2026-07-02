#!/usr/bin/env python3
"""
inspect_training_data.py

Summarize NESTFUL curriculum JSONL for GRPO training readiness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if DATA_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(DATA_DIR))

from context_budget import DEFAULT_TARGET_PROMPT_TOKENS, estimate_prompt_tokens  # noqa: E402
from prepare_dataset import FINAL_FIELDS, build_chat_prompt, coerce_json, parse_row  # noqa: E402


def truncate(s: str, n: int = 400) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect curriculum JSONL for training")
    ap.add_argument("--path", required=True)
    ap.add_argument("--max_preview", type=int, default=2)
    ap.add_argument("--max_prompt_tokens", type=int, default=DEFAULT_TARGET_PROMPT_TOKENS)
    ap.add_argument("--grpo_max_prompt_length", type=int, default=4096)
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    malformed = Counter()
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
                continue
            if not isinstance(row, dict):
                malformed["not_object"] += 1
                continue
            if set(row.keys()) != set(FINAL_FIELDS):
                malformed["bad_fields"] += 1
                continue
            parsed = parse_row(row)
            if parsed is None:
                malformed["parse_failed"] += 1
                continue
            rows.append(parsed)

    print(f"Valid rows: {len(rows)}")
    if malformed:
        print(f"Malformed rows: {sum(malformed.values())}")
        for k, v in malformed.most_common():
            print(f"  {k}: {v}")

    if not rows:
        print("[data] no valid rows", file=sys.stderr)
        sys.exit(1)

    out_lens = Counter(len(r["gold_output"]) for r in rows)
    tool_counts = Counter(len(r["tools"]) for r in rows)
    prompt_est = [estimate_prompt_tokens(r["input"], r["tools"]) for r in rows]
    over_budget = sum(1 for p in prompt_est if p > args.max_prompt_tokens)
    over_grpo = sum(1 for p in prompt_est if p > args.grpo_max_prompt_length)

    print("Output length distribution:")
    for k in sorted(out_lens):
        print(f"  len={k}: {out_lens[k]}")
    print("Tools per sample distribution:")
    for k in sorted(tool_counts):
        print(f"  tools={k}: {tool_counts[k]}")
    print(
        f"Prompt tokens est (char//3): min={min(prompt_est)} max={max(prompt_est)} "
        f"avg={sum(prompt_est)//len(prompt_est)}"
    )
    print(f"Over curriculum budget (>{args.max_prompt_tokens}): {over_budget}/{len(rows)}")
    print(f"Over GRPO max_prompt_length (>{args.grpo_max_prompt_length}): {over_grpo}/{len(rows)}")

    gold_leaks = 0
    for rec in rows:
        chat = build_chat_prompt(rec["input"], rec["tools"])
        user_tools = json.dumps(chat[1:], ensure_ascii=False)
        gold_out_blob = json.dumps(rec["gold_output"], ensure_ascii=False, separators=(",", ":"))
        if gold_out_blob in user_tools:
            gold_leaks += 1
        elif len(rec["gold_answer"]) >= 8 and rec["gold_answer"] in user_tools:
            gold_leaks += 1
    print(f"Prompt gold leakage check: {gold_leaks}/{len(rows)} rows contain gold output/answer in user+tools")

    print("\nSample prompt previews:")
    for rec in rows[: args.max_preview]:
        chat = build_chat_prompt(rec["input"], rec["tools"])
        user = chat[1]["content"] if len(chat) > 1 else ""
        print("---")
        print(f"id: {rec['sample_id']} num_calls={rec['num_calls']}")
        print(f"prompt_tokens_est: {estimate_prompt_tokens(rec['input'], rec['tools'])}")
        print(f"user: {truncate(user, 500)}")


if __name__ == "__main__":
    main()
