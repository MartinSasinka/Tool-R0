#!/usr/bin/env python3
"""
step3_select_curriculum.py

Select up to n_final verified candidates and write NESTFUL-compatible JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from context_budget import (
    DEFAULT_TARGET_MAX_COMPLETION_TOKENS,
    DEFAULT_TARGET_PROMPT_TOKENS,
    compact_tools_list,
    estimate_training_context,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")

_VAR_REF_RE = re.compile(r"^\$var_(\d+)(?:\.([A-Za-z_][\w]*))?\$$")
FINAL_FIELDS = ("sample_id", "input", "tools", "output", "gold_answer")


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\d", "0", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def coerce_json(val: Any) -> Optional[Any]:
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return None
    return None


def output_fingerprint(calls: List[Dict[str, Any]]) -> str:
    return sha1(json.dumps(calls, sort_keys=True, ensure_ascii=False))


def input_fingerprint(inp: str) -> str:
    return sha1(normalize_text(inp))


def extract_references_from_value(value: Any) -> List[tuple]:
    refs = []
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

    return max(depth(i) for i in range(1, n + 1))


def count_references(calls: List[Dict[str, Any]]) -> int:
    n = 0
    for c in calls:
        args = c.get("arguments", {})
        if isinstance(args, dict):
            for v in args.values():
                n += len(extract_references_from_value(v))
    return n


def call2_refs_call1(calls: List[Dict[str, Any]]) -> bool:
    if len(calls) < 2:
        return False
    args = calls[1].get("arguments", {})
    if not isinstance(args, dict):
        return False
    for v in args.values():
        for ref_idx, _ in extract_references_from_value(v):
            if ref_idx == 1:
                return True
    return False


def call_name_pattern(calls: List[Dict[str, Any]]) -> str:
    return "->".join(str(c.get("name", "")) for c in calls)


def score_sample(sample: Dict[str, Any], epoch: int, target_prompt: int, target_completion: int) -> float:
    calls = sample.get("output")
    if isinstance(calls, str):
        calls = coerce_json(calls)
    if not isinstance(calls, list):
        return 0.0

    tools = sample.get("tools")
    if isinstance(tools, str):
        tools = coerce_json(tools)
    inp = sample.get("input", "")

    score = 1.0
    if isinstance(inp, str) and len(inp.strip()) < 25:
        score -= 0.1

    if isinstance(tools, list) and isinstance(inp, str):
        ctx = estimate_training_context(inp, tools, calls)
        meta = sample.get("meta") or {}
        if meta.get("context_est"):
            ctx = meta["context_est"]
        # Prefer samples comfortably under GRPO budget
        prompt_ratio = ctx["prompt_tokens_est"] / max(1, target_prompt)
        completion_ratio = ctx["completion_tokens_est"] / max(1, target_completion)
        if prompt_ratio > 1.0:
            score -= 2.0 * (prompt_ratio - 1.0)
        elif prompt_ratio < 0.75:
            score += 0.15
        if completion_ratio > 1.0:
            score -= 2.0 * (completion_ratio - 1.0)
        elif completion_ratio < 0.6:
            score += 0.1

    depth = compute_dependency_depth(calls)
    score += depth * 0.1

    if epoch >= 2:
        ref_count = count_references(calls)
        score += ref_count * 0.15
        if epoch == 2 and call2_refs_call1(calls):
            score += 0.5

    return score


def load_verified(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("verified JSON must be a list")
    return [x for x in data if isinstance(x, dict)]


def select_samples(
    items: List[Dict[str, Any]],
    n_final: int,
    epoch: int,
    seed: int,
    target_prompt: int,
    target_completion: int,
    pattern_cap_ratio: float = 0.05,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rng = random.Random(seed)
    scored = [
        (score_sample(it, epoch, target_prompt, target_completion), it) for it in items
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    buckets: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for sc, it in scored:
        buckets[sc].append(it)
    ordered: List[Dict[str, Any]] = []
    for sc in sorted(buckets.keys(), reverse=True):
        bucket = buckets[sc][:]
        bucket.sort(key=lambda x: (x.get("meta") or {}).get("candidate_id", x.get("input", "")))
        rng.shuffle(bucket)
        ordered.extend(bucket)

    selected: List[Dict[str, Any]] = []
    seen_inputs: Set[str] = set()
    seen_outputs: Set[str] = set()
    pattern_counts: Counter = Counter()
    max_per_pattern = max(1, int(n_final * pattern_cap_ratio))
    skip_stats = Counter()

    for it in ordered:
        if len(selected) >= n_final:
            break
        inp = it.get("input", "")
        calls = it.get("output")
        if isinstance(calls, str):
            calls = coerce_json(calls)
        if not isinstance(inp, str) or not isinstance(calls, list):
            skip_stats["invalid_record"] += 1
            continue

        ifp = input_fingerprint(inp)
        ofp = output_fingerprint(calls)
        if ifp in seen_inputs:
            skip_stats["duplicate_input"] += 1
            continue
        if ofp in seen_outputs:
            skip_stats["duplicate_output"] += 1
            continue

        pattern = call_name_pattern(calls)
        if pattern_counts[pattern] >= max_per_pattern:
            skip_stats["pattern_cap"] += 1
            continue

        seen_inputs.add(ifp)
        seen_outputs.add(ofp)
        pattern_counts[pattern] += 1
        selected.append(it)

    return selected, dict(skip_stats)


def format_gold_answer(gold: Any) -> str:
    if isinstance(gold, str):
        return gold
    return json.dumps(gold, ensure_ascii=False)


def to_nestful_row(sample: Dict[str, Any], epoch: int, idx: int) -> Dict[str, str]:
    tools = sample["tools"]
    output = sample["output"]
    if isinstance(tools, str):
        tools = coerce_json(tools)
    if isinstance(output, str):
        output = coerce_json(output)
    if isinstance(tools, list):
        tools = compact_tools_list(tools)
    row = {
        "sample_id": f"synthetic-epoch{epoch}-{idx:06d}",
        "input": sample["input"],
        "tools": json.dumps(tools, ensure_ascii=False, separators=(",", ":")),
        "output": json.dumps(output, ensure_ascii=False, separators=(",", ":")),
        "gold_answer": format_gold_answer(sample["gold_answer"]),
    }
    if set(row.keys()) != set(FINAL_FIELDS):
        raise ValueError("internal error: final row has wrong fields")
    return row


def write_reports(
    epoch: int,
    summary: Dict[str, Any],
    previews: List[Dict[str, Any]],
) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    txt_path = os.path.join(REPORTS_DIR, f"step3_epoch{epoch}_report.txt")
    json_path = os.path.join(REPORTS_DIR, f"step3_epoch{epoch}_summary.json")

    lines = [
        f"Step3 selection report — epoch {epoch}",
        "",
        f"Verified input: {summary.get('verified_input')}",
        f"Selected: {summary.get('selected_count')}",
        f"Target n_final: {summary.get('n_final')}",
        f"Shortfall: {summary.get('shortfall')}",
        "",
        "Selection skips:",
    ]
    for k, v in summary.get("selection_skips", {}).items():
        lines.append(f"  {k}: {v}")
    lines.extend(["", "Output length distribution (selected):"])
    for k, v in summary.get("output_length_distribution", {}).items():
        lines.append(f"  len={k}: {v}")
    lines.extend(["", "Dependency depth distribution (selected):"])
    for k, v in summary.get("dependency_depth_distribution", {}).items():
        lines.append(f"  depth={k}: {v}")
    lines.extend(["", "Top call-name sequences:"])
    for pat, cnt in summary.get("top_call_patterns", [])[:10]:
        lines.append(f"  {pat}: {cnt}")
    lines.extend(["", "Top tool names:"])
    for name, cnt in summary.get("top_tool_names", [])[:15]:
        lines.append(f"  {name}: {cnt}")
    lines.extend(["", "Selected previews (up to 5):"])
    for i, ex in enumerate(previews[:5], 1):
        lines.append(f"--- {i} ---")
        lines.append(json.dumps(ex, ensure_ascii=False)[:800])

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[select] reports -> {txt_path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Select NESTFUL curriculum samples")
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--n_final", type=int, default=400)
    ap.add_argument("--epoch", type=int, choices=list(range(1, 8)), required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_prompt_tokens", type=int, default=DEFAULT_TARGET_PROMPT_TOKENS)
    ap.add_argument(
        "--target_max_completion_tokens",
        type=int,
        default=DEFAULT_TARGET_MAX_COMPLETION_TOKENS,
    )
    args = ap.parse_args()

    items = load_verified(args.in_json)
    selected, skip_stats = select_samples(
        items,
        args.n_final,
        args.epoch,
        args.seed,
        args.target_prompt_tokens,
        args.target_max_completion_tokens,
    )

    shortfall = max(0, args.n_final - len(selected))
    if shortfall > 0:
        print(
            f"[select] WARNING: SHORTFALL of {shortfall} samples "
            f"(selected {len(selected)} / target {args.n_final}) — not padding",
            file=sys.stderr,
        )

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    tool_counter: Counter = Counter()
    out_len_counter: Counter = Counter()
    depth_counter: Counter = Counter()
    pattern_counter: Counter = Counter()
    previews: List[Dict[str, Any]] = []

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(selected, start=1):
            row = to_nestful_row(sample, args.epoch, idx)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            calls = json.loads(row["output"])
            out_len_counter[str(len(calls))] += 1
            depth_counter[str(compute_dependency_depth(calls))] += 1
            pattern_counter[call_name_pattern(calls)] += 1
            for c in calls:
                if isinstance(c.get("name"), str):
                    tool_counter[c["name"]] += 1
            meta = sample.get("meta") or {}
            if len(previews) < 5:
                previews.append(
                    {
                        "sample_id": row["sample_id"],
                        "candidate_id": meta.get("candidate_id"),
                        "raw_id": meta.get("raw_id"),
                        "input": row["input"][:200],
                        "output_len": len(calls),
                        "dependency_depth": compute_dependency_depth(calls),
                    }
                )

    summary = {
        "stage": "step3_select_curriculum",
        "epoch": args.epoch,
        "verified_input": len(items),
        "selected_count": len(selected),
        "n_final": args.n_final,
        "shortfall": shortfall,
        "selection_skips": skip_stats,
        "output_length_distribution": dict(out_len_counter),
        "dependency_depth_distribution": dict(depth_counter),
        "top_call_patterns": pattern_counter.most_common(20),
        "unique_tool_names": len(tool_counter),
        "top_tool_names": tool_counter.most_common(20),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "out_jsonl": args.out_jsonl,
    }
    write_reports(args.epoch, summary, previews)
    print(f"[select] wrote {len(selected)} rows -> {args.out_jsonl}", file=sys.stderr)


if __name__ == "__main__":
    main()
