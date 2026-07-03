#!/usr/bin/env python3
"""Validate semantic alignment between question text and gold_calls (v3.1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import load_jsonl, repo_root, write_csv  # noqa: E402
from question_templates_v3_1 import (  # noqa: E402
    check_constant_reference_consistency,
    check_unresolved_placeholders,
    compute_tool_usage_stats,
    is_incomplete_question,
    validate_question_trace_alignment,
)

STAGE_GLOBS = [
    "stage1_1call_atomic.jsonl",
    "stage2_2call_dependency.jsonl",
    "stage3_3call_composition.jsonl",
    "stage4_4to6call_persistence.jsonl",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3_1",
    )
    ap.add_argument("--use-filtered", action="store_true", default=True)
    args = ap.parse_args()

    base = args.in_dir / "filtered" if args.use_filtered else args.in_dir
    failures: List[dict] = []
    all_samples: List[dict] = []
    total = 0
    ambiguous = 0
    incomplete = 0
    unresolved = 0
    const_ref_mismatch = 0
    stage4_questions: set = set()

    for fname in STAGE_GLOBS:
        path = base / fname
        if not path.is_file():
            continue
        for sample in load_jsonl(path):
            total += 1
            all_samples.append(sample)
            q = sample.get("question", "")
            calls = sample.get("gold_calls") or []
            n_calls = sample.get("num_calls", len(calls))
            errs = validate_question_trace_alignment(q, calls, num_calls=n_calls)
            if is_incomplete_question(q):
                incomplete += 1
            if check_unresolved_placeholders(q):
                unresolved += 1
            if check_constant_reference_consistency(q, calls):
                const_ref_mismatch += 1
            if "stage4" in fname:
                stage4_questions.add(q)
            if errs:
                if "incomplete_or_short_question" in errs:
                    ambiguous += 1
                failures.append({
                    "sample_id": sample.get("sample_id"),
                    "stage": sample.get("stage"),
                    "question": q[:160],
                    "tools": "->".join(c.get("name", "") for c in calls),
                    "errors": ";".join(errs),
                })

    tool_stats = compute_tool_usage_stats(all_samples)
    s4_path = base / "stage4_4to6call_persistence.jsonl"
    s4_total = len(load_jsonl(s4_path)) if s4_path.is_file() else 0

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "total_samples": total,
        "question_trace_alignment_failures": len(failures),
        "ambiguous_question_count": ambiguous,
        "incomplete_question_count": incomplete,
        "unresolved_placeholders": unresolved,
        "constant_reference_mismatch": const_ref_mismatch,
        "stage4_unique_questions": len(stage4_questions),
        "stage4_total": s4_total,
        **tool_stats,
    }

    args.in_dir.mkdir(parents=True, exist_ok=True)
    (args.in_dir / "question_trace_alignment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(
        args.in_dir / "question_trace_alignment_failures.csv",
        failures,
        ["sample_id", "stage", "question", "tools", "errors"],
    )
    report = [
        "# Question Trace Alignment Report (v3.1)",
        "",
        f"Status: **{summary['status']}**",
        "",
        "## Alignment gates",
        f"- total_samples: {total}",
        f"- alignment_failures: {len(failures)}",
        f"- unresolved_placeholders: {unresolved}",
        f"- constant_reference_mismatch: {const_ref_mismatch}",
        f"- ambiguous_question_count: {ambiguous}",
        f"- incomplete_question_count: {incomplete}",
        "",
        "## Question diversity",
        f"- stage4_unique_questions: {len(stage4_questions)} / {s4_total}",
        "",
        "## Tool usage",
        f"- offered_tool_diversity: {tool_stats['offered_tool_diversity']}",
        f"- used_tool_diversity: {tool_stats['used_tool_diversity']}",
        f"- used_tool_family_count: {tool_stats['used_tool_family_count']}",
        "",
        "Hard gates: question_trace_alignment_failures=0, unresolved_placeholders=0, constant_reference_mismatch=0",
    ]
    (args.in_dir / "QUESTION_TRACE_ALIGNMENT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        f"[validate_question_trace_alignment] status={summary['status']} failures={len(failures)} "
        f"unresolved={unresolved} const_ref={const_ref_mismatch} "
        f"used_tools={tool_stats['used_tool_diversity']}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
