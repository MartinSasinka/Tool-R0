#!/usr/bin/env python3
"""Analyze dataset uniqueness for curriculum v3.1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import load_jsonl, repo_root, write_csv  # noqa: E402
from uniqueness_utils_v3_1 import STAGE_FILES, analyze_all_stages  # noqa: E402


def _load_stages(base: Path) -> dict:
    out = {}
    for stage, fname in STAGE_FILES.items():
        path = base / fname
        if path.is_file():
            out[stage] = load_jsonl(path)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3_1",
    )
    ap.add_argument("--use-filtered", action="store_true", default=True)
    args = ap.parse_args()

    filtered = args.in_dir / "filtered"
    base = filtered if args.use_filtered and filtered.is_dir() else args.in_dir
    stage_samples = _load_stages(base)
    if not stage_samples:
        print("[analyze_uniqueness] no stage files found", file=sys.stderr)
        return 1

    analysis = analyze_all_stages(stage_samples)
    dup_rows = []
    for stage, data in analysis["per_stage"].items():
        for row in data.get("duplicate_sample_examples", []):
            dup_rows.append(row)

    summary = {
        "status": analysis["status"],
        "source_dir": str(base),
        **analysis["overall"],
        "per_stage": {
            k: {kk: vv for kk, vv in v.items() if kk != "duplicate_sample_examples"}
            for k, v in analysis["per_stage"].items()
        },
    }

    args.in_dir.mkdir(parents=True, exist_ok=True)
    (args.in_dir / "dataset_uniqueness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(
        args.in_dir / "dataset_duplicate_samples.csv",
        dup_rows,
        ["sample_id", "stage", "exact_count", "trace_count", "question", "tool_sequence", "trajectory_id"],
    )

    lines = [
        "# Dataset Uniqueness Report (v3.1)",
        "",
        f"Overall status: **{analysis['status']}**",
        f"- exact_duplicate_count (all stages): {summary['exact_duplicate_count']}",
        f"- mean_unique_question_ratio: {summary['mean_unique_question_ratio']}",
        f"- mean_trace_duplicate_ratio: {summary['mean_trace_duplicate_ratio']}",
        "",
        "## Per-stage summary",
        "",
        "| Stage | Status | N | Unique Q ratio | Exact dup | Trace dup ratio | Tools |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for stage, data in analysis["per_stage"].items():
        short = stage.replace("_", " ")[:28]
        lines.append(
            f"| {short} | {data['status']} | {data['total_samples']} | "
            f"{data['unique_question_ratio']:.3f} | {data['exact_duplicate_count']} | "
            f"{data['trace_duplicate_ratio']:.3f} | {data['used_tool_count']} |"
        )

    lines += ["", "## Most duplicated signature types", ""]
    for stage, data in analysis["per_stage"].items():
        lines.append(f"### {stage}")
        if data["top_trace_duplicates"]:
            lines.append("- Top trace duplicates:")
            for item in data["top_trace_duplicates"][:5]:
                lines.append(f"  - count={item['count']} hash={item['hash'][:12]}...")
        if data["top_template_duplicates"]:
            lines.append("- Top question-template duplicates:")
            for item in data["top_template_duplicates"][:5]:
                lines.append(f"  - count={item['count']}: `{item['template']}`")
        if data["top_tool_sequence_duplicates"]:
            lines.append("- Top tool-sequence duplicates:")
            for item in data["top_tool_sequence_duplicates"][:5]:
                lines.append(f"  - {item['sequence']} (count={item['count']})")
        lines.append("")

    lines += [
        "## Recommendations",
        "- Regenerate upsampled stage2/3/4 slots with dedup-aware trajectory generation.",
        "- Allow skill repetition (motif/tool sequence) but reject exact and excessive trace duplicates.",
        "- Increase question-template variants per tool family.",
        "",
        "## Gates",
        "- Hard fail: exact_duplicate_count > 0, unique_question_ratio < 0.40, stage count < 800",
        "- Soft warn: trace_duplicate_ratio > 0.05, template_duplicate_ratio > 0.30",
    ]
    (args.in_dir / "DATASET_UNIQUENESS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"[analyze_uniqueness] status={analysis['status']} exact_dup={summary['exact_duplicate_count']} "
        f"mean_uq={summary['mean_unique_question_ratio']}"
    )
    return 1 if analysis["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
