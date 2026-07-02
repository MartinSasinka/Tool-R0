#!/usr/bin/env python3
"""Validate synthetic v3 tasks (structure, refs, leakage)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import (  # noqa: E402
    VALID_MOTIF_TYPES,
    graph_matches_refs,
    load_blocked_ids,
    load_jsonl,
    repo_root,
    validate_references,
    write_csv,
)

REQUIRED = (
    "task_id", "question", "tools", "gold_calls", "gold_answer",
    "num_calls", "motif_type", "dependency_graph", "reference_pattern",
    "output_type", "difficulty_score",
)


def validate_task(task: dict, blocked_ids: set) -> list:
    errs = []
    tid = task.get("task_id", "")
    if tid in blocked_ids:
        errs.append("leakage:dev_or_test_id")
    for f in REQUIRED:
        if f not in task or task[f] is None:
            errs.append(f"missing:{f}")
    if task.get("gold_answer") is None:
        errs.append("missing:gold_answer_value")
    calls = task.get("gold_calls") or []
    if not calls:
        errs.append("empty:gold_calls")
    tools = {t.get("name") for t in (task.get("tools") or [])}
    for i, c in enumerate(calls):
        if c.get("name") not in tools:
            errs.append(f"call_{i+1}:tool_not_in_tools")
    errs.extend(validate_references(calls))
    if not graph_matches_refs(calls, task.get("dependency_graph") or {}):
        errs.append("dependency_graph_mismatch")
    mt = task.get("motif_type")
    if mt and mt not in VALID_MOTIF_TYPES:
        errs.append(f"unknown_motif_type:{mt}")
    ds = task.get("difficulty_score")
    if ds is not None and not (0.0 <= float(ds) <= 1.0):
        errs.append("difficulty_score_out_of_range")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=None,
                    help="single jsonl or curriculum dir")
    ap.add_argument("--out_dir", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs")
    args = ap.parse_args()

    paths = []
    if args.input and args.input.is_file():
        paths = [args.input]
    elif args.input and args.input.is_dir():
        paths = sorted(args.input.glob("*.jsonl"))
    else:
        cur = repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3"
        manifest = cur / "curriculum_manifest.json"
        if manifest.is_file():
            # Validate staged curriculum only (avoids duplicate IDs vs raw pool).
            paths = sorted(cur.glob("stage*.jsonl"))
        elif cur.is_dir():
            paths = sorted(cur.glob("*.jsonl"))
        if not paths:
            syn = repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/synthetic_motif_tasks.jsonl"
            if syn.is_file():
                paths = [syn]

    if not paths:
        print("ERROR: no input files", file=sys.stderr)
        return 1

    blocked = load_blocked_ids()
    failures = []
    seen_ids = set()
    total = 0
    for p in paths:
        for row in load_jsonl(p):
            total += 1
            tid = row.get("task_id", "")
            if tid in seen_ids:
                failures.append({"task_id": tid, "file": str(p), "errors": "duplicate_task_id"})
            seen_ids.add(tid)
            errs = validate_task(row, blocked)
            if errs:
                failures.append({"task_id": tid, "file": str(p), "errors": ";".join(errs)})

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "synthetic_validation_failures.csv", failures,
              ["task_id", "file", "errors"])

    rate = len(failures) / max(total, 1)
    report = [
        "# Synthetic Validation Report",
        "",
        f"Files checked: {len(paths)}",
        f"Tasks checked: {total}",
        f"Failures: {len(failures)} ({100*rate:.2f}%)",
        f"Duplicate IDs: {sum(1 for f in failures if 'duplicate' in f.get('errors', ''))}",
        "",
        "Result: **PASS**" if not failures else "Result: **FAIL**",
    ]
    (out_dir / "synthetic_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[validate_synthetic_tasks] {total} tasks, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
