#!/usr/bin/env python3
"""Process-filter prefix samples for v3.1 curriculum quality."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import graph_matches_refs, load_jsonl, repo_root, validate_references  # noqa: E402
from traj_utils_v3_1 import replay_calls, validate_trajectory  # noqa: E402
from tool_registry_v3_1 import ALL_TOOL_NAMES, infer_output_type  # noqa: E402

STAGE_EXACT = {
    "stage1_1call_atomic": 1,
    "stage2_2call_dependency": 2,
    "stage3_3call_composition": 3,
}
STAGE_RANGE = {
    "stage4_4to6call_persistence": (4, 6),
}

STAGE_FILES = [
    "stage1_1call_atomic.jsonl",
    "stage2_2call_dependency.jsonl",
    "stage3_3call_composition.jsonl",
    "stage4_4to6call_persistence.jsonl",
]


def check_sample(sample: Dict[str, Any], stage: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    sid = sample.get("sample_id", "?")

    calls = sample.get("gold_calls") or []
    n = len(calls)
    if stage in STAGE_EXACT and n != STAGE_EXACT[stage]:
        errors.append(f"exact_num_calls: expected {STAGE_EXACT[stage]} got {n}")
    if stage in STAGE_RANGE:
        lo, hi = STAGE_RANGE[stage]
        if not (lo <= n <= hi):
            errors.append(f"call_range: expected [{lo},{hi}] got {n}")

    if sample.get("prefix_length") != n:
        errors.append(f"prefix_length mismatch: {sample.get('prefix_length')} vs {n}")

    tool_names = {t.get("name") for t in sample.get("tools") or []}
    for c in calls:
        if c.get("name") not in ALL_TOOL_NAMES:
            errors.append(f"unknown tool: {c.get('name')}")
        elif c.get("name") not in tool_names:
            errors.append(f"tool not in pool: {c.get('name')}")

    ref_errs = validate_references(calls)
    errors.extend(ref_errs)

    graph = sample.get("dependency_graph") or {}
    if not graph_matches_refs(calls, graph):
        errors.append("dependency_graph mismatch")

    _, _, replay_errs = replay_calls(calls)
    errors.extend(replay_errs)

    if sample.get("gold_answer") is None:
        errors.append("gold_answer is null — training samples require prefix-level answer")

    labels = sample.get("process_labels") or []
    if len(labels) != n:
        errors.append(f"process_labels length {len(labels)} != {n}")

    seq = sample.get("output_type_sequence") or []
    if seq and len(seq) != n:
        errors.append("output_type_sequence length mismatch")

    return len(errors) == 0, errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3_1")
    args = ap.parse_args()

    filtered_dir = args.in_dir / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    all_ids: set = set()
    dup_count = 0
    total_in = 0
    total_pass = 0
    total_fail = 0
    failures: List[dict] = []
    stage_stats: Dict[str, dict] = {}

    for fname in STAGE_FILES:
        stage = fname.replace(".jsonl", "")
        path = args.in_dir / fname
        if not path.is_file():
            continue
        samples = load_jsonl(path)
        passed: List[dict] = []
        stage_fail = 0
        for s in samples:
            total_in += 1
            sid = s.get("sample_id", "")
            if sid in all_ids:
                dup_count += 1
                failures.append({"sample_id": sid, "stage": stage, "errors": ["duplicate sample_id"]})
                total_fail += 1
                stage_fail += 1
                continue
            all_ids.add(sid)
            ok, errs = check_sample(s, stage)
            if ok:
                passed.append(s)
                total_pass += 1
            else:
                total_fail += 1
                stage_fail += 1
                failures.append({"sample_id": sid, "stage": stage, "errors": errs})
        out_path = filtered_dir / fname
        with open(out_path, "w", encoding="utf-8") as fh:
            for s in passed:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        stage_stats[stage] = {"input": len(samples), "passed": len(passed), "failed": stage_fail}

    pass_rate = total_pass / max(total_in, 1)
    summary = {
        "total_input": total_in,
        "total_passed": total_pass,
        "total_failed": total_fail,
        "pass_rate": pass_rate,
        "duplicate_ids": dup_count,
        "stage_stats": stage_stats,
        "hard_fail": pass_rate < 1.0 or dup_count > 0,
    }
    (args.in_dir / "process_filter_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = [
        "# Process Filter Report (v3.1)",
        "",
        f"- pass_rate: **{pass_rate:.4f}**",
        f"- total_input: {total_in}",
        f"- total_passed: {total_pass}",
        f"- total_failed: {total_fail}",
        f"- duplicate_ids: {dup_count}",
        "",
        "## Stage stats",
    ]
    for stage, st in stage_stats.items():
        report.append(f"- {stage}: {st['passed']}/{st['input']} passed")
    report += [
        "",
        "## Notes",
        "- Optional LLM judge for ambiguous alternative traces — TODO.",
        "",
        "## Hard fail gates",
        f"- pass_rate == 1.0: {'PASS' if pass_rate >= 1.0 else 'FAIL'}",
        f"- duplicate_ids == 0: {'PASS' if dup_count == 0 else 'FAIL'}",
    ]
    (args.in_dir / "PROCESS_FILTER_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[process_filter] pass_rate={pass_rate:.4f} failed={total_fail}")
    return 1 if summary["hard_fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
