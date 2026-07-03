#!/usr/bin/env python3
"""Validate v3.1 curriculum call-count integrity and quality gates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import load_blocked_ids, load_jsonl, repo_root, validate_references  # noqa: E402
from traj_utils_v3_1 import replay_calls  # noqa: E402

STAGE_RULES = {
    "stage1_1call_atomic": lambda n: n == 1,
    "stage2_2call_dependency": lambda n: n == 2,
    "stage3_3call_composition": lambda n: n == 3,
    "stage4_4to6call_persistence": lambda n: 4 <= n <= 6,
}

STAGE_FILES = {
    "stage1_1call_atomic": "stage1_1call_atomic.jsonl",
    "stage2_2call_dependency": "stage2_2call_dependency.jsonl",
    "stage3_3call_composition": "stage3_3call_composition.jsonl",
    "stage4_4to6call_persistence": "stage4_4to6call_persistence.jsonl",
}

REQUIRED_FIELDS = {
    "sample_id", "trajectory_id", "stage", "num_calls", "prefix_length",
    "source_prefix_length", "target_full_motif", "source_failure_cluster",
    "prefix_of_motif", "question", "tools", "gold_calls", "observations",
    "dependency_graph", "motif_type", "process_labels", "terminal_stage",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3_1")
    ap.add_argument("--use-filtered", action="store_true", default=True)
    ap.add_argument("--min-per-stage", type=int, default=800)
    args = ap.parse_args()

    base = args.in_dir / "filtered" if args.use_filtered else args.in_dir
    blocked = load_blocked_ids()
    hard_fails: List[str] = []
    stage_counts: Dict[str, int] = {}
    all_ids: set = set()
    invalid_refs = 0
    call_violations = 0
    replay_failures = 0
    missing_fields = 0
    leakage = False

    pf_summary_path = args.in_dir / "process_filter_summary.json"
    pf_rate = 1.0
    if pf_summary_path.is_file():
        pf_rate = json.loads(pf_summary_path.read_text(encoding="utf-8")).get("pass_rate", 1.0)

    for stage, fname in STAGE_FILES.items():
        path = base / fname
        if not path.is_file():
            hard_fails.append(f"missing stage file: {fname}")
            continue
        samples = load_jsonl(path)
        stage_counts[stage] = len(samples)
        rule = STAGE_RULES[stage]
        for s in samples:
            sid = s.get("sample_id", "")
            if sid in all_ids:
                hard_fails.append(f"duplicate sample_id: {sid}")
            all_ids.add(sid)
            if sid in blocked or s.get("trajectory_id") in blocked:
                leakage = True
            missing = REQUIRED_FIELDS - set(s.keys())
            if missing:
                missing_fields += 1
            n = int(s.get("num_calls", 0))
            if not rule(n):
                call_violations += 1
            if s.get("prefix_length") != n or s.get("source_prefix_length") != n:
                call_violations += 1
            if not s.get("prefix_of_motif"):
                call_violations += 1
            ref_errs = validate_references(s.get("gold_calls") or [])
            invalid_refs += len(ref_errs)
            _, _, errs = replay_calls(s.get("gold_calls") or [])
            replay_failures += len(errs)

    for stage, cnt in stage_counts.items():
        if cnt < args.min_per_stage:
            hard_fails.append(f"{stage} count {cnt} < {args.min_per_stage}")

    if call_violations > 0:
        hard_fails.append(f"exact_num_calls violations: {call_violations}")
    if invalid_refs > 0:
        hard_fails.append(f"invalid_references: {invalid_refs}")
    if replay_failures > 0:
        hard_fails.append(f"replay_failures: {replay_failures}")
    if leakage:
        hard_fails.append("dev/test leakage detected")
    if pf_rate < 1.0:
        hard_fails.append(f"process_filter pass_rate={pf_rate}")

    replay_summary = args.in_dir / "synthetic_gold_replay_summary.json"
    gold_rate = 1.0
    if replay_summary.is_file():
        gold_rate = json.loads(replay_summary.read_text(encoding="utf-8")).get("gold_replay_success_rate", 0)
    if gold_rate < 1.0:
        hard_fails.append(f"gold_replay_success_rate={gold_rate}")

    status = "PASS" if not hard_fails else "FAIL"
    summary = {
        "status": status,
        "stage_counts": stage_counts,
        "exact_num_calls_integrity": call_violations == 0,
        "invalid_references": invalid_refs,
        "gold_replay_success_rate": gold_rate,
        "process_filter_pass_rate": pf_rate,
        "dev_test_leakage": leakage,
        "hard_fails": hard_fails,
    }
    (args.in_dir / "curriculum_integrity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = [
        "# Curriculum Integrity Report (v3.1)",
        "",
        f"Status: **{status}**",
        "",
        "## Stage counts",
    ]
    for stage, cnt in stage_counts.items():
        report.append(f"- {stage}: {cnt}")
    report += [
        "",
        f"- exact_num_calls violations: {call_violations}",
        f"- invalid_references: {invalid_refs}",
        f"- gold_replay_success_rate: {gold_rate}",
        f"- process_filter_pass_rate: {pf_rate}",
        "",
        "## Hard failures",
    ]
    report += [f"- {x}" for x in hard_fails] or ["- (none)"]
    (args.in_dir / "CURRICULUM_INTEGRITY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[validate_integrity_v3_1] status={status}")
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
