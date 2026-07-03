#!/usr/bin/env python3
"""Unified preflight gates before curriculum v3 training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import load_jsonl, repo_root  # noqa: E402


def _load_json(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _validation_stats(out_dir: Path) -> dict:
    report = out_dir / "synthetic_validation_report.md"
    text = report.read_text(encoding="utf-8") if report.is_file() else ""
    failures = 0
    total = 0
    dup = 0
    if "Tasks checked:" in text:
        for line in text.splitlines():
            if line.startswith("Tasks checked:"):
                total = int(line.split(":")[1].strip())
            if line.startswith("Failures:"):
                failures = int(line.split(":")[1].strip().split()[0])
            if line.startswith("Duplicate IDs:"):
                dup = int(line.split(":")[1].strip())
    invalid_rate = failures / max(total, 1)
    return {
        "total_tasks": total,
        "validation_failures": failures,
        "invalid_task_rate": invalid_rate,
        "duplicate_task_ids": dup,
    }


def _invalid_ref_rate(out_dir: Path) -> float:
    summary = _load_json(out_dir / "synthetic_gold_replay_summary.json")
    inv = summary.get("invalid_reference_count", 0)
    n = summary.get("tasks", 0)
    return inv / max(n, 1)


def _leakage_detected(out_dir: Path) -> bool:
    csv_path = out_dir / "synthetic_validation_failures.csv"
    if not csv_path.is_file():
        return False
    text = csv_path.read_text(encoding="utf-8")
    return "leakage:dev_or_test_id" in text


def decide_status(
    *,
    prototype_only: bool,
    hard_fails: list,
    soft_fails: list,
    tool_status: str,
    pilot_ready: bool = False,
) -> str:
    if hard_fails:
        return "FAIL"
    if tool_status in ("final_ready", "final_experiment_ready") and not soft_fails:
        return "PASS_FINAL_READY"
    if pilot_ready and tool_status == "pilot_ready" and not hard_fails:
        return "PASS_PILOT_READY"
    if prototype_only or soft_fails or tool_status in (
        "prototype_only", "pilot_ready", "math_only", "mixed_synthetic_prototype", "partial_tool_realism",
    ):
        return "PASS_PROTOTYPE_ONLY"
    return "PASS_FINAL_READY"


def _preflight_v3_1(out_dir: Path, prototype_only: bool, stages: set) -> int:
    curr_dir = out_dir / "curriculum_v3_1"
    replay = _load_json(curr_dir / "synthetic_gold_replay_summary.json")
    tool = _load_json(curr_dir / "tool_output_realism_summary.json")
    integrity = _load_json(curr_dir / "curriculum_integrity_summary.json")
    pf = _load_json(curr_dir / "process_filter_summary.json")
    manifest = _load_json(curr_dir / "curriculum_v3_1_manifest.json")
    alignment = _load_json(curr_dir / "question_trace_alignment_summary.json")
    uniqueness = _load_json(curr_dir / "dataset_uniqueness_summary.json")
    final_audit = _load_json(curr_dir / "final_dataset_audit_summary.json")

    hard_fails = []
    soft_fails = []

    audit_status = final_audit.get("status")
    if audit_status is None:
        hard_fails.append("missing final_dataset_audit_summary.json — run final_dataset_audit_v3_1.py")
    elif audit_status == "FAIL":
        hard_fails.append(f"final_dataset_audit status=FAIL")

    if replay.get("gold_replay_success_rate", 0) < 1.0:
        hard_fails.append(f"gold_replay_success_rate={replay.get('gold_replay_success_rate', 0)}")
    if replay.get("invalid_reference_count", 0) > 0:
        hard_fails.append("invalid_reference_count > 0")
    if pf.get("pass_rate", 0) < 1.0:
        hard_fails.append(f"process_filter_pass_rate={pf.get('pass_rate', 0)}")
    if pf.get("duplicate_ids", 0) > 0:
        hard_fails.append(f"duplicate_ids={pf.get('duplicate_ids')}")
    dup_sample_ids = final_audit.get("duplicate_sample_id_count")
    if dup_sample_ids is not None and dup_sample_ids > 0:
        hard_fails.append(f"duplicate_sample_id_count={dup_sample_ids}")
    if integrity.get("status") == "FAIL":
        hard_fails.extend(integrity.get("hard_fails", []))
    manifest_diversity = manifest.get("diversity") or {}
    if manifest_diversity.get("null_gold_answer_count", 0) > 0:
        hard_fails.append("gold_answer_null_in_training_samples")
    if integrity.get("dev_test_leakage"):
        hard_fails.append("dev/test leakage detected")

    q_align_failures = alignment.get("question_trace_alignment_failures")
    if q_align_failures is None:
        hard_fails.append("missing question_trace_alignment_summary.json — run validate_question_trace_alignment_v3_1.py")
    elif q_align_failures != 0:
        hard_fails.append(f"question_trace_alignment_failures={q_align_failures}")
    if alignment.get("ambiguous_question_count", 0) != 0:
        hard_fails.append(f"ambiguous_question_count={alignment.get('ambiguous_question_count')}")
    if alignment.get("incomplete_question_count", 0) != 0:
        hard_fails.append(f"incomplete_question_count={alignment.get('incomplete_question_count')}")
    if alignment.get("unresolved_placeholders", 0) != 0:
        hard_fails.append(f"unresolved_placeholders={alignment.get('unresolved_placeholders')}")
    if alignment.get("constant_reference_mismatch", 0) != 0:
        hard_fails.append(f"constant_reference_mismatch={alignment.get('constant_reference_mismatch')}")
    if final_audit.get("metadata_leakage_count", 0) > 0:
        hard_fails.append(f"metadata_leakage_count={final_audit.get('metadata_leakage_count')}")
    if final_audit.get("unresolved_placeholder_count", 0) > 0:
        hard_fails.append(f"unresolved_placeholder_count={final_audit.get('unresolved_placeholder_count')}")

    stage_counts = manifest.get("stages") or integrity.get("stage_counts") or {}
    exact_dup = uniqueness.get("exact_duplicate_count")
    if exact_dup is None:
        hard_fails.append("missing dataset_uniqueness_summary.json — run analyze_dataset_uniqueness_v3_1.py")
    elif exact_dup > 0:
        hard_fails.append(f"exact_duplicate_count={exact_dup}")
    per_stage_uq = uniqueness.get("per_stage") or {}
    for stage, minimum in (
        ("stage1_1call_atomic", 800),
        ("stage2_2call_dependency", 800),
        ("stage3_3call_composition", 800),
        ("stage4_4to6call_persistence", 800),
    ):
        sd = per_stage_uq.get(stage, {})
        if sd.get("total_samples", stage_counts.get(stage, 0)) < minimum:
            hard_fails.append(f"{stage} count {sd.get('total_samples', 0)} < {minimum}")
        uq_ratio = sd.get("unique_question_ratio", 1.0)
        if uq_ratio < 0.40:
            hard_fails.append(f"{stage} unique_question_ratio={uq_ratio} < 0.40")
        if stage.startswith("stage1") and uq_ratio < 0.60:
            soft_fails.append(f"{stage} unique_question_ratio={uq_ratio} below pilot target 0.60")
        elif stage.startswith("stage2") and uq_ratio < 0.60:
            soft_fails.append(f"{stage} unique_question_ratio={uq_ratio} below pilot target 0.60")
        elif stage.startswith("stage3") and uq_ratio < 0.60:
            soft_fails.append(f"{stage} unique_question_ratio={uq_ratio} below pilot target 0.60")
        elif stage.startswith("stage4") and uq_ratio < 0.50:
            soft_fails.append(f"{stage} unique_question_ratio={uq_ratio} below pilot target 0.50")
        if sd.get("trace_duplicate_ratio", 0) > 0.15:
            soft_fails.append(f"{stage} trace_duplicate_ratio={sd.get('trace_duplicate_ratio')} > 0.15")
        if sd.get("question_template_duplicate_ratio", 0) > 0.30:
            soft_fails.append(f"{stage} question_template_duplicate_ratio > 0.30")
        if sd.get("max_trajectory_id_count", 0) > 6:
            soft_fails.append(f"{stage} max_trajectory_id_count={sd.get('max_trajectory_id_count')} > 6")
        if sd.get("max_tool_sequence_share", 0) > 0.15:
            soft_fails.append(f"{stage} max_tool_sequence_share={sd.get('max_tool_sequence_share')} > 0.15")
        if sd.get("used_tool_count", 99) < 20:
            soft_fails.append(f"{stage} used_tool_count={sd.get('used_tool_count')} < 20")
    ns_stage2plus = manifest_diversity.get("non_scalar_gold_answer_share_stage2_plus", 0)
    if ns_stage2plus < 0.30:
        soft_fails.append(f"non_scalar_gold_answer_share_stage2_plus={ns_stage2plus} < 0.30")
    s3_ns = (final_audit.get("per_stage") or {}).get("stage3_3call_composition", {}).get("non_scalar_output_share")
    if s3_ns is not None and s3_ns < 0.25:
        soft_fails.append(f"stage3 non_scalar_output_share={s3_ns} < 0.25")
    if final_audit.get("status") == "WARN":
        soft_fails.append("final_dataset_audit status=WARN")

    for stage, minimum in (
        ("stage1_1call_atomic", 800),
        ("stage2_2call_dependency", 800),
        ("stage3_3call_composition", 800),
    ):
        if stage_counts.get(stage, 0) < minimum:
            hard_fails.append(f"{stage} count {stage_counts.get(stage, 0)} < {minimum}")

    reward_cfg = repo_root() / "experiments/nestful_synthetic_curriculum_v3/configs/reward_v3_1_stepwise.yaml"
    if not reward_cfg.is_file():
        hard_fails.append("missing reward_v3_1_stepwise.yaml")

    run_sh = repo_root() / "experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh"
    if run_sh.is_file():
        text = run_sh.read_text(encoding="utf-8")
        if "CURRICULUM_VERSION=v3_1" not in text and "v3_1" not in text:
            hard_fails.append("run_curriculum_v3.sh missing v3_1 support")
    else:
        hard_fails.append("missing run_curriculum_v3.sh")

    tool_status = tool.get("status", "prototype_only")
    if tool_status == "prototype_only" and not prototype_only:
        soft_fails.append("tool realism = prototype_only")
    if tool_status not in (
        "pilot_ready", "final_experiment_ready", "prototype_only",
        "partial_tool_realism", "mixed_synthetic_prototype",
    ):
        hard_fails.append(f"tool_status={tool_status}")

    if "3" in stages or "4" in stages:
        soft_fails.append("stage3/4 require advance_gates on pod")

    pilot_ready = tool_status == "pilot_ready"
    status = decide_status(
        prototype_only=prototype_only,
        hard_fails=hard_fails,
        soft_fails=soft_fails,
        tool_status=tool_status,
        pilot_ready=pilot_ready,
    )

    summary = {
        "status": status,
        "curriculum_version": "v3_1",
        "prototype_only_mode": prototype_only,
        "hard_fails": hard_fails,
        "soft_fails": soft_fails,
        "gold_replay_success_rate": replay.get("gold_replay_success_rate"),
        "process_filter_pass_rate": pf.get("pass_rate"),
        "exact_num_calls_integrity": integrity.get("exact_num_calls_integrity"),
        "tool_family_realism_status": tool_status,
        "question_trace_alignment_failures": q_align_failures,
        "ambiguous_question_count": alignment.get("ambiguous_question_count", 0),
        "incomplete_question_count": alignment.get("incomplete_question_count", 0),
        "unresolved_placeholders": alignment.get("unresolved_placeholders", 0),
        "constant_reference_mismatch": alignment.get("constant_reference_mismatch", 0),
        "used_tool_diversity": alignment.get("used_tool_diversity"),
        "offered_tool_diversity": alignment.get("offered_tool_diversity"),
        "used_tool_family_count": alignment.get("used_tool_family_count"),
        "final_dataset_audit_status": audit_status,
        "final_dataset_audit_decision": final_audit.get("pilot_decision"),
        "exact_duplicate_count": exact_dup,
        "stage4_unique_questions": alignment.get("stage4_unique_questions"),
        "curriculum_stages": stage_counts,
        "requested_stages": sorted(stages),
    }
    curr_dir.mkdir(parents=True, exist_ok=True)
    (curr_dir / "preflight_gates_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Preflight Gates Report (v3.1)",
        "",
        f"Status: **{status}**",
        "",
        f"- gold_replay_success_rate: {replay.get('gold_replay_success_rate', 'n/a')}",
        f"- process_filter_pass_rate: {pf.get('pass_rate', 'n/a')}",
        f"- exact_num_calls_integrity: {integrity.get('exact_num_calls_integrity', 'n/a')}",
        f"- tool_family_realism: {tool_status}",
        f"- question_trace_alignment_failures: {q_align_failures if q_align_failures is not None else 'missing'}",
        f"- unresolved_placeholders: {alignment.get('unresolved_placeholders', 'n/a')}",
        f"- constant_reference_mismatch: {alignment.get('constant_reference_mismatch', 'n/a')}",
        f"- ambiguous_question_count: {alignment.get('ambiguous_question_count', 'n/a')}",
        f"- incomplete_question_count: {alignment.get('incomplete_question_count', 'n/a')}",
        f"- used_tool_diversity: {alignment.get('used_tool_diversity', 'n/a')}",
        f"- stage4_unique_questions: {alignment.get('stage4_unique_questions', 'n/a')}",
        f"- final_dataset_audit: {audit_status if audit_status is not None else 'missing'}",
        f"- pilot_decision: {final_audit.get('pilot_decision', 'n/a')}",
        "",
        "## Hard failures",
    ]
    report += [f"- {x}" for x in hard_fails] or ["- (none)"]
    report += ["", "## Soft warnings"]
    report += [f"- {x}" for x in soft_fails] or ["- (none)"]
    (curr_dir / "PREFLIGHT_GATES_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[run_preflight_gates] v3_1 status={status} hard_fails={len(hard_fails)}")
    return 1 if status == "FAIL" else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs",
    )
    ap.add_argument(
        "--prototype-only",
        action="store_true",
        help="Relax motif coverage gates for explicit prototype runs",
    )
    ap.add_argument("--stages", default="1 2", help="Requested training stages")
    ap.add_argument("--curriculum-version", default="v3", choices=["v3", "v3_1"])
    args = ap.parse_args()

    out_dir = args.out_dir
    prototype_only = args.prototype_only or (
        __import__("os").environ.get("ALLOW_PROTOTYPE_TRAINING", "") == "1"
    )
    stages = set(args.stages.split())

    if args.curriculum_version == "v3_1":
        return _preflight_v3_1(out_dir, prototype_only, stages)

    val = _validation_stats(out_dir)
    audit = _load_json(out_dir / "distribution_audit_summary.json")
    replay = _load_json(out_dir / "synthetic_gold_replay_summary.json")
    tool = _load_json(out_dir / "tool_family_realism_summary.json")
    manifest = _load_json(out_dir / "curriculum_v3/curriculum_manifest.json")

    hard_fails = []
    soft_fails = []

    if val["invalid_task_rate"] > 0:
        hard_fails.append(f"invalid_task_rate={val['invalid_task_rate']:.4f}")
    if val["duplicate_task_ids"] > 0:
        hard_fails.append(f"duplicate_task_ids={val['duplicate_task_ids']}")
    if _invalid_ref_rate(out_dir) > 0:
        hard_fails.append("invalid_reference_rate > 0")
    if replay.get("gold_replay_success_rate", 0) < 1.0:
        hard_fails.append(f"gold_replay_success_rate={replay.get('gold_replay_success_rate', 0)}")
    if _leakage_detected(out_dir):
        hard_fails.append("dev/test leakage detected")

    motif_cov = float(audit.get("motif_coverage", 0))
    bf_cov = float(audit.get("baseline_failure_motif_coverage", 0))
    if not prototype_only:
        if motif_cov < 0.80:
            hard_fails.append(f"motif_coverage={motif_cov:.2%} < 80%")
        if bf_cov < 0.80:
            hard_fails.append(f"baseline_failure_motif_coverage={bf_cov:.2%} < 80%")

    tool_status = tool.get("status", "prototype_only")
    if tool_status == "prototype_only" and not prototype_only:
        soft_fails.append("tool-family realism = prototype_only (math registry)")
    elif tool_status in ("math_only", "mixed_synthetic_prototype", "partial_tool_realism") and not prototype_only:
        soft_fails.append(f"tool-family realism = {tool_status}")
    if tool_status not in (
        "pilot_ready", "final_ready", "final_experiment_ready", "prototype_only",
        "math_only", "mixed_synthetic_prototype", "partial_tool_realism",
    ):
        hard_fails.append(f"tool_family_realism_status={tool_status}")

    stages = set(args.stages.split())
    if ("3" in stages or "4" in stages) and prototype_only:
        soft_fails.append("stage3/4 requested — requires stage2 dev gates on pod")

    status = decide_status(
        prototype_only=prototype_only,
        hard_fails=hard_fails,
        soft_fails=soft_fails,
        tool_status=tool_status,
    )

    summary = {
        "status": status,
        "prototype_only_mode": prototype_only,
        "hard_fails": hard_fails,
        "soft_fails": soft_fails,
        "validation": val,
        "motif_coverage": motif_cov,
        "baseline_failure_motif_coverage": bf_cov,
        "gold_replay_success_rate": replay.get("gold_replay_success_rate"),
        "tool_family_realism_status": tool_status,
        "curriculum_stages": manifest.get("stages", {}),
        "requested_stages": sorted(stages),
    }
    (out_dir / "preflight_gates_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = [
        "# Preflight Gates Report",
        "",
        f"Status: **{status}**",
        f"- prototype_only mode: {prototype_only}",
        "",
        "## Gate results",
        f"- invalid_task_rate: {val['invalid_task_rate']:.4f}",
        f"- duplicate_task_ids: {val['duplicate_task_ids']}",
        f"- invalid_reference_rate: {_invalid_ref_rate(out_dir):.4f}",
        f"- gold_replay_success_rate: {replay.get('gold_replay_success_rate', 'n/a')}",
        f"- motif_coverage: {motif_cov:.1%}",
        f"- baseline_failure_motif_coverage: {bf_cov:.1%}",
        f"- tool_family_realism: {tool_status}",
        f"- dev/test leakage: {_leakage_detected(out_dir)}",
        "",
        "## Hard failures",
    ]
    report += [f"- {x}" for x in hard_fails] or ["- (none)"]
    report += ["", "## Soft warnings"]
    report += [f"- {x}" for x in soft_fails] or ["- (none)"]
    report += [
        "",
        "## Training policy",
        "- `FAIL`: training must NOT start",
        "- `PASS_PROTOTYPE_ONLY`: training allowed only with `ALLOW_PROTOTYPE_TRAINING=1`",
        "- `PASS_FINAL_READY`: training allowed without prototype override",
        "",
        f"Recommended: **{status}**",
    ]
    (out_dir / "PREFLIGHT_GATES_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[run_preflight_gates] status={status} hard_fails={len(hard_fails)}")
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
