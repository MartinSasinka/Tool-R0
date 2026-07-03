#!/usr/bin/env python3
"""Read-only final dataset audit for curriculum v3.1 (pre-pilot hard gates)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import load_jsonl, repo_root, validate_references, write_csv  # noqa: E402
from question_templates_v3_1 import (  # noqa: E402
    check_unresolved_placeholders,
    compute_tool_usage_stats,
    is_non_scalar_answer,
    stage2_task_category,
    validate_question_trace_alignment,
)
from replay_synthetic_gold_traces_v3_1 import replay_sample  # noqa: E402
from tool_registry_v3_1 import ALL_TOOL_NAMES, infer_answer_type, tool_family  # noqa: E402
from uniqueness_utils_v3_1 import STAGE_FILES, analyze_all_stages, compute_signatures  # noqa: E402

STAGE_CALL_RULES = {
    "stage1_1call_atomic": lambda n: n == 1,
    "stage2_2call_dependency": lambda n: n == 2,
    "stage3_3call_composition": lambda n: n == 3,
    "stage4_4to6call_persistence": lambda n: 4 <= n <= 6,
}

METADATA_LEAK_PATTERNS = (
    re.compile(r"cluster\s*=", re.I),
    re.compile(r"synthetic_gap", re.I),
    re.compile(r"long_chain__", re.I),
    re.compile(r"traj_v3_1_", re.I),
    re.compile(r"prefix_v3_1_", re.I),
    re.compile(r"\bdev/test\b", re.I),
    re.compile(r"\bnestful_test\b", re.I),
)


def _resolve_stage_dir(in_dir: Path, use_filtered: bool) -> Path:
    filtered = in_dir / "filtered"
    if use_filtered and filtered.is_dir():
        return filtered
    return in_dir


def _load_stages(base: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for stage, fname in STAGE_FILES.items():
        path = base / fname
        if path.is_file():
            out[stage] = load_jsonl(path)
    return out


def _call_count_histogram(samples: List[dict]) -> Dict[str, int]:
    return dict(Counter(str(s.get("num_calls", 0)) for s in samples))


def _dependency_pattern(sample: dict) -> str:
    stage = sample.get("stage", "")
    if stage == "stage2_2call_dependency":
        return stage2_task_category(sample)
    calls = sample.get("gold_calls") or []
    families = "->".join(tool_family(c.get("name", "")) for c in calls)
    ref_count = sum(
        1
        for c in calls
        for v in (c.get("arguments") or {}).values()
        if isinstance(v, str) and "$var_" in v
    )
    return f"{families}|refs={ref_count}"


def _audit_stage(stage: str, samples: List[dict]) -> dict:
    rule = STAGE_CALL_RULES[stage]
    exact_counts: Counter = Counter()
    trace_counts: Counter = Counter()
    tpl_counts: Counter = Counter()
    sample_ids: Counter = Counter()
    questions: set = set()
    tools_used: Counter = Counter()
    families_used: Counter = Counter()
    output_types: Counter = Counter()
    clusters: Counter = Counter()
    motifs: Counter = Counter()
    tool_seq: Counter = Counter()
    traj: Counter = Counter()
    dep_patterns: Counter = Counter()

    null_gold = 0
    call_violations = 0
    align_failures = 0
    unresolved = 0
    metadata_leak = 0
    invalid_refs = 0
    replay_ok = 0
    replay_fail = 0
    csv_rows: List[dict] = []

    for s in samples:
        sid = s.get("sample_id", "")
        sample_ids[sid] += 1
        q = s.get("question", "")
        calls = s.get("gold_calls") or []
        n_calls = s.get("num_calls", len(calls))
        questions.add(q)

        if s.get("gold_answer") is None:
            null_gold += 1
        if not rule(n_calls):
            call_violations += 1
        if validate_references(calls):
            invalid_refs += 1
        if check_unresolved_placeholders(q):
            unresolved += 1
        if any(p.search(q or "") for p in METADATA_LEAK_PATTERNS):
            metadata_leak += 1

        errs = validate_question_trace_alignment(q, calls, num_calls=n_calls)
        if errs:
            align_failures += 1
            csv_rows.append({
                "sample_id": sid,
                "stage": stage,
                "issue": "alignment",
                "detail": ";".join(errs),
                "question": q[:120],
            })

        ok, err = replay_sample(s)
        if ok:
            replay_ok += 1
        else:
            replay_fail += 1
            csv_rows.append({
                "sample_id": sid,
                "stage": stage,
                "issue": "replay",
                "detail": err,
                "question": q[:120],
            })

        sigs = compute_signatures(s)
        exact_counts[sigs["exact"]] += 1
        trace_counts[sigs["trace"]] += 1
        tpl_counts[sigs["question_template"]] += 1
        tool_seq[tuple(c.get("name", "") for c in calls)] += 1
        traj[s.get("trajectory_id", "")] += 1
        dep_patterns[_dependency_pattern(s)] += 1

        for c in calls:
            name = c.get("name", "")
            tools_used[name] += 1
            families_used[tool_family(name)] += 1
        ot = s.get("answer_type") or infer_answer_type(s.get("gold_answer"))
        output_types[ot] += 1
        clusters[s.get("source_failure_cluster", "unknown")] += 1
        motifs[s.get("target_full_motif", "unknown")] += 1

    total = len(samples)
    exact_dup = sum(c - 1 for c in exact_counts.values() if c > 1)
    trace_dup = sum(c - 1 for c in trace_counts.values() if c > 1)
    tpl_dup = sum(c - 1 for c in tpl_counts.values() if c > 1)
    dup_ids = sum(c - 1 for c in sample_ids.values() if c > 1)
    uq_ratio = len(questions) / max(total, 1)
    trace_dup_ratio = trace_dup / max(total, 1)
    tpl_dup_ratio = tpl_dup / max(total, 1)
    ns_count = sum(1 for s in samples if is_non_scalar_answer(s.get("gold_answer")))
    ns_share = ns_count / max(total, 1)
    replay_rate = replay_ok / max(total, 1)
    max_tool_seq_share = max((c / max(total, 1) for c in tool_seq.values()), default=0)
    max_cluster_share = max((c / max(total, 1) for c in clusters.values()), default=0)
    max_traj = max(traj.values()) if traj else 0

    hard_fails: List[str] = []
    soft_warns: List[str] = []

    if total < 800:
        hard_fails.append(f"count={total}<800")
    if call_violations > 0:
        hard_fails.append(f"call_count_violations={call_violations}")
    if null_gold > 0:
        hard_fails.append(f"null_gold_answer={null_gold}")
    if dup_ids > 0:
        hard_fails.append(f"duplicate_sample_id={dup_ids}")
    if exact_dup > 0:
        hard_fails.append(f"exact_duplicates={exact_dup}")
    if invalid_refs > 0:
        hard_fails.append(f"invalid_references={invalid_refs}")
    if replay_rate < 1.0:
        hard_fails.append(f"gold_replay_rate={replay_rate:.4f}")
    if align_failures > 0:
        hard_fails.append(f"alignment_failures={align_failures}")
    if unresolved > 0:
        hard_fails.append(f"unresolved_placeholders={unresolved}")
    if metadata_leak > 0:
        hard_fails.append(f"metadata_leakage={metadata_leak}")

    if trace_dup_ratio > 0.05:
        soft_warns.append(f"trace_duplicate_ratio={trace_dup_ratio:.4f}>0.05")
    if uq_ratio < 0.95:
        soft_warns.append(f"unique_question_ratio={uq_ratio:.4f}<0.95")
    if len(tools_used) < 22:
        soft_warns.append(f"used_tool_names={len(tools_used)}<22")
    if len(families_used) < 5:
        soft_warns.append(f"used_tool_families={len(families_used)}<5")
    if stage != "stage1_1call_atomic" and ns_share < 0.30:
        soft_warns.append(f"non_scalar_share={ns_share:.4f}<0.30")
    if max_tool_seq_share > 0.20:
        soft_warns.append(f"max_tool_sequence_share={max_tool_seq_share:.4f}>0.20")
    if max_cluster_share > 0.50:
        soft_warns.append(f"max_failure_cluster_share={max_cluster_share:.4f}>0.50")

    status = "FAIL" if hard_fails else ("WARN" if soft_warns else "PASS")

    return {
        "stage": stage,
        "status": status,
        "hard_fails": hard_fails,
        "soft_warns": soft_warns,
        "sample_count": total,
        "call_count_histogram": _call_count_histogram(samples),
        "call_count_integrity": call_violations == 0,
        "null_gold_answer_count": null_gold,
        "duplicate_sample_id_count": dup_ids,
        "exact_duplicate_count": exact_dup,
        "trace_duplicate_count": trace_dup,
        "trace_duplicate_ratio": round(trace_dup_ratio, 4),
        "unique_question_count": len(questions),
        "unique_question_ratio": round(uq_ratio, 4),
        "question_template_duplicate_ratio": round(tpl_dup_ratio, 4),
        "unresolved_placeholder_count": unresolved,
        "metadata_leakage_count": metadata_leak,
        "question_trace_alignment_failure_count": align_failures,
        "invalid_reference_count": invalid_refs,
        "gold_replay_success_rate": round(replay_rate, 4),
        "used_tool_names": sorted(tools_used.keys()),
        "used_tool_count": len(tools_used),
        "used_tool_families": sorted(families_used.keys()),
        "used_tool_family_count": len(families_used),
        "output_type_distribution": dict(output_types),
        "non_scalar_output_share": round(ns_share, 4),
        "top_tool_sequences": [
            {"sequence": "->".join(k), "count": c, "share": round(c / max(total, 1), 4)}
            for k, c in tool_seq.most_common(20)
        ],
        "source_failure_cluster_distribution": dict(clusters.most_common(15)),
        "target_full_motif_distribution": dict(motifs.most_common(15)),
        "dependency_pattern_distribution": dict(dep_patterns.most_common(15)),
        "trajectory_concentration": {
            "unique_trajectory_ids": len(traj),
            "max_samples_from_one_trajectory": max_traj,
            "top_trajectories": [{"trajectory_id": k, "count": v} for k, v in traj.most_common(10)],
        },
        "issue_rows": csv_rows,
    }


def _overall_status(per_stage: Dict[str, dict]) -> str:
    if any(p["status"] == "FAIL" for p in per_stage.values()):
        return "FAIL"
    if any(p["status"] == "WARN" for p in per_stage.values()):
        return "WARN"
    return "PASS"


def _pilot_decision(summary: dict) -> str:
    if summary["status"] == "FAIL":
        return "NOT_READY_FIX_REQUIRED"
    preflight_ok = summary.get("preflight_status") in ("PASS_PILOT_READY", "PASS_PROTOTYPE_ONLY", "PASS_FINAL_READY")
    if summary["status"] in ("PASS", "WARN") and preflight_ok:
        return "READY_FOR_POD_DRY_RUN"
    if summary["status"] in ("PASS", "WARN"):
        return "READY_FOR_POD_DRY_RUN"
    return "NOT_READY_FIX_REQUIRED"


def _write_readiness_report(out_dir: Path, summary: dict) -> None:
    ps = summary.get("per_stage", {})
    tool_stats = summary.get("tool_stats", {})
    s2plus_ns = summary.get("stage2_plus_non_scalar_share", 0)
    decision = summary.get("pilot_decision", "NOT_READY_FIX_REQUIRED")

    lines = [
        "# Final Pilot Readiness Report (v3.1)",
        "",
        f"**Overall audit status:** {summary['status']}",
        f"**Pilot decision:** {decision}",
        "",
        "## A. Dataset summary",
        "",
        "| metric | value | status |",
        "|---|---:|---|",
    ]
    for stage in STAGE_FILES:
        d = ps.get(stage, {})
        short = stage.replace("_", " ")[:28]
        lines.append(f"| {short} count | {d.get('sample_count', 0)} | {d.get('status', 'n/a')} |")
        lines.append(
            f"| {short} call-count integrity | "
            f"{'PASS' if d.get('call_count_integrity') else 'FAIL'} | "
            f"{d.get('status', 'n/a')} |"
        )
    lines += [
        "",
        "## B. Quality gates",
        "",
        "| gate | value | status |",
        "|---|---:|---|",
        f"| gold replay success | {summary.get('gold_replay_success_rate', 'n/a')} | "
        f"{'PASS' if summary.get('gold_replay_success_rate', 0) >= 1.0 else 'FAIL'} |",
        f"| process filter pass | {summary.get('process_filter_pass_rate', 'n/a')} | "
        f"{'PASS' if summary.get('process_filter_pass_rate', 1) >= 1.0 else 'FAIL'} |",
        f"| question-trace alignment failures | {summary.get('question_trace_alignment_failures', 0)} | "
        f"{'PASS' if summary.get('question_trace_alignment_failures', 0) == 0 else 'FAIL'} |",
        f"| unresolved placeholders | {summary.get('unresolved_placeholder_count', 0)} | "
        f"{'PASS' if summary.get('unresolved_placeholder_count', 0) == 0 else 'FAIL'} |",
        f"| metadata leakage | {summary.get('metadata_leakage_count', 0)} | "
        f"{'PASS' if summary.get('metadata_leakage_count', 0) == 0 else 'FAIL'} |",
        f"| exact duplicates | {summary.get('exact_duplicate_count', 0)} | "
        f"{'PASS' if summary.get('exact_duplicate_count', 0) == 0 else 'FAIL'} |",
        f"| trace duplicate ratio (mean) | {summary.get('mean_trace_duplicate_ratio', 'n/a')} | "
        f"{'PASS' if summary.get('mean_trace_duplicate_ratio', 0) <= 0.05 else 'WARN'} |",
        f"| invalid references | {summary.get('invalid_reference_count', 0)} | "
        f"{'PASS' if summary.get('invalid_reference_count', 0) == 0 else 'FAIL'} |",
        f"| duplicate sample IDs | {summary.get('duplicate_sample_id_count', 0)} | "
        f"{'PASS' if summary.get('duplicate_sample_id_count', 0) == 0 else 'FAIL'} |",
        f"| preflight status | {summary.get('preflight_status', 'not run')} | — |",
        "",
        "## C. Diversity",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| unique questions | {summary.get('unique_questions', 0)} |",
        f"| unique question ratio (mean) | {summary.get('mean_unique_question_ratio', 0)} |",
        f"| used tool names | {tool_stats.get('used_tool_diversity', 'n/a')} |",
        f"| offered tool names | {tool_stats.get('offered_tool_diversity', 'n/a')} |",
        f"| used tool families | {tool_stats.get('used_tool_family_count', 'n/a')} |",
        f"| stage2+ non-scalar share | {s2plus_ns} |",
        f"| stage3 non-scalar share | {ps.get('stage3_3call_composition', {}).get('non_scalar_output_share', 'n/a')} |",
        f"| stage4 non-scalar share | {ps.get('stage4_4to6call_persistence', {}).get('non_scalar_output_share', 'n/a')} |",
        "",
        "## D. Remaining caveats",
        "",
        "- Synthetic data remains synthetic; tool-family realism is **pilot_ready**, not final NESTFUL overlap.",
        "- NESTFUL test split was not used for generation or training.",
        "- Final test eval has not been run.",
        "- **Training has not started.**",
        "- Stage3/4 training remains gated behind stage1–2 pilot success on dev.",
        "",
        "## E. Decision",
        "",
        f"**{decision}**",
        "",
    ]
    if decision == "READY_FOR_POD_DRY_RUN":
        lines += [
            "### Local build and validation",
            "",
            "```bash",
            "python experiments/nestful_synthetic_curriculum_v3/scripts/build_curriculum_v3_1_pipeline.py",
            "```",
            "",
            "### Pod dry-run",
            "",
            "```bash",
            "cd /workspace/Tool-R0",
            "",
            'DRY_RUN=1 ALLOW_PROTOTYPE_TRAINING=1 CURRICULUM_VERSION=v3_1 STAGES="1 2" \\',
            "  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh",
            "```",
            "",
            "### Stage1 pilot",
            "",
            "```bash",
            "cd /workspace/Tool-R0",
            "",
            'ALLOW_PROTOTYPE_TRAINING=1 USE_VLLM=1 ROLLOUT_DP_GPUS="1,2,3" DP_LEARNER_GPU=0 \\',
            '  CURRICULUM_VERSION=v3_1 STAGES="1" MAX_EPOCHS_PER_STAGE=2 \\',
            "  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh",
            "```",
        ]
    (out_dir / "FINAL_PILOT_READINESS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3_1",
    )
    ap.add_argument("--use-filtered", action="store_true", default=True)
    args = ap.parse_args()

    base = _resolve_stage_dir(args.in_dir, args.use_filtered)
    stage_samples = _load_stages(base)
    if not stage_samples:
        print("[final_audit] no stage files found", file=sys.stderr)
        return 1

    per_stage = {stage: _audit_stage(stage, samples) for stage, samples in stage_samples.items()}
    all_samples = [s for samples in stage_samples.values() for s in samples]
    tool_stats = compute_tool_usage_stats(all_samples)
    uniqueness = analyze_all_stages(stage_samples)

    pf_path = args.in_dir / "preflight_gates_summary.json"
    pf = json.loads(pf_path.read_text(encoding="utf-8")) if pf_path.is_file() else {}
    pf_filter = args.in_dir / "process_filter_summary.json"
    pf_rate = json.loads(pf_filter.read_text(encoding="utf-8")).get("pass_rate", 1.0) if pf_filter.is_file() else 1.0

    s2plus = [s for s in all_samples if int(s.get("num_calls", 0)) >= 2]
    s2plus_ns = sum(1 for s in s2plus if is_non_scalar_answer(s.get("gold_answer"))) / max(len(s2plus), 1)

    overall_status = _overall_status(per_stage)
    summary = {
        "status": overall_status,
        "source_dir": str(base),
        "total_samples": len(all_samples),
        "stage_counts": {k: v["sample_count"] for k, v in per_stage.items()},
        "exact_duplicate_count": sum(v["exact_duplicate_count"] for v in per_stage.values()),
        "duplicate_sample_id_count": sum(v["duplicate_sample_id_count"] for v in per_stage.values()),
        "invalid_reference_count": sum(v["invalid_reference_count"] for v in per_stage.values()),
        "question_trace_alignment_failures": sum(
            v["question_trace_alignment_failure_count"] for v in per_stage.values()
        ),
        "unresolved_placeholder_count": sum(v["unresolved_placeholder_count"] for v in per_stage.values()),
        "metadata_leakage_count": sum(v["metadata_leakage_count"] for v in per_stage.values()),
        "gold_replay_success_rate": round(
            sum(v["gold_replay_success_rate"] * v["sample_count"] for v in per_stage.values())
            / max(len(all_samples), 1),
            4,
        ),
        "mean_trace_duplicate_ratio": uniqueness["overall"]["mean_trace_duplicate_ratio"],
        "mean_unique_question_ratio": uniqueness["overall"]["mean_unique_question_ratio"],
        "unique_questions": sum(v["unique_question_count"] for v in per_stage.values()),
        "stage2_plus_non_scalar_share": round(s2plus_ns, 4),
        "process_filter_pass_rate": pf_rate,
        "preflight_status": pf.get("status"),
        "uniqueness_status": uniqueness["status"],
        "tool_stats": tool_stats,
        "per_stage": {
            k: {kk: vv for kk, vv in v.items() if kk != "issue_rows"}
            for k, v in per_stage.items()
        },
    }
    summary["pilot_decision"] = _pilot_decision(summary)

    issue_rows: List[dict] = []
    for stage, data in per_stage.items():
        for row in data.get("issue_rows", []):
            issue_rows.append(row)

    args.in_dir.mkdir(parents=True, exist_ok=True)
    (args.in_dir / "final_dataset_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(
        args.in_dir / "final_dataset_audit_samples.csv",
        issue_rows,
        ["sample_id", "stage", "issue", "detail", "question"],
    )

    report = [
        "# Final Dataset Audit (v3.1)",
        "",
        f"Overall status: **{overall_status}**",
        f"Pilot decision: **{summary['pilot_decision']}**",
        f"Source: `{base}`",
        "",
        "## Per-stage summary",
        "",
        "| Stage | Status | N | Call integrity | Exact dup | Trace dup ratio | UQ ratio | NS share | Replay |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for stage, d in per_stage.items():
        short = stage.replace("_", " ")[:24]
        report.append(
            f"| {short} | {d['status']} | {d['sample_count']} | "
            f"{'PASS' if d['call_count_integrity'] else 'FAIL'} | {d['exact_duplicate_count']} | "
            f"{d['trace_duplicate_ratio']:.4f} | {d['unique_question_ratio']:.4f} | "
            f"{d['non_scalar_output_share']:.4f} | {d['gold_replay_success_rate']:.4f} |"
        )

    report += ["", "## Hard failures (aggregate)", ""]
    all_hard = []
    for d in per_stage.values():
        all_hard.extend(d["hard_fails"])
    report += [f"- {x}" for x in all_hard] or ["- (none)"]

    report += ["", "## Soft warnings (aggregate)", ""]
    all_soft = []
    for d in per_stage.values():
        all_soft.extend(d["soft_warns"])
    report += [f"- {x}" for x in all_soft] or ["- (none)"]

    report += [
        "",
        "## Global",
        f"- used tool names: {tool_stats.get('used_tool_diversity')}",
        f"- offered tool names: {tool_stats.get('offered_tool_diversity')}",
        f"- stage2+ non-scalar share: {s2plus_ns:.4f}",
        f"- preflight: {pf.get('status', 'not run')}",
        "",
        "See `FINAL_PILOT_READINESS_REPORT.md` for pilot commands.",
    ]
    (args.in_dir / "FINAL_DATASET_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _write_readiness_report(args.in_dir, summary)

    print(
        f"[final_audit] status={overall_status} decision={summary['pilot_decision']} "
        f"exact_dup={summary['exact_duplicate_count']} align_fail={summary['question_trace_alignment_failures']}"
    )
    return 1 if overall_status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
