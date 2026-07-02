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
) -> str:
    if hard_fails:
        return "FAIL"
    if tool_status in ("final_ready", "final_experiment_ready") and not soft_fails:
        return "PASS_FINAL_READY"
    if prototype_only or soft_fails or tool_status in (
        "prototype_only", "pilot_ready", "math_only", "mixed_synthetic_prototype", "partial_tool_realism",
    ):
        return "PASS_PROTOTYPE_ONLY"
    return "PASS_FINAL_READY"


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
    args = ap.parse_args()

    out_dir = args.out_dir
    prototype_only = args.prototype_only or (
        __import__("os").environ.get("ALLOW_PROTOTYPE_TRAINING", "") == "1"
    )

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
