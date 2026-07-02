#!/usr/bin/env python3
"""Distribution audit: NESTFUL vs old synthetic vs v3 synthetic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import (  # noqa: E402
    aggregate_distribution,
    kl_divergence,
    load_jsonl,
    normalize_dist,
    repo_root,
)


def _load_dir_jsonl(d: Path) -> list:
    rows = []
    if d.is_dir():
        for p in sorted(d.glob("*.jsonl")):
            rows.extend(load_jsonl(p))
    elif d.is_file():
        rows = load_jsonl(d)
    return rows


def _baseline_failure_coverage(v3_dist: dict, specs_path: Path) -> tuple[float, list]:
    if not specs_path.is_file():
        return 1.0, []
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    v3_motifs = set(v3_dist.get("motif_type", {}))
    missing = [s["motif_type"] for s in specs if s.get("motif_type") not in v3_motifs]
    covered = sum(1 for s in specs if s.get("motif_type") in v3_motifs)
    rate = covered / max(len(specs), 1)
    return rate, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs")
    args = ap.parse_args()

    out_dir = args.out_dir
    nest_dist_path = out_dir / "nestful_motif_distribution.json"
    if not nest_dist_path.is_file():
        print("ERROR: run analyze_nestful_motifs.py first", file=sys.stderr)
        return 1

    nest_dist = json.loads(nest_dist_path.read_text(encoding="utf-8"))
    syn_old_summary = out_dir / "synthetic_vs_nestful_summary.json"
    old_gap = json.loads(syn_old_summary.read_text(encoding="utf-8")) if syn_old_summary.is_file() else {}

    v3_dir = out_dir / "curriculum_v3"
    v3_tasks = _load_dir_jsonl(v3_dir)
    if not v3_tasks:
        v3_tasks = _load_dir_jsonl(out_dir / "synthetic_motif_tasks.jsonl")
    v3_dist = aggregate_distribution(v3_tasks) if v3_tasks else {}

    nest_m = normalize_dist(nest_dist.get("motif_type", {}))
    v3_m = normalize_dist(v3_dist.get("motif_type", {}))
    kl = kl_divergence(nest_m, v3_m) if v3_m else float("inf")

    covered = sum(1 for k in nest_m if v3_m.get(k, 0) >= 0.5 * nest_m.get(k, 0.001))
    coverage = covered / max(len(nest_m), 1)

    val_report = out_dir / "synthetic_validation_report.md"
    invalid = "FAIL" in val_report.read_text(encoding="utf-8") if val_report.is_file() else False

    failure_specs = out_dir / "baseline_failure_motif_specs.json"
    bf_rate, missing_failure_motifs = _baseline_failure_coverage(v3_dist, failure_specs)

    status = "PASS"
    warnings = []
    if coverage < 0.80:
        warnings.append(f"motif coverage {coverage:.1%} < 80%")
    if bf_rate < 0.80:
        warnings.append(f"baseline failure motif coverage {bf_rate:.1%} < 80%")
    if missing_failure_motifs:
        warnings.append(f"baseline failure motifs not covered: {missing_failure_motifs}")
    if invalid:
        status = "FAIL"
        warnings.append("invalid synthetic tasks > 0")

    summary = {
        "status": status,
        "v3_tasks": len(v3_tasks),
        "motif_coverage": round(coverage, 4),
        "baseline_failure_motif_coverage": round(bf_rate, 4),
        "motif_kl": round(kl, 4),
        "covered_motif_types": covered,
        "total_nestful_motif_types": len(nest_m),
        "missing_failure_motifs": missing_failure_motifs,
        "v3_motif_type_counts": v3_dist.get("motif_type", {}),
        "warnings": warnings,
    }
    (out_dir / "distribution_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = [
        "# Distribution Audit Report",
        "",
        f"Status: **{status}**",
        "",
        "## NESTFUL motif distribution",
        f"- tasks analyzed: {nest_dist.get('num_tasks', '?')}",
        f"- top motifs: {list(nest_dist.get('motif_type', {}).items())[:8]}",
        "",
        "## Old synthetic vs NESTFUL gaps",
        f"- KL (from compare script): {old_gap.get('kl_motif_type', 'n/a')}",
        f"- missing motifs: {old_gap.get('missing', [])}",
        "",
        "## New synthetic v3 vs NESTFUL",
        f"- v3 tasks: {len(v3_tasks)}",
        f"- motif KL(nestful||v3): {kl:.4f}",
        f"- coverage (v3 >= 50% nestful share): {coverage:.1%}",
        "",
        "## Baseline failure motif coverage",
        f"- rate: {bf_rate:.1%}",
        f"- uncovered: {missing_failure_motifs or '(none)'}",
        "",
        "## Warnings",
    ]
    report += [f"- {w}" for w in warnings] or ["- (none)"]
    report += [
        "",
        "## Recommendations",
        "- Use weighted NESTFUL sampling (not equal per-family split).",
        "- Ensure independent_calls generator is active.",
        "- Run baseline dev eval before training to refresh failure specs.",
        "- Do NOT use nestful_test.jsonl for generation or validation inputs.",
    ]
    (out_dir / "DISTRIBUTION_AUDIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[run_distribution_audit] status={status} coverage={coverage:.1%} bf={bf_rate:.1%}")
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
