#!/usr/bin/env python3
"""Compare old synthetic curriculum distribution vs real NESTFUL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import (  # noqa: E402
    aggregate_distribution,
    default_nestful_path,
    extract_motifs,
    histogram,
    kl_divergence,
    load_jsonl,
    load_task_row,
    normalize_dist,
    repo_root,
    write_csv,
)


def _load_synthetic(base: Path) -> list:
    tasks = []
    for p in sorted(base.glob("epoch_*_*call.jsonl")):
        for row in load_jsonl(p):
            tasks.append(load_task_row(row))
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic_dir", type=Path, default=None)
    ap.add_argument("--nestful", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs")
    args = ap.parse_args()

    syn_base = args.synthetic_dir
    if syn_base is None:
        clean = repo_root() / "experiments/nestful_mtgrpo_minimal/data/clean_curriculum"
        # legacy dataset B fallback, archived by cleanup Phase K
        filt = (repo_root() / "experiments/nestful_synthetic_curriculum_v3/archive"
                / "legacy_dataset_B_filtered_toolr0_synthetic")
        syn_base = clean if clean.is_dir() else filt

    nestful_path = args.nestful or default_nestful_path()
    syn_tasks = _load_synthetic(syn_base)
    nest_tasks = [load_task_row(r) for r in load_jsonl(nestful_path)]
    syn_dist = aggregate_distribution(syn_tasks)
    nest_dist = aggregate_distribution(nest_tasks)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_motifs = set(syn_dist["motif_type"]) | set(nest_dist["motif_type"])
    syn_m = normalize_dist(syn_dist["motif_type"])
    nest_m = normalize_dist(nest_dist["motif_type"])
    for mt in sorted(all_motifs):
        rows.append({
            "motif_type": mt,
            "synthetic_count": syn_dist["motif_type"].get(mt, 0),
            "nestful_count": nest_dist["motif_type"].get(mt, 0),
            "synthetic_pct": round(100 * syn_m.get(mt, 0), 2),
            "nestful_pct": round(100 * nest_m.get(mt, 0), 2),
            "delta_pct": round(100 * (syn_m.get(mt, 0) - nest_m.get(mt, 0)), 2),
        })
    write_csv(out_dir / "synthetic_vs_nestful_distribution.csv", rows, list(rows[0].keys()))

    heatmap = []
    for metric in ("num_calls", "motif_type", "output_type", "dependency_depth"):
        s = normalize_dist(syn_dist.get(metric, {}))
        n = normalize_dist(nest_dist.get(metric, {}))
        for k in sorted(set(s) | set(n)):
            heatmap.append({
                "metric": metric,
                "bucket": k,
                "synthetic": round(s.get(k, 0), 4),
                "nestful": round(n.get(k, 0), 4),
                "abs_gap": round(abs(s.get(k, 0) - n.get(k, 0)), 4),
            })
    write_csv(out_dir / "motif_coverage_heatmap.csv", heatmap,
              ["metric", "bucket", "synthetic", "nestful", "abs_gap"])

    kl = kl_divergence(nest_m, syn_m)
    report = [
        "# Synthetic vs NESTFUL Distribution Gaps",
        "",
        f"Synthetic source: `{syn_base}` ({len(syn_tasks)} tasks)",
        f"NESTFUL source: `{nestful_path}` ({len(nest_tasks)} tasks)",
        f"Motif-type KL(nestful||synthetic): {kl:.4f}",
        "",
        "## Covered by synthetic (within 5pp of NESTFUL share)",
    ]
    covered, under, over, missing = [], [], [], []
    for r in rows:
        if r["synthetic_count"] == 0 and r["nestful_count"] > 0:
            missing.append(r["motif_type"])
        elif abs(r["delta_pct"]) <= 5:
            covered.append(r["motif_type"])
        elif r["delta_pct"] < -5:
            under.append(f"{r['motif_type']} ({r['nestful_pct']:.1f}% nestful vs {r['synthetic_pct']:.1f}% syn)")
        else:
            over.append(f"{r['motif_type']} ({r['synthetic_pct']:.1f}% syn vs {r['nestful_pct']:.1f}% nestful)")

    report.append("- " + ", ".join(covered) if covered else "- (none)")
    report += ["", "## Underrepresented in synthetic"]
    report += [f"- {x}" for x in under] or ["- (none)"]
    report += ["", "## Overrepresented in synthetic"]
    report += [f"- {x}" for x in over] or ["- (none)"]
    report += ["", "## Missing entirely from synthetic"]
    report += [f"- {x}" for x in missing] or ["- (none)"]
    report += [
        "",
        "## Stage coverage issue",
        "Old N-call stages (epoch_N_Ncall.jsonl) conflate depth with structure — e.g. stage 2",
        "includes both linear chains and independent calls, but rarely fan-in/fan-out patterns",
        "that dominate harder NESTFUL tasks.",
    ]
    (out_dir / "synthetic_vs_nestful_gaps.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    with open(out_dir / "synthetic_vs_nestful_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"kl_motif_type": kl, "covered": covered, "under": under, "over": over, "missing": missing}, fh, indent=2)
    print(f"[compare_synthetic_vs_nestful] wrote reports -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
