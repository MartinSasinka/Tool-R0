#!/usr/bin/env python3
"""Analyze real NESTFUL dataset and extract structural motif properties."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import (  # noqa: E402
    aggregate_distribution,
    call_count_bucket,
    default_dev_path,
    default_nestful_path,
    extract_motifs,
    load_jsonl,
    load_task_row,
    repo_root,
    write_csv,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs")
    ap.add_argument("--split", choices=("full", "dev"), default="full")
    args = ap.parse_args()

    if args.input:
        inp = args.input
    elif args.split == "dev":
        inp = default_dev_path()
    else:
        inp = default_nestful_path()

    if not inp.is_file():
        print(f"ERROR: input not found: {inp}", file=sys.stderr)
        return 1

    rows = load_jsonl(inp)
    tasks = [load_task_row(r) for r in rows]
    motifs = [extract_motifs(t) for t in tasks]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for m in motifs:
        summary_rows.append({
            "task_id": m["task_id"],
            "num_calls": m["num_calls"],
            "call_bucket": call_count_bucket(m["num_calls"]),
            "dependency_depth": m["dependency_depth"],
            "motif_type": m["motif_type"],
            "output_type": m["output_type"],
            "answer_type": m["answer_type"],
            "num_references": m["reference_pattern"]["num_references"],
            "fan_in": int(m["fan_in"]),
            "fan_out": int(m["fan_out"]),
            "reference_reuse": int(m["reference_reuse"]),
            "difficulty_score": m["difficulty_score"],
            "tool_sequence": m["tool_sequence"],
        })
    write_csv(out_dir / "nestful_motif_summary.csv", summary_rows, list(summary_rows[0].keys()))

    dist = aggregate_distribution(tasks)
    with open(out_dir / "nestful_motif_distribution.json", "w", encoding="utf-8") as fh:
        json.dump(dist, fh, indent=2)

    bigram_ctr = Counter(m["tool_sequence_bigram"] for m in motifs if m["tool_sequence_bigram"])
    tri_ctr = Counter(m["tool_sequence_trigram"] for m in motifs if m["tool_sequence_trigram"])
    write_csv(out_dir / "nestful_tool_sequence_motifs.csv",
              [{"motif": k, "count": v, "order": "bigram"} for k, v in bigram_ctr.most_common(200)]
              + [{"motif": k, "count": v, "order": "trigram"} for k, v in tri_ctr.most_common(200)],
              ["motif", "count", "order"])

    dep_rows = [{"motif_type": k, "count": v} for k, v in Counter(m["motif_type"] for m in motifs).items()]
    write_csv(out_dir / "nestful_dependency_motifs.csv", dep_rows, ["motif_type", "count"])

    nc = [m["num_calls"] for m in motifs]
    report = [
        "# NESTFUL Motif Analysis",
        "",
        f"Input: `{inp}` ({len(tasks)} tasks)",
        "",
        "## num_calls",
        f"- min={min(nc)}, max={max(nc)}, mean={statistics.mean(nc):.2f}, median={statistics.median(nc):.1f}",
        "",
        "### buckets",
    ]
    for b, c in sorted(dist["num_calls"].items(), key=lambda x: x[0]):
        report.append(f"- {b}: {c} ({100*c/len(tasks):.1f}%)")
    report += ["", "## Top motif types"]
    for mt, c in Counter(m["motif_type"] for m in motifs).most_common(15):
        report.append(f"- {mt}: {c}")
    report += ["", "## Dependency depth distribution"]
    for d, c in sorted(dist["dependency_depth"].items(), key=lambda x: int(x[0])):
        report.append(f"- depth {d}: {c}")
    report += ["", "## Most complex tasks (top 5 difficulty)"]
    for m in sorted(motifs, key=lambda x: -x["difficulty_score"])[:5]:
        report.append(f"- {m['task_id']}: difficulty={m['difficulty_score']}, calls={m['num_calls']}, "
                      f"motif={m['motif_type']}, seq={m['tool_sequence']}")
    report += [
        "",
        "## Implications for synthetic generation",
        "- Match call-count buckets but prioritize fan-in/fan-out and reference reuse (underrepresented in N-call curriculum).",
        "- Include object/list output types — common in real NESTFUL answers.",
        "- Stage by structural motif, not call count alone.",
        "- Mine baseline failures on dev to oversample hard motif clusters.",
    ]
    (out_dir / "NESTFUL_MOTIF_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[analyze_nestful_motifs] wrote {len(tasks)} tasks -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
