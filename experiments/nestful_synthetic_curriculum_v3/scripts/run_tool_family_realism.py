#!/usr/bin/env python3
"""Tool-family realism report: synthetic v3 vs real NESTFUL."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import (  # noqa: E402
    aggregate_distribution,
    default_nestful_path,
    extract_motifs,
    histogram,
    load_jsonl,
    load_task_row,
    normalize_dist,
    repo_root,
    write_csv,
)


from synthetic_tool_registry import is_math_only_toolset  # noqa: E402


def _tool_names(tasks: list) -> Counter:
    c = Counter()
    for t in tasks:
        row = load_task_row(t)
        for tl in row.get("tools") or []:
            if tl.get("name"):
                c[str(tl["name"])] += 1
    return c


def _distractor_counts(tasks: list) -> list[int]:
    return [extract_motifs(t)["distractor_tools"] for t in tasks]


def _ngram_overlap(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    inter = sum(min(a[k], b[k]) for k in set(a) & set(b))
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b))
    return inter / max(union, 1)


def _classify_status(
    math_only: bool,
    family_overlap: float,
    bigram_overlap: float,
    tool_diversity: int,
    non_scalar_share: float,
) -> str:
    if family_overlap >= 0.5 and bigram_overlap >= 0.3:
        return "final_ready"
    if not math_only and (family_overlap >= 0.25 or bigram_overlap >= 0.08 or non_scalar_share >= 0.15):
        return "partial_tool_realism"
    if not math_only and tool_diversity >= 15:
        return "mixed_synthetic_prototype"
    if math_only:
        return "math_only"
    return "mixed_synthetic_prototype"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nestful", type=Path, default=None)
    ap.add_argument("--v3", type=Path, default=None)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs",
    )
    args = ap.parse_args()

    nest_path = args.nestful or default_nestful_path()
    v3_path = args.v3 or (repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3")
    if v3_path.is_dir():
        v3_tasks = []
        for p in sorted(v3_path.glob("stage*.jsonl")):
            v3_tasks.extend(load_jsonl(p))
        if not v3_tasks:
            v3_tasks = load_jsonl(
                repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/synthetic_motif_tasks.jsonl"
            )
    else:
        v3_tasks = load_jsonl(v3_path)

    nest_tasks = [load_task_row(r) for r in load_jsonl(nest_path)]
    v3_norm = [load_task_row(t) for t in v3_tasks]
    nest_dist = aggregate_distribution(nest_tasks)
    v3_dist = aggregate_distribution(v3_norm)

    nest_tools = _tool_names(nest_tasks)
    v3_tools = _tool_names(v3_norm)
    v3_tool_set = set(v3_tools)
    math_only = is_math_only_toolset(v3_tool_set)

    v3_out = normalize_dist(v3_dist.get("output_type", {}))
    scalar_share = v3_out.get("scalar", 0.0)
    non_scalar_share = 1.0 - scalar_share

    nest_families = Counter()
    v3_families = Counter()
    for t in nest_tasks:
        nest_families.update(extract_motifs(t)["tool_family"].split(","))
    for t in v3_norm:
        v3_families.update(extract_motifs(t)["tool_family"].split(","))

    nest_bigrams = Counter(nest_dist.get("tool_bigrams", {}))
    v3_bigrams = Counter(v3_dist.get("tool_bigrams", {}))
    nest_trigrams = Counter()
    v3_trigrams = Counter()
    for t in nest_tasks:
        m = extract_motifs(t)
        if m["tool_sequence_trigram"]:
            for tg in m["tool_sequence_trigram"].split("|"):
                nest_trigrams[tg] += 1
    for t in v3_norm:
        m = extract_motifs(t)
        if m["tool_sequence_trigram"]:
            for tg in m["tool_sequence_trigram"].split("|"):
                v3_trigrams[tg] += 1

    family_overlap = _ngram_overlap(nest_families, v3_families)
    bigram_overlap = _ngram_overlap(nest_bigrams, v3_bigrams)
    trigram_overlap = _ngram_overlap(nest_trigrams, v3_trigrams)
    status = _classify_status(math_only, family_overlap, bigram_overlap, len(v3_tools), non_scalar_share)

    rows = []
    metrics = [
        ("tool_family_distribution", nest_families, v3_families),
        ("tool_name_diversity", nest_tools, v3_tools),
        ("output_type_distribution", nest_dist.get("output_type", {}), v3_dist.get("output_type", {})),
        ("answer_type_distribution", nest_dist.get("answer_type", {}), v3_dist.get("answer_type", {})),
    ]
    for metric, nest_c, v3_c in metrics:
        keys = sorted(set(nest_c) | set(v3_c))
        for k in keys:
            rows.append({
                "metric": metric,
                "bucket": k,
                "nestful": nest_c.get(k, 0) if isinstance(nest_c, Counter) else nest_c.get(k, 0),
                "synthetic_v3": v3_c.get(k, 0) if isinstance(v3_c, Counter) else v3_c.get(k, 0),
            })

    rows.append({
        "metric": "distractor_tool_count_mean",
        "bucket": "all",
        "nestful": round(sum(_distractor_counts(nest_tasks)) / max(len(nest_tasks), 1), 4),
        "synthetic_v3": round(sum(_distractor_counts(v3_norm)) / max(len(v3_norm), 1), 4),
    })
    rows.append({
        "metric": "tool_sequence_bigram_overlap",
        "bucket": "jaccard",
        "nestful": round(bigram_overlap, 4),
        "synthetic_v3": round(bigram_overlap, 4),
    })
    rows.append({
        "metric": "tool_sequence_trigram_overlap",
        "bucket": "jaccard",
        "nestful": round(trigram_overlap, 4),
        "synthetic_v3": round(trigram_overlap, 4),
    })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "tool_family_coverage.csv", rows,
              ["metric", "bucket", "nestful", "synthetic_v3"])

    summary = {
        "status": status,
        "math_only_synthetic": math_only,
        "tool_name_diversity_nestful": len(nest_tools),
        "tool_name_diversity_v3": len(v3_tools),
        "family_overlap": round(family_overlap, 4),
        "bigram_overlap": round(bigram_overlap, 4),
        "trigram_overlap": round(trigram_overlap, 4),
        "scalar_output_share": round(scalar_share, 4),
        "non_scalar_output_share": round(non_scalar_share, 4),
    }
    (args.out_dir / "tool_family_realism_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = [
        "# Tool Family Realism Report",
        "",
        f"Status: **{status}**",
        "",
        f"- NESTFUL tasks: {len(nest_tasks)}",
        f"- Synthetic v3 tasks: {len(v3_norm)}",
        f"- Tool name diversity (nestful / v3): {len(nest_tools)} / {len(v3_tools)}",
        f"- Tool family overlap (Jaccard): {family_overlap:.4f}",
        f"- Tool sequence bigram overlap: {bigram_overlap:.4f}",
        f"- Tool sequence trigram overlap: {trigram_overlap:.4f}",
        f"- Mean distractor tools (nestful / v3): "
        f"{rows[-3]['nestful']:.2f} / {rows[-3]['synthetic_v3']:.2f}",
        "",
        "## Output / answer type",
        f"- NESTFUL output types: {nest_dist.get('output_type', {})}",
        f"- v3 output types: {v3_dist.get('output_type', {})}",
        "",
    ]
    if math_only:
        report += [
            "> **Prototype math-tool synthetic curriculum may be structurally motif-aligned "
            "but tool-family shifted. This is suitable for pipeline validation or prototype "
            "pilot, not final evidence of NESTFUL transfer.**",
            "",
        ]
    report += [
        "## Status meanings",
        "- `math_only`: only math/distractor tools — pipeline validation only",
        "- `mixed_synthetic_prototype`: multi-family synthetic tools, low NESTFUL overlap",
        "- `partial_tool_realism`: improved diversity/overlap — pilot with caveats",
        "- `final_ready`: high overlap — suitable for transfer claims",
        "",
        f"Current classification: **{status}**",
        f"- scalar output share: {scalar_share:.1%}",
        f"- non-scalar output share: {non_scalar_share:.1%}",
    ]
    (args.out_dir / "tool_family_realism_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[run_tool_family_realism] status={status} math_only={math_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
