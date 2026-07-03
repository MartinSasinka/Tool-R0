#!/usr/bin/env python3
"""Tool/output realism audit for v3.1 vs NESTFUL."""
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
    load_jsonl,
    load_task_row,
    normalize_dist,
    repo_root,
    write_csv,
)
from tool_registry_v3_1 import ALL_TOOL_NAMES, FAMILY_MAP, infer_answer_type, is_math_only_toolset  # noqa: E402


def _tool_names(tasks: list) -> Counter:
    c = Counter()
    for t in tasks:
        row = load_task_row(t)
        for tl in row.get("tools") or []:
            if tl.get("name"):
                c[str(tl["name"])] += 1
    return c


def _ngram_overlap(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    inter = sum(min(a[k], b[k]) for k in set(a) & set(b))
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b))
    return inter / max(union, 1)


def _non_scalar_share_stage2_plus(tasks: list) -> float:
    s2plus = [t for t in tasks if int(t.get("num_calls") or load_task_row(t)["num_calls"]) >= 2]
    if not s2plus:
        return 0.0
    non_scalar = 0
    for t in s2plus:
        ans = t.get("gold_answer")
        if ans is not None:
            if infer_answer_type(ans) not in ("scalar", "unknown"):
                non_scalar += 1
            continue
        ot = t.get("answer_type") or t.get("output_type") or extract_motifs(t)["output_type"]
        if ot not in ("scalar", "unknown"):
            non_scalar += 1
    return non_scalar / len(s2plus)


def _classify_status(
    tool_diversity: int,
    family_count: int,
    non_scalar_s2: float,
    scalar_share: float,
    bigram_overlap: float,
    trigram_overlap: float,
) -> str:
    meets_pilot = (
        family_count >= 5
        and tool_diversity >= 30
        and non_scalar_s2 >= 0.30
        and scalar_share < 0.70
    )
    meets_final = (
        meets_pilot
        and bigram_overlap >= 0.15
        and trigram_overlap >= 0.10
    )
    if meets_final:
        return "final_experiment_ready"
    if meets_pilot:
        return "pilot_ready"
    if not is_math_only_toolset(set(ALL_TOOL_NAMES)):
        return "prototype_only"
    return "prototype_only"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nestful", type=Path, default=None)
    ap.add_argument("--in-dir", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3_1")
    ap.add_argument("--use-filtered", action="store_true", default=True)
    args = ap.parse_args()

    nest_path = args.nestful or default_nestful_path()
    base = args.in_dir / "filtered" if args.use_filtered else args.in_dir
    v31_tasks = []
    for p in sorted(base.glob("stage*.jsonl")):
        v31_tasks.extend(load_jsonl(p))

    nest_tasks = [load_task_row(r) for r in load_jsonl(nest_path)]
    v31_norm = [load_task_row(t) for t in v31_tasks]
    nest_dist = aggregate_distribution(nest_tasks)
    v31_dist = aggregate_distribution(v31_norm)

    nest_tools = _tool_names(nest_tasks)
    v31_tools = _tool_names(v31_norm)
    tool_diversity = len(set(v31_tools))
    v31_families = set(FAMILY_MAP.values())
    family_count = len(v31_families)

    nest_families = Counter()
    for t in nest_tasks:
        nest_families.update(extract_motifs(t)["tool_family"].split(","))
    v31_tool_families = Counter()
    for name in v31_tools:
        v31_tool_families[FAMILY_MAP.get(name, "other")] += v31_tools[name]

    nest_bigrams = Counter(nest_dist.get("tool_bigrams", {}))
    v31_bigrams = Counter(v31_dist.get("tool_bigrams", {}))
    nest_trigrams = Counter()
    v31_trigrams = Counter()
    for t in nest_tasks:
        m = extract_motifs(t)
        if m.get("tool_sequence_trigram"):
            nest_trigrams[m["tool_sequence_trigram"]] += 1
    for t in v31_norm:
        m = extract_motifs(t)
        if m.get("tool_sequence_trigram"):
            v31_trigrams[m["tool_sequence_trigram"]] += 1

    bigram_ov = _ngram_overlap(v31_bigrams, nest_bigrams)
    trigram_ov = _ngram_overlap(v31_trigrams, nest_trigrams)
    family_overlap = _ngram_overlap(v31_tool_families, nest_families)

    v31_out = normalize_dist(v31_dist.get("output_type", {}))
    scalar_share = v31_out.get("scalar", 0.0)
    non_scalar_s2 = _non_scalar_share_stage2_plus(v31_tasks)
    distractor_counts = [extract_motifs(t)["distractor_tools"] for t in v31_norm]

    status = _classify_status(
        tool_diversity, family_count, non_scalar_s2, scalar_share, bigram_ov, trigram_ov
    )

    summary = {
        "status": status,
        "tool_name_diversity": tool_diversity,
        "tool_family_count": family_count,
        "tool_family_overlap": round(family_overlap, 4),
        "bigram_overlap": round(bigram_ov, 4),
        "trigram_overlap": round(trigram_ov, 4),
        "scalar_output_share": round(scalar_share, 4),
        "non_scalar_output_share_stage2_plus": round(non_scalar_s2, 4),
        "mean_distractor_tool_count": round(sum(distractor_counts) / max(len(distractor_counts), 1), 2),
        "output_type_distribution": v31_dist.get("output_type", {}),
        "answer_type_distribution": v31_dist.get("answer_type", {}),
        "targets": {
            "tool_family_count_ge": 5,
            "tool_name_diversity_ge": 30,
            "non_scalar_output_share_stage2_plus_ge": 0.30,
            "scalar_output_share_lt": 0.70,
        },
    }
    args.in_dir.mkdir(parents=True, exist_ok=True)
    (args.in_dir / "tool_output_realism_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    rows = [
        {"metric": "tool_name_diversity", "v3_1": tool_diversity, "nestful": len(nest_tools)},
        {"metric": "tool_family_count", "v3_1": family_count, "nestful": len(nest_families)},
        {"metric": "bigram_overlap", "v3_1": bigram_ov, "nestful": 1.0},
        {"metric": "non_scalar_share_s2plus", "v3_1": non_scalar_s2, "nestful": "n/a"},
    ]
    write_csv(args.in_dir / "tool_output_coverage.csv", rows, ["metric", "v3_1", "nestful"])

    report = [
        "# Tool Output Realism Report (v3.1)",
        "",
        f"Status: **{status}**",
        "",
        f"- tool_name_diversity: {tool_diversity}",
        f"- tool_family_count: {family_count}",
        f"- tool_family_overlap: {family_overlap:.4f}",
        f"- bigram_overlap: {bigram_ov:.4f}",
        f"- trigram_overlap: {trigram_ov:.4f}",
        f"- scalar_output_share: {scalar_share:.4f}",
        f"- non_scalar_output_share_stage2_plus: {non_scalar_s2:.4f}",
        "",
        "## Targets",
        "- tool_family_count >= 5",
        "- tool_name_diversity >= 30",
        "- non_scalar_output_share_stage2_plus >= 0.30",
        "- scalar_output_share < 0.70",
    ]
    (args.in_dir / "TOOL_OUTPUT_REALISM_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[run_tool_family_realism_v3_1] status={status} diversity={tool_diversity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
