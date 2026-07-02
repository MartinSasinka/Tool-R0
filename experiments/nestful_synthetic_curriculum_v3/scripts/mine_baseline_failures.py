#!/usr/bin/env python3
"""Mine baseline failure motifs from dev trajectories (abstract recipes only)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
MINIMAL = ROOT / "experiments/nestful_mtgrpo_minimal"
sys.path.insert(0, str(MINIMAL))

from motif_lib import (  # noqa: E402
    default_dev_path,
    extract_motifs,
    load_jsonl,
    load_task_row,
    repo_root,
    write_csv,
)


def _load_win_map(csv_path: Path) -> dict:
    wins = {}
    if not csv_path.is_file():
        return wins
    import csv
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tid = row.get("task_id") or row.get("sample_id")
            if tid and "official_win" in row:
                wins[str(tid)] = float(row["official_win"]) >= 1.0
    return wins


def _infer_failure_mode(traj: dict) -> str:
    if traj.get("zero_tool_calls") or traj.get("num_tool_calls", 1) == 0:
        return "zero_tool_calls"
    if traj.get("parse_error") or traj.get("parse_valid") is False:
        return "parse_error"
    gold_n = traj.get("gold_num_calls") or traj.get("num_gold_calls") or 0
    pred_n = traj.get("num_tool_calls") or len(traj.get("pred_calls") or [])
    if pred_n < gold_n:
        return "too_few_calls"
    if not traj.get("official_win") and not traj.get("tool_final_answer_pass"):
        return "wrong_answer"
    if not traj.get("strict_gold_trace_pass"):
        return "trace_drift"
    return "wrong_answer"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", type=Path, default=default_dev_path())
    ap.add_argument("--trajectories", type=Path, default=None)
    ap.add_argument("--win_csv", type=Path,
                    default=ROOT / "experiments/comparison/per_sample_official_win.csv")
    ap.add_argument("--out_dir", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs")
    args = ap.parse_args()

    traj_path = args.trajectories or (
        ROOT / "experiments/nestful_mtgrpo_partial/outputs/final_eval_v2/baseline/final_eval_trajectories.jsonl"
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_tasks = {load_task_row(r)["task_id"]: load_task_row(r) for r in load_jsonl(args.dev)}
    win_map = _load_win_map(args.win_csv)

    clusters: dict = defaultdict(list)
    examples = []
    skeleton_mode = not traj_path.is_file()

    if skeleton_mode:
        print(f"[mine_baseline_failures] WARNING: trajectories not found: {traj_path}", file=sys.stderr)
    else:
        for line in open(traj_path, encoding="utf-8"):
            traj = json.loads(line)
            tid = str(traj.get("task_id") or traj.get("sample_id") or "")
            if tid not in dev_tasks:
                continue
            won = win_map.get(tid)
            if won is None:
                won = bool(traj.get("official_win"))
            if won:
                continue
            task = dev_tasks[tid]
            motifs = extract_motifs(task)
            mode = _infer_failure_mode(traj)
            key = (motifs["motif_type"], mode)
            clusters[key].append({"task_id": tid, "motifs": motifs, "traj": traj})
            if len(examples) < 50:
                examples.append({"task_id": tid, "motif_type": motifs["motif_type"],
                                 "failure_mode": mode, "num_calls": motifs["num_calls"]})

    cluster_specs = []
    for (motif_type, mode), items in sorted(clusters.items(), key=lambda x: -len(x[1])):
        cid = f"{motif_type}__{mode}"
        avg_calls = sum(i["motifs"]["num_calls"] for i in items) / max(len(items), 1)
        cluster_specs.append({
            "cluster_id": cid,
            "motif_type": motif_type,
            "num_examples": len(items),
            "typical_num_calls": round(avg_calls, 1),
            "dependency_pattern": motif_type,
            "reference_pattern": "reuse" if motif_type in ("reference_reuse", "fan_in") else "linear",
            "tool_family_pattern": "mixed",
            "output_type_pattern": Counter(i["motifs"]["output_type"] for i in items).most_common(1)[0][0]
            if items else "unknown",
            "common_failure_mode": mode,
            "synthetic_generation_recipe": (
                f"Generate {motif_type} tasks with {mode} failure mode; "
                f"target ~{int(round(avg_calls))} calls; include distractors and reference chains."
            ),
        })

    write_csv(out_dir / "baseline_failure_clusters.csv", cluster_specs,
              list(cluster_specs[0].keys()) if cluster_specs else [
                  "cluster_id", "motif_type", "num_examples", "typical_num_calls",
                  "dependency_pattern", "reference_pattern", "tool_family_pattern",
                  "output_type_pattern", "common_failure_mode", "synthetic_generation_recipe"])

    with open(out_dir / "baseline_failure_motif_specs.json", "w", encoding="utf-8") as fh:
        json.dump(cluster_specs, fh, indent=2)

    with open(out_dir / "baseline_failure_examples.jsonl", "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    report = [
        "# Baseline Failure Mining",
        "",
        f"Dev split: `{args.dev}` ({len(dev_tasks)} tasks)",
        f"Trajectories: `{traj_path}` ({'MISSING — skeleton mode' if skeleton_mode else 'found'})",
        "",
        f"Failure clusters found: {len(cluster_specs)}",
        "",
    ]
    for spec in cluster_specs[:20]:
        report.append(f"- **{spec['cluster_id']}**: n={spec['num_examples']}, recipe={spec['synthetic_generation_recipe']}")
    if skeleton_mode:
        report += [
            "",
            "## TODO",
            "Run baseline ReAct eval on dev split and point `--trajectories` to those outputs.",
            "Dev task IDs must overlap trajectory task_ids. Never copy dev tasks into training JSONL.",
        ]
    report += [
        "",
        "## Leakage policy",
        "Only abstract recipes exported. Original dev tasks are NOT written to synthetic training data.",
    ]
    (out_dir / "BASELINE_FAILURE_MINING.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[mine_baseline_failures] clusters={len(cluster_specs)} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
