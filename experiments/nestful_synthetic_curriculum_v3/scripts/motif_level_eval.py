#!/usr/bin/env python3
"""Motif-level evaluation: baseline vs model Win broken down by motif type."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import (  # noqa: E402
    call_count_bucket,
    default_dev_path,
    default_nestful_path,
    default_test_path,
    extract_motifs,
    load_jsonl,
    repo_root,
    write_csv,
)


def _load_motif_labels(path: Path, split_ids: Optional[set] = None) -> Dict[str, dict]:
    labels = {}
    for row in load_jsonl(path):
        tid = str(row.get("task_id") or row.get("sample_id") or row.get("id") or "")
        if split_ids is not None and tid not in split_ids:
            continue
        labels[tid] = extract_motifs(row)
    return labels


def _load_win_csv(path: Path) -> Dict[str, float]:
    wins: Dict[str, float] = {}
    if not path.is_file():
        return wins
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = row.get("sample_id") or row.get("task_id") or row.get("id")
            if not sid:
                continue
            for key in ("official_win", "win", "win_rate"):
                if key in row and row[key] not in (None, ""):
                    wins[str(sid)] = float(row[key])
                    break
    return wins


def _load_trajectory_wins(path: Path) -> Dict[str, float]:
    wins: Dict[str, float] = {}
    if not path.is_file():
        return wins
    for row in load_jsonl(path):
        tid = str(row.get("task_id") or row.get("sample_id") or row.get("id") or "")
        if "official_win" in row:
            wins[tid] = float(row["official_win"])
    return wins


def _find_eval_artifacts(repo: Path) -> tuple[Dict[str, float], Dict[str, float]]:
    baseline: Dict[str, float] = {}
    model: Dict[str, float] = {}
    candidates = [
        repo / "experiments/nestful_mtgrpo_partial/outputs/final_eval_v2",
        repo / "experiments/nestful_synthetic_curriculum_v3/outputs/final_eval_v3",
    ]
    for base in candidates:
        if not base.is_dir():
            continue
        for run in sorted(base.glob("run_*"), reverse=True):
            for cell in run.iterdir():
                traj = cell / "final_eval_trajectories.jsonl"
                if "baseline" in cell.name.lower():
                    baseline.update(_load_trajectory_wins(traj))
                else:
                    model.update(_load_trajectory_wins(traj))
            if baseline or model:
                break
        if baseline or model:
            break
    return baseline, model


def _conclusion(delta: float, n: int) -> str:
    if n < 5:
        return "insufficient_n"
    if delta >= 0.05:
        return "model_better"
    if delta <= -0.05:
        return "model_worse"
    return "neutral"


def _aggregate_bucket(
    ids: List[str],
    labels: Dict[str, dict],
    baseline: Dict[str, float],
    model: Dict[str, float],
    key_fn,
) -> List[dict]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for tid in ids:
        if tid in labels:
            groups[key_fn(labels[tid])].append(tid)

    rows = []
    for bucket, tids in sorted(groups.items()):
        b_wins = [baseline[t] for t in tids if t in baseline]
        m_wins = [model[t] for t in tids if t in model]
        paired = [(baseline[t], model[t]) for t in tids if t in baseline and t in model]
        if not paired:
            continue
        b_mean = sum(p[0] for p in paired) / len(paired)
        m_mean = sum(p[1] for p in paired) / len(paired)
        bfm = sum(1 for b, m in paired if b >= 0.5 and m < 0.5)
        bmf = sum(1 for b, m in paired if b < 0.5 and m >= 0.5)
        rows.append({
            "bucket_type": key_fn.__name__.replace("_key", ""),
            "bucket": bucket,
            "n": len(paired),
            "baseline_win": round(b_mean, 4),
            "model_win": round(m_mean, 4),
            "delta": round(m_mean - b_mean, 4),
            "baseline_fail_model_win": bfm,
            "baseline_win_model_fail": bmf,
            "net_gain": bfm - bmf,
            "conclusion": _conclusion(m_mean - b_mean, len(paired)),
        })
    return rows


def _motif_key(m: dict) -> str:
    return m.get("motif_type", "unknown")


def _calls_key(m: dict) -> str:
    return call_count_bucket(m.get("num_calls", 0))


def _depth_key(m: dict) -> str:
    return str(m.get("dependency_depth", 0))


def _output_key(m: dict) -> str:
    return m.get("output_type", "unknown")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nestful", type=Path, default=None)
    ap.add_argument("--split", type=Path, default=None, help="dev or test split jsonl")
    ap.add_argument("--baseline_wins", type=Path, default=None)
    ap.add_argument("--model_wins", type=Path, default=None)
    ap.add_argument("--failure_clusters", type=Path, default=None)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs",
    )
    args = ap.parse_args()

    split_path = args.split or default_dev_path()
    split_ids = {str(r.get("task_id") or r.get("sample_id") or r.get("id"))
                 for r in load_jsonl(split_path)} if split_path.is_file() else None

    nest_path = args.nestful or default_nestful_path()
    labels = _load_motif_labels(nest_path, split_ids)

    baseline = _load_win_csv(args.baseline_wins) if args.baseline_wins else {}
    model = _load_win_csv(args.model_wins) if args.model_wins else {}
    if not baseline and not model:
        baseline, model = _find_eval_artifacts(repo_root())

    paired_ids = [tid for tid in labels if tid in baseline and tid in model]
    motif_rows = []
    for motif in sorted({labels[t]["motif_type"] for t in paired_ids}):
        tids = [t for t in paired_ids if labels[t]["motif_type"] == motif]
        paired = [(baseline[t], model[t]) for t in tids]
        b_mean = sum(p[0] for p in paired) / len(paired)
        m_mean = sum(p[1] for p in paired) / len(paired)
        bfm = sum(1 for b, m in paired if b >= 0.5 and m < 0.5)
        bmf = sum(1 for b, m in paired if b < 0.5 and m >= 0.5)
        motif_rows.append({
            "motif_type": motif,
            "n": len(paired),
            "baseline_win": round(b_mean, 4),
            "model_win": round(m_mean, 4),
            "delta": round(m_mean - b_mean, 4),
            "baseline_fail_model_win": bfm,
            "baseline_win_model_fail": bmf,
            "net_gain": bfm - bmf,
            "conclusion": _conclusion(m_mean - b_mean, len(paired)),
        })

    bucket_rows = []
    bucket_rows += _aggregate_bucket(paired_ids, labels, baseline, model, _calls_key)
    bucket_rows += _aggregate_bucket(paired_ids, labels, baseline, model, _depth_key)
    bucket_rows += _aggregate_bucket(paired_ids, labels, baseline, model, _output_key)

    cluster_path = args.failure_clusters or (args.out_dir / "baseline_failure_motif_specs.json")
    if cluster_path.is_file():
        specs = json.loads(cluster_path.read_text(encoding="utf-8"))
        for spec in specs:
            motif = spec.get("motif_type")
            tids = [t for t in paired_ids if labels[t]["motif_type"] == motif]
            if not tids:
                continue
            paired = [(baseline[t], model[t]) for t in tids]
            b_mean = sum(p[0] for p in paired) / len(paired)
            m_mean = sum(p[1] for p in paired) / len(paired)
            bucket_rows.append({
                "bucket_type": "failure_cluster",
                "bucket": spec.get("cluster_id", motif),
                "n": len(paired),
                "baseline_win": round(b_mean, 4),
                "model_win": round(m_mean, 4),
                "delta": round(m_mean - b_mean, 4),
                "baseline_fail_model_win": sum(1 for b, m in paired if b >= 0.5 and m < 0.5),
                "baseline_win_model_fail": sum(1 for b, m in paired if b < 0.5 and m >= 0.5),
                "net_gain": 0,
                "conclusion": _conclusion(m_mean - b_mean, len(paired)),
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "motif_level_eval.csv",
        motif_rows,
        ["motif_type", "n", "baseline_win", "model_win", "delta",
         "baseline_fail_model_win", "baseline_win_model_fail", "net_gain", "conclusion"],
    )
    if bucket_rows:
        write_csv(
            args.out_dir / "motif_level_eval_buckets.csv",
            bucket_rows,
            ["bucket_type", "bucket", "n", "baseline_win", "model_win", "delta",
             "baseline_fail_model_win", "baseline_win_model_fail", "net_gain", "conclusion"],
        )

    cpu_only = not paired_ids
    report = [
        "# Motif-Level Eval Report",
        "",
        f"Split: `{split_path}` ({len(split_ids or labels)} tasks with labels)",
        f"Paired baseline+model samples: {len(paired_ids)}",
        "",
    ]
    if cpu_only:
        report += [
            "## CPU fallback mode",
            "No per-sample Win CSV/trajectories found for the requested split.",
            "Motif labels loaded; run GPU eval then re-run this script.",
            "",
            f"Available motif types in split: {sorted({labels[t]['motif_type'] for t in labels})}",
        ]
    else:
        report += [
            "## Motif-type table",
            "",
            "| motif_type | n | baseline_win | model_win | delta | conclusion |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for r in motif_rows:
            report.append(
                f"| {r['motif_type']} | {r['n']} | {r['baseline_win']:.3f} | "
                f"{r['model_win']:.3f} | {r['delta']:+.3f} | {r['conclusion']} |"
            )

    (args.out_dir / "MOTIF_LEVEL_EVAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[motif_level_eval] paired={len(paired_ids)} motifs={len(motif_rows)} cpu_only={cpu_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
