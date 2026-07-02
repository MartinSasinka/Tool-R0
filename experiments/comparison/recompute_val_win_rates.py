#!/usr/bin/env python3
"""Recompute val_eval react_win_rate from saved trajectories (offline).

Uses per-sample ``official_win`` already stored in final_eval_trajectories.jsonl
(written during val_eval). This is the authoritative number when the inline
batch scorer crashed but per-sample scoring succeeded — or to audit epochs that
logged react_win_rate=null.

Also writes metrics_epoch_<E>.rescored.json alongside the original metrics file.

Usage:
  python experiments/comparison/recompute_val_win_rates.py \\
      --run-root experiments/nestful_mtgrpo_partial/outputs/execution_v2_mixed_replay_full
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _stage_epoch_from_path(path: str) -> Tuple[int, int]:
    parts = path.replace("\\", "/").split("/")
    stage = epoch = -1
    for i, p in enumerate(parts):
        if p.startswith("stage_"):
            try:
                stage = int(p.split("_", 1)[1])
                if i + 1 < len(parts) and parts[i + 1].startswith("epoch_"):
                    epoch = int(parts[i + 1].split("_", 1)[1])
            except (ValueError, IndexError):
                pass
            break
    return stage, epoch


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _win_from_trajectories(traj_path: str) -> Optional[float]:
    wins: List[float] = []
    with open(traj_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            traj = row.get("_traj", {}) or {}
            ow = traj.get("official_win")
            if ow is None:
                ow = row.get("internal_win_rate")
            if ow is not None:
                wins.append(float(ow))
    return (sum(wins) / len(wins)) if wins else None


def recompute(run_root: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pattern = os.path.join(run_root, "stage_*", "epoch_*", "val_eval", "final_eval_trajectories.jsonl")
    for traj_path in sorted(glob.glob(pattern)):
        stage, epoch = _stage_epoch_from_path(traj_path)
        val_dir = os.path.dirname(traj_path)
        orig_path = os.path.join(val_dir, f"metrics_epoch_{epoch}.json")
        orig_metrics = _load_json(orig_path) or {}

        recomputed = _win_from_trajectories(traj_path)
        out_path = os.path.join(val_dir, f"metrics_epoch_{epoch}.rescored.json")
        payload = {
            "react_win_rate": recomputed,
            "metric": "react_win_rate",
            "epoch": epoch,
            "stage": stage,
            "eval_path": orig_metrics.get("eval_path"),
            "rescored_from_trajectories": True,
            "n_samples": orig_metrics.get("subset_size") or None,
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        rows.append({
            "stage": stage,
            "epoch": epoch,
            "orig_react_win_rate": orig_metrics.get("react_win_rate"),
            "rescored_react_win_rate": recomputed,
            "eval_path": orig_metrics.get("eval_path", ""),
            "trajectories": traj_path,
            "metrics_out": out_path,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Recompute val_eval Win from trajectories")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    rows = recompute(args.run_root)
    out_csv = args.out_csv or os.path.join(args.run_root, "val_win_rescore_summary.csv")
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[recompute_val_win_rates] wrote {out_csv} ({len(rows)} epochs)")
        for r in rows:
            print(f"  stage{r['stage']}/e{r['epoch']}: "
                  f"orig={r['orig_react_win_rate']} -> rescored={r['rescored_react_win_rate']}")
    else:
        print("[recompute_val_win_rates] no val_eval trajectories found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
