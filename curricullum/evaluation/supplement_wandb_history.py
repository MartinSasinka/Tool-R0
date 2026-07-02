#!/usr/bin/env python3
"""Patch wandb_grpo_signal_summary.json with train history + local val metrics."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "curricullum/evaluation/wandb_analysis"
VAL_DIR = REPO / "curricullum/training/results"

ENTITY = "sasinka-martin"
PROJECT = "nestful-curriculum-toolr0"

HISTORY_KEYS = [
    "train/reward",
    "train/reward_std",
    "train/frac_reward_zero_std",
    "train/completions/clipped_ratio",
    "train/rewards/toolr0_reward_func/mean",
]


def _history_stats(run) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for row in run.scan_history(keys=HISTORY_KEYS, page_size=500):
        rows.append(row)
    if not rows:
        return {"history_steps": 0}

    df = pd.DataFrame(rows)
    out: Dict[str, Any] = {"history_steps": len(df)}

    def _series(col: str) -> Optional[pd.Series]:
        if col not in df.columns:
            return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return s if len(s) else None

    for col, key_mean, key_final, key_max in [
        ("train/reward", "reward_mean", "reward_final", None),
        ("train/frac_reward_zero_std", "frac_zero_std_mean", "frac_zero_std_final", "frac_zero_std_max"),
        ("train/reward_std", "reward_std_mean", None, None),
        ("train/completions/clipped_ratio", "clipped_ratio_mean", None, None),
        ("train/rewards/toolr0_reward_func/mean", "toolr0_reward_mean", None, None),
    ]:
        s = _series(col)
        if s is None:
            continue
        if key_mean:
            out[key_mean] = float(s.mean())
        if key_final:
            out[key_final] = float(s.iloc[-1])
        if key_max:
            out[key_max] = float(s.max())

    return out


def _local_val(stage: int, epoch: int) -> Dict[str, Optional[float]]:
    path = VAL_DIR / f"stage_{stage}_epoch{epoch}_val.json"
    if not path.exists():
        return {"curriculum_exec_pass": None, "curriculum_parse_fail": None}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "curriculum_exec_pass": data.get("exec_pass_rate"),
        "curriculum_parse_fail": data.get("parse_fail_rate"),
    }


def main() -> None:
    import wandb

    summary_path = OUT_DIR / "wandb_grpo_signal_summary.json"
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    api = wandb.Api(timeout=120)

    print(f"Patching history for {len(rows)} runs...")
    for i, row in enumerate(rows, 1):
        run_id = row["run_id"]
        stage, epoch = row.get("stage"), row.get("epoch")
        print(f"[{i}/{len(rows)}] {row['run_name']}", flush=True)
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        hist = _history_stats(run)
        row.update(hist)
        if stage is not None and epoch is not None:
            row.update(_local_val(stage, epoch))

    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    by_stage: Dict[int, list] = {}
    for r in rows:
        if r.get("stage") is not None:
            by_stage.setdefault(r["stage"], []).append(r)

    rollup = {}
    for stage, rs in sorted(by_stage.items()):
        rs = sorted(rs, key=lambda x: x.get("epoch") or 0)
        rollup[f"stage_{stage}"] = {
            "epochs": rs,
            "frac_zero_std_mean_avg": statistics.mean(
                [r["frac_zero_std_mean"] for r in rs if r.get("frac_zero_std_mean") is not None]
            )
            if any(r.get("frac_zero_std_mean") is not None for r in rs)
            else None,
            "frac_zero_std_mean_worst": max(
                (r["frac_zero_std_mean"] for r in rs if r.get("frac_zero_std_mean") is not None),
                default=None,
            ),
            "steps_dead_pct_avg": statistics.mean(
                [r["steps_dead_pct"] for r in rs if r.get("steps_dead_pct") is not None]
            )
            if any(r.get("steps_dead_pct") is not None for r in rs)
            else None,
            "exec_pass_final": rs[-1].get("curriculum_exec_pass") if rs else None,
            "parse_fail_final": rs[-1].get("curriculum_parse_fail") if rs else None,
        }

    rollup_path = OUT_DIR / "wandb_grpo_signal_by_stage.json"
    rollup_path.write_text(json.dumps(rollup, indent=2), encoding="utf-8")
    print(f"Updated {summary_path}")
    print(f"Updated {rollup_path}")


if __name__ == "__main__":
    main()
