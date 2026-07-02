#!/usr/bin/env python3
"""Offline diagnosis of a completed v2 curriculum run + final_eval_v2 cells.

Emits TRAINING_RUN_DIAGNOSIS.md under --run-root with:
  - val react_win_rate timeline (original + rescored if present)
  - rollout_eval strict_gold_trace_pass timeline
  - train dead_group_rate per epoch (from train_log.jsonl)
  - best_react_win_adapter meta
  - pointer to CHECKPOINT_REEVAL_REPORT.md

Usage:
  python experiments/comparison/training_run_diagnosis.py \\
      --run-root experiments/nestful_mtgrpo_partial/outputs/execution_v2_mixed_replay_full \\
      --eval-root experiments/nestful_mtgrpo_partial/outputs/final_eval_v2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _dead_group_rate(log_path: str) -> Optional[float]:
    if not os.path.isfile(log_path):
        return None
    dead = total = 0
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "dead_group" in rec and "task_id" in rec:
                total += 1
                if rec.get("dead_group"):
                    dead += 1
    return (dead / total) if total else None


def _collect_val_rows(run_root: str) -> List[Dict[str, Any]]:
    rows = []
    for mp in sorted(glob.glob(os.path.join(run_root, "stage_*", "epoch_*", "val_eval", "metrics_epoch_*.json"))):
        if mp.endswith(".rescored.json"):
            continue
        d = _load_json(mp) or {}
        stage = d.get("stage")
        epoch = d.get("epoch")
        if stage is None or epoch is None:
            stage, epoch = _stage_epoch_from_path(mp)
        val_dir = os.path.dirname(mp)
        rescored = _load_json(os.path.join(val_dir, f"metrics_epoch_{epoch}.rescored.json"))
        rows.append({
            "stage": stage, "epoch": epoch,
            "react_win_rate": d.get("react_win_rate"),
            "rescored_win_rate": (rescored or {}).get("react_win_rate"),
            "eval_path": d.get("eval_path", ""),
        })
    return rows


def _stage_epoch_from_path(path: str) -> tuple:
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


def _collect_rollout_rows(run_root: str) -> List[Dict[str, Any]]:
    rows = []
    for mp in sorted(glob.glob(os.path.join(run_root, "stage_*", "epoch_*", "eval", "metrics.json"))):
        d = _load_json(mp) or {}
        stage, epoch = _stage_epoch_from_path(mp)
        rows.append({
            "stage": stage,
            "epoch": epoch,
            "strict_gold_trace_pass": d.get("strict_gold_trace_pass"),
            "final_answer_pass": d.get("final_answer_pass"),
            "zero_tool_calls": d.get("zero_tool_calls"),
            "clipped_completion_rate": d.get("clipped_completion_rate"),
        })
    return rows


def _collect_train_rows(run_root: str) -> List[Dict[str, Any]]:
    rows = []
    for lp in sorted(glob.glob(os.path.join(run_root, "stage_*", "epoch_*", "train_log.jsonl"))):
        stage, epoch = _stage_epoch_from_path(lp)
        ts = _load_json(os.path.join(os.path.dirname(lp), "train_summary.json")) or {}
        rows.append({
            "stage": stage,
            "epoch": epoch,
            "dead_group_rate": _dead_group_rate(lp),
            "reward_policy": ts.get("reward_train_policy"),
            "mean_reward_last": None,  # filled below if present
        })
        # last mean_reward from log
        last_mr = None
        with open(lp, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    if "mean_reward" in rec:
                        last_mr = rec["mean_reward"]
                except json.JSONDecodeError:
                    pass
        rows[-1]["mean_reward_last"] = last_mr
    return rows


def build_report(run_root: str, eval_root: str) -> str:
    best_meta = _load_json(os.path.join(run_root, "best_react_win_adapter", "best_meta.json")) or {}
    val_rows = _collect_val_rows(run_root)
    rollout_rows = _collect_rollout_rows(run_root)
    train_rows = _collect_train_rows(run_root)

    lines = [
        "# TRAINING RUN DIAGNOSIS",
        "",
        f"Run root: `{run_root}`",
        f"Eval root: `{eval_root}`",
        "",
        "## Auto-selected checkpoint (best_react_win_adapter)",
        "",
        f"```json\n{json.dumps(best_meta, indent=2)}\n```",
        "",
        "## Validation ReAct Win timeline",
        "",
        "| stage | epoch | orig react_win | rescored win | val set |",
        "|---|---|---|---|---|",
    ]
    for r in val_rows:
        ep = os.path.basename(r.get("eval_path") or "")
        lines.append(
            f"| {r.get('stage')} | {r.get('epoch')} | {r.get('react_win_rate')} "
            f"| {r.get('rescored_win_rate')} | `{ep}` |"
        )
    null_count = sum(1 for r in val_rows if r.get("react_win_rate") is None)
    lines.extend([
        "",
        f"**{null_count}/{len(val_rows)}** epochs had `react_win_rate=null` in the original run "
        "(official scorer crash — see ROOT_CAUSE_ANALYSIS.md #1).",
        "",
        "## Rollout eval (proxy) timeline",
        "",
        "| stage | epoch | strict_trace | final_answer | zero_calls | clipped |",
        "|---|---|---|---|---|---|",
    ])
    for r in rollout_rows:
        lines.append(
            f"| {r['stage']} | {r['epoch']} | {r.get('strict_gold_trace_pass')} "
            f"| {r.get('final_answer_pass')} | {r.get('zero_tool_calls')} "
            f"| {r.get('clipped_completion_rate')} |"
        )
    lines.extend([
        "",
        "## Training dynamics (dead groups)",
        "",
        "| stage | epoch | dead_group_rate | reward_policy |",
        "|---|---|---|---|",
    ])
    for r in train_rows:
        dgr = r.get("dead_group_rate")
        dgr_s = f"{dgr:.3f}" if isinstance(dgr, float) else "-"
        lines.append(
            f"| {r['stage']} | {r['epoch']} | {dgr_s} | {r.get('reward_policy')} |"
        )
    reeval = os.path.join(eval_root, "CHECKPOINT_REEVAL_REPORT.md")
    if os.path.isfile(reeval):
        lines.extend([
            "",
            "## Full-test official eval",
            "",
            f"See [`CHECKPOINT_REEVAL_REPORT.md`]({reeval.replace(chr(92), '/')}).",
        ])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = build_report(args.run_root, args.eval_root)
    out = args.out or os.path.join(args.run_root, "TRAINING_RUN_DIAGNOSIS.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"[training_run_diagnosis] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
