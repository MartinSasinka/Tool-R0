#!/usr/bin/env python3
"""Build STAGE2B_TEACHER_FORCED_REPORT.md comparing Stage2 BEFORE (ordinary
full-trajectory training) vs AFTER (teacher-forced continuation training).

Both the "before" and "after" eval numbers come from ordinary, non-forced
rollout_eval (run.py's mode_rollout_eval / rollout.run_episode never applies
teacher forcing) — so this is an honest before/after comparison of real
task-solving ability, not an artifact of the training-time intervention.

Usage:
  python build_stage2b_teacher_forced_report.py \
      --after-run-dir experiments/nestful_synthetic_curriculum_v3/outputs/runs/<id>_stage2b_teacher_forced \
      --before-run-dir experiments/nestful_synthetic_curriculum_v3/outputs/runs/<id>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, Optional


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _latest(pattern: str) -> Optional[str]:
    cands = sorted(glob.glob(pattern))
    return cands[-1] if cands else None


def _stage2_eval_metrics(run_dir: str) -> Optional[Dict[str, Any]]:
    """Last epoch's stage_2/epoch_*/eval/metrics.json for `run_dir`."""
    path = _latest(os.path.join(run_dir, "stage_2", "epoch_*", "eval", "metrics.json"))
    return _load_json(path) if path else None


def _stage2_train_summary(run_dir: str) -> Optional[Dict[str, Any]]:
    path = _latest(os.path.join(run_dir, "stage_2", "epoch_*", "train_summary.json"))
    return _load_json(path) if path else None


def _stage2_val_metrics(run_dir: str) -> Optional[Dict[str, Any]]:
    path = _latest(os.path.join(run_dir, "stage_2", "epoch_*", "val_eval", "metrics_epoch_*.json"))
    return _load_json(path) if path else None


def fmt(v, digits=4):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _delta(after, before):
    if after is None or before is None:
        return None
    try:
        return float(after) - float(before)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--after-run-dir", required=True,
                     help="Stage2b run's OUTPUT_ROOT (teacher-forced training)")
    ap.add_argument("--before-run-dir", default=None,
                     help="Stage2a run's OUTPUT_ROOT (ordinary training), for the diff")
    args = ap.parse_args()

    after_dir = os.path.abspath(args.after_run_dir)
    if not os.path.isdir(after_dir):
        print(f"[report] ERROR: --after-run-dir not found: {after_dir}", file=sys.stderr)
        return 1

    after_eval = _stage2_eval_metrics(after_dir) or {}
    after_train = _stage2_train_summary(after_dir) or {}
    after_val = _stage2_val_metrics(after_dir) or {}

    before_dir = os.path.abspath(args.before_run_dir) if args.before_run_dir else None
    before_eval = _stage2_eval_metrics(before_dir) if before_dir else None
    before_val = _stage2_val_metrics(before_dir) if before_dir else None
    before_eval = before_eval or {}
    before_val = before_val or {}

    lines: list[str] = []
    w = lines.append
    w("# STAGE2B TEACHER-FORCED CONTINUATION REPORT")
    w("")
    w(f"After run  (teacher-forced): `{after_dir}`")
    if before_dir:
        w(f"Before run (ordinary)      : `{before_dir}`")
    else:
        w("Before run: not provided (`--before-run-dir`) — showing AFTER numbers only.")
    w("")
    w("Both eval columns come from ordinary, non-forced `rollout_eval` "
      "(evaluation NEVER uses teacher forcing) — this is an honest comparison "
      "of real task-solving ability.")
    w("")

    w("## A. Training configuration")
    w("")
    w("| field | value |")
    w("|---|---|")
    w(f"| teacher_forced_prefix_calls | {fmt(after_train.get('teacher_forced_prefix_calls'))} |")
    w(f"| reward policy resolved | {fmt(after_train.get('reward_policy_resolved'))} |")
    w(f"| optimizer steps | {fmt(after_train.get('steps'))} |")
    w(f"| dead_group_rate | {fmt(after_train.get('dead_group_rate'))} |")
    w(f"| contributing_turns_total | {fmt(after_train.get('contributing_turns_total'))} |")
    w("")

    w("## B. Non-forced eval on Stage2 data (real continuation ability)")
    w("")
    w("| metric | before (2a) | after (2b) | delta |")
    w("|---|---|---|---|")
    rows = [
        ("strict_gold_trace_pass", after_eval.get("strict_gold_trace_pass"),
         before_eval.get("strict_gold_trace_pass")),
        ("too_few_calls_rate", after_eval.get("too_few_calls_rate"),
         before_eval.get("too_few_calls_rate")),
        ("avg_predicted_calls", after_eval.get("avg_predicted_calls"),
         before_eval.get("avg_predicted_calls")),
        ("zero_tool_calls", after_eval.get("zero_tool_calls"),
         before_eval.get("zero_tool_calls")),
        ("clipped_completion_rate", after_eval.get("clipped_completion_rate"),
         before_eval.get("clipped_completion_rate")),
    ]
    for name, av, bv in rows:
        d = _delta(av, bv)
        w(f"| {name} | {fmt(bv)} | {fmt(av)} | {fmt(d)} |")
    w("")
    if not before_dir:
        w("_(Pass `--before-run-dir <stage2a run dir>` to populate the `before` "
          "column and deltas.)_")
        w("")

    w("## C. Dev ReAct Win (validation subset)")
    w("")
    w("| | before (2a) | after (2b) | delta |")
    w("|---|---|---|---|")
    av_win = after_val.get("react_win_rate")
    bv_win = before_val.get("react_win_rate")
    w(f"| react_win_rate | {fmt(bv_win)} | {fmt(av_win)} | {fmt(_delta(av_win, bv_win))} |")
    w("")

    w("## D. Decision")
    w("")
    too_few_after = after_eval.get("too_few_calls_rate")
    too_few_before = before_eval.get("too_few_calls_rate")
    strict_after = after_eval.get("strict_gold_trace_pass")
    strict_before = before_eval.get("strict_gold_trace_pass")
    if not after_eval:
        decision = "STAGE2B_NO_EVAL_ARTIFACTS"
        w(f"**{decision}** — no `stage_2/epoch_*/eval/metrics.json` found under the "
          f"after-run dir; training likely failed or gates stopped it before eval ran.")
    elif before_dir and too_few_before is not None and too_few_after is not None:
        improved_too_few = too_few_after < too_few_before - 1e-9
        improved_strict = (strict_after is not None and strict_before is not None
                            and strict_after > strict_before + 1e-9)
        if improved_too_few and improved_strict:
            decision = "TEACHER_FORCING_IMPROVED_CONTINUATION"
        elif improved_too_few and not improved_strict:
            decision = "TEACHER_FORCING_REDUCED_TOO_FEW_CALLS_BUT_NOT_STRICT_PASS"
        elif not improved_too_few and improved_strict:
            decision = "STRICT_PASS_IMPROVED_WITHOUT_TOO_FEW_CALLS_CHANGE"
        else:
            decision = "TEACHER_FORCING_DID_NOT_HELP"
        w(f"**{decision}**")
    else:
        decision = "STAGE2B_RAN_NO_BASELINE_COMPARISON"
        w(f"**{decision}** — after-run eval metrics are present but no before-run "
          f"comparison was provided.")
    w("")
    w("## E. Paper-safe interpretation")
    w("")
    w("- This is a targeted ablation on a SHORT follow-up run (1 epoch, dev "
      "subset only) — treat effect sizes as suggestive, not conclusive.")
    w("- Numbers in section B/C come from ordinary full-generation eval; the "
      "training-time forcing mechanism itself is never evaluated directly.")
    w("- This run does NOT use the NESTFUL test split.")
    w("")

    out_path = os.path.join(after_dir, "STAGE2B_TEACHER_FORCED_REPORT.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[report] wrote {out_path}")
    print(f"[report] decision: {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
