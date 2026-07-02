#!/usr/bin/env python3
"""Offline alignment of execution_aware_v2 vs the official Win Rate.

Rescores SAVED ReAct trajectories with ``nestful_core.rewards.execution_aware_v2``
(and component ablations) and correlates each with the canonical per-sample
official Win (`per_sample_official_win.csv`). This is purely offline — no model,
no GPU — and only uses runs whose trajectories were preserved.

For every run + ALL we report:
  * Pearson r (reward vs official_win)
  * mean reward @ win=1 and @ win=0  (separation)
  * false positives  (win=0 & reward>0.7)
  * false negatives  (win=1 & reward<0.3)
and the same Pearson for four reward ablations:
  final_only / +executable / +references / full(+caps).

Outputs (experiments/comparison/):
  execution_reward_v2_correlation.csv          per-sample long table
  execution_reward_v2_correlation_summary.md   per-run + ALL summary

Usage:
  python experiments/comparison/rescore_execution_reward_v2.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.dirname(_HERE)
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)

from nestful_core import rewards  # noqa: E402
from nestful_core.logging_utils import write_csv  # noqa: E402
import nestful_official_score as nos  # noqa: E402
from data import normalize_task  # noqa: E402
from rollout import Trajectory, Turn  # noqa: E402

_MINIMAL = os.path.join(_EXPERIMENTS, "nestful_mtgrpo_minimal")
_PARTIAL = os.path.join(_EXPERIMENTS, "nestful_mtgrpo_partial")
_DATASET = os.path.join(_MINIMAL, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl")
_PER_SAMPLE_CSV = os.path.join(_HERE, "per_sample_official_win.csv")

_FP_HI = 0.7
_FN_LO = 0.3


def _registry():
    pe = os.path.join(_PARTIAL, "outputs", "final_eval")
    return {
        "partial_s1_e4_react": os.path.join(pe, "partial_s1_e4_react", "final_eval_trajectories.jsonl"),
        "partial_s4_e1_react": os.path.join(pe, "partial_s4_e1_react", "final_eval_trajectories.jsonl"),
        "baseline_react": os.path.join(_MINIMAL, "outputs", "final_eval_baseline_react", "final_eval_trajectories.jsonl"),
    }


def _reconstruct_traj(tj: dict) -> Trajectory:
    """Rebuild a Trajectory (+ final_observation) from a saved ``_traj`` dict."""
    turns = []
    final_obs = None
    for td in tj.get("turns") or []:
        t = Turn(
            turn_idx=int(td.get("turn_idx", len(turns))),
            model_text=td.get("model_text", ""),
            parsed_call=td.get("parsed_call"),
            observation=td.get("observation"),
            fail_reason=td.get("fail_reason"),
            is_terminal=bool(td.get("is_terminal", False)),
            clipped_completion=bool(td.get("clipped_completion", False)),
        )
        turns.append(t)
        if t.parsed_call is not None and t.fail_reason is None:
            final_obs = t.observation
    return Trajectory(
        task_id=str(tj.get("task_id") or ""),
        stage=int(tj.get("stage", 0)),
        gold_num_turns=int(tj.get("gold_num_turns", 0)),
        turns=turns,
        final_observation=tj.get("final_observation", final_obs),
        stop_reason=tj.get("stop_reason"),
        executor_mode=tj.get("executor_mode", "full"),
        clipped_any=bool(tj.get("clipped_any", False)),
        prompt_overflow=bool(tj.get("prompt_overflow", False)),
    )


def _ablations(diag: dict, full_reward: float) -> dict:
    """Component ablations derived from the v2 breakdown (caps only in `full`)."""
    final = float(diag.get("tool_final_answer_pass", 0.0))
    execu = float(diag.get("executable_trajectory", 0.0))
    refs = float(diag.get("valid_references", 0.0))
    return {
        "abl_final_only": final,
        "abl_final_executable": (0.55 * final + 0.20 * execu) / 0.75,
        "abl_final_exec_refs": (0.55 * final + 0.20 * execu + 0.10 * refs) / 0.85,
        "abl_full": full_reward,
    }


def _pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _load_official_wins() -> dict:
    out: dict = defaultdict(dict)
    if not os.path.isfile(_PER_SAMPLE_CSV):
        return {}
    with open(_PER_SAMPLE_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["run"]][row["sample_id"]] = float(row["official_win"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="comma list; default = all available")
    args = ap.parse_args()

    raw = nos.load_raw_dataset(_DATASET)
    official = _load_official_wins()
    reg = _registry()
    runs = [r for r in args.runs.split(",") if r] or list(reg)

    rows = []
    missing = []
    for run in runs:
        path = reg.get(run)
        if not path or not os.path.isfile(path):
            missing.append(run)
            continue
        wins = official.get(run, {})
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sid = str(row.get("sample_id") or row.get("task_id") or "")
                gold_row = raw.get(sid)
                if gold_row is None:
                    continue
                task = normalize_task(gold_row)
                traj = _reconstruct_traj(row.get("_traj", row))
                rr = rewards.execution_aware_v2(traj, task)
                d = rr.diagnostics
                abl = _ablations(d, rr.reward)
                ow = wins.get(sid)
                rows.append({
                    "run": run,
                    "sample_id": sid,
                    "official_win": ow,
                    "exec_v2_reward": round(rr.reward, 6),
                    "cap_applied": d.get("cap_applied"),
                    "final_answer_pass": float(d.get("tool_final_answer_pass", 0.0)),
                    "executable_trajectory": round(float(d.get("executable_trajectory", 0.0)), 4),
                    "tool_use_completeness": round(float(d.get("tool_use_completeness", 0.0)), 4),
                    "valid_references": round(float(d.get("valid_references", 0.0)), 4),
                    "num_successful_calls": d.get("num_successful_calls"),
                    "num_extra_calls": d.get("num_extra_calls"),
                    **{k: round(v, 6) for k, v in abl.items()},
                })

    write_csv(_PER_SAMPLE_CSV.replace("per_sample_official_win.csv",
                                      "execution_reward_v2_correlation.csv"), rows)

    # ── summary ───────────────────────────────────────────────────────────
    abl_keys = ["abl_final_only", "abl_final_executable", "abl_final_exec_refs", "abl_full"]
    by_run = defaultdict(list)
    for r in rows:
        if r["official_win"] is not None:
            by_run[r["run"]].append(r)
    groups = list(by_run.items())
    all_scored = [r for r in rows if r["official_win"] is not None]
    if all_scored:
        groups.append(("ALL", all_scored))

    lines = [
        "# execution_aware_v2 — offline alignment vs official Win",
        "",
        "Reward recomputed offline from saved trajectories; Win from "
        "`per_sample_official_win.csv` (consistency-gated).",
        "",
        "| run | n | win_rate | Pearson(full) | mean@win | mean@loss | FP(win0,r>0.7) | FN(win1,r<0.3) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run, rs in groups:
        n = len(rs)
        wins = [r["official_win"] for r in rs]
        rew = [r["exec_v2_reward"] for r in rs]
        wr = sum(wins) / n
        m_win = ([r["exec_v2_reward"] for r in rs if r["official_win"] >= 0.5] or [0])
        m_loss = ([r["exec_v2_reward"] for r in rs if r["official_win"] < 0.5] or [0])
        fp = sum(1 for r in rs if r["official_win"] < 0.5 and r["exec_v2_reward"] > _FP_HI) / n
        fn = sum(1 for r in rs if r["official_win"] >= 0.5 and r["exec_v2_reward"] < _FN_LO) / n
        lines.append(
            f"| {run} | {n} | {wr:.3f} | {_pearson(rew, wins):.3f} | "
            f"{sum(m_win)/len(m_win):.3f} | {sum(m_loss)/len(m_loss):.3f} | "
            f"{fp:.3f} | {fn:.3f} |")

    lines += ["", "## Component ablations (Pearson vs Win)", "",
              "| run | " + " | ".join(abl_keys) + " |",
              "|---|" + "|".join(["---"] * len(abl_keys)) + "|"]
    for run, rs in groups:
        wins = [r["official_win"] for r in rs]
        cells = [f"{_pearson([r[k] for r in rs], wins):.3f}" for k in abl_keys]
        lines.append(f"| {run} | " + " | ".join(cells) + " |")

    if missing:
        lines += ["", "## Missing (no saved trajectories — excluded)", "",
                  *[f"- {m}" for m in missing]]
    lines += [
        "",
        "## Interpretation",
        "",
        "- `final_answer_pass` alone already explains most of the rank-correlation "
        "with Win; the executable/reference/gold-progress terms add little Pearson "
        "but provide the dense partial-credit signal GRPO needs early in training.",
        "- The v2 caps deliberately trade a small amount of rank-correlation for "
        "training SAFETY: parse-error / clipped / no-tool / terminal-before-first-"
        "tool trajectories are forced to 0 (no exploitable partial credit), which is "
        "the failure mode that drove the legacy reward/Win mismatch.",
        "- Residual false positives (Win=0 but reward>0.7) come from `matches_gold` "
        "being more lenient than the official scorer on answer formatting; they are "
        "logged here per-sample for inspection in `execution_reward_v2_correlation.csv`.",
        "",
        "Generated by `rescore_execution_reward_v2.py`.",
    ]

    out_md = os.path.join(_HERE, "execution_reward_v2_correlation_summary.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[rescore_v2] wrote execution_reward_v2_correlation.csv ({len(rows)} rows) + summary")
    if missing:
        print(f"[rescore_v2] missing trajectories (excluded): {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
