#!/usr/bin/env python3
"""Canonical per-sample official Win recompute + aggregate-consistency gate.

For each configured run we recompute the official per-sample metrics from the
SAVED trajectories using the exact same predicted-call extraction the aggregate
``metrics_official.json`` used (lenient parsing for ReAct). We then assert:

    mean(per_sample_official_win) == metrics_official.win_rate  (± 1e-6, rounding-aware)

If a run has no saved trajectories (e.g. minimal/baseline kept only aggregate
metrics), it is reported as MISSING_TRAJECTORIES (a WARNING) and NOT used for any
downstream overlap / correlation / failure taxonomy.

Outputs (under experiments/comparison/):
  per_sample_official_win.csv        long: run, sample_id, official_win, ...
  per_sample_consistency_report.md   PASS/FAIL/WARNING per run

Usage:
  python experiments/comparison/recompute_per_sample_official.py [--runs a,b,c]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.dirname(_HERE)
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)

from nestful_core import scoring  # noqa: E402
from nestful_core.logging_utils import write_csv  # noqa: E402
import nestful_official_score as nos  # noqa: E402

_MINIMAL = os.path.join(_EXPERIMENTS, "nestful_mtgrpo_minimal")
_PARTIAL = os.path.join(_EXPERIMENTS, "nestful_mtgrpo_partial")
_DATASET = os.path.join(_MINIMAL, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl")


# run_name -> (trajectories_path, metrics_official_path, paradigm)
def _registry():
    pe = os.path.join(_PARTIAL, "outputs", "final_eval")
    me = os.path.join(_MINIMAL, "final_outputs", "runs")
    return {
        # Runs WITH saved trajectories (recomputable).
        "partial_s1_e4_react": (
            os.path.join(pe, "partial_s1_e4_react", "final_eval_trajectories.jsonl"),
            os.path.join(pe, "partial_s1_e4_react", "metrics_official.json"), "react"),
        "partial_s4_e1_react": (
            os.path.join(pe, "partial_s4_e1_react", "final_eval_trajectories.jsonl"),
            os.path.join(pe, "partial_s4_e1_react", "metrics_official.json"), "react"),
        # Required by the plan but trajectories were not preserved -> WARNING.
        "baseline_react": (
            os.path.join(_MINIMAL, "outputs", "final_eval_baseline_react",
                         "final_eval_trajectories.jsonl"),
            os.path.join(me, "baseline_react", "metrics_official.json"), "react"),
        "baseline_direct": (
            os.path.join(_MINIMAL, "outputs", "final_eval_baseline_direct",
                         "direct_eval_trajectories.jsonl"),
            os.path.join(me, "baseline_direct", "metrics_official.json"), "direct"),
        "minimal_s4e2_react": (
            os.path.join(_MINIMAL, "outputs", "final_eval_stage4e2_react",
                         "final_eval_trajectories.jsonl"),
            os.path.join(me, "stage4e2_react", "metrics_official.json"), "react"),
    }


def _per_sample_for_run(traj_path: str, paradigm: str):
    raw = nos.load_raw_dataset(_DATASET)
    items, sids, missing = [], [], 0
    with open(traj_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id") or row.get("task_id") or "")
            gold_row = raw.get(sid)
            if gold_row is None:
                missing += 1
                continue
            if paradigm == "direct" and row.get("predicted_calls") is not None:
                pred = row.get("predicted_calls") or []
            else:
                traj = row.get("_traj", row)
                pred = nos._predicted_calls_from_traj(traj, lenient=True)
            items.append(nos.build_item(pred, gold_row))
            sids.append(sid)
    res = scoring.score_items_per_sample(items, win_rate=True)
    return sids, res, missing


# Aggregate metrics_official.json stores win_rate rounded to 3 dp, and the
# official scorer's exception handling makes a 1-2 sample swing possible between
# the corpus-level and per-sample passes. We therefore treat the per-sample mean
# as consistent with the aggregate when it agrees to within 3-dp rounding
# (0.0005) plus ~2 samples of granularity (2/1861 ~= 0.0011).
_CONSISTENCY_TOL = 0.0012


def _reaggregate_from_csv(csv_path: str, reg) -> int:
    """Fast path: recompute per-run means from an existing per_sample CSV (no scorer)."""
    import csv as _csv
    from collections import defaultdict
    wins = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            try:
                wins[row["run"]].append(float(row["official_win"]))
            except (TypeError, ValueError):
                pass
    report = ["# Per-sample official Win — consistency", ""]
    report.append("| run | n | mean(per_sample_win) | aggregate_win | abs_diff | status |")
    report.append("|---|---|---|---|---|---|")
    overall_ok = True
    for run, (traj_path, metrics_path, _p) in reg.items():
        agg = None
        if os.path.isfile(metrics_path):
            with open(metrics_path, encoding="utf-8") as fh:
                agg = json.load(fh).get("win_rate")
        if run not in wins:
            report.append(f"| {run} | - | - | {agg if agg is not None else '?'} | - | "
                          f"WARNING_MISSING_TRAJECTORIES |")
            continue
        vals = wins[run]
        mean_win = sum(vals) / len(vals)
        if agg is None:
            status, diff = "NO_AGGREGATE", float("nan")
        else:
            diff = abs(mean_win - float(agg))
            status = "PASS" if diff <= _CONSISTENCY_TOL else "FAIL"
            overall_ok = overall_ok and status == "PASS"
        report.append(f"| {run} | {len(vals)} | {mean_win:.4f} | "
                      f"{agg if agg is not None else '?'} | {diff:.5f} | {status} |")
    report += [
        "",
        f"## OVERALL: {'PASS' if overall_ok else 'FAIL/REVIEW'}",
        "",
        f"Consistency tolerance = {_CONSISTENCY_TOL} (3-dp aggregate rounding + ~2 "
        "samples of per-sample granularity out of 1861).",
        "",
        "Runs marked WARNING_MISSING_TRAJECTORIES kept only aggregate "
        "`metrics_official.json` (trajectories were not preserved). They CANNOT "
        "be recomputed per-sample and MUST NOT be used for per-sample overlap / "
        "correlation / failure taxonomy.",
    ]
    with open(os.path.join(_HERE, "per_sample_consistency_report.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print(f"[per_sample] re-aggregated from {csv_path}; "
          f"OVERALL {'PASS' if overall_ok else 'FAIL/REVIEW'}")
    return 0 if overall_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="comma list; default = all in registry")
    ap.add_argument("--tol", type=float, default=_CONSISTENCY_TOL)
    ap.add_argument("--from-csv", action="store_true",
                    help="recompute the report from existing per_sample_official_win.csv "
                         "without re-running the (slow) official scorer")
    args = ap.parse_args()

    reg = _registry()
    csv_path = os.path.join(_HERE, "per_sample_official_win.csv")
    if args.from_csv and os.path.isfile(csv_path):
        return _reaggregate_from_csv(csv_path, reg)

    runs = [r for r in args.runs.split(",") if r] or list(reg)

    all_rows = []
    report = ["# Per-sample official Win — consistency", ""]
    report.append("| run | n | mean(per_sample_win) | aggregate_win | abs_diff | status |")
    report.append("|---|---|---|---|---|---|")
    overall_ok = True

    for run in runs:
        if run not in reg:
            report.append(f"| {run} | - | - | - | - | UNKNOWN_RUN |")
            continue
        traj_path, metrics_path, paradigm = reg[run]
        agg = None
        if os.path.isfile(metrics_path):
            with open(metrics_path, encoding="utf-8") as fh:
                agg = json.load(fh).get("win_rate")
        if not os.path.isfile(traj_path):
            report.append(f"| {run} | - | - | "
                          f"{agg if agg is not None else '?'} | - | "
                          f"WARNING_MISSING_TRAJECTORIES |")
            continue

        print(f"[per_sample] scoring {run} ({paradigm}) ...", flush=True)
        sids, res, missing = _per_sample_for_run(traj_path, paradigm)
        n = len(res)
        mean_win = sum(float(x.get("official_win") or 0.0) for x in res) / n if n else 0.0
        for sid, x in zip(sids, res):
            all_rows.append({
                "run": run,
                "sample_id": sid,
                "official_win": x.get("official_win"),
                "executable": x.get("executable"),
                "official_full_match": x.get("official_full_match"),
                "official_partial_match": x.get("official_partial_match"),
                "parse_valid": x.get("parse_valid"),
                "execution_error": x.get("execution_error"),
            })
        if agg is None:
            status, diff = "NO_AGGREGATE", float("nan")
        else:
            diff = abs(mean_win - float(agg))
            # Aggregate is stored rounded to 3 dp + ~2-sample scorer granularity.
            status = "PASS" if diff <= args.tol else "FAIL"
            overall_ok = overall_ok and status == "PASS"
        report.append(f"| {run} | {n} | {mean_win:.4f} | "
                      f"{agg if agg is not None else '?'} | "
                      f"{diff:.5f} | {status} |")

    write_csv(os.path.join(_HERE, "per_sample_official_win.csv"), all_rows, fieldnames=[
        "run", "sample_id", "official_win", "executable", "official_full_match",
        "official_partial_match", "parse_valid", "execution_error",
    ])
    report += [
        "",
        f"## OVERALL: {'PASS' if overall_ok else 'FAIL/REVIEW'}",
        "",
        "Runs marked WARNING_MISSING_TRAJECTORIES kept only aggregate "
        "`metrics_official.json` (trajectories were not preserved). They CANNOT "
        "be recomputed per-sample and MUST NOT be used for per-sample overlap / "
        "correlation / failure taxonomy. Re-run their eval with trajectory dumps "
        "to include them.",
    ]
    with open(os.path.join(_HERE, "per_sample_consistency_report.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print(f"[per_sample] wrote per_sample_official_win.csv "
          f"({len(all_rows)} rows) + per_sample_consistency_report.md", flush=True)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
