#!/usr/bin/env python3
"""Full gold-trace replay over ALL NESTFUL tasks (sanity gate before training).

Feeds each task's GOLD calls as the prediction through the official scorer and
checks that official_win / official_full_match / parse_valid / executable are all
~= 1.0. If the executor + scorer cannot reproduce the gold answer for a task, that
task is unscoreable and any Win-Rate comparison built on it is suspect — so we
refuse to proceed to training.

Outputs (under experiments/comparison/):
  gold_replay_full_report.md
  gold_replay_failures.csv   (one row per task with official_win < 1.0)

Usage:
  python experiments/comparison/gold_replay_full.py [--limit N] [--dataset PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.dirname(_HERE)
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)

from nestful_core import scoring  # noqa: E402
from nestful_core.logging_utils import write_csv  # noqa: E402

_DEFAULT_DATASET = os.path.join(
    _EXPERIMENTS, "nestful_mtgrpo_minimal", "data", "NESTFUL-main", "data_v2",
    "nestful_data.jsonl",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=_DEFAULT_DATASET)
    ap.add_argument("--limit", type=int, default=0, help="0 = all tasks")
    ap.add_argument("--report", default=os.path.join(_HERE, "gold_replay_full_report.md"))
    ap.add_argument("--failures", default=os.path.join(_HERE, "gold_replay_failures.csv"))
    args = ap.parse_args()

    raw = []
    with open(args.dataset, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    if args.limit:
        raw = raw[: args.limit]
    n = len(raw)
    print(f"[gold_replay_full] scoring {n} tasks (gold calls AS prediction)...", flush=True)

    items = []
    sample_ids = []
    for r in raw:
        out = r.get("output")
        if isinstance(out, str):
            out = json.loads(out)
        items.append(scoring.build_item(out, r))
        sample_ids.append(r.get("sample_id") or r.get("task_id") or r.get("id"))

    t0 = time.time()
    res = scoring.score_items_per_sample(items, win_rate=True)
    dt = time.time() - t0

    def _mean(key):
        vals = [float(x.get(key) or 0.0) for x in res]
        return sum(vals) / len(vals) if vals else 0.0

    win = _mean("official_win")
    full = _mean("official_full_match")
    parse = sum(1 for x in res if x.get("parse_valid")) / n
    execu = sum(1 for x in res if x.get("executable")) / n
    exec_err_rate = sum(1 for x in res if x.get("execution_error")) / n

    failures = []
    for sid, x in zip(sample_ids, res):
        if float(x.get("official_win") or 0.0) < 1.0:
            failures.append({
                "sample_id": sid,
                "official_win": x.get("official_win"),
                "official_full_match": x.get("official_full_match"),
                "executable": x.get("executable"),
                "parse_valid": x.get("parse_valid"),
                "execution_error": x.get("execution_error"),
                "n_pred_calls": x.get("n_pred_calls"),
            })
    write_csv(args.failures, failures, fieldnames=[
        "sample_id", "official_win", "official_full_match", "executable",
        "parse_valid", "execution_error", "n_pred_calls",
    ])

    gate = (win >= 0.98 and parse >= 0.98 and execu >= 0.98)
    status = "PASS" if gate else "FAIL"
    lines = [
        "# Gold-trace replay (full dataset)",
        "",
        f"- dataset: `{args.dataset}`",
        f"- tasks scored: **{n}**",
        f"- scoring time: {dt:.1f}s",
        "",
        "| metric | value | expected |",
        "|---|---|---|",
        f"| official_win | {win:.4f} | ~1.0 |",
        f"| official_full_match | {full:.4f} | ~1.0 |",
        f"| parse_valid | {parse:.4f} | ~1.0 |",
        f"| executable | {execu:.4f} | ~1.0 |",
        f"| execution_error_rate | {exec_err_rate:.4f} | ~0.0 |",
        "",
        f"- tasks with official_win < 1.0: **{len(failures)}** "
        f"(see `gold_replay_failures.csv`)",
        "",
        f"## GATE: {status}",
        "",
        ("Gold replay reproduces the gold answer via our executor + the official "
         "scorer, so Win Rate built on this pipeline is trustworthy."
         if gate else
         "Gold replay did NOT reproduce gold answers for enough tasks. Do NOT "
         "proceed to training until the failing tasks are understood."),
    ]
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[gold_replay_full] {status} win={win:.4f} parse={parse:.4f} "
          f"exec={execu:.4f} fails={len(failures)}", flush=True)
    print(f"[gold_replay_full] wrote {args.report} and {args.failures}", flush=True)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
