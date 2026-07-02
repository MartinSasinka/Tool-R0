"""Consolidate rescored curriculum metrics into a single CSV.

Reads:
  - rescored_official.json   (produced by curricullum/evaluation/rescore_official.py)
  - the original *_multiturn_summary.json files (for final_answer_accuracy + run meta)

Writes:
  - curriculum_official_metrics.csv

The point of this folder (experiments/nestful_grpo) is to express the ORIGINAL
curriculum (plain GRPO/SFT) runs in TODAY's official NESTFUL metrics, so they sit
apples-to-apples next to nestful_mtgrpo_minimal and nestful_mtgrpo_partial.
"""
from __future__ import annotations

import csv
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_EVAL_DIR = os.path.join(_REPO_ROOT, "curricullum", "evaluation")

_RESCORED = os.path.join(_HERE, "rescored_official.json")
_OUT_CSV = os.path.join(_HERE, "curriculum_official_metrics.csv")


def _summary_path(profile: str) -> str:
    """'results1/curriculum_baseline' -> .../results1/curriculum_baseline_multiturn_summary.json"""
    parent, base = profile.split("/", 1)
    return os.path.join(_EVAL_DIR, parent, f"{base}_multiturn_summary.json")


def _load_summary(profile: str) -> dict:
    p = _summary_path(profile)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> int:
    with open(_RESCORED, "r", encoding="utf-8") as fh:
        rescored = json.load(fh)

    cols = [
        "profile", "parser",
        "num_tasks", "num_rollouts_per_task",
        "final_answer_accuracy",            # original metric (executor pass-rate)
        "f1_func", "f1_param",
        "partial_sequence_accuracy", "full_sequence_accuracy",
        "win_rate",                         # official; null until Linux rescore
        "avg_pred_calls",
    ]

    rows = []
    for profile, entry in rescored.items():
        summ = _load_summary(profile)
        faa = summ.get("final_answer_accuracy")
        nrp = summ.get("num_rollouts_per_task")
        for parser in ("stored", "reparse"):
            m = entry.get(parser)
            if not m:
                continue
            rows.append({
                "profile": profile,
                "parser": parser,
                "num_tasks": entry.get("num_tasks"),
                "num_rollouts_per_task": nrp,
                "final_answer_accuracy": faa,
                "f1_func": m.get("f1_func"),
                "f1_param": m.get("f1_param"),
                "partial_sequence_accuracy": m.get("partial_sequence_accuracy"),
                "full_sequence_accuracy": m.get("full_sequence_accuracy"),
                "win_rate": m.get("win_rate"),
                "avg_pred_calls": m.get("avg_pred_calls"),
            })

    with open(_OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {_OUT_CSV}  ({len(rows)} rows)")

    # Console preview
    hdr = f"{'profile':<42}{'parser':<9}{'FAA':>7}{'F1func':>8}{'F1par':>7}{'Part':>7}{'Full':>7}{'Win':>7}{'calls':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def f(x, p=4):
            return f"{x:.{p}f}" if isinstance(x, (int, float)) else "-"
        print(f"{r['profile']:<42}{r['parser']:<9}{f(r['final_answer_accuracy'],3):>7}"
              f"{f(r['f1_func']):>8}{f(r['f1_param']):>7}{f(r['partial_sequence_accuracy']):>7}"
              f"{f(r['full_sequence_accuracy']):>7}{f(r['win_rate']):>7}{f(r['avg_pred_calls'],2):>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
