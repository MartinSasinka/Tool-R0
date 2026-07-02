#!/usr/bin/env python3
"""Parser diagnostics over saved trajectories using the canonical parser.

Reports, per run, the rate at which model turns parse under the STRICT gate vs.
the LENIENT recovery, and how often lenient recovers a turn strict rejected
(``parse_recovery_rate``). This is the eval-side counterpart of the v2 unified
parser policy (``nestful_core.parser.parse_canonical``): training gates on
strict, eval logs both.

Output: experiments/comparison/parser_diagnostics.csv
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

from nestful_core.parser import parse_canonical  # noqa: E402
from nestful_core.logging_utils import write_csv  # noqa: E402

_PARTIAL = os.path.join(_EXPERIMENTS, "nestful_mtgrpo_partial", "outputs", "final_eval")
_MIN = os.path.join(_EXPERIMENTS, "nestful_mtgrpo_minimal", "outputs")


def _default_runs():
    runs = {
        "partial_s1_e4_react": os.path.join(_PARTIAL, "partial_s1_e4_react",
                                            "final_eval_trajectories.jsonl"),
        "partial_s4_e1_react": os.path.join(_PARTIAL, "partial_s4_e1_react",
                                            "final_eval_trajectories.jsonl"),
        "baseline_react": os.path.join(_MIN, "final_eval_baseline_react",
                                       "final_eval_trajectories.jsonl"),
    }
    return {k: v for k, v in runs.items() if os.path.isfile(v)}


def _diag_for_file(path: str):
    n_turns = strict_ok = lenient_ok = recovery = terminal = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            traj = row.get("_traj", row)
            for t in traj.get("turns", []):
                txt = t.get("model_text", "")
                if not txt:
                    continue
                pr = parse_canonical(txt)
                n_turns += 1
                if pr["is_terminal"]:
                    terminal += 1
                if pr["strict_ok"]:
                    strict_ok += 1
                if pr["lenient_ok"]:
                    lenient_ok += 1
                if pr["parse_recovery"]:
                    recovery += 1
    if n_turns == 0:
        return None
    return {
        "n_turns": n_turns,
        "strict_parse_rate": round(strict_ok / n_turns, 4),
        "lenient_parse_rate": round(lenient_ok / n_turns, 4),
        "parse_recovery_rate": round(recovery / n_turns, 4),
        "terminal_rate": round(terminal / n_turns, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "parser_diagnostics.csv"))
    args = ap.parse_args()
    rows = []
    for run, path in _default_runs().items():
        d = _diag_for_file(path)
        if d:
            d["run"] = run
            rows.append(d)
            print(f"[parser_diag] {run}: {d}")
    write_csv(args.out, rows, fieldnames=[
        "run", "n_turns", "strict_parse_rate", "lenient_parse_rate",
        "parse_recovery_rate", "terminal_rate",
    ])
    print(f"[parser_diag] wrote {args.out} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
