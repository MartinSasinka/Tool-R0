#!/usr/bin/env python3
"""Success-gate checker for the v2 pipeline.

Two modes:

  preflight   Offline gates that MUST pass before any training is launched:
                - reward v2 unit tests + report-loadable + parser round-trip
                - gold_replay_full GATE == PASS
                - per_sample_consistency OVERALL == PASS
              The run_pilot_v2 / run_full_v2 scripts call this first and abort
              on failure.

  pilot       Post-pilot gates (request §16) evaluated from a pilot metrics JSON
              ({baseline_react_win, validation_react_win, no_tool_call_rate, ...}
              with *_start / *_end where relevant). Flags reward-hacking when the
              training reward rises while validation Win falls.

Usage:
  python experiments/comparison/check_gates.py preflight
  python experiments/comparison/check_gates.py pilot --metrics pilot_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


def _ok(name, passed, detail=""):
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return passed


def _file_contains(path, needle):
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as fh:
        return needle in fh.read()


def preflight() -> int:
    print("== preflight gates ==")
    results = []

    # 1. Unit tests (rewards v2 + reports loadable + parser).
    tests = [
        os.path.join(_REPO, "tests", "test_execution_reward_v2.py"),
        os.path.join(_REPO, "tests", "test_rewards_v2.py"),
        os.path.join(_REPO, "tests", "test_reports_loadable.py"),
    ]
    tests = [t for t in tests if os.path.isfile(t)]
    rc = subprocess.call([sys.executable, "-m", "pytest", "-q", *tests], cwd=_REPO)
    results.append(_ok("reward/report unit tests", rc == 0, f"pytest rc={rc}"))

    # 2. Gold replay full.
    gr = os.path.join(_HERE, "gold_replay_full_report.md")
    results.append(_ok("gold_replay_full GATE PASS", _file_contains(gr, "## GATE: PASS"),
                       gr if os.path.isfile(gr) else "(missing - run gold_replay_full.py)"))

    # 3. Per-sample consistency.
    ps = os.path.join(_HERE, "per_sample_consistency_report.md")
    results.append(_ok("per_sample_consistency OVERALL PASS",
                       _file_contains(ps, "## OVERALL: PASS"),
                       ps if os.path.isfile(ps) else "(missing - run recompute_per_sample_official.py)"))

    passed = all(results)
    print(f"\nPREFLIGHT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def pilot(metrics_path: str) -> int:
    print("== pilot success gates ==")
    if not os.path.isfile(metrics_path):
        print(f"FAIL: pilot metrics not found: {metrics_path}")
        return 1
    with open(metrics_path, encoding="utf-8") as fh:
        m = json.load(fh)

    base = float(m.get("baseline_react_win", 0.0))
    val = float(m.get("validation_react_win", 0.0))
    reward_start = m.get("reward_total_start")
    reward_end = m.get("reward_total_end")

    def _not_rising(key):
        s, e = m.get(f"{key}_start"), m.get(f"{key}_end")
        if s is None or e is None:
            return True, "n/a"
        return (float(e) <= float(s) + 1e-9), f"{s}->{e}"

    results = []
    results.append(_ok("validation_react_win >= baseline - 1pp", val >= base - 0.01,
                       f"val={val:.3f} baseline={base:.3f}"))
    for key in ("no_tool_call_rate", "too_few_calls_rate", "parse_error_rate",
                "invalid_reference_rate"):
        ok, det = _not_rising(key)
        results.append(_ok(f"{key} not rising", ok, det))
    # executable_trajectory_rate should not fall.
    es, ee = m.get("executable_trajectory_rate_start"), m.get("executable_trajectory_rate_end")
    if es is not None and ee is not None:
        results.append(_ok("executable_trajectory_rate not falling",
                           float(ee) >= float(es) - 1e-9, f"{es}->{ee}"))

    # Reward-hacking flag: reward up while validation Win down vs baseline.
    if reward_start is not None and reward_end is not None:
        hacking = (float(reward_end) > float(reward_start)) and (val < base - 0.02)
        results.append(_ok("no reward-hacking (reward up & Win down)", not hacking,
                           f"reward {reward_start}->{reward_end}, val={val:.3f} vs base={base:.3f}"))
        if hacking:
            print("  >> REWARD HACKING SUSPECTED: stop, export failure samples.")

    passed = all(results)
    print(f"\nPILOT GATES: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("preflight")
    p = sub.add_parser("pilot")
    p.add_argument("--metrics", required=True)
    args = ap.parse_args()
    if args.mode == "preflight":
        return preflight()
    return pilot(args.metrics)


if __name__ == "__main__":
    raise SystemExit(main())
