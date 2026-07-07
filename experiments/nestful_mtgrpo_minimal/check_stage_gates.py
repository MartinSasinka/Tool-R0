"""Stage advancement gates for the gated curriculum smoke pilot.

Decides whether stage N passed and the run may proceed to stage N+1.
Called by run_curriculum.sh after each stage when STAGE_GATES=1.

Gates (from the post-audit pilot spec):

  stage1 -> stage2:
    steps > 0, contributing_turns > 0
    dead_group_rate < 0.70, dead_group_rate_first_50 < 0.90
    resolved reward == configured reward (no fallback)
    fractional_rewards_present == true (for graded rewards)
    dev_win >= baseline_dev_win - 0.02
    no catastrophic no_tool_call / too_few_calls rates

  stage2 -> stage3 (additionally):
    position_artifact_group_rate < 0.20
    dev_win >= previous_stage_dev_win - 0.02
    dev_win >= baseline_dev_win - 0.03
    avg_predicted_calls not collapsed

Writes <stage_out>/stage_gate_report.json and prints PASS/FAIL per gate.
Exit code: 0 = gates pass, 4 = gates fail (pilot must stop).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, Optional

STRICT_ALIASES = ("strict", "strict_gold_trace", "strict_gold_trace_legacy")


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def _latest_train_summary(stage_out: str) -> Optional[Dict[str, Any]]:
    candidates = sorted(glob.glob(os.path.join(stage_out, "epoch_*", "train_summary.json")))
    if not candidates:
        return None
    return _load_json(candidates[-1])


def _best_stage_win(stage_out: str) -> Optional[float]:
    path = os.path.join(stage_out, "epoch_summary.jsonl")
    best = None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                w = row.get("react_win_rate")
                if w is not None and (best is None or float(w) > best):
                    best = float(w)
    except Exception:  # noqa: BLE001
        return None
    return best


def _opt_float(v) -> Optional[float]:
    if v is None or v == "" or str(v).lower() in ("none", "null"):
        return None
    x = float(v)
    return None if x < 0 else x


def check_stage_gates(
    stage: int,
    train_summary: Dict[str, Any],
    dev_win: Optional[float],
    baseline_win: Optional[float],
    prev_stage_win: Optional[float],
    *,
    dead_group_max: float = 0.70,
    first50_dead_max: float = 0.90,
    position_artifact_max: float = 0.20,
    no_tool_call_max: float = 0.60,
    too_few_calls_max: float = 0.90,
) -> Dict[str, Any]:
    """Pure gate evaluation — returns the gate report dict."""
    gates: Dict[str, Dict[str, Any]] = {}

    def gate(name: str, ok: Optional[bool], detail: str) -> None:
        gates[name] = {"pass": ok, "detail": detail}

    steps = int(train_summary.get("steps", 0) or 0)
    gate("steps_gt_0", steps > 0, f"steps={steps}")

    contributing = train_summary.get("contributing_turns_total")
    if contributing is None:
        gate("contributing_turns_gt_0", None, "not recorded (skipped)")
    else:
        gate("contributing_turns_gt_0", int(contributing) > 0,
             f"contributing_turns={contributing}")

    dead = train_summary.get("dead_group_rate")
    gate("dead_group_rate_lt_max",
         (float(dead) < dead_group_max) if dead is not None else None,
         f"dead_group_rate={dead} (max {dead_group_max})")

    d50 = train_summary.get("dead_group_rate_first_50")
    gate("first_50_dead_group_rate_lt_max",
         (float(d50) < first50_dead_max) if d50 is not None else None,
         f"dead_group_rate_first_50={d50} (max {first50_dead_max})")

    configured = str(train_summary.get("reward_policy_configured", "") or "")
    resolved = str(train_summary.get("reward_policy_resolved", "") or "")
    fallback = bool(train_summary.get("reward_fallback_used", False))
    dispatch_ok = (not fallback) and (
        configured.lower() == resolved.lower()
        or resolved.lower() in configured.lower()
        or configured.lower() in resolved.lower())
    gate("resolved_reward_matches_configured", dispatch_ok,
         f"configured={configured} resolved={resolved} fallback={fallback}")

    graded = configured.lower() not in STRICT_ALIASES
    frac = train_summary.get("fractional_rewards_present")
    if graded:
        gate("fractional_rewards_present", bool(frac),
             f"fractional_rewards_present={frac}")
    else:
        gate("fractional_rewards_present", None,
             "strict reward configured (skipped)")

    if dev_win is not None and baseline_win is not None:
        margin = 0.02 if stage <= 1 else 0.03
        gate("dev_win_vs_baseline", float(dev_win) >= float(baseline_win) - margin,
             f"dev_win={dev_win} baseline={baseline_win} (allowed drop {margin})")
    else:
        gate("dev_win_vs_baseline", None,
             f"dev_win={dev_win} baseline={baseline_win} (not measurable)")

    ntc = train_summary.get("no_tool_call_rate")
    gate("no_catastrophic_no_tool_call",
         (float(ntc) <= no_tool_call_max) if ntc is not None else None,
         f"no_tool_call_rate={ntc} (max {no_tool_call_max})")

    tfc = train_summary.get("too_few_calls_rate")
    gate("no_catastrophic_too_few_calls",
         (float(tfc) <= too_few_calls_max) if tfc is not None else None,
         f"too_few_calls_rate={tfc} (max {too_few_calls_max})")

    if stage >= 2:
        par = train_summary.get("position_artifact_group_rate")
        gate("position_artifact_rate_lt_max",
             (float(par) < position_artifact_max) if par is not None else None,
             f"position_artifact_group_rate={par} (max {position_artifact_max})")

        if dev_win is not None and prev_stage_win is not None:
            gate("dev_win_vs_previous_stage",
                 float(dev_win) >= float(prev_stage_win) - 0.02,
                 f"dev_win={dev_win} prev_stage={prev_stage_win} (allowed drop 0.02)")
        else:
            gate("dev_win_vs_previous_stage", None,
                 f"dev_win={dev_win} prev_stage={prev_stage_win} (not measurable)")

        apc = train_summary.get("avg_predicted_calls")
        # Collapse guard: predicting < half the expected calls for this stage.
        gate("avg_predicted_calls_not_collapsed",
             (float(apc) >= 0.5 * stage) if apc is not None else None,
             f"avg_predicted_calls={apc} (min {0.5 * stage})")

    hard_fail = [k for k, v in gates.items() if v["pass"] is False]
    skipped = [k for k, v in gates.items() if v["pass"] is None]
    return {
        "stage": stage,
        "gates": gates,
        "hard_failures": hard_fail,
        "skipped": skipped,
        "pass": not hard_fail,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--stage-out", required=True)
    ap.add_argument("--dev-win", default=None)
    ap.add_argument("--baseline-win", default=None)
    ap.add_argument("--prev-stage-win", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    train_summary = _latest_train_summary(args.stage_out)
    if train_summary is None:
        print(f"[stage_gates] FAIL: no train_summary.json under {args.stage_out}")
        report = {"stage": args.stage, "pass": False,
                  "hard_failures": ["train_summary_missing"], "gates": {}}
    else:
        dev_win = _opt_float(args.dev_win)
        if dev_win is None:
            dev_win = _best_stage_win(args.stage_out)
        report = check_stage_gates(
            args.stage, train_summary,
            dev_win=dev_win,
            baseline_win=_opt_float(args.baseline_win),
            prev_stage_win=_opt_float(args.prev_stage_win),
        )

    out = args.out or os.path.join(args.stage_out, "stage_gate_report.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    for name, g in (report.get("gates") or {}).items():
        status = "PASS" if g["pass"] else ("SKIP" if g["pass"] is None else "FAIL")
        print(f"[stage_gates] stage{args.stage} {status:4s} {name}: {g['detail']}")
    print(f"[stage_gates] stage{args.stage} overall: "
          f"{'PASS' if report['pass'] else 'FAIL'} -> {out}")
    return 0 if report["pass"] else 4


if __name__ == "__main__":
    sys.exit(main())
