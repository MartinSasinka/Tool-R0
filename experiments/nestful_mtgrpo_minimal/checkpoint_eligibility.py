"""Checkpoint eligibility for best-adapter crowning (audit Bugs 4 & 5).

A checkpoint may be crowned best_react_win_adapter ONLY when ALL hold:
  * optimizer steps > 0                (a 0-step adapter is NOT a trained model)
  * contributing turns > 0
  * dead_group_rate < threshold        (default 0.95)
  * reward dispatch verified           (resolved == configured, no fallback)
  * dev ReAct Win improves the persisted GLOBAL best
  * regression guard passes            (win >= baseline + margin) when enabled

Used by run_curriculum.sh via the CLI; pure Python so it is unit-testable.

CLI:
    python checkpoint_eligibility.py \
        --train-summary <epoch_out>/train_summary.json \
        --react-win 0.31 --global-best 0.30 \
        --baseline-win 0.305 --regression-guard 1 --regression-margin 0.0 \
        [--out eligibility.json]
Prints "ELIGIBLE" or "INELIGIBLE <reason>"; exit code 0/1 respectively.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional, Tuple

DEAD_GROUP_MAX = 0.95


def evaluate_eligibility(
    train_summary: Dict[str, Any],
    react_win: Optional[float],
    global_best: Optional[float] = None,
    baseline_win: Optional[float] = None,
    regression_guard: bool = True,
    regression_margin: float = 0.0,
    dead_group_max: float = DEAD_GROUP_MAX,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Returns (eligible, reason, meta) — meta holds fields for best_meta.json."""
    steps = int(train_summary.get("steps", 0) or 0)
    contributing = train_summary.get("contributing_turns_total")
    if contributing is None:
        contributing = train_summary.get("contributing_turns", None)
    dead_rate = train_summary.get("dead_group_rate")
    if dead_rate is None:
        dead_rate = train_summary.get("dead_group_rate_last_epoch")
    fallback = bool(train_summary.get("reward_fallback_used", False))
    configured = train_summary.get("reward_policy_configured")
    resolved = train_summary.get("reward_policy_resolved")

    meta = {
        "steps": steps,
        "contributing_turns": contributing,
        "dead_group_rate": dead_rate,
        "reward_policy": configured,
        "resolved_reward_fn": (
            f"{train_summary.get('reward_fn_module', '?')}."
            f"{train_summary.get('reward_fn_name', '?')}"),
        "reward_fallback_used": fallback,
        "trained": steps > 0,
        "react_win_rate": react_win,
        "baseline_win": baseline_win,
        "global_best_win": global_best,
        "regression_guard": bool(regression_guard),
    }

    def _check() -> Tuple[bool, str]:
        if steps <= 0:
            return False, "steps==0 (untrained checkpoint must never be crowned)"
        if contributing is not None and int(contributing) <= 0:
            return False, "contributing_turns==0"
        if dead_rate is not None and float(dead_rate) >= dead_group_max:
            return False, f"dead_group_rate={float(dead_rate):.3f} >= {dead_group_max}"
        if fallback:
            return False, "reward fallback was used during training"
        if configured and resolved and str(configured).lower() != str(resolved).lower():
            # Alias-resolved policies carry the canonical name; a real mismatch
            # means the configured reward did not run.
            if str(resolved).lower() not in str(configured).lower() and \
                    str(configured).lower() not in str(resolved).lower():
                return False, (f"reward policy mismatch: configured={configured} "
                               f"resolved={resolved}")
        if react_win is None:
            return False, "no dev ReAct Win measured"
        if regression_guard and baseline_win is not None and \
                float(react_win) < float(baseline_win) + float(regression_margin):
            return False, (f"regression guard: win={react_win} < baseline="
                           f"{baseline_win}+{regression_margin}")
        if global_best is not None and float(react_win) <= float(global_best):
            return False, (f"win={react_win} does not improve global best="
                           f"{global_best}")
        return True, "ok"

    eligible, reason = _check()
    meta["eligible_for_best"] = eligible
    meta["reason"] = reason
    if not eligible:
        meta["ineligible_reason"] = reason
    return eligible, reason, meta


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _opt_float(v: Optional[str]) -> Optional[float]:
    if v is None or v == "" or str(v).lower() in ("none", "null", "-1", "-1.0"):
        return None
    return float(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-summary", required=True)
    ap.add_argument("--react-win", default=None)
    ap.add_argument("--global-best", default=None)
    ap.add_argument("--baseline-win", default=None)
    ap.add_argument("--regression-guard", default="1")
    ap.add_argument("--regression-margin", default="0.0")
    ap.add_argument("--dead-group-max", default=str(DEAD_GROUP_MAX))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        summary = _load_json(args.train_summary)
    except Exception as exc:  # noqa: BLE001
        print(f"INELIGIBLE cannot_read_train_summary: {exc}")
        return 1

    eligible, reason, meta = evaluate_eligibility(
        summary,
        react_win=_opt_float(args.react_win),
        global_best=_opt_float(args.global_best),
        baseline_win=_opt_float(args.baseline_win),
        regression_guard=(str(args.regression_guard) == "1"),
        regression_margin=float(args.regression_margin),
        dead_group_max=float(args.dead_group_max),
    )
    meta["eligible_for_best"] = eligible
    meta["reason"] = reason
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"ELIGIBLE {reason}" if eligible else f"INELIGIBLE {reason}")
    return 0 if eligible else 1


if __name__ == "__main__":
    sys.exit(main())
