#!/usr/bin/env python3
"""Reward ablation — hard gates + lexicographic arm selection (spec §12-13).

Consumes the Round N cross-arm summary written by
`summarize_reward_ablation.py round-summary` (ROUND<N>_SUMMARY.json) plus
each arm's training diagnostics (dead_group_rate, terminal inversions from
`lib.reward_ablation_registry` invariants, parse/executability rates) and:

  1. applies the HARD GATES (spec §12) -> PASS / CONDITIONAL / FAIL per arm;
  2. ranks PASS/CONDITIONAL arms lexicographically (spec §13) — NEVER by a
     single weighted score and NEVER by training reward mean;
  3. writes reports/reward_ablation/ROUND<N>_DECISION.{md,json};
  4. for Round 1, additionally writes reports/reward_ablation/ROUND2_PLAN.json
     selecting A0_R0_CURRENT + the top 2 non-A0 arms that PASSed, with the
     Round 2 seed (20260725) and exact `run_reward_ablation.py` commands —
     Round 2 itself is never launched automatically (spec §2).

This script performs NO training/eval itself — it is pure post-hoc decision
logic over already-computed metrics, so it is fully deterministic and
testable on CPU with synthetic input (see tests/test_select_reward_arms.py).

Usage:
  python select_reward_arms.py --round 1 \\
      --summary reports/reward_ablation/round1/ROUND1_SUMMARY.json \\
      --training-diagnostics reports/reward_ablation/round1/training_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
REPORTS_DIR = _V3 / "reports" / "reward_ablation"

CONTROL_ARM = "A0_R0_CURRENT"
SCIENTIFIC_CONTROL_ARM = "A1_OUTCOME_ONLY"  # may fail gates by design (spec §12)
ROUND2_SEED = 20260725


def _get(d: Dict[str, Any], *path, default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def evaluate_gates(arm: str, entry: Dict[str, Any], control_entry: Dict[str, Any],
                    training_diag: Dict[str, Any]) -> Dict[str, Any]:
    """Returns {"verdict": PASS|CONDITIONAL|FAIL, "reasons": [...], "checks": {...}}."""
    reasons: List[str] = []
    checks: Dict[str, Any] = {}

    terminal_inversions = int(training_diag.get("terminal_inversions", 0) or 0)
    checks["terminal_inversions"] = terminal_inversions
    if terminal_inversions > 0:
        reasons.append(f"terminal_inversion_count={terminal_inversions} > 0")

    nan_or_inf = bool(training_diag.get("nan_or_inf_detected", False))
    checks["nan_or_inf_detected"] = nan_or_inf
    if nan_or_inf:
        reasons.append("NaN/Inf detected in rewards")

    official_loss_beats_success = bool(training_diag.get("official_loss_beats_success_in_group", False))
    checks["official_loss_beats_success_in_group"] = official_loss_beats_success
    if official_loss_beats_success:
        reasons.append("an official loss outranked an official success within the same rollout group")

    dead_rate = _get(entry, "metrics", "diagnostics", "dead_group_rate")
    dead_rate = training_diag.get("dead_group_rate", dead_rate)
    control_dead_rate = training_diag.get("control_dead_group_rate")
    checks["dead_group_rate"] = dead_rate
    checks["control_dead_group_rate"] = control_dead_rate
    if dead_rate is not None and control_dead_rate is not None:
        if dead_rate - control_dead_rate > 0.05:
            # A1_OUTCOME_ONLY is EXPECTED to have a high dead-group rate (no
            # process tie-break -> more literal reward ties); still recorded
            # as a reason so the verdict-collapse logic below can downgrade
            # it to CONDITIONAL instead of FAIL (spec §12).
            reasons.append(f"dead_group_rate {dead_rate} worse than control {control_dead_rate} by >5pp")

    parse_rate = _get(entry, "metrics", "diagnostics", "f1_func")  # placeholder if parse_rate absent
    parse_rate = training_diag.get("parse_rate", parse_rate)
    control_parse_rate = training_diag.get("control_parse_rate")
    checks["parse_rate"] = parse_rate
    checks["control_parse_rate"] = control_parse_rate
    if parse_rate is not None and control_parse_rate is not None:
        if control_parse_rate - parse_rate > 0.02:
            reasons.append(f"parse_rate {parse_rate} worse than control {control_parse_rate} by >2pp")

    exec_rate = _get(entry, "metrics", "diagnostics", "executable_rate")
    control_exec = _get(control_entry, "metrics", "diagnostics", "executable_rate")
    checks["executable_rate"] = exec_rate
    checks["control_executable_rate"] = control_exec
    if exec_rate is not None and control_exec is not None:
        if control_exec - exec_rate > 0.02:
            reasons.append(f"executable_rate {exec_rate} worse than control {control_exec} by >2pp")

    synthetic_success_drop = training_diag.get("synthetic_terminal_success_drop")
    checks["synthetic_terminal_success_drop"] = synthetic_success_drop
    if synthetic_success_drop is not None and synthetic_success_drop > 0.05:
        reasons.append(f"synthetic path-invariant terminal success dropped by {synthetic_success_drop}")

    exec_wrong_pos_adv = training_diag.get("executable_wrong_positive_advantage_rate")
    control_exec_wrong_pos_adv = training_diag.get("control_executable_wrong_positive_advantage_rate")
    checks["executable_wrong_positive_advantage_rate"] = exec_wrong_pos_adv
    if (exec_wrong_pos_adv is not None and control_exec_wrong_pos_adv is not None
            and exec_wrong_pos_adv > control_exec_wrong_pos_adv * 1.5
            and exec_wrong_pos_adv > 0.05):
        reasons.append("executable_wrong positive-advantage rate rose sharply vs control")

    reward_up_success_down = bool(training_diag.get("reward_up_but_terminal_success_down", False))
    checks["reward_up_but_terminal_success_down"] = reward_up_success_down
    if reward_up_success_down:
        reasons.append("training reward increased while synthetic terminal success decreased")

    reward_hacking_flag = bool(training_diag.get("reward_hacking_suspected", False))
    checks["reward_hacking_suspected"] = reward_hacking_flag
    if reward_hacking_flag:
        reasons.append("reward-hacking / degenerate strategy suspected")

    hard_fail = bool(reasons)
    if hard_fail and arm == SCIENTIFIC_CONTROL_ARM and all(
        r.startswith("dead_group_rate") for r in reasons
    ):
        # A1 may be kept as a scientific control despite high dead-group rate
        # (spec §12) but is still marked CONDITIONAL, never auto-PASS, and
        # never advances to Round 2 on its own merits.
        return {"verdict": "CONDITIONAL", "reasons": reasons, "checks": checks,
                "note": "scientific control — high dead-group rate tolerated, not eligible for Round 2 selection"}
    if hard_fail:
        return {"verdict": "FAIL", "reasons": reasons, "checks": checks}
    return {"verdict": "PASS", "reasons": [], "checks": checks}


def _lexicographic_key(arm: str, entry: Dict[str, Any], training_diag: Dict[str, Any]) -> Tuple:
    """Spec §13 ordering, most-important-first. Returns a tuple sortable
    descending (i.e. caller should sort with reverse=True) where every
    element is a "higher is better" numeric proxy; None -> treated as worst
    (-inf) so incomplete data never wins by omission."""
    def s(x: Optional[float]) -> float:
        return float("-inf") if x is None else float(x)

    synthetic_success = training_diag.get("synthetic_terminal_success_rate")
    off = _get(entry, "metrics", "official", default={}) or {}
    win_rate = off.get("win_rate")
    pc0 = entry.get("paired_vs_c0", {}) or {}
    gained_minus_regressed = (pc0.get("n_gained") or 0) - (pc0.get("n_regressed") or 0)
    exec_wrong_rate = _get(entry, "metrics", "diagnostics", "executable_rate")
    parse_stability = training_diag.get("parse_rate")
    turn2_cond_acc = training_diag.get("turn2_conditional_tool_accuracy")
    turn3_cond_acc = training_diag.get("turn3_conditional_tool_accuracy")
    dead_mixed_signal = -(training_diag.get("dead_group_rate") or 0.0)  # lower dead rate is better
    seed_stability = training_diag.get("seed_stability_score")  # filled after Round 2
    simplicity = -{"A1_OUTCOME_ONLY": 0, "A0_R0_CURRENT": 1, "A2_R3_OUTCOME_FIRST": 2,
                   "A3_VERIFIABLE_PROCESS": 3, "A4_GATED_VERIFIABLE": 3}.get(arm, 2)

    return (
        s(synthetic_success),
        s(win_rate),
        s(gained_minus_regressed),
        s(exec_wrong_rate),
        s(parse_stability),
        s(turn2_cond_acc),
        s(turn3_cond_acc),
        s(dead_mixed_signal),
        s(seed_stability),
        s(simplicity),
    )


def rank_arms(gate_results: Dict[str, Dict[str, Any]], summary: Dict[str, Any],
              training_diag_by_arm: Dict[str, Dict[str, Any]]) -> List[str]:
    eligible = [a for a, g in gate_results.items() if g["verdict"] in ("PASS", "CONDITIONAL")
                and a != SCIENTIFIC_CONTROL_ARM]
    eligible.sort(key=lambda a: _lexicographic_key(a, summary["arms"].get(a, {}), training_diag_by_arm.get(a, {})),
                  reverse=True)
    return eligible


def build_round2_plan(round1_ranked: List[str], gate_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    top2 = [a for a in round1_ranked if a != CONTROL_ARM and gate_results[a]["verdict"] == "PASS"][:2]
    arms = [CONTROL_ARM] + top2
    commands = []
    for arm in arms:
        commands.append(
            "python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation.py "
            f"--round 2 --reward-arm {arm} --seed {ROUND2_SEED} "
            "--train-subset experiments/nestful_synthetic_curriculum_v3/reports/reward_ablation/data/train_subset_160.jsonl "
            "--eval-subset experiments/nestful_synthetic_curriculum_v3/reports/reward_ablation/data/nestful_diagnostic_500_ids.json"
        )
    return {
        "round": 2,
        "not_auto_launched": True,
        "selection_basis": "Round 1 lexicographic ranking (spec §13), gates from spec §12",
        "seed": ROUND2_SEED,
        "arms": arms,
        "top2_candidates": top2,
        "train_subset": "reports/reward_ablation/data/train_subset_160.jsonl",
        "eval_subset": "reports/reward_ablation/data/nestful_diagnostic_500_ids.json",
        "commands": commands,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--summary", type=Path, required=True,
                    help="ROUND<N>_SUMMARY.json from summarize_reward_ablation.py round-summary")
    ap.add_argument("--training-diagnostics", type=Path, default=None,
                    help="JSON: {arm_id: {dead_group_rate, terminal_inversions, ...}}")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    training_diag_by_arm = (json.loads(args.training_diagnostics.read_text(encoding="utf-8"))
                            if args.training_diagnostics and args.training_diagnostics.is_file() else {})

    control_entry = summary["arms"].get(CONTROL_ARM, {})
    gate_results = {}
    for arm, entry in summary["arms"].items():
        gate_results[arm] = evaluate_gates(arm, entry, control_entry, training_diag_by_arm.get(arm, {}))

    ranked = rank_arms(gate_results, summary, training_diag_by_arm)

    out_dir = args.out_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    decision = {
        "round": args.round,
        "gate_results": gate_results,
        "lexicographic_ranking": ranked,
        "control_arm": CONTROL_ARM,
        "scientific_control_arm": SCIENTIFIC_CONTROL_ARM,
        "note": "Ranking excludes A0 (control, always retained for comparison) and "
                "A1 (scientific control, informative regardless of gate outcome). "
                "Training reward mean is NOT part of this decision (spec §13).",
    }
    (out_dir / f"ROUND{args.round}_DECISION.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# Round {args.round} — Decision", "", "## Gate verdicts", "",
             "| Arm | Verdict | Reasons |", "|---|---|---|"]
    for arm, g in gate_results.items():
        lines.append(f"| {arm} | {g['verdict']} | {'; '.join(g['reasons']) or '—'} |")
    lines += ["", "## Lexicographic ranking (excl. A0/A1)", ""]
    for i, arm in enumerate(ranked, 1):
        lines.append(f"{i}. {arm}")
    (out_dir / f"ROUND{args.round}_DECISION.md").write_text("\n".join(lines), encoding="utf-8")

    if args.round == 1:
        plan = build_round2_plan(ranked, gate_results)
        (out_dir / "ROUND2_PLAN.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[select_reward_arms] Round 2 plan (NOT auto-launched) -> {out_dir / 'ROUND2_PLAN.json'}")

    print(f"[select_reward_arms] decision -> {out_dir / f'ROUND{args.round}_DECISION.json'}")
    print(f"[select_reward_arms] ranking: {ranked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
