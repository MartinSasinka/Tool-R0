#!/usr/bin/env python3
"""One-off local re-analysis of downloaded Round 1 zips with true shared C0 eval."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
if str(_V3) not in sys.path:
    sys.path.insert(0, str(_V3))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "summarize_reward_ablation",
    _HERE / "summarize_reward_ablation.py",
)
SUM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SUM)  # type: ignore[union-attr]

_spec2 = importlib.util.spec_from_file_location(
    "select_reward_arms",
    _HERE / "select_reward_arms.py",
)
SEL = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(SEL)  # type: ignore[union-attr]

ROOT = _V3 / "outputs" / "runs" / "_local_round1_analysis"
OUT = _V3 / "reports" / "reward_ablation" / "round1_corrected"
SEED = "20260724"

ARMS = [
    "A0_R0_CURRENT",
    "A1_OUTCOME_ONLY",
    "A2_R3_OUTCOME_FIRST",
    "A3_VERIFIABLE_PROCESS",
    "A4_GATED_VERIFIABLE",
]


def eval_dir(arm: str) -> Path:
    run = ROOT / f"reward_ablation_r1_{arm}_seed{SEED}" / f"reward_ablation_r1_{arm}_seed{SEED}"
    return run / "eval" / arm / SEED


def c0_dir() -> Path:
    return ROOT / "shared_C0_eval_500" / "shared_C0_eval_500" / "eval" / "C0" / SEED


def train_summary_path(arm: str) -> Path:
    run = ROOT / f"reward_ablation_r1_{arm}_seed{SEED}" / f"reward_ablation_r1_{arm}_seed{SEED}"
    return run / "train" / "train_summary.json"


def build_training_diagnostics() -> dict:
    out = {}
    control = None
    for arm in ARMS:
        p = train_summary_path(arm)
        if not p.is_file():
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        dead = s.get("dead_group_rate")
        if dead is None and "dead_groups" in s:
            dead = s.get("dead_group_rate")
        diag = {
            "dead_group_rate": s.get("dead_group_rate"),
            "terminal_inversions": s.get("terminal_inversions", 0),
            "nan_or_inf_detected": s.get("nan_or_inf_detected", False),
            "official_loss_beats_success_in_group": s.get(
                "official_loss_beats_success_in_group", False
            ),
        }
        out[arm] = diag
        if arm == "A0_R0_CURRENT":
            control = diag
    if control:
        for arm in out:
            out[arm]["control_dead_group_rate"] = control.get("dead_group_rate")
    return out


def main() -> int:
    c0 = c0_dir()
    a0 = eval_dir("A0_R0_CURRENT")
    if not (c0 / "final_eval_trajectories.jsonl").is_file():
        raise SystemExit(f"missing C0 trajectories: {c0}")
    OUT.mkdir(parents=True, exist_ok=True)
    arm_dirs = {}
    rows = []
    for arm in ARMS:
        d = eval_dir(arm)
        if not (d / "final_eval_trajectories.jsonl").is_file():
            print(f"SKIP {arm}: no trajectories in {d}")
            continue
        SUM.summarize_arm(arm, d, d, c0, a0 if arm != "A0_R0_CURRENT" else None)
        arm_dirs[arm] = d
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        pc0 = json.loads((d / "paired_vs_c0.json").read_text(encoding="utf-8"))
        pr0 = {}
        if (d / "paired_vs_r0.json").is_file():
            pr0 = json.loads((d / "paired_vs_r0.json").read_text(encoding="utf-8"))
        off = m["official"]
        diag = m["diagnostics"]
        rows.append({
            "arm": arm,
            "win_rate": off["win_rate"],
            "full_seq": off["full_sequence_accuracy"],
            "f1_param": off["f1_param"],
            "executable_rate": diag.get("executable_rate"),
            "under_calling": diag.get("under_calling_rate"),
            "vs_c0_gained": pc0.get("n_gained"),
            "vs_c0_regressed": pc0.get("n_regressed"),
            "vs_c0_delta_mean": pc0.get("win_delta_mean"),
            "vs_c0_mcnemar_p": (pc0.get("mcnemar") or {}).get("p_value"),
            "vs_r0_gained": pr0.get("n_gained"),
            "vs_r0_regressed": pr0.get("n_regressed"),
            "vs_r0_delta_mean": pr0.get("win_delta_mean"),
            "vs_r0_mcnemar_p": (pr0.get("mcnemar") or {}).get("p_value"),
        })

    SUM.REPORTS_DIR = OUT  # type: ignore[attr-defined]
    summary = SUM.round_summary(1, arm_dirs, c0_dir=c0, a0_dir=a0)

    td_path = OUT / "training_diagnostics.json"
    td_path.write_text(json.dumps(build_training_diagnostics(), indent=2), encoding="utf-8")
    SEL.REPORTS_DIR = OUT  # type: ignore[attr-defined]
    SEL.main = lambda: None  # noqa — we call pieces manually
    summary_path = OUT / "ROUND1_SUMMARY.json"
    training_diag_by_arm = json.loads(td_path.read_text(encoding="utf-8"))
    control_entry = summary["arms"].get("A0_R0_CURRENT", {})
    gate_results = {}
    for arm, entry in summary["arms"].items():
        gate_results[arm] = SEL.evaluate_gates(
            arm, entry, control_entry, training_diag_by_arm.get(arm, {})
        )
    ranked = SEL.rank_arms(gate_results, summary, training_diag_by_arm)
    decision = {"round": 1, "gate_results": gate_results, "lexicographic_ranking": ranked}
    (OUT / "ROUND1_DECISION.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plan = SEL.build_round2_plan(ranked, gate_results)
    (OUT / "ROUND2_PLAN.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_lines = [
        "# Round 1 — corrected comparison (true shared C0 baseline)",
        "",
        f"C0 eval: `{c0}`",
        f"A0 (R0-trained) eval: `{a0}`",
        "",
        "| Arm | win_rate | vs C0 Δmean | gained | regressed | McNemar p | vs R0 Δmean | gained | regressed | exec_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['arm']} | {r['win_rate']:.3f} | {r['vs_c0_delta_mean']} | "
            f"{r['vs_c0_gained']} | {r['vs_c0_regressed']} | {r['vs_c0_mcnemar_p']} | "
            f"{r.get('vs_r0_delta_mean', '—')} | {r.get('vs_r0_gained', '—')} | "
            f"{r.get('vs_r0_regressed', '—')} | {r['executable_rate']} |"
        )
    md_lines += ["", "## Gate verdicts (corrected)", ""]
    for arm, g in gate_results.items():
        md_lines.append(f"- **{arm}**: {g['verdict']} — {'; '.join(g['reasons']) or 'ok'}")
    md_lines += ["", "## Lexicographic ranking (excl. A1 scientific control)", ""]
    for i, arm in enumerate(ranked, 1):
        md_lines.append(f"{i}. {arm}")
    md_lines += ["", "## Round 2 plan (if you proceed)", "", f"Arms: {plan['arms']}", f"Seed: {plan['seed']}"]
    (OUT / "ROUND1_CORRECTED_REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps({"rows": rows, "ranking": ranked, "top2": plan["top2_candidates"]}, indent=2))
    print(f"Wrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
