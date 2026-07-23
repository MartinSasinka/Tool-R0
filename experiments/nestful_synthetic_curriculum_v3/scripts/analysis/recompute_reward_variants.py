#!/usr/bin/env python3
"""Deterministic offline recomputation of reward variants R0–R3 on stored eval trajectories.

Compares:
  R0 = execution_aware_v3_2_dense (current)
  R1 = outcome-only / outcome-first (no process tie-break)
  R2 = R1 + small process epsilon (0.05)
  R3 = wider executable_wrong gap + process epsilon (0.04)

Outputs under reports/pure_stage3_reward_variant_audit/ by default.
No GPU, no LLM, no training changes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
_REPO = _V3.parents[1]
_MINIMAL = _REPO / "experiments/nestful_mtgrpo_minimal"
sys.path.insert(0, str(_MINIMAL))
sys.path.insert(0, str(_V3))
sys.path.append(str(_V3 / "scripts"))

from group_stats import compute_group_stats  # noqa: E402
from grpo_train import _turn_returns  # noqa: E402
from lib.reward_variants_offline import (  # noqa: E402
    DEFAULT_EPS_R2,
    DEFAULT_EPS_R3,
    score_variants,
    variant_to_dict,
)
from motif_lib import default_test_path  # noqa: E402
from scripts.analysis.pure_stage3_diag_utils import (  # noqa: E402
    load_tasks,
    load_traj_rows,
    traj_from_dict,
)
from scripts.analysis.pure_stage3_diagnostic_pack import (  # noqa: E402
    ARM_DIRS,
    DEFAULT_RUN,
    pseudo_group_metrics,
    score_all_variants,
)
from scripts.analysis.two_phase_root_cause_analysis import official_win  # noqa: E402

DEFAULT_OUT = _V3 / "reports/pure_stage3_reward_variant_audit"
GAMMA = 1.0
LAMBDA_EP = 1.0
VARIANTS = ("R0", "R1", "R2", "R3")
LABELS = {
    "R0": "Current (execution_aware_v3_2_dense)",
    "R1": "Outcome-only",
    "R2": "Outcome-first + ε process (0.05)",
    "R3": "Outcome-first + wider executable_wrong gap (0.04 ε)",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _official_win_row(row: dict) -> bool:
    return official_win(row) == 1.0


def _terminal_ok(vn: str, v: dict, official: bool) -> bool:
    if not official:
        return True
    tc = v.get("terminal_class") or ""
    if vn == "R0":
        return tc in ("fully_correct", "too_few_calls") or float(v.get("total_reward") or 0) >= 0.89
    return tc == "official_success"


def global_trajectory_metrics(
    ids: List[str],
    arms: Dict[str, Dict[str, dict]],
    all_variants: Dict[str, Dict[str, dict]],
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for vn in VARIANTS:
        n = 0
        official_wins = 0
        official_win_recognized = 0
        official_loss_high = 0
        too_few_on_official_win = 0
        exec_wrong_high = 0
        for sid in ids:
            for arm in ("C0", "E1", "E2"):
                n += 1
                row = arms[arm][sid]
                v = all_variants[sid][arm][vn]
                ow = _official_win_row(row)
                if ow:
                    official_wins += 1
                    if _terminal_ok(vn, v, True):
                        official_win_recognized += 1
                    if vn == "R0" and v.get("terminal_class") == "too_few_calls":
                        too_few_on_official_win += 1
                if vn == "R0":
                    if v.get("terminal_class") == "executable_wrong_final" and float(v["total_reward"]) >= 0.52:
                        exec_wrong_high += 1
                else:
                    if v.get("terminal_class") == "executable_wrong_outcome" and float(v["total_reward"]) >= 0.45:
                        exec_wrong_high += 1
            c0w = _official_win_row(arms["C0"][sid])
            e2w = _official_win_row(arms["E2"][sid])
            if not c0w and e2w:
                r_c0 = all_variants[sid]["C0"][vn]["total_reward"]
                r_e2 = all_variants[sid]["E2"][vn]["total_reward"]
                if r_e2 > r_c0:
                    official_loss_high += 1
        out[vn] = {
            "n_trajectories": n,
            "official_wins": official_wins,
            "official_win_recognized": official_win_recognized,
            "official_win_recognition_rate": round(
                official_win_recognized / official_wins, 4
            ) if official_wins else None,
            "official_win_labeled_failure": official_wins - official_win_recognized,
            "valid_shorter_path_penalized_too_few": too_few_on_official_win,
            "executable_wrong_high_reward": exec_wrong_high,
            "official_loss_reward_above_win": official_loss_high,
        }
    return out


def advantage_summary(per_task_rows: List[dict]) -> Dict[str, dict]:
    by_variant: Dict[str, List[dict]] = defaultdict(list)
    for row in per_task_rows:
        by_variant[row["variant"]].append(row)

    summary: Dict[str, dict] = {}
    for vn in VARIANTS:
        rows = by_variant[vn]
        n = len(rows)
        dead = sum(1 for r in rows if r.get("dead_group"))
        adv_e2 = [abs(r["advantages_t0"]["E2"]) for r in rows]
        pos_e2 = sum(1 for r in rows if r["advantages_t0"]["E2"] > 1e-9)
        neg_e2 = sum(1 for r in rows if r["advantages_t0"]["E2"] < -1e-9)
        zero_e2 = n - pos_e2 - neg_e2
        summary[vn] = {
            "dead_group_rate": round(dead / n, 4) if n else None,
            "mean_abs_adv_E2": round(sum(adv_e2) / n, 4) if n else None,
            "E2_adv_positive_rate": round(pos_e2 / n, 4) if n else None,
            "E2_adv_negative_rate": round(neg_e2 / n, 4) if n else None,
            "E2_adv_zero_rate": round(zero_e2 / n, 4) if n else None,
        }

    r0_by_task = {r["task_id"]: r for r in by_variant["R0"]}
    for vn in ("R1", "R2", "R3"):
        flips = 0
        for r in by_variant[vn]:
            a0 = r0_by_task[r["task_id"]]["advantages_t0"]["E2"]
            a1 = r["advantages_t0"]["E2"]
            if (a0 >= 0) != (a1 >= 0):
                flips += 1
        summary[vn]["E2_adv_sign_flip_vs_R0"] = flips
    return summary


def c0_win_e2_loss_table(
    ids: List[str],
    arms: Dict[str, Dict[str, dict]],
    all_variants: Dict[str, Dict[str, dict]],
) -> Dict[str, dict]:
    cohort_ids = [
        sid for sid in ids
        if _official_win_row(arms["C0"][sid]) and not _official_win_row(arms["E2"][sid])
    ]
    table: Dict[str, dict] = {}
    for vn in VARIANTS:
        wrong = correct = tie = 0
        for sid in cohort_ids:
            r0 = all_variants[sid]["C0"][vn]["total_reward"]
            r2 = all_variants[sid]["E2"][vn]["total_reward"]
            if r2 > r0:
                wrong += 1
            elif r2 < r0:
                correct += 1
            else:
                tie += 1
        table[vn] = {
            "n_cohort": len(cohort_ids),
            "wrong_order_E2_gt_C0": wrong,
            "correct_order_E2_lt_C0": correct,
            "tie": tie,
            "fixed_vs_R0": None,
        }
    base_wrong = table["R0"]["wrong_order_E2_gt_C0"]
    for vn in VARIANTS:
        table[vn]["fixed_vs_R0"] = base_wrong - table[vn]["wrong_order_E2_gt_C0"]
    return table


def render_report(
    *,
    run_id: str,
    global_metrics: Dict[str, dict],
    pseudo_summary: Dict[str, dict],
    adv_summary: Dict[str, dict],
    c0_table: Dict[str, dict],
    eps_r2: float,
    eps_r3: float,
    n_tasks: int,
) -> str:
    lines = [
        "# Reward variant audit (deterministic, eval trajectories)",
        "",
        f"**Generated:** {_now()}",
        f"**Run:** `{run_id}`",
        f"**Tasks:** {n_tasks} (pseudo-group = C0/E1/E2 per task, n={n_tasks})",
        f"**Policy:** R0=`execution_aware_v3_2_dense`, R2 ε={eps_r2}, R3 ε={eps_r3}",
        "",
        "No LLM. Official Nestful win is authority for outcome variants.",
        "",
        "## 1. Official wins recognized by reward",
        "",
        "| Variant | Official wins | Recognized | Rate | Mislabeled | Too-few on win (R0 only) |",
        "|---------|--------------:|-----------:|-----:|-----------:|-------------------------:|",
    ]
    for vn in VARIANTS:
        g = global_metrics[vn]
        lines.append(
            f"| {vn} | {g['official_wins']} | {g['official_win_recognized']} | "
            f"{g['official_win_recognition_rate']:.1%} | {g['official_win_labeled_failure']} | "
            f"{g.get('valid_shorter_path_penalized_too_few', 0)} |"
        )

    lines += [
        "",
        "## 2. C0-win / E2-loss ordering (reward prefers loser?)",
        "",
        "| Variant | Cohort n | Wrong (E2>C0) | Correct | Tie | Fixed vs R0 |",
        "|---------|--------:|--------------:|--------:|----:|------------:|",
    ]
    for vn in VARIANTS:
        t = c0_table[vn]
        lines.append(
            f"| {vn} | {t['n_cohort']} | {t['wrong_order_E2_gt_C0']} | "
            f"{t['correct_order_E2_lt_C0']} | {t['tie']} | {t['fixed_vs_R0']} |"
        )

    lines += [
        "",
        "## 3. Pseudo-group dead rate & advantages (C0/E1/E2)",
        "",
        "| Variant | Dead group rate | Δ vs R0 | Mean |adv| E2 | E2 adv sign flip vs R0 |",
        "|---------|----------------:|--------:|---------------:|------------------------:|",
    ]
    r0_dead = adv_summary["R0"]["dead_group_rate"]
    for vn in VARIANTS:
        a = adv_summary[vn]
        delta = round(a["dead_group_rate"] - r0_dead, 4) if a["dead_group_rate"] is not None else None
        flip = a.get("E2_adv_sign_flip_vs_R0")
        flip_s = str(flip) if flip is not None else "—"
        lines.append(
            f"| {vn} | {a['dead_group_rate']:.1%} | {delta:+.1%} | "
            f"{a['mean_abs_adv_E2']:.3f} | {flip_s} |"
        )

    lines += [
        "",
        "## 4. Executable-wrong high reward (trajectory-level)",
        "",
        "| Variant | Count (reward too high for wrong executable outcome) |",
        "|---------|-----------------------------------------------------:|",
    ]
    for vn in VARIANTS:
        lines.append(
            f"| {vn} | {global_metrics[vn]['executable_wrong_high_reward']} |"
        )

    lines += [
        "",
        "## Interpretation (deterministic only)",
        "",
        "- **R1** fixes C0/E2 ordering and official-win labeling but **increases dead groups** sharply "
        "(outcome bands collapse many pseudo-groups).",
        "- **R2/R3** keep dead-group rate near R0 while fixing C0-win/E2-loss wrong ordering.",
        "- **R3** additionally separates executable-wrong from official-success (see section 4).",
        "- Advantage sign flips vs R0 measure how much GRPO credit would shift on E2 under each variant.",
        "",
        "## Variant definitions",
        "",
    ]
    for vn, label in LABELS.items():
        lines.append(f"- **{vn}**: {label}")

    return "\n".join(lines)


def run_audit(run_dir: Path, out_dir: Path) -> dict:
    os.environ["TRAIN_STAGE"] = "3"
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(default_test_path())
    arms = {a: load_traj_rows(run_dir / "eval" / d) for a, d in ARM_DIRS.items()}
    ids = sorted(set.intersection(*(set(v) for v in arms.values())))
    print(f"[audit] scoring {len(ids)} tasks × 3 arms × 4 variants…")
    all_variants, _ = score_all_variants(ids, arms, tasks)

    print("[audit] pseudo-group metrics…")
    per_task_rows, pseudo_summary = pseudo_group_metrics(ids, all_variants, arms)
    global_metrics = global_trajectory_metrics(ids, arms, all_variants)
    adv_summary = advantage_summary(per_task_rows)
    c0_table = c0_win_e2_loss_table(ids, arms, all_variants)

    payload = {
        "generated_at": _now(),
        "run_id": run_dir.name,
        "n_tasks": len(ids),
        "epsilon_R2": DEFAULT_EPS_R2,
        "epsilon_R3": DEFAULT_EPS_R3,
        "variants": LABELS,
        "global_trajectory_metrics": global_metrics,
        "pseudo_group_summary": pseudo_summary,
        "advantage_summary": adv_summary,
        "c0_win_e2_loss_ordering": c0_table,
        "delta_vs_R0": {
            vn: {
                "dead_group_rate_delta": round(
                    adv_summary[vn]["dead_group_rate"] - adv_summary["R0"]["dead_group_rate"], 4
                ),
                "c0_e2_wrong_order_fixed": c0_table[vn]["fixed_vs_R0"],
                "official_win_mislabel_delta": (
                    global_metrics[vn]["official_win_labeled_failure"]
                    - global_metrics["R0"]["official_win_labeled_failure"]
                ),
            }
            for vn in VARIANTS
        },
    }

    _write_json(out_dir / "reward_variant_audit.json", payload)
    _write_jsonl(out_dir / "reward_variant_per_task.jsonl", per_task_rows)
    report = render_report(
        run_id=run_dir.name,
        global_metrics=global_metrics,
        pseudo_summary=pseudo_summary,
        adv_summary=adv_summary,
        c0_table=c0_table,
        eps_r2=DEFAULT_EPS_R2,
        eps_r3=DEFAULT_EPS_R3,
        n_tasks=len(ids),
    )
    (out_dir / "REWARD_VARIANT_AUDIT.md").write_text(report, encoding="utf-8")
    print(f"[audit] wrote {out_dir / 'REWARD_VARIANT_AUDIT.md'}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    run_audit(args.run_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
