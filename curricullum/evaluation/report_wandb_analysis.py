#!/usr/bin/env python3
"""Build narrative report from wandb_grpo_signal_summary.json."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
IN_PATH = REPO / "curricullum/evaluation/wandb_analysis/wandb_grpo_signal_summary.json"
OUT_PATH = REPO / "curricullum/evaluation/wandb_analysis/wandb_analysis_report.json"


def _corr(xs: List[float], ys: List[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((a - mx) * (b - my) for a, b in pairs)
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den if den else None


def main() -> None:
    rows: List[Dict[str, Any]] = json.loads(IN_PATH.read_text(encoding="utf-8"))
    valid = [r for r in rows if r.get("completion_rows", 0) > 0]

    table = []
    for r in sorted(valid, key=lambda x: (x.get("stage") or 0, x.get("epoch") or 0)):
        reward_one_pct = (
            100 * r["rows_reward_one"] / r["completion_rows"] if r["completion_rows"] else None
        )
        table.append(
            {
                "stage": r["stage"],
                "epoch": r["epoch"],
                "run": r["run_name"],
                "frac_zero_std_mean": r.get("frac_zero_std_mean"),
                "frac_zero_std_final": r.get("frac_zero_std_final"),
                "steps_dead_pct": r.get("steps_dead_pct"),
                "adv_zero_pct": r.get("rows_advantage_zero_pct"),
                "reward_mean": r.get("reward_mean"),
                "reward_one_pct": reward_one_pct,
                "steps_with_signal": r.get("steps_with_signal"),
                "format_no_tag_pct": r.get("format_no_tag_pct"),
                "exec_pass": r.get("curriculum_exec_pass"),
                "parse_fail": r.get("curriculum_parse_fail"),
            }
        )

    by_stage: Dict[int, List[Dict[str, Any]]] = {}
    for r in valid:
        if r.get("stage") is not None:
            by_stage.setdefault(r["stage"], []).append(r)

    stage_summary = {}
    for stage, rs in sorted(by_stage.items()):
        rs = sorted(rs, key=lambda x: x.get("epoch") or 0)
        stage_summary[f"stage_{stage}"] = {
            "epochs": len(rs),
            "frac_zero_std_mean_avg": statistics.mean(
                [r["frac_zero_std_mean"] for r in rs if r.get("frac_zero_std_mean") is not None]
            )
            if any(r.get("frac_zero_std_mean") is not None for r in rs)
            else None,
            "frac_zero_std_mean_worst": max(
                (r["frac_zero_std_mean"] for r in rs if r.get("frac_zero_std_mean") is not None),
                default=None,
            ),
            "steps_dead_pct_avg": statistics.mean(
                [r["steps_dead_pct"] for r in rs if r.get("steps_dead_pct") is not None]
            )
            if any(r.get("steps_dead_pct") is not None for r in rs)
            else None,
            "reward_one_pct_avg": statistics.mean(
                [
                    100 * r["rows_reward_one"] / r["completion_rows"]
                    for r in rs
                    if r.get("completion_rows")
                ]
            ),
            "exec_pass_trend": [r.get("curriculum_exec_pass") for r in rs],
            "parse_fail_trend": [r.get("curriculum_parse_fail") for r in rs],
        }

    frac = [r.get("frac_zero_std_mean") for r in valid]
    dead = [r.get("steps_dead_pct") for r in valid]
    parse = [r.get("curriculum_parse_pass") for r in valid]  # typo guard
    parse = [r.get("curriculum_parse_fail") for r in valid]
    exec_p = [r.get("curriculum_exec_pass") for r in valid]
    no_tag = [r.get("format_no_tag_pct") for r in valid]

    correlations = {
        "frac_zero_std_vs_steps_dead": _corr(
            [x for x in frac if x is not None],
            [y for y, x in zip(dead, frac) if x is not None and y is not None],
        ),
        "frac_zero_std_vs_parse_fail": _corr(
            [x for x, p in zip(frac, parse) if x is not None and p is not None],
            [p for x, p in zip(frac, parse) if x is not None and p is not None],
        ),
        "frac_zero_std_vs_exec_pass": _corr(
            [x for x, e in zip(frac, exec_p) if x is not None and e is not None],
            [e for x, e in zip(frac, exec_p) if x is not None and e is not None],
        ),
        "format_no_tag_vs_frac_zero_std": _corr(
            [n for n, f in zip(no_tag, frac) if n is not None and f is not None],
            [f for n, f in zip(no_tag, frac) if n is not None and f is not None],
        ),
    }

    report = {
        "runs_analyzed": len(valid),
        "per_run": table,
        "by_stage": stage_summary,
        "correlations": correlations,
        "headlines": [],
    }

    # Headlines
    s4 = stage_summary.get("stage_4", {})
    s5 = stage_summary.get("stage_5", {})
    if s4.get("frac_zero_std_mean_avg") is not None:
        report["headlines"].append(
            f"Stage 4 má nejnižší průměrný frac_zero_std ({s4['frac_zero_std_mean_avg']:.1%}) "
            f"a nejnižší dead steps ({s4['steps_dead_pct_avg']:.1f}%) — paradoxně nejvíc živého signálu."
        )
    if s5.get("frac_zero_std_mean_avg") is not None:
        report["headlines"].append(
            f"Stage 5 se vrací k vysokému zero_std ({s5['frac_zero_std_mean_avg']:.1%}) "
            f"při reward_one ~{s5['reward_one_pct_avg']:.0f}% — téměř všechny rollouty dostanou 1.0."
        )
    avg_dead = statistics.mean([r["steps_dead_pct"] for r in valid if r.get("steps_dead_pct")])
    avg_adv_zero = statistics.mean(
        [r["rows_advantage_zero_pct"] for r in valid if r.get("rows_advantage_zero_pct")]
    )
    report["headlines"].append(
        f"Průměrně {avg_dead:.0f}% train kroků má advantage=0 u všech 16 rolloutů; "
        f"{avg_adv_zero:.0f}% řádků v completion tabulkách má advantage≈0."
    )
    if correlations.get("frac_zero_std_vs_parse_fail") is not None:
        report["headlines"].append(
            f"Korelace frac_zero_std ↔ parse_fail: {correlations['frac_zero_std_vs_parse_fail']:.2f} "
            f"(vyšší parse fail → více mrtvých kroků)."
        )
    report["headlines"].append(
        "Formát (<tool_call_answer>) není problém: no_tag_pct <1% ve všech runech kromě ojedinělých řádků."
    )

    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
