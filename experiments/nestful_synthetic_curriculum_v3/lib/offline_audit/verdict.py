from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _dispatch_mismatches(ctx: Dict[str, Any]) -> List[str]:
    arms = (ctx.get("discovery") or {}).get("arms") or {}
    bad = []
    for arm, entry in arms.items():
        rdch = entry.get("reward_dispatch") or {}
        if rdch.get("dispatch_ok") is False:
            bad.append(f"{arm}: declared {rdch.get('expected_policy')!r}, "
                       f"train log ran {rdch.get('logged_policies')}")
    return bad


def _rewards_identical_on_hash_matched(ctx: Dict[str, Any]) -> bool:
    """True when hash-matched completions received IDENTICAL rewards in every
    compared arm pair (reward_pearson==1 to numerical precision) — the raw
    signature of a single shared reward function, NOT of 'distinct rewards'."""
    pairs = (ctx.get("pairwise") or {}).get("pairs") or []
    checked = [p for p in pairs if p.get("n_hash_matched_rollouts")]
    if not checked:
        return False
    return all((p.get("reward_pearson_hash_matched") or 0) > 0.999999
               and (p.get("reward_spearman_hash_matched") or 0) > 0.999999
               for p in checked)


def decide_verdict(ctx: Dict[str, Any]) -> Dict[str, Any]:
    pairwise = ctx.get("pairwise") or {}
    pairs = {f"{p['arm_a']}|{p['arm_b']}": p for p in pairwise.get("pairs", [])}
    a0_a4 = pairs.get("A0_R0_CURRENT|A4_GATED_VERIFIABLE") or pairs.get(
        "A4_GATED_VERIFIABLE|A0_R0_CURRENT"
    )
    on_policy = ctx.get("on_policy") or {}
    optimizer = ctx.get("optimizer") or {}
    progress = ctx.get("progress") or {}
    adapters = ctx.get("adapters") or {}
    counter_status = (ctx.get("pairwise") or {}).get("status", {})

    verdict = "INSUFFICIENT_LOG_COVERAGE"
    reasons: List[str] = []
    recommendation = "Log full trajectories for counterfactual registry re-scoring."
    do_not = ["New Round-1 GRPO re-runs without trajectory logging"]

    # ── Highest priority: was the declared reward actually dispatched? ──────
    # Round-1 (2026-07-24) regression: every arm trained with
    # execution_aware_v3_2_dense; all downstream cross-arm comparisons were
    # comparisons of sampling noise. Detect and short-circuit.
    mismatches = _dispatch_mismatches(ctx)
    if mismatches:
        return {
            "verdict": "REWARD_DISPATCH_BUG",
            "reasons": ["Declared reward policy did not run:"] + mismatches,
            "recommended_next_experiment": (
                "Fix reward dispatch (config policy must win over REWARD_POLICY "
                "env default), then re-run the ablation arms."),
            "do_not_run_now": ["Any cross-arm reward conclusion from this round"],
            "a0_a4_pairwise": a0_a4,
            "rewards_identical_on_hash_matched": _rewards_identical_on_hash_matched(ctx),
        }

    equiv_count = sum(1 for p in pairs.values() if p.get("diagnostic_effectively_equivalent"))
    if counter_status.get("mode") == "PARTIAL" and a0_a4:
        cos = a0_a4.get("advantage_cosine_hash_matched")
        if cos is not None and cos >= 0.95 and a0_a4.get("n_hash_matched_rollouts", 0) > 0:
            verdict = "REWARD_SIGNALS_EFFECTIVELY_EQUIVALENT"
            reasons.append("Hash-matched A0/A4 advantages nearly identical")
            recommendation = "Pivot to credit-assignment or data-mix experiments; skip reward band tuning."
        elif any("WEAK_UPDATE_SIGNAL" in (optimizer.get(a, {}).get("flags") or "") for a in optimizer):
            verdict = "UPDATE_TOO_WEAK"
            reasons.append("Sparse/dead groups and weak KL/update flags")
            recommendation = "Short isolated LR/KL/update-strength ablation."
        elif any(
            progress.get(a, {}).get("reward_proxy_warning") for a in progress.get("per_arm", {})
        ):
            verdict = "REWARD_PROXY_OPTIMIZATION"
            reasons.append("Reward slope up while terminal success flat/down")
            recommendation = "Terminal constraint / outcome redesign."
        else:
            best_arm = max(on_policy.items(), key=lambda kv: kv[1].get("synthetic_terminal_success_rate") or 0)[0]
            reasons.append(f"Best on-policy synthetic success: {best_arm}")
            verdict = "SYNTHETIC_LEARNING_WITH_TRANSFER_GAP"
            recommendation = "Held-out 166 inference + Stage2/3 mix analysis."
            do_not.append("Declaring NESTFUL winner from train logs alone")

    credit = ctx.get("credit") or {}
    metrics = credit.get("metrics") or {}
    if any((m.get("local_good_negative_advantage_rate") or 0) > 0.15 for m in metrics.values()):
        if verdict not in ("REWARD_SIGNALS_EFFECTIVELY_EQUIVALENT",):
            verdict = "CREDIT_ASSIGNMENT_SUSPECT"
            reasons.append("Elevated good-turn negative advantage rate")
            recommendation = "Credit-assignment ablation."

    return {
        "verdict": verdict,
        "reasons": reasons,
        "recommended_next_experiment": recommendation,
        "do_not_run_now": do_not,
        "a0_a4_pairwise": a0_a4,
        "rewards_identical_on_hash_matched": _rewards_identical_on_hash_matched(ctx),
    }


def summarize_reports(reports_dir: Path, ctx: Dict[str, Any]) -> None:
    verdict = decide_verdict(ctx)
    report = {
        "discovery": ctx.get("discovery"),
        "on_policy": ctx.get("on_policy"),
        "pairwise": ctx.get("pairwise"),
        "optimizer": ctx.get("optimizer"),
        "eval_behavior": ctx.get("eval_behavior"),
        "adapters": ctx.get("adapters"),
        "verdict": verdict,
    }
    (reports_dir / "OFFLINE_AUDIT_REPORT.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    op = ctx.get("on_policy") or {}
    if verdict.get("rewards_identical_on_hash_matched"):
        reward_line = ("1. Hash-matched completions received IDENTICAL rewards in every "
                       "arm pair (reward_pearson=1.0) — arms shared ONE reward function; "
                       "see PAIRWISE_SIGNAL_SIMILARITY.md and the dispatch check in "
                       "DISCOVERY.md")
    else:
        reward_line = ("1. Rewards on hash-matched completions differ across arms: "
                       "see PAIRWISE_SIGNAL_SIMILARITY.md")
    exec_lines = [
        "# Executive summary",
        "",
        reward_line,
        f"2. Best synthetic on-policy success: {max(op.items(), key=lambda kv: kv[1].get('synthetic_terminal_success_rate') or 0)[0] if op else 'n/a'}",
        f"3. Proxy warning arms: {[a for a,d in (ctx.get('progress') or {}).get('per_arm', {}).items() if d.get('reward_proxy_warning')]}",
        f"4. Update strength: see OPTIMIZER_SIGNAL_AUDIT.md",
        f"5. A0≈A4 heuristic: {verdict.get('a0_a4_pairwise', {}).get('diagnostic_effectively_equivalent')}",
        f"6. Main suspicion: **{verdict['verdict']}**",
        f"7. Next experiment: {verdict['recommended_next_experiment']}",
        f"8. Do not run: {', '.join(verdict['do_not_run_now'])}",
    ]
    (reports_dir / "EXECUTIVE_SUMMARY.md").write_text("\n".join(exec_lines), encoding="utf-8")
    (reports_dir / "NEXT_EXPERIMENT_DECISION.md").write_text(
        "\n".join(
            [
                "# Next experiment decision",
                "",
                f"**Verdict:** `{verdict['verdict']}`",
                "",
                "## Reasons",
                *[f"- {r}" for r in verdict["reasons"]],
                "",
                f"**Recommended:** {verdict['recommended_next_experiment']}",
                "",
                "## Do not run now",
                *[f"- {x}" for x in verdict["do_not_run_now"]],
            ]
        ),
        encoding="utf-8",
    )
    md = [
        "# Offline audit report",
        "",
        f"Diagnostic verdict: **{verdict['verdict']}**",
        "",
        "See EXECUTIVE_SUMMARY.md and section CSV/MD files in this directory.",
    ]
    (reports_dir / "OFFLINE_AUDIT_REPORT.md").write_text("\n".join(md), encoding="utf-8")
