"""Final Pilot3 forensic markdown/json report assembly."""
from __future__ import annotations

from typing import Any, Dict, List


def build_final_report(ctx: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    h = ctx.get("headline") or {}
    integrity = ctx.get("integrity_checks") or {}
    reward = ctx.get("reward") or {}
    bottlenecks = ctx.get("bottlenecks") or []
    recs = ctx.get("generation_cells") or []
    topo = ctx.get("topology") or {}
    subset = ctx.get("subset") or {}
    surface = ctx.get("surface") or {}
    cov = ctx.get("coverage") or {}
    dist = ctx.get("distribution") or {}
    quality = ctx.get("quality") or {}
    divergence = ctx.get("divergence_summary") or {}
    gained_patterns = ctx.get("gained_lost_patterns") or []

    facts = [
        f"Matched C0-vLLM wins {h.get('wins_c0')}/{h.get('n')} ({100*float(h.get('win_rate_c0') or 0):.1f}%).",
        f"Matched D1-vLLM wins {h.get('wins_d1')}/{h.get('n')} ({100*float(h.get('win_rate_d1') or 0):.1f}%).",
        f"Paired flips: loss→win={h.get('loss_to_win')}, win→loss={h.get('win_to_loss')}.",
        f"McNemar exact p={((h.get('mcnemar') or {}).get('exact_p'))}.",
        f"Paired bootstrap 95% CI pp={[(h.get('bootstrap_paired') or {}).get('ci95_lo_pp'), (h.get('bootstrap_paired') or {}).get('ci95_hi_pp')]}",
        f"Train-log dead_group_rate={(reward.get('train_log_aggregates') or {}).get('aggregates', {}).get('dead_group_rate')}",
        "diagnostic-500 is balanced by call-count buckets (100 each for 2/3/4/5/6+).",
        f"D1 subset identity vs local full train: status={subset.get('subset_identity_status')} "
        f"overlap={subset.get('n_overlap_subset_ids_with_local_full_train')}/{subset.get('n_subset')}.",
    ]
    interpretations = [
        "Matched-engine point estimate is small (+~2pp) and not statistically conclusive at conventional thresholds.",
        "Surface/schema mismatch is a candidate bottleneck because exact tool namespace overlap with diagnostic is low.",
        "Reward degeneracy is a candidate bottleneck given aggregate dead_group_rate≈0.5, but all-success vs all-fail split is not identifiable without rollouts.",
        "First-300 subset selection may be a data-selection bottleneck if cell distributions diverge from rest-300.",
    ]
    hypotheses = [
        "D1 gains concentrate on coverage-friendly tasks while regressions are noise or path confound.",
        "Increasing topology novelty without improving distractor realism will not transfer.",
        "Paired A/G renderers will improve surface-invariant dependency skill.",
    ]
    not_id = [
        "Causal training effect magnitude after removing all inference-path confounds.",
        "Per-cell effective GRPO group rates for Pilot3 D1.",
        "Whether more data alone would help without composition changes.",
        "Semantic equivalence of proxy-mapped tools.",
        "Natural NESTFUL official win rate (diagnostic is not a natural sample).",
    ]

    md: List[str] = []
    md += [
        "# PILOT3_FORENSIC_ANALYSIS",
        "",
        "## 1. Executive summary",
        "",
        "### VERIFIED FACTS",
        "",
    ]
    for x in facts:
        md.append(f"- {x}")
    md += ["", "### SUPPORTED INTERPRETATIONS", ""]
    for x in interpretations:
        md.append(f"- {x}")
    md += ["", "### OPEN HYPOTHESES", ""]
    for x in hypotheses:
        md.append(f"- {x}")
    md += ["", "### NOT IDENTIFIABLE", ""]
    for x in not_id:
        md.append(f"- {x}")

    md += [
        "",
        "## 2. Scope and non-goals",
        "",
        "- Offline analysis only: no training, no model inference, no new NESTFUL eval, no new synthetic generation.",
        "- Main contrast is C0-vLLM vs D1-vLLM on diagnostic-500.",
        "- C0-HF is reported separately when present and is not mixed into the training-effect contrast.",
        "- Goal: concrete Targeted Tool Data Factory changes, not a generic dashboard.",
        "",
        "## 3. Input artifacts and integrity",
        "",
        f"- Pairing status: `{(integrity.get('pairing') or {}).get('status')}`",
        f"- Pairing OK: `{(integrity.get('pairing') or {}).get('pairing_ok')}`",
        f"- Eval manifest parity: `{(integrity.get('eval_manifest_parity') or {}).get('status')}`",
        f"- Rollouts: `{(integrity.get('rollouts') or {}).get('status')}`",
        "- See `INPUT_MANIFEST.json` and `INPUT_INTEGRITY.md`.",
        "",
        "## 4. Reproduction of matched C0 vs D1 result",
        "",
        f"- Δ = **{h.get('delta_pp'):+.2f} pp** (matched-engine point estimate).",
        f"- Stratified bootstrap CI pp: `{[(h.get('bootstrap_stratified_by_call_bucket') or {}).get('ci95_lo_pp'), (h.get('bootstrap_stratified_by_call_bucket') or {}).get('ci95_hi_pp')]}`",
        "- Overall diagnostic win is a macro average across call-count buckets, not a natural NESTFUL estimate.",
        "- Residual LoRA inference-path confound cannot be removed from these two trajectory sets alone.",
        "",
        "## 5. What changed in trajectories",
        "",
        f"- Divergence category counts: `{(divergence.get('category_counts') or {})}`",
        f"- Mean first divergent turn (where defined): `{(divergence.get('mean_first_divergent_turn') or None)}`",
        "- Details: `TRAJECTORY_PAIR_FEATURES.csv`, `TRAJECTORY_DIVERGENCE_SUMMARY.md`.",
        "",
        "## 6. Gained vs lost task analysis",
        "",
        f"- n_gained={h.get('loss_to_win')}, n_lost={h.get('win_to_loss')} (small n; avoid overclaiming significance).",
        f"- Top patterns: `{gained_patterns[:8]}`",
        "- See `GAINED_LOST_AUDIT.md` and representative example markdowns.",
        "",
        "## 7. Failure taxonomy",
        "",
        "- Absolute counts in `FAILURE_ANALYSIS.md` / `FAILURE_TAXONOMY_PER_TASK.csv`.",
        "- Primary category uses documented priority; secondary flags may co-occur.",
        "",
        "## 8. Reward-signal observability and dead groups",
        "",
        f"- Aggregates: `{(reward.get('train_log_aggregates') or {}).get('aggregates')}`",
        f"- Per-rollout available: `{reward.get('per_rollout_available')}`",
        "- If per-rollout missing, dead-group composition and cell-level reward health are NOT IDENTIFIABLE.",
        "- See `REWARD_AUDIT.md` and `MISSING_OBSERVABILITY.md`.",
        "",
        "## 9. Train subset representativeness",
        "",
        f"- Shuffle interpretation: `{subset.get('shuffle_interpretation')}`",
        f"- Missing generation cells in first300: `{subset.get('n_missing_cells_in_first300')}`",
        f"- generation_cell JSD: `{((subset.get('categorical') or {}).get('generation_cell') or {}).get('jsd')}`",
        "- See `TRAIN_SUBSET_SELECTION_AUDIT.md`.",
        "",
        "## 10. Topology diversity and coverage",
        "",
        f"- Train300 summary: `{topo.get('train300')}`",
        f"- Coverage vs diagnostic: `{topo.get('coverage_train300_vs_diag')}`",
        "- Topology hash is call-order-indexed shape hash (see methodology limitations).",
        "",
        "## 11. Surface, schema and reference mismatch",
        "",
        f"- Namespace overlap: `{surface.get('namespace')}`",
        f"- Distractor hardness: `{surface.get('distractor')}`",
        "- Overlaps are lexical/schema proxies, not semantic equivalence.",
        "",
        "## 12. Registry coverage",
        "",
        f"- By-outcome preview: `{(cov.get('by_outcome') or [])[:6]}`",
        "- Proxy mappings labeled EXACT/HIGH_PROXY/MEDIUM_PROXY/LOW_PROXY/UNMAPPED.",
        "",
        "## 13. Joint-distribution and OOD analysis",
        "",
        f"- Summary: `{dist.get('summary')}`",
        "- Distance is transparent Gower-like mixed distance; models are associative only.",
        "",
        "## 14. Data quality and shortcut risks",
        "",
        f"- Quality summary: `{quality}`",
        "",
        "## 15. Ranked bottlenecks",
        "",
    ]
    for b in bottlenecks[:12]:
        md.append(
            f"- **#{b.get('rank')} `{b.get('category')}`** — {b.get('status')} / {b.get('evidence_strength')}: {b.get('argument_for')}"
        )

    md += ["", "## 16. Concrete changes to data generation", ""]
    for r in recs:
        if r.get("priority") in ("P0", "P1"):
            md.append(
                f"- `{r['recommendation_id']}` [{r['mode']}/{r['priority']}]: {r.get('topology_requirement') or r.get('target_skill')} "
                f"— {r.get('recommended_relative_quota')}"
            )

    md += [
        "",
        "## 17. Proposed generation cells",
        "",
        "- Machine-readable: `RECOMMENDED_GENERATION_CELLS.json` / `.csv`.",
        "- PROFILE_SAFE vs DIAGNOSTIC_INFORMED_EXPLORATORY are separated; the latter carries an explicit contamination disclaimer.",
        "",
        "## 18. What can be concluded",
        "",
        "- Matched-engine D1−C0 delta is small and not statistically conclusive.",
        "- Multiple measurable factory mismatches exist (surface, joint cells, possible subset bias, reward aggregates).",
        "- Next data generation should prioritize composition/constraints over naive scale-up.",
        "",
        "## 19. What cannot be concluded",
        "",
    ]
    for x in not_id:
        md.append(f"- {x}")

    md += [
        "",
        "## 20. Missing artifacts and future logging",
        "",
        "- Pilot3 per-rollout reward groups / canary rollouts for D1.",
        "- Multi-seed D1 runs.",
        "- Null-LoRA matched inference control.",
        "- See `MISSING_OBSERVABILITY.md`.",
        "",
    ]

    payload = {
        "schema_version": "pilot3_forensics.final.v1",
        "verified_facts": facts,
        "supported_interpretations": interpretations,
        "open_hypotheses": hypotheses,
        "not_identifiable": not_id,
        "headline": h,
        "bottlenecks": bottlenecks,
        "generation_cell_recommendations": recs,
        "topology": topo,
        "subset": subset,
        "surface": surface,
        "coverage": cov,
        "distribution": dist,
        "reward": {
            "aggregates": (reward.get("train_log_aggregates") or {}).get("aggregates"),
            "per_rollout_available": reward.get("per_rollout_available"),
        },
        "quality": quality,
    }
    return "\n".join(md) + "\n", payload
