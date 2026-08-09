"""Evidence-based data-factory recommendations (no generation)."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence


def rank_bottlenecks(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    headline = ctx.get("headline") or {}
    reward = ctx.get("reward") or {}
    topo = ctx.get("topology") or {}
    subset = ctx.get("subset") or {}
    surface = ctx.get("surface") or {}
    coverage = ctx.get("coverage") or {}
    dist = ctx.get("distribution") or {}
    quality = ctx.get("quality") or {}

    delta = float(headline.get("delta_pp") or 0.0)
    ci = headline.get("bootstrap_paired") or {}
    ci_lo = float(ci.get("ci95_lo_pp") or 0.0)
    ci_hi = float(ci.get("ci95_hi_pp") or 0.0)
    p_mc = float((headline.get("mcnemar") or {}).get("exact_p") or 1.0)
    dead = (reward.get("train_log_aggregates") or {}).get("aggregates", {}).get("dead_group_rate")
    per_rollout = bool(reward.get("per_rollout_available"))
    top1 = float((topo.get("train300") or {}).get("top1_share") or 0.0)
    unseen_topo = float((topo.get("coverage_train300_vs_diag") or {}).get("diagnostic_unseen_topology_rate") or 0.0)
    subset_jsd = float(((subset.get("categorical") or {}).get("generation_cell") or {}).get("jsd") or 0.0)
    exact_overlap = float((surface.get("namespace") or {}).get("exact_overlap_rate_vs_diag") or 0.0)
    unseen_joint = float((dist.get("summary") or {}).get("unseen_combination_rate") or 0.0)

    items = []

    items.append({
        "category": "EVAL_PROTOCOL",
        "status": "SUPPORTED",
        "evidence_strength": "HIGH",
        "metrics": {
            "c0_hf_vs_c0_vllm_if_present": (headline.get("three_arm") or {}),
            "matched_delta_pp": delta,
            "bootstrap_ci": [ci_lo, ci_hi],
        },
        "argument_for": "Original +8.8pp used mismatched HF vs vLLM backends; matched contrast is smaller.",
        "argument_against": "Matched C0/D1 still share residual LoRA/vLLM path differences.",
        "alternative_explanations": ["True small training effect masked by power."],
        "fixable_by_data_only": False,
        "next_step": "Keep matched vLLM eval as the only training-effect contrast; freeze eval script hash.",
        "requires_new_train_or_eval": True,
        "rank_score": 95,
    })

    items.append({
        "category": "LORA_INFERENCE_PATH",
        "status": "NOT_IDENTIFIABLE",
        "evidence_strength": "LOW",
        "metrics": {},
        "argument_for": "D1 uses adapter; C0 is base. Path differences can shift decoding even at T=0.",
        "argument_against": "Cannot quantify from stored trajectories alone.",
        "alternative_explanations": ["Genuine policy change from GRPO."],
        "fixable_by_data_only": False,
        "next_step": "Future A/B: base+null-LoRA vs D1 under identical vLLM loader.",
        "requires_new_train_or_eval": True,
        "rank_score": 80,
    })

    reward_status = "PARTIALLY_SUPPORTED" if dead is not None else "NOT_IDENTIFIABLE"
    items.append({
        "category": "REWARD_SIGNAL",
        "status": reward_status,
        "evidence_strength": "MEDIUM" if dead is not None else "LOW",
        "metrics": {"dead_group_rate": dead, "per_rollout_available": per_rollout, **((reward.get("train_log_aggregates") or {}).get("aggregates") or {})},
        "argument_for": f"Aggregate dead_group_rate={dead} and low unique rewards imply many non-informative GRPO groups." if dead is not None else "No aggregate.",
        "argument_against": "Without per-rollout groups cannot separate all-success vs all-fail cells.",
        "alternative_explanations": ["Data too easy", "epsilon process too weak", "gating collapses distinctions"],
        "fixable_by_data_only": True,
        "next_step": "Add difficulty-targeted cells + persist per-rollout rewards; balance effective-group rate.",
        "requires_new_train_or_eval": True,
        "rank_score": 88 if (dead is not None and float(dead) >= 0.4) else 60,
    })

    identity = str(subset.get("subset_identity_status") or "")
    overlap = subset.get("n_overlap_subset_ids_with_local_full_train")
    items.append({
        "category": "TRAIN_SUBSET_SELECTION",
        "status": (
            "SUPPORTED" if identity == "INCONSISTENT"
            else ("PARTIALLY_SUPPORTED" if subset else "NOT_IDENTIFIABLE")
        ),
        "evidence_strength": (
            "HIGH" if identity == "INCONSISTENT"
            else ("MEDIUM" if subset_jsd >= 0.05 else "LOW")
        ),
        "metrics": {
            "generation_cell_jsd": subset_jsd,
            "shuffle_interpretation": subset.get("shuffle_interpretation"),
            "n_missing_cells": subset.get("n_missing_cells_in_first300"),
            "subset_identity_status": identity,
            "n_overlap_with_local_full_train": overlap,
        },
        "argument_for": (
            "D1 train_subset_300 is not identical to the local train_grpo_pilot3 prefix/freeze; "
            "selection provenance is broken or the export was regenerated."
            if identity == "INCONSISTENT"
            else "First-300 may be contiguous cell blocks / non-representative vs rest-300."
        ),
        "argument_against": "Even a balanced 300 may still yield small transfer.",
        "alternative_explanations": ["n=300 underpowered regardless of balance.", "Artifact sync drift after RunPod."],
        "fixable_by_data_only": True,
        "next_step": "Freeze SHA256 of exact D1 subset; nested stratified subset by cell×call_bucket×motif×track; never silent file-prefix slices.",
        "requires_new_train_or_eval": True,
        "rank_score": 92 if identity == "INCONSISTENT" else (84 if subset_jsd >= 0.05 or (subset.get("n_missing_cells_in_first300") or 0) > 0 else 55),
    })

    items.append({
        "category": "TOPOLOGY_DIVERSITY",
        "status": "PARTIALLY_SUPPORTED" if top1 or unseen_topo else "NOT_IDENTIFIABLE",
        "evidence_strength": "MEDIUM" if (top1 >= 0.15 or unseen_topo >= 0.5) else "LOW",
        "metrics": {"train300_top1_share": top1, "diag_unseen_topology_rate": unseen_topo},
        "argument_for": "High top-1 topology share and/or high diagnostic unseen-topology rate indicate shape mismatch risk.",
        "argument_against": "Topology coverage may not associate with gained/lost if surface mismatch dominates.",
        "alternative_explanations": ["Surface/schema mismatch", "reward degeneracy"],
        "fixable_by_data_only": True,
        "next_step": "Cap top-1 topology family share; raise unique topology quota; joint topology×call_count matching.",
        "requires_new_train_or_eval": False,
        "rank_score": 78 if (top1 >= 0.15 or unseen_topo >= 0.5) else 50,
    })

    items.append({
        "category": "JOINT_DISTRIBUTION_MISMATCH",
        "status": "PARTIALLY_SUPPORTED" if unseen_joint else "NOT_IDENTIFIABLE",
        "evidence_strength": "MEDIUM" if unseen_joint >= 0.3 else "LOW",
        "metrics": {"unseen_combination_rate": unseen_joint},
        "argument_for": "Margin match can hide unseen joint cells (topology×calls×answer×track).",
        "argument_against": "Nearest-neighbor OOD may not predict flips at n=27/16.",
        "alternative_explanations": ["Call-count stratification already dominates."],
        "fixable_by_data_only": True,
        "next_step": "Selection objective: joint deficit matching + rare-cell floors.",
        "requires_new_train_or_eval": False,
        "rank_score": 76 if unseen_joint >= 0.3 else 48,
    })

    items.append({
        "category": "SURFACE_SCHEMA_MISMATCH",
        "status": "PARTIALLY_SUPPORTED",
        "evidence_strength": "HIGH" if exact_overlap < 0.2 else "MEDIUM",
        "metrics": surface.get("namespace") or {},
        "argument_for": "Diagnostic gold tools largely outside factory exact namespace; transfer relies on schema/lexical proxies.",
        "argument_against": "Factory intentionally uses synthetic surfaces; some transfer still observed.",
        "alternative_explanations": ["Registry semantic gap", "distractor realism"],
        "fixable_by_data_only": True,
        "next_step": "Paired A-native/G-general renderers; NESTFUL-like output keys; harder schema-compatible distractors.",
        "requires_new_train_or_eval": False,
        "rank_score": 90 if exact_overlap < 0.2 else 70,
    })

    items.append({
        "category": "REFERENCE_SYNTAX_MISMATCH",
        "status": "PARTIALLY_SUPPORTED",
        "evidence_strength": "MEDIUM",
        "metrics": ctx.get("reference_syntax") or {},
        "argument_for": "Train and diagnostic may differ in $var vs $var_ and output key conventions.",
        "argument_against": "Parser accepts multiple formats; may not drive official_win.",
        "alternative_explanations": ["Argument value errors mislabeled as reference."],
        "fixable_by_data_only": True,
        "next_step": "Validation gate on reference syntax + output-key distribution vs TargetProfile.",
        "requires_new_train_or_eval": False,
        "rank_score": 72,
    })

    items.append({
        "category": "DISTRACTOR_REALISM",
        "status": "PARTIALLY_SUPPORTED",
        "evidence_strength": "MEDIUM",
        "metrics": {"train_mean_hardness": (surface.get("distractor") or {}).get("train_mean"), "diag_mean_hardness": (surface.get("distractor") or {}).get("diag_mean")},
        "argument_for": "If train distractors are lexically far / type-impossible, model learns weak discrimination.",
        "argument_against": "Hardness proxies are not semantic.",
        "alternative_explanations": ["Offered-tool count mismatch."],
        "fixable_by_data_only": True,
        "next_step": "Minimum distractor hardness gate; schema-compatible near-miss distractors.",
        "requires_new_train_or_eval": False,
        "rank_score": 74,
    })

    items.append({
        "category": "REGISTRY_SEMANTIC_COVERAGE",
        "status": "PARTIALLY_SUPPORTED",
        "evidence_strength": "MEDIUM",
        "metrics": {"n_coverage_rows": len(coverage.get("task_rows") or []), "by_outcome_preview": (coverage.get("by_outcome") or [])[:5]},
        "argument_for": "Unmapped diagnostic gold tools on critical path associate with persistent failures (proxy).",
        "argument_against": "Registry size alone does not explain gap; exact IBM clone is not required.",
        "alternative_explanations": ["Surface renderer", "topology"],
        "fixable_by_data_only": True,
        "next_step": "Expand abstract operation families with high eval frequency + low coverage + low D1 gain.",
        "requires_new_train_or_eval": False,
        "rank_score": 77,
    })

    items.append({
        "category": "DATA_SCALE",
        "status": "PARTIALLY_SUPPORTED",
        "evidence_strength": "MEDIUM",
        "metrics": {"train_n": 300, "delta_pp": delta, "mcnemar_p": p_mc, "ci_includes_0": ci_lo <= 0 <= ci_hi},
        "argument_for": "n=300 with 51% dead groups yields few effective updates; underpowered for +2pp.",
        "argument_against": "More easy data can worsen dead groups.",
        "alternative_explanations": ["Quality/composition > scale."],
        "fixable_by_data_only": True,
        "next_step": "Scale only after effective-group and joint-coverage constraints.",
        "requires_new_train_or_eval": True,
        "rank_score": 73,
    })

    items.append({
        "category": "TRAINING_SEED_VARIANCE",
        "status": "NOT_IDENTIFIABLE",
        "evidence_strength": "LOW",
        "metrics": {},
        "argument_for": "Single seed run; flips could be seed noise.",
        "argument_against": "No multi-seed artifacts available.",
        "alternative_explanations": ["Stable small effect."],
        "fixable_by_data_only": False,
        "next_step": "Multi-seed train only after data composition fix.",
        "requires_new_train_or_eval": True,
        "rank_score": 40,
    })

    items.append({
        "category": "OTHER",
        "status": "PARTIALLY_SUPPORTED" if quality else "NOT_IDENTIFIABLE",
        "evidence_strength": "LOW",
        "metrics": quality,
        "argument_for": "Template concentration / shortcut cues can inflate train reward without transfer.",
        "argument_against": "Shortcuts may be limited after paraphrase.",
        "alternative_explanations": ["Genuine skill learning on narrow family."],
        "fixable_by_data_only": True,
        "next_step": "Template-skeleton concentration gate; anti-leak checks in V7+.",
        "requires_new_train_or_eval": False,
        "rank_score": 58,
    })

    items.sort(key=lambda x: -x["rank_score"])
    for i, it in enumerate(items, 1):
        it["rank"] = i
    return items


def build_generation_cells(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    topo = ctx.get("topology") or {}
    top1 = float((topo.get("train300") or {}).get("top1_share") or 0.0)
    unseen = float((topo.get("coverage_train300_vs_diag") or {}).get("diagnostic_unseen_topology_rate") or 0.0)
    dead = (ctx.get("reward") or {}).get("train_log_aggregates", {}).get("aggregates", {}).get("dead_group_rate")
    subset = ctx.get("subset") or {}
    surface = ctx.get("surface") or {}
    exact = float((surface.get("namespace") or {}).get("exact_overlap_rate_vs_diag") or 0.0)

    recs: List[Dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        base = {
            "recommendation_id": kwargs.pop("recommendation_id"),
            "mode": kwargs.get("mode", "PROFILE_SAFE"),
            "priority": kwargs.get("priority", "P1"),
            "track": kwargs.get("track", "paired"),
            "call_count_bucket": kwargs.get("call_count_bucket", "any"),
            "topology_requirement": kwargs.get("topology_requirement", ""),
            "motif": kwargs.get("motif", "any"),
            "target_skill": kwargs.get("target_skill", ""),
            "target_failure_mode": kwargs.get("target_failure_mode", ""),
            "reference_requirement": kwargs.get("reference_requirement", ""),
            "surface_requirement": kwargs.get("surface_requirement", ""),
            "offered_tool_range": kwargs.get("offered_tool_range", "8-16"),
            "distractor_requirement": kwargs.get("distractor_requirement", ""),
            "registry_family_requirement": kwargs.get("registry_family_requirement", ""),
            "current_support": kwargs.get("current_support", 0),
            "recommended_relative_quota": kwargs.get("recommended_relative_quota", "directional"),
            "evidence": kwargs.get("evidence", []),
            "expected_mechanism": kwargs.get("expected_mechanism", ""),
            "confidence": kwargs.get("confidence", "medium"),
            "risk": kwargs.get("risk", ""),
            "requires_new_primitive": kwargs.get("requires_new_primitive", False),
            "requires_new_surface": kwargs.get("requires_new_surface", False),
            "requires_new_validator": kwargs.get("requires_new_validator", False),
        }
        # mode-specific disclaimer
        if base["mode"] == "DIAGNOSTIC_INFORMED_EXPLORATORY":
            base["diagnostic_loop_disclaimer"] = (
                "This recommendation makes diagnostic-500 part of the development loop. "
                "It must not be validated as a final confirmatory claim on the same 500 tasks."
            )
        recs.append(base)

    add(
        recommendation_id="REC_001",
        mode="PROFILE_SAFE",
        priority="P0",
        track="paired",
        call_count_bucket="any",
        topology_requirement="increase unique topology hashes; cap top-1 family share below 10%",
        motif="reduce_linear_dominance",
        target_skill="multi_hop_dependency",
        target_failure_mode="wrong_tool_sequence",
        reference_requirement="mixed $varN.result$ / NESTFUL-like output keys",
        surface_requirement="A-native and G-general paired renders of same program",
        distractor_requirement="schema-compatible near-miss; mean hardness proxy increase 1.5–2×",
        evidence=[
            f"train300 top1 topology share={top1:.3f}",
            f"diagnostic unseen topology rate vs train300={unseen:.3f}",
            f"exact namespace overlap vs diagnostic={exact:.3f}",
        ],
        expected_mechanism="Reduce shape collapse and force surface-invariant dependency skill.",
        confidence="high" if top1 >= 0.15 or exact < 0.2 else "medium",
        risk="Over-diversifying topologies without reward signal wastes capacity.",
        requires_new_surface=True,
        requires_new_validator=True,
        recommended_relative_quota="increase unique topologies ≥2×; cap top-1 topology share ≤10%",
        current_support=0,
    )

    add(
        recommendation_id="REC_002",
        mode="PROFILE_SAFE",
        priority="P0",
        track="any",
        call_count_bucket="3|4|5|6+",
        topology_requirement="fan_in and reuse (max_outdegree>=2) minimum floor",
        motif="fan_in|branch_aggregate",
        target_skill="aggregation_arity",
        target_failure_mode="reference_wrong_source",
        reference_requirement="multi-parent refs with distinct output keys",
        surface_requirement="preserve argument key diversity",
        distractor_requirement="same arity / type signature distractors",
        evidence=[
            "linear chains dominate many factory curricula historically",
            f"unseen topology rate={unseen:.3f}",
        ],
        expected_mechanism="Teach non-linear dependency / fan-in that diagnostic programs use.",
        confidence="medium",
        risk="Harder graphs may raise all-fail dead groups if too hard.",
        recommended_relative_quota="increase fan-in/reuse cells 1.5–2× vs current linear share",
        requires_new_validator=True,
    )

    add(
        recommendation_id="REC_003",
        mode="PROFILE_SAFE",
        priority="P0",
        track="any",
        call_count_bucket="any",
        topology_requirement="any",
        motif="any",
        target_skill="reward_effective_group",
        target_failure_mode="all_success_saturation",
        reference_requirement="any",
        surface_requirement="any",
        distractor_requirement="calibrated difficulty bands",
        registry_family_requirement="keep families but stratified by probe difficulty later",
        evidence=[
            f"dead_group_rate={dead}",
            "mean_unique_rewards low in train log aggregates",
            f"subset shuffle_interpretation={subset.get('shuffle_interpretation')}",
        ],
        expected_mechanism="Increase fraction of mixed-reward groups so GRPO updates are informative.",
        confidence="high" if dead is not None and float(dead) >= 0.4 else "medium",
        risk="Without rollouts, difficulty targeting may overshoot to all-fail.",
        recommended_relative_quota="reserve 20–30% cells for mid-difficulty; reduce trivially saturated cells",
        requires_new_validator=True,
    )

    add(
        recommendation_id="REC_004",
        mode="PROFILE_SAFE",
        priority="P1",
        track="A",
        call_count_bucket="any",
        topology_requirement="any",
        motif="any",
        target_skill="nestful_reference_surface",
        target_failure_mode="reference_syntax",
        reference_requirement="NESTFUL-native output keys (result/output_0) distribution matched to TargetProfile",
        surface_requirement="longer tool descriptions; IBM-like parameter naming without copying functions",
        distractor_requirement="lexically similar names with incompatible required keys",
        evidence=[
            "reference syntax audit differences train vs diagnostic",
            f"exact overlap={exact:.3f}",
        ],
        expected_mechanism="Close surface/schema gap that blocks transfer of dependency skill.",
        confidence="high",
        risk="Overfitting to diagnostic lexical style if using DIAGNOSTIC_INFORMED mode accidentally.",
        requires_new_surface=True,
        requires_new_validator=True,
        recommended_relative_quota="≥50% of A-track with NESTFUL-like keys",
    )

    add(
        recommendation_id="REC_005",
        mode="DIAGNOSTIC_INFORMED_EXPLORATORY",
        priority="P1",
        track="paired",
        call_count_bucket="4|5|6+",
        topology_requirement="match high-frequency diagnostic unseen topology classes (abstract shapes only)",
        motif="mixed|fan_in|branch_aggregate",
        target_skill="ood_joint_cell",
        target_failure_mode="persistent_c0_d1_failure",
        reference_requirement="high reference density",
        surface_requirement="low exact namespace overlap by design",
        distractor_requirement="hardness ≥ diagnostic median proxy",
        registry_family_requirement="families with high diagnostic frequency and low train coverage (abstract)",
        evidence=[
            "coverage×outcome tables",
            "gained/lost pattern table",
            "joint unseen combination rate",
        ],
        expected_mechanism="Directly fill joint OOD cells associated with unchanged failures.",
        confidence="medium",
        risk="Adaptive overfitting to diagnostic-500.",
        recommended_relative_quota="allocate 15–25% exploratory mass to top unmet joint cells",
        requires_new_primitive=True,
        requires_new_surface=True,
        requires_new_validator=True,
    )

    add(
        recommendation_id="REC_006",
        mode="PROFILE_SAFE",
        priority="P1",
        track="G",
        call_count_bucket="2|3",
        topology_requirement="keep short programs but non-trivial distractors",
        motif="linear|fan_in",
        target_skill="tool_discrimination",
        target_failure_mode="wrong_first_tool",
        reference_requirement="optional",
        surface_requirement="G-general names with schema-matched distractors",
        distractor_requirement="raise near-miss rate; avoid type-impossible distractors",
        evidence=["wrong_first_tool transitions in failure taxonomy", "distractor hardness train vs diag"],
        expected_mechanism="Improve first-tool selection under realistic confusion sets.",
        confidence="medium",
        risk="2-call saturation → dead all-success.",
        recommended_relative_quota="cap easy 2-call all-success-prone cells; keep discrimination-focused 2/3-call",
        requires_new_validator=True,
    )

    add(
        recommendation_id="REC_007",
        mode="PROFILE_SAFE",
        priority="P2",
        track="any",
        call_count_bucket="any",
        topology_requirement="same graph template with constant-only variants limited",
        motif="any",
        target_skill="anti_shortcut",
        target_failure_mode="template_leak",
        reference_requirement="any",
        surface_requirement="paraphrase with skeleton diversity",
        distractor_requirement="any",
        evidence=["anti-shortcut audit template concentration"],
        expected_mechanism="Prevent reward hacking via question template cues.",
        confidence="medium",
        risk="Paraphrase drift breaking executability.",
        recommended_relative_quota="cap same question skeleton share below 5%",
        requires_new_validator=True,
    )

    return recs


def selection_constraints(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "objectives": [
            "joint deficit matching (call_bucket×motif×topology_hash×answer_type×track)",
            "penalize top-1 topology concentration",
            "minimum topology novelty vs already selected",
            "minimum tool-combination novelty",
            "minimum distractor hardness proxy",
            "cap semantic_program_family share",
            "coverage constraints for reference-syntax bins",
            "nested stratified subset creation for train-n",
            "separate train difficulty target from TargetProfile similarity",
        ],
        "suggested_thresholds_directional": {
            "top1_topology_share_max": "≤10%",
            "min_unique_topologies_per_100": "≥20",
            "max_program_family_share": "≤5%",
            "min_mean_distractor_hardness_proxy": "≥ diagnostic-or-profile median",
        },
        "evidence_basis": "Pilot3 forensic distribution/topology/subset audits",
    }


def validation_gates(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "V7_topology_concentration_gate": {
            "rule": "fail if top-1 topology share > 10% or unique topologies < 20 per 100 tasks",
            "requires_model": False,
        },
        "V8_joint_cell_coverage_gate": {
            "rule": "fail if selected set misses >X% of target joint cells from TargetProfile expansion",
            "requires_model": False,
        },
        "V9_reference_syntax_distribution_gate": {
            "rule": "match output-key and $var_ vs $var patterns to profile bins within TV≤0.15",
            "requires_model": False,
        },
        "V10_distractor_hardness_gate": {
            "rule": "mean hardness proxy and near-miss rate above floor",
            "requires_model": False,
        },
        "V11_template_skeleton_concentration_gate": {
            "rule": "max skeleton share ≤5%; near-duplicate rate ≤2%",
            "requires_model": False,
        },
        "V12_executable_difficulty_probe": {
            "rule": "future: base-model rollout probe for mixed-group rate bands (NOT run in this analysis)",
            "requires_model": True,
            "status": "specified_only",
        },
        "V13_effective_reward_group_probe": {
            "rule": "future: estimate dead_group_rate proxy before full train",
            "requires_model": True,
            "status": "specified_only",
        },
        "V14_surface_renderer_consistency": {
            "rule": "paired A/G renders must share topology_hash and gold answer",
            "requires_model": False,
        },
    }


def registry_gap_priorities(coverage_by_outcome: Sequence[Dict[str, Any]], task_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Aggregate unmapped tools from mapping_summary is limited; prioritize buckets
    rows = []
    for b in coverage_by_outcome:
        if not str(b.get("bucket", "")).startswith("exact"):
            continue
        rows.append({
            "gap_bucket": b.get("bucket"),
            "n_tasks": b.get("n"),
            "c0_win_rate": b.get("c0_win_rate"),
            "d1_win_rate": b.get("d1_win_rate"),
            "net_gain": b.get("net_gain"),
            "priority": "P0" if b.get("bucket") == "exact_none" and (b.get("n") or 0) >= 20 else "P1",
            "recommendation": "Add abstract schema-compatible primitives/surfaces for unmapped families; do not copy IBM function list.",
            "note": "proxy evidence only",
        })
    # count unmapped-heavy tasks
    n_unmapped = sum(1 for r in task_rows if int(r.get("n_unmapped_gold_tools") or 0) > 0)
    rows.append({
        "gap_bucket": "tasks_with_any_unmapped_gold_tool",
        "n_tasks": n_unmapped,
        "c0_win_rate": None,
        "d1_win_rate": None,
        "net_gain": None,
        "priority": "P0" if n_unmapped >= 50 else "P1",
        "recommendation": "Prioritize operation families on critical path with unmapped proxy labels.",
        "note": "Do not emit concrete IBM APIs to copy.",
    })
    return rows
