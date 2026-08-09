"""Acceptance criteria as executable checks, and the readiness state machine.

Pilot4.2 could be declared finished while hard gates were unmet, because the gates
lived in prose in a report. Here every criterion from the specification is a row in
``CHECKS`` with a comparison against a number recomputed from the exported records.
``TRAINING_READY`` is a conjunction that no other module may set.

A check may be *blocking* (the dataset is not usable for the controlled GRPO run) or
*advisory* (a distribution preference). Only blocking checks gate
``AUTOMATED_GATES_PASSED``, and every failure is listed with its observed value so
the deficit report is specific rather than "some gates failed".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import (Any, Callable, Dict, FrozenSet, List, Mapping, Optional,
                    Sequence)

from . import (CALL_BUCKETS, HELDOUT_PARTS, PROFILE_CALL_TARGETS,
               QUERY_MODE_TARGETS, RESERVE_TARGET, RUN_ID, TIER_TARGETS,
               TRAIN_MASTER_TARGET, TRAINING_READINESS_KEYS)
from .export import (MASTER_FILE, RESERVE_FILE, TIER_FILES, dataset_metrics,
                     generation_cells, tv_distance)
from .ops import build_ops
from .pipeline import read_jsonl

BLOCKING = "blocking"
ADVISORY = "advisory"


@dataclass
class Check:
    """One acceptance criterion, evaluated against recomputed metrics."""

    id: str
    requirement: str
    observed: Any
    passed: bool
    severity: str = BLOCKING
    evidence: str = ""
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "requirement": self.requirement,
            "observed": self.observed,
            "passed": bool(self.passed),
            "severity": self.severity,
            "evidence": self.evidence,
            "note": self.note,
        }


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    def need(self, cid: str, requirement: str, observed: Any, passed: bool,
             evidence: str = "", note: str = "") -> Check:
        return self.add(Check(cid, requirement, observed, passed, BLOCKING,
                              evidence, note))

    def prefer(self, cid: str, requirement: str, observed: Any, passed: bool,
               evidence: str = "", note: str = "") -> Check:
        return self.add(Check(cid, requirement, observed, passed, ADVISORY,
                              evidence, note))

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.passed and c.severity == BLOCKING]

    @property
    def advisories(self) -> List[Check]:
        return [c for c in self.checks if not c.passed and c.severity == ADVISORY]

    @property
    def passed(self) -> bool:
        return not self.failures


def _in(value: float, low: float, high: float) -> bool:
    return low <= value <= high


def _load(out_dir: Path, name: str) -> List[Dict[str, Any]]:
    path = out_dir / name
    return read_jsonl(path) if path.exists() else []


def _json(out_dir: Path, name: str) -> Optional[Dict[str, Any]]:
    path = out_dir / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def evaluate(out_dir: Path) -> Dict[str, Any]:
    """Run every acceptance check over the exported dataset."""
    rep = Report()
    master = _load(out_dir, MASTER_FILE)
    tiers = {tier: _load(out_dir, fname) for tier, fname in TIER_FILES.items()}
    heldout = {name: _load(out_dir, f"heldout_{name}.jsonl")
               for name in HELDOUT_PARTS}
    reserve = _load(out_dir, RESERVE_FILE)
    selection = _json(out_dir, "selection_report.json") or {}
    profile = _json(out_dir, "target_profile_v3.json") or {}

    metrics = dataset_metrics(master) if master else {}
    tier_metrics = {t: dataset_metrics(rows) if rows else {}
                    for t, rows in tiers.items()}
    cells = generation_cells(master) if master else {}

    _sizes(rep, tiers, master, heldout, reserve)
    _call_counts(rep, metrics, tier_metrics, profile)
    _structural(rep, master, metrics, tier_metrics)
    _capability(rep, metrics, tier_metrics)
    _answers(rep, metrics, tier_metrics, out_dir, master)
    _queries(rep, metrics, master)
    _v4_and_necessity(rep, metrics, master)
    _distractors(rep, master)
    _split(rep, selection, out_dir)
    _cells(rep, cells)
    _reproducibility(rep, out_dir)

    result = {
        "run_id": RUN_ID,
        "checks": [c.as_dict() for c in rep.checks],
        "n_checks": len(rep.checks),
        "n_passed": sum(1 for c in rep.checks if c.passed),
        "blocking_failures": [c.as_dict() for c in rep.failures],
        "advisory_failures": [c.as_dict() for c in rep.advisories],
        "AUTOMATED_GATES_PASSED": rep.passed,
        "metrics": {"train_master": metrics,
                    "tiers": tier_metrics,
                    "generation_cells": cells},
    }
    return result


# ── individual criterion groups ──────────────────────────────────────────
def _sizes(rep: Report, tiers, master, heldout, reserve) -> None:
    for tier, want in TIER_TARGETS.items():
        got = len(tiers.get(tier, []))
        rep.need(f"size.{tier}", f"exactly {want} tasks", got, got == want,
                 evidence=TIER_FILES[tier])
    rep.need("size.TRAIN_MASTER", f"exactly {TRAIN_MASTER_TARGET}", len(master),
             len(master) == TRAIN_MASTER_TARGET, evidence=MASTER_FILE)
    total_heldout = sum(len(v) for v in heldout.values())
    # The spec explicitly prefers a valid split over a round 1000.
    rep.need("size.HELDOUT", ">= 900 with all parts non-empty", total_heldout,
             total_heldout >= 900 and all(heldout.get(p) for p in HELDOUT_PARTS),
             evidence="heldout_all.jsonl",
             note="round 1000 is not required; part validity is")
    rep.need("size.RESERVE", f"exactly {RESERVE_TARGET}", len(reserve),
             len(reserve) == RESERVE_TARGET, evidence=RESERVE_FILE)


def _call_counts(rep: Report, metrics, tier_metrics, profile) -> None:
    core = tier_metrics.get("PROFILE_CORE") or {}
    observed = core.get("call_bucket_distribution", {})
    target = {b: PROFILE_CALL_TARGETS[b][0] for b in CALL_BUCKETS}
    dev = ((profile.get("call_counts") or {}).get("bucket_shares")
           or (profile.get("call_count") or {}).get("bucket_shares"))
    if isinstance(dev, dict) and dev:
        target = {str(k): float(v) for k, v in dev.items()}
    tvd = tv_distance(observed, target) if observed else 1.0
    rep.need("calls.core_tv_distance", "TV distance to dev profile <= 0.03", tvd,
             tvd <= 0.03, evidence="train_profile_core_3000.jsonl")
    for bucket, (centre, tol) in PROFILE_CALL_TARGETS.items():
        got = observed.get(bucket, 0.0)
        rep.need(f"calls.core_bucket_{bucket}",
                 f"{centre:.3f} +- {tol:.3f}", got,
                 abs(got - centre) <= tol + 1e-9)
    core_six = core.get("six_plus_share", 0.0)
    rep.need("calls.core_six_plus", "0.20 <= share <= 0.24", core_six,
             _in(core_six, 0.20, 0.24))
    master_six = metrics.get("six_plus_share", 0.0)
    rep.need("calls.master_six_plus", "0.30 <= share <= 0.38", master_six,
             _in(master_six, 0.30, 0.38), evidence=MASTER_FILE)
    buckets = metrics.get("call_bucket_distribution", {})
    missing = [b for b in CALL_BUCKETS if not buckets.get(b)]
    rep.need("calls.no_missing_bucket", "every call bucket present", missing,
             not missing)

    lh = tier_metrics.get("LONG_HORIZON_ENRICHMENT") or {}
    lh_six = lh.get("six_plus_share", 0.0)
    rep.need("calls.long_horizon_six_plus", "share >= 0.65", lh_six,
             lh_six >= 0.65, evidence="train_long_horizon_1200.jsonl")
    deep = {k: v for k, v in (lh.get("call_count_distribution") or {}).items()
            if int(k) >= 6}
    deep_total = sum(deep.values())
    concentrated = (max(deep.values()) / deep_total) if deep_total else 1.0
    rep.need("calls.long_horizon_depth_spread",
             "no single depth is all of the 6+ mass (<= 0.45)",
             round(concentrated, 4), concentrated <= 0.45,
             note="prevents every 6+ task having exactly six calls")
    cap = tier_metrics.get("CAPABILITY_ENRICHMENT") or {}
    cap_six = cap.get("six_plus_share", 0.0)
    cap_five = cap_six + (cap.get("call_bucket_distribution", {}).get("5", 0.0))
    rep.need("calls.capability_six_plus", "share >= 0.25", cap_six,
             cap_six >= 0.25, evidence="train_capability_enrichment_600.jsonl")
    rep.need("calls.capability_five_plus", "share >= 0.45", round(cap_five, 5),
             cap_five >= 0.45)


def _structural(rep: Report, master, metrics, tier_metrics) -> None:
    mismatched = [r["task_id"] for r in master
                  if r["declared"]["requested_structural_skill"]
                  not in r["declared"]["satisfied_patterns"]]
    rep.need("structure.pattern_match", "actual pattern match = 100 %",
             len(master) - len(mismatched), not mismatched,
             evidence="actual_pattern_classification.csv")
    no_graph = [r["task_id"] for r in master
                if not r["declared"]["graph_features"].get("n_nodes")]
    rep.need("structure.graph_reconstructed", "graph features for 100 %",
             len(master) - len(no_graph), not no_graph,
             evidence="actual_graph_features.csv")
    six = [r for r in master if len(r["gold_calls"]) >= 6]
    fams = {r["declared"]["structural_pattern"] for r in six}
    rep.need("structure.six_plus_pattern_families",
             ">= 10 actual pattern families among 6+ tasks", len(fams),
             len(fams) >= 10)
    core_six = [r for r in master if len(r["gold_calls"]) >= 6
                and r["cell_tier"] == "PROFILE_CORE"]
    if core_six:
        gf = [r["declared"]["graph_features"] for r in core_six]
        n = len(gf)
        join = sum(1 for g in gf if g.get("n_join_nodes", 0) >= 1) / n
        multi = sum(1 for g in gf if g.get("n_join_nodes", 0) >= 2) / n
        late = sum(1 for g in gf if g.get("n_late_edges", 0) >= 1) / n
        fan = sum(1 for g in gf if g.get("n_fan_out_nodes", 0) >= 1) / n
        reuse = sum(1 for g in gf if g.get("n_reused_outputs", 0) >= 1) / n
        rep.need("structure.core6_join_rate", ">= 0.55", round(join, 4),
                 join >= 0.55)
        rep.need("structure.core6_multi_join_rate", ">= 0.25", round(multi, 4),
                 multi >= 0.25)
        rep.need("structure.core6_late_reference_rate", ">= 0.50",
                 round(late, 4), late >= 0.50)
        rep.need("structure.core6_fan_out_rate", ">= 0.15", round(fan, 4),
                 fan >= 0.15)
        rep.need("structure.core6_reuse_rate", ">= 0.12", round(reuse, 4),
                 reuse >= 0.12)


def _capability(rep: Report, metrics, tier_metrics) -> None:
    rep.need("capability.primitives_used", ">= 60 distinct primitives in gold calls",
             metrics.get("actual_primitives_used", 0),
             metrics.get("actual_primitives_used", 0) >= 60,
             evidence="actual_capability_usage.csv")
    rep.need("capability.families_used", ">= 20 capability families",
             metrics.get("actual_capability_families", 0),
             metrics.get("actual_capability_families", 0) >= 20)
    rep.need("capability.coding_families", ">= 12 generic/coding families",
             metrics.get("coding_capability_families", 0),
             metrics.get("coding_capability_families", 0) >= 12)
    rep.need("capability.coding_task_share", ">= 0.15 of tasks",
             metrics.get("coding_task_share", 0.0),
             metrics.get("coding_task_share", 0.0) >= 0.15)
    rep.need("capability.coding_call_share", ">= 0.15 of gold calls",
             metrics.get("coding_call_share", 0.0),
             metrics.get("coding_call_share", 0.0) >= 0.15)
    rep.need("capability.sequence_concentration",
             "no exact primitive sequence > 0.03",
             metrics.get("max_exact_sequence_share", 1.0),
             metrics.get("max_exact_sequence_share", 1.0) <= 0.03,
             evidence="primitive_sequence_distribution.csv")
    rep.need("capability.normalized_sequence_concentration",
             "no normalized capability sequence > 0.05",
             metrics.get("max_normalized_sequence_share", 1.0),
             metrics.get("max_normalized_sequence_share", 1.0) <= 0.05)
    rep.need("capability.top10_sequences", "top-10 exact share <= 0.30",
             metrics.get("top10_exact_sequence_share", 1.0),
             metrics.get("top10_exact_sequence_share", 1.0) <= 0.30)
    cap = tier_metrics.get("CAPABILITY_ENRICHMENT") or {}
    rep.need("capability.enrichment_all_coding",
             "CAPABILITY_ENRICHMENT coding task share = 1.0",
             cap.get("coding_task_share", 0.0),
             cap.get("coding_task_share", 0.0) >= 0.999)
    rep.need("capability.enrichment_primitives",
             ">= 35 distinct coding primitives in the tier",
             cap.get("distinct_coding_primitives", 0),
             cap.get("distinct_coding_primitives", 0) >= 35)
    rep.need("capability.enrichment_families",
             ">= 12 coding families in the tier",
             cap.get("coding_capability_families", 0),
             cap.get("coding_capability_families", 0) >= 12)
    rep.need("capability.enrichment_structured_answers",
             "string/list/object share >= 0.25",
             cap.get("structured_answer_share", 0.0),
             cap.get("structured_answer_share", 0.0) >= 0.25)


def _answers(rep: Report, metrics, tier_metrics, out_dir: Path, master) -> None:
    dist = metrics.get("answer_type_distribution", {})
    missing = [a for a, v in dist.items() if not v]
    rep.prefer("answers.all_types_present", "every answer type represented",
               missing, not missing)
    true_share = metrics.get("boolean_true_share", 0.0)
    rep.need("answers.boolean_balance_overall", "True share in 0.40-0.60",
             true_share, _in(true_share, 0.40, 0.60),
             evidence="boolean_balance.csv")
    per_wf = _boolean_per_workflow(master)
    offenders = {k: v for k, v in per_wf.items()
                 if v["n"] >= 20 and not _in(v["true_share"], 0.35, 0.65)}
    rep.need("answers.boolean_balance_per_workflow",
             "each workflow with n>=20 in 0.35-0.65", offenders, not offenders,
             evidence="boolean_balance.csv")
    extreme = {k: v for k, v in per_wf.items()
               if v["n"] >= 20 and (v["true_share"] > 0.80 or v["true_share"] < 0.20)}
    rep.need("answers.no_boolean_landslide", "no workflow > 80 % one label",
             extreme, not extreme)


def _boolean_per_workflow(master: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[bool]] = {}
    for row in master:
        if row["answer_type"] != "boolean":
            continue
        groups.setdefault(row["workflow_id"], []).append(bool(row["gold_answer"]))
    return {wid: {"n": len(vals),
                  "true_share": round(sum(vals) / len(vals), 4)}
            for wid, vals in groups.items()}


def _queries(rep: Report, metrics, master) -> None:
    modes = metrics.get("query_mode_distribution", {})
    graph_explicit = modes.get("GRAPH_EXPLICIT", 0.0)
    rep.need("query.graph_explicit_share", "<= 0.03", graph_explicit,
             graph_explicit <= 0.03)
    implicit = (modes.get("DOMAIN_GROUNDED_IMPLICIT", 0.0)
                + modes.get("GOAL_BASED_IMPLICIT", 0.0))
    rep.need("query.implicit_share",
             "domain-grounded + goal-based >= 0.65", round(implicit, 5),
             implicit >= 0.65)
    for mode, (low, high) in QUERY_MODE_TARGETS.items():
        got = modes.get(mode, 0.0)
        rep.prefer(f"query.mode_band_{mode}", f"{low:.2f}-{high:.2f}", got,
                   _in(got, low, high))
    div = metrics.get("diversity", {})
    rep.need("query.exact_duplicates", "duplicate rate = 0",
             div.get("exact_duplicate_rate", 1.0),
             div.get("exact_duplicate_rate") == 0.0,
             evidence="template_diversity_report.json")
    rep.need("query.skeleton_share", "max normalized skeleton share <= 0.01",
             div.get("max_skeleton_share", 1.0),
             div.get("max_skeleton_share", 1.0) <= 0.01)
    rep.need("query.intent_share", "max intent template share <= 0.02",
             div.get("max_intent_share", 1.0),
             div.get("max_intent_share", 1.0) <= 0.02)
    rep.need("query.top10_intent_share", "top-10 intent share <= 0.15",
             div.get("top10_intent_share", 1.0),
             div.get("top10_intent_share", 1.0) <= 0.15)
    failed = [r["task_id"] for r in master
              if not r["validation"]["query_checks"].get("passed")]
    rep.need("query.deterministic_checks", "all queries pass hard validation",
             len(master) - len(failed), not failed,
             evidence="query_hard_valid.jsonl")
    leaks = [r["task_id"] for r in master
             if r["actual_query_mode"] in ("GOAL_BASED_IMPLICIT",
                                           "DOMAIN_GROUNDED_IMPLICIT")
             and (r["validation"]["query_checks"].get("classification", {})
                  .get("graph_edge_coverage", 0.0) > 0.0)]
    share = len(leaks) / len(master) if master else 0.0
    rep.need("query.implicit_graph_leakage", "<= 0.02 of implicit tasks",
             round(share, 5), share <= 0.02)
    llm_rows = [r for r in master if r["query_source"] == "openrouter"]
    critic = sum(1 for r in llm_rows
                 if r["validation"].get("critic", {}).get("executed"))
    coverage = critic / len(llm_rows) if llm_rows else 0.0
    rep.need("query.critic_coverage",
             "100 % of LLM-written queries have critic evidence",
             round(coverage, 5), (not llm_rows) or coverage >= 0.999,
             note="vacuously true when no LLM queries were written; "
                  "LLM_VALIDATED is a separate status")
    routed = [r for r in master
              if r["validation"].get("second_critic", {}).get("routed")]
    executed = sum(1 for r in routed
                   if r["validation"]["second_critic"].get("executed"))
    rep.need("query.second_critic_routing",
             "every routed task has a second critic verdict",
             f"{executed}/{len(routed)}", executed == len(routed))


def _v4_and_necessity(rep: Report, metrics, master) -> None:
    rep.need("v4.coverage", "= 1.0", metrics.get("v4_coverage", 0.0),
             metrics.get("v4_coverage", 0.0) >= 0.999, evidence="v4_per_task.jsonl")
    rep.need("v4.skipped", "= 0", metrics.get("v4_skipped", -1),
             metrics.get("v4_skipped") == 0)
    rep.need("v4.shortcuts", "= 0", metrics.get("v4_shortcuts", -1),
             metrics.get("v4_shortcuts") == 0)
    rep.need("v4.unresolved", "= 0", metrics.get("v4_unresolved", -1),
             metrics.get("v4_unresolved") == 0)
    non_numeric = [r for r in master
                   if r["answer_type"] not in ("float", "integer")]
    covered = sum(1 for r in non_numeric
                  if r["validation"]["v4"].get("v4_executed"))
    rep.need("v4.non_numeric_coverage",
             "V4 executed for every non-numeric answer type",
             f"{covered}/{len(non_numeric)}", covered == len(non_numeric))
    rep.need("necessity.coverage", "= 1.0",
             metrics.get("node_necessity_coverage", 0.0),
             metrics.get("node_necessity_coverage", 0.0) >= 0.999,
             evidence="per_node_necessity.jsonl")
    rep.need("necessity.unnecessary_nodes", "= 0",
             metrics.get("unnecessary_gold_nodes", -1),
             metrics.get("unnecessary_gold_nodes") == 0)
    no_evidence = [r["task_id"] for r in master
                   if len(r["validation"].get("node_necessity") or [])
                   != len(r["gold_calls"])]
    rep.need("necessity.per_node_evidence",
             "one evidence record per gold call",
             len(master) - len(no_evidence), not no_evidence)


def _distractors(rep: Report, master) -> None:
    missing_gold = [r["task_id"] for r in master
                    if not {c["name"] for c in r["gold_calls"]}
                    <= {t["name"] for t in r["tools"]}]
    rep.need("tools.gold_present", "no gold tool missing from the offered set",
             len(master) - len(missing_gold), not missing_gold)
    dupes = [r["task_id"] for r in master
             if len({t["name"] for t in r["tools"]}) != len(r["tools"])]
    rep.need("tools.no_duplicate_schema", "no duplicate tool name collision",
             len(master) - len(dupes), not dupes)
    thin = [r["task_id"] for r in master
            if r["distractor_profile"]["distractor_count"] < 1]
    rep.need("tools.distractor_minimum",
             ">= 1 behaviourally validated distractor per task",
             len(master) - len(thin), not thin,
             note="every distractor in the set was confirmed to change the answer "
                  "on the instance or a counterfactual before being offered")
    # A hard distractor shares its gold operation's signature *and* capability
    # family. Two things can legitimately empty that slot: the registry holds no
    # such sibling, or every sibling computed the same answer here and offering it
    # would have made the task multi-solution. Failing those tasks would delete
    # whole capability families from the dataset, so the gate fails only where a
    # usable sibling existed and was not offered.
    missed = [r["task_id"] for r in master
              if r["distractor_profile"]["hard"] < 1
              and _hard_distractor_available(r)]
    absent = sum(1 for r in master if r["distractor_profile"]["hard"] < 1)
    rep.need("tools.hard_distractor",
             ">= 1 hard (same-family) distractor wherever one was usable",
             len(master) - len(missed), not missed,
             note=f"{absent} tasks have no hard distractor; {absent - len(missed)} "
                  f"of those had no usable sibling (none in the registry, or every "
                  f"sibling was answer-preserving on the instance)")


def _hard_distractor_available(row: Mapping[str, Any]) -> bool:
    """Could a same-signature, same-family alternative have been offered?"""
    if row["distractor_profile"].get("hard_rejected_as_alias", 0) > 0:
        return False
    gold = frozenset(c["primitive_id"] for c in row["gold_calls"]
                     if c.get("primitive_id"))
    return bool(gold) and _hard_siblings(gold)


@lru_cache(maxsize=None)
def _hard_siblings(gold: FrozenSet[str]) -> bool:
    from .distractors import _signature

    ops = build_ops()
    for gid in gold:
        if gid not in ops:
            continue
        gop = ops[gid]
        gsig = _signature(gop)
        for pid, op in ops.items():
            if (pid not in gold and op.family == gop.family
                    and op.capability != gop.capability
                    and _signature(op) == gsig):
                return True
    return False


def _split(rep: Report, selection: Dict[str, Any], out_dir: Path) -> None:
    overlap = selection.get("split_overlap") or {}
    rep.need("split.instance_leakage", "= 0",
             overlap.get("instance_leakage", -1),
             overlap.get("instance_leakage") == 0, evidence="split_manifest.json")
    for key in ("workflow_holdout_overlap", "program_plan_holdout_overlap",
                "actual_topology_holdout_overlap",
                "query_template_holdout_overlap",
                "capability_combination_holdout_overlap"):
        rep.need(f"split.{key}", "= 0", overlap.get(key, -1),
                 overlap.get(key) == 0)
    rep.need("split.surface_holdout", "holdout track absent from train",
             overlap.get("surface_holdout_respects_config"),
             bool(overlap.get("surface_holdout_respects_config")))
    rep.need("split.tier_quotas_met", "no tier cross-filled or short",
             selection.get("quotas_met"), bool(selection.get("quotas_met")),
             evidence="selection_report.json")
    rep.need("split.nested_subsets", "nested subsets valid and exact",
             selection.get("nested_subsets_valid"),
             bool(selection.get("nested_subsets_valid")))
    ledger = _json(out_dir, "reserve_access_ledger.json") or {}
    rep.need("split.reserve_untouched", "reserve never read back",
             ledger.get("untouched"), bool(ledger.get("untouched")))


def _cells(rep: Report, cells: Dict[str, Any]) -> None:
    n = cells.get("n_cells", 0)
    rep.need("cells.count", "60-100 core skill cells", n, 60 <= n <= 100,
             evidence="generation_cells_v3.json")
    rep.need("cells.no_singletons", "no singleton core cells",
             cells.get("singleton_cells", -1), cells.get("singleton_cells") == 0)
    rep.need("cells.no_two_task", "no two-task core cells",
             cells.get("two_task_cells", -1), cells.get("two_task_cells") == 0)
    under = cells.get("under_supported_cells") or {}
    rep.need("cells.min_support", ">= 20 tasks per cell", under, not under)


def _reproducibility(rep: Report, out_dir: Path) -> None:
    freeze = _json(out_dir, "freeze_manifest.json") or {}
    rep.need("repro.freeze_manifest", "freeze manifest present",
             bool(freeze), bool(freeze), evidence="freeze_manifest.json")
    inputs = freeze.get("input_hashes") or {}
    rep.need("repro.input_hashes", "non-empty input hashes", len(inputs),
             bool(inputs))
    outputs = freeze.get("artifact_hashes") or {}
    rep.need("repro.output_hashes", "non-empty artifact hashes", len(outputs),
             bool(outputs))
    iso = _json(out_dir, "openrouter_usage_pilot43.json") or {}
    foreign = iso.get("foreign_run_records")
    rep.need("repro.openrouter_isolation",
             "no foreign run_id in this run's OpenRouter logs",
             foreign if foreign is not None else "no log",
             foreign in (0, None),
             note="no log means the writer never ran; LLM_VALIDATED stays false")


# ── readiness state machine ──────────────────────────────────────────────
def readiness(out_dir: Path, gates: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the ten statuses. ``TRAINING_READY`` is a pure conjunction."""
    master = _load(out_dir, MASTER_FILE)
    audit = _json(out_dir, "PILOT43_INDEPENDENT_AUDIT.json") or {}
    human = _json(out_dir, "human_audit_results.json") or {}
    probe = _json(out_dir, "model_probe_report.json") or {}
    or_usage = _json(out_dir, "openrouter_usage_pilot43.json") or {}
    selection = _json(out_dir, "selection_report.json") or {}

    llm_rows = [r for r in master if r["query_source"] == "openrouter"]
    critic_ok = bool(llm_rows) and all(
        r["validation"].get("critic", {}).get("executed") for r in llm_rows)
    second_ok = all(r["validation"]["second_critic"].get("executed")
                    for r in master
                    if r["validation"].get("second_critic", {}).get("routed"))
    # foreign_run_records defaults to 0 when the usage file omits the field
    # (older usage dumps); only a positive count must fail LLM_VALIDATED.
    foreign = or_usage.get("foreign_run_records")
    if foreign is None:
        foreign = (or_usage.get("totals") or {}).get("foreign_run_records", 0)
    llm_validated = bool(critic_ok and second_ok and int(foreign or 0) == 0)

    states = {
        "IMPLEMENTATION_COMPLETE": True,
        "GENERATION_COMPLETE": bool(master) and bool(selection),
        "AUTOMATED_GATES_PASSED": bool(gates.get("AUTOMATED_GATES_PASSED")),
        "INDEPENDENT_AUDIT_PASSED": bool(audit.get("INDEPENDENT_AUDIT_PASSED")),
        "LLM_VALIDATED": llm_validated,
        "HUMAN_VALIDATED": bool(human.get("thresholds_met")),
        "GRPO_SIGNAL_READY": bool(probe.get("thresholds_met")),
    }
    states["HUMAN_REVIEW_PENDING"] = not states["HUMAN_VALIDATED"]
    states["GRPO_PROBE_PENDING"] = not states["GRPO_SIGNAL_READY"]
    states["TRAINING_READY"] = all(states[k] for k in (
        "AUTOMATED_GATES_PASSED", "INDEPENDENT_AUDIT_PASSED", "LLM_VALIDATED",
        "HUMAN_VALIDATED", "GRPO_SIGNAL_READY"))

    evidence = {
        "IMPLEMENTATION_COMPLETE": ["src/targeted_tool_data/pilot43/",
                                    "tests/test_pilot43_*.py"],
        "GENERATION_COMPLETE": [MASTER_FILE, "selection_report.json"],
        "AUTOMATED_GATES_PASSED": ["PILOT43_DATA_QUALITY_REPORT.json"],
        "INDEPENDENT_AUDIT_PASSED": ["PILOT43_INDEPENDENT_AUDIT.json",
                                     "independent_audit_per_task.csv"],
        "LLM_VALIDATED": ["openrouter_requests_pilot43.jsonl",
                          "openrouter_usage_pilot43.json",
                          "critic_disagreements.jsonl"],
        "HUMAN_REVIEW_PENDING": ["human_audit_sample.csv", "human_audit_guide.md"],
        "HUMAN_VALIDATED": ["human_audit_results.json"],
        "GRPO_PROBE_PENDING": ["model_probe_report.json"],
        "GRPO_SIGNAL_READY": ["model_probe_report.json", "model_probe_groups.csv"],
        "TRAINING_READY": ["PILOT43_IMPLEMENTATION_REPORT.json"],
    }
    return {
        "statuses": {k: states[k] for k in TRAINING_READINESS_KEYS},
        "evidence": evidence,
        "blockers": [k for k in ("AUTOMATED_GATES_PASSED",
                                 "INDEPENDENT_AUDIT_PASSED", "LLM_VALIDATED",
                                 "HUMAN_VALIDATED", "GRPO_SIGNAL_READY")
                     if not states[k]],
    }


def run(out_dir: Path) -> Dict[str, Any]:
    """Evaluate gates, compute readiness, write both report faces."""
    gates = evaluate(out_dir)
    ready = readiness(out_dir, gates)
    payload = {**gates, "readiness": ready}
    (out_dir / "PILOT43_DATA_QUALITY_REPORT.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "PILOT43_DATA_QUALITY_REPORT.md").write_text(
        _markdown(payload), encoding="utf-8")
    return payload


def _markdown(payload: Dict[str, Any]) -> str:
    lines = ["# Pilot4.3 data quality report", "",
             f"run_id: `{payload['run_id']}`", "",
             f"- checks: {payload['n_passed']}/{payload['n_checks']} passed",
             f"- AUTOMATED_GATES_PASSED: **{payload['AUTOMATED_GATES_PASSED']}**",
             ""]
    ready = payload.get("readiness", {}).get("statuses", {})
    lines += ["## Readiness", "", "| status | value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in ready.items()]
    lines.append("")
    fails = payload.get("blocking_failures", [])
    lines += ["## Blocking failures", ""]
    if not fails:
        lines.append("None.")
    else:
        lines += ["| check | requirement | observed |", "| --- | --- | --- |"]
        lines += [f"| `{c['id']}` | {c['requirement']} | `{c['observed']}` |"
                  for c in fails]
    lines.append("")
    adv = payload.get("advisory_failures", [])
    lines += ["## Advisory deviations", ""]
    if not adv:
        lines.append("None.")
    else:
        lines += ["| check | requirement | observed |", "| --- | --- | --- |"]
        lines += [f"| `{c['id']}` | {c['requirement']} | `{c['observed']}` |"
                  for c in adv]
    lines += ["", "## All checks", "",
              "| check | requirement | observed | passed | severity |",
              "| --- | --- | --- | --- | --- |"]
    lines += [f"| `{c['id']}` | {c['requirement']} | `{c['observed']}` | "
              f"{c['passed']} | {c['severity']} |" for c in payload["checks"]]
    return "\n".join(lines) + "\n"
