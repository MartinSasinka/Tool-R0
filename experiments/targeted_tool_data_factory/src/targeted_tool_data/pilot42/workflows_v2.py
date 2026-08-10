"""Auditable workflow blueprints: the generative source for Pilot4.2."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..repro import sha256_obj, write_json
from . import SCHEMA_VERSION

_CHAIN = ("LINEAR_CHAIN", "TYPE_TRANSITION_CHAIN", "REPEATED_PRIMITIVE")
_JOIN = ("FAN_IN_SINGLE", "LINEAR_CHAIN", "REUSE_EARLY_OUTPUT")
_RICH = ("FAN_IN_SINGLE", "DIAMOND", "PARALLEL_THEN_MERGE", "REUSE_EARLY_OUTPUT",
         "LATE_REFERENCE", "MULTI_JOIN", "TWO_STAGE_AGGREGATION", "LINEAR_CHAIN")


@dataclass(frozen=True)
class PlanNode:
    capability: str
    input_roles: Tuple[str, ...]
    output_role: str
    input_semantic_types: Tuple[str, ...]
    output_semantic_type: str


@dataclass(frozen=True)
class WorkflowBlueprint:
    workflow_id: str
    domain: str
    user_goal: str
    input_roles: Tuple[str, ...]
    role_semantic_types: Dict[str, str]
    target_role: str
    target_semantic_type: str
    plan_template: Tuple[PlanNode, ...]
    allowed_structural_patterns: Tuple[str, ...]
    allowed_call_count_range: Tuple[int, int]
    entity_pools: Tuple[str, ...]
    value_generator_hints: Dict[str, Any] = field(default_factory=dict)
    distractor_hints: Tuple[str, ...] = ("swap_operands", "neighbor_capability",
                                         "wrong_percentage_direction")
    difficulty_variants: Tuple[str, ...] = ("easy", "medium", "hard")
    natural_language_assets: Dict[str, Any] = field(default_factory=dict)
    coding_like: bool = False

    def as_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["schema_version"] = SCHEMA_VERSION
        return row


def _pn(cap, ins, out, in_types, out_type) -> PlanNode:
    return PlanNode(cap, tuple(ins), out, tuple(in_types), out_type)


def _wf(wid, domain, goal, roles, types, target, target_type, plan, patterns,
        entities, *, coding=False, assets=None) -> WorkflowBlueprint:
    n = len(plan)
    return WorkflowBlueprint(
        workflow_id=wid, domain=domain, user_goal=goal,
        input_roles=tuple(roles), role_semantic_types=dict(types),
        target_role=target, target_semantic_type=target_type,
        plan_template=tuple(plan),
        allowed_structural_patterns=tuple(patterns),
        allowed_call_count_range=(n, n),
        entity_pools=tuple(entities),
        value_generator_hints={"amount": [80, 1800], "rate": [5, 35]},
        natural_language_assets=assets or {
            "fact_order": list(roles), "target_phrase": target.replace("_", " ")},
        coding_like=coding,
    )


def build_default_workflows() -> List[WorkflowBlueprint]:
    w: List[WorkflowBlueprint] = []

    # --- 2-call core commerce/finance ---
    w.append(_wf(
        "commerce.discount_then_fee", "commerce",
        "determine the final price after a discount and a fee",
        ["base_amount", "reduction_rate", "fee"],
        {"base_amount": "Money", "reduction_rate": "Percentage", "fee": "Money",
         "reduced_amount": "Money", "final_amount": "Money"},
        "final_amount", "Money",
        [_pn("arithmetic.decrease_by_percent", ["base_amount", "reduction_rate"],
             "reduced_amount", ["Money", "Percentage"], "Money"),
         _pn("arithmetic.add", ["reduced_amount", "fee"], "final_amount",
             ["Money", "Money"], "Money")],
        _JOIN, ["invoice", "cart", "checkout"]))
    w.append(_wf(
        "commerce.portion_then_remaining", "commerce",
        "find the remaining amount after allocating a percentage of the total",
        ["total_amount", "allocation_rate"],
        {"total_amount": "Money", "allocation_rate": "Percentage",
         "allocated_amount": "Money", "remaining_amount": "Money"},
        "remaining_amount", "Money",
        [_pn("arithmetic.percentage_of", ["allocation_rate", "total_amount"],
             "allocated_amount", ["Percentage", "Money"], "Money"),
         _pn("arithmetic.subtract", ["total_amount", "allocated_amount"],
             "remaining_amount", ["Money", "Money"], "Money")],
        _JOIN, ["budget", "wallet"]))
    w.append(_wf(
        "personal_finance.adjust_compare", "personal_finance",
        "calculate an adjusted amount and test whether it exceeds a limit",
        ["starting_amount", "growth_rate", "limit"],
        {"starting_amount": "Money", "growth_rate": "Percentage", "limit": "Money",
         "adjusted_amount": "Money", "exceeds_limit": "Boolean"},
        "exceeds_limit", "Boolean",
        [_pn("arithmetic.increase_by_percent", ["starting_amount", "growth_rate"],
             "adjusted_amount", ["Money", "Percentage"], "Money"),
         _pn("comparison.greater_than", ["adjusted_amount", "limit"],
             "exceeds_limit", ["Money", "Money"], "Boolean")],
        _CHAIN, ["account", "budget"]))

    # --- 3-call ---
    w.append(_wf(
        "commerce.discount_tax_total", "commerce",
        "determine the final price after a discount and tax",
        ["base_price", "discount_rate", "tax_rate"],
        {"base_price": "Money", "discount_rate": "Percentage", "tax_rate": "Percentage",
         "discounted": "Money", "tax_amount": "Money", "final_price": "Money"},
        "final_price", "Money",
        [_pn("arithmetic.decrease_by_percent", ["base_price", "discount_rate"],
             "discounted", ["Money", "Percentage"], "Money"),
         _pn("arithmetic.percentage_of", ["tax_rate", "discounted"],
             "tax_amount", ["Percentage", "Money"], "Money"),
         _pn("arithmetic.add", ["discounted", "tax_amount"], "final_price",
             ["Money", "Money"], "Money")],
        _RICH, ["invoice", "SKU", "cart"]))
    w.append(_wf(
        "finance.interest_fee_total", "personal_finance",
        "compute the total due after interest and a fixed fee",
        ["principal", "interest_rate", "fee"],
        {"principal": "Money", "interest_rate": "Percentage", "fee": "Money",
         "with_interest": "Money", "total_due": "Money", "interest_part": "Money"},
        "total_due", "Money",
        [_pn("arithmetic.percentage_of", ["interest_rate", "principal"],
             "interest_part", ["Percentage", "Money"], "Money"),
         _pn("arithmetic.add", ["principal", "interest_part"], "with_interest",
             ["Money", "Money"], "Money"),
         _pn("arithmetic.add", ["with_interest", "fee"], "total_due",
             ["Money", "Money"], "Money")],
        _JOIN, ["loan", "credit line"]))
    w.append(_wf(
        "inventory.available_after_allocations", "inventory",
        "compute available units after two successive allocations",
        ["on_hand", "alloc_a", "alloc_b"],
        {"on_hand": "Count", "alloc_a": "Count", "alloc_b": "Count",
         "after_a": "Count", "available": "Count"},
        "available", "Count",
        [_pn("arithmetic.subtract", ["on_hand", "alloc_a"], "after_a",
             ["Count", "Count"], "Count"),
         _pn("arithmetic.subtract", ["after_a", "alloc_b"], "available",
             ["Count", "Count"], "Count"),
         _pn("comparison.greater_than", ["available", "alloc_b"],
             "available_ok_unused", ["Count", "Count"], "Boolean")],
        _CHAIN, ["warehouse bin", "SKU"]))
    # Fix inventory - last node shouldn't change sink. Rebuild cleanly:
    w[-1] = _wf(
        "inventory.available_after_two_picks", "inventory",
        "compute remaining stock after two picks from on-hand inventory",
        ["on_hand", "pick_a", "pick_b"],
        {"on_hand": "Count", "pick_a": "Count", "pick_b": "Count",
         "after_a": "Count", "remaining": "Count"},
        "remaining", "Count",
        [_pn("arithmetic.subtract", ["on_hand", "pick_a"], "after_a",
             ["Count", "Count"], "Count"),
         _pn("arithmetic.subtract", ["after_a", "pick_b"], "remaining",
             ["Count", "Count"], "Count")],
        _JOIN, ["warehouse bin", "SKU"])
    # Need 3-call inventory properly
    w.append(_wf(
        "inventory.reserve_then_ship", "inventory",
        "start from on-hand, reserve a portion, then subtract a shipment",
        ["on_hand", "reserve_rate", "shipment"],
        {"on_hand": "Count", "reserve_rate": "Percentage", "shipment": "Count",
         "reserved": "Count", "after_reserve": "Count", "remaining": "Count"},
        "remaining", "Count",
        [_pn("arithmetic.percentage_of", ["reserve_rate", "on_hand"],
             "reserved", ["Percentage", "Count"], "Count"),
         _pn("arithmetic.subtract", ["on_hand", "reserved"], "after_reserve",
             ["Count", "Count"], "Count"),
         _pn("arithmetic.subtract", ["after_reserve", "shipment"], "remaining",
             ["Count", "Count"], "Count")],
        _RICH, ["fulfillment center", "SKU"]))

    # --- 4-call ---
    w.append(_wf(
        "commerce.unit_price_quantity_discount_total", "commerce",
        "compute line total from unit price and quantity after a discount",
        ["unit_price", "quantity", "discount_rate"],
        {"unit_price": "Money", "quantity": "Count", "discount_rate": "Percentage",
         "gross": "Money", "discounted": "Money", "discount_amt": "Money",
         "line_total": "Money"},
        "line_total", "Money",
        [_pn("arithmetic.multiply", ["unit_price", "quantity"], "gross",
             ["Money", "Count"], "Money"),
         _pn("arithmetic.percentage_of", ["discount_rate", "gross"],
             "discount_amt", ["Percentage", "Money"], "Money"),
         _pn("arithmetic.subtract", ["gross", "discount_amt"], "discounted",
             ["Money", "Money"], "Money"),
         _pn("arithmetic.add", ["discounted", "discount_amt"], "line_total",
             ["Money", "Money"], "Money")],
        _RICH, ["order line", "catalog item"]))
    # The last add is wrong semantically (adds discount back). Fix:
    w[-1] = _wf(
        "commerce.unit_qty_discount_fee", "commerce",
        "compute final line amount from unit price, quantity, discount, and fee",
        ["unit_price", "quantity", "discount_rate", "fee"],
        {"unit_price": "Money", "quantity": "Count", "discount_rate": "Percentage",
         "fee": "Money", "gross": "Money", "discounted": "Money",
                                 "final_amount": "Money"},
        "final_amount", "Money",
        [_pn("arithmetic.multiply", ["unit_price", "quantity"], "gross",
             ["Money", "Count"], "Money"),
         _pn("arithmetic.decrease_by_percent", ["gross", "discount_rate"],
             "discounted", ["Money", "Percentage"], "Money"),
         _pn("arithmetic.add", ["discounted", "fee"], "final_amount",
             ["Money", "Money"], "Money")],
        _RICH, ["order line", "catalog item"])
    # That's 3-call - add 4-call properly
    w.append(_wf(
        "rates.scale_aggregate_compare", "rates_and_ratios",
        "scale a base by a rate, add an adjustment, then compare to a threshold",
        ["base", "rate", "adjustment", "threshold"],
        {"base": "GenericScalar", "rate": "Percentage", "adjustment": "GenericScalar",
         "threshold": "GenericScalar", "scaled": "GenericScalar",
         "combined": "GenericScalar", "over_threshold": "Boolean"},
        "over_threshold", "Boolean",
        [_pn("arithmetic.percentage_of", ["rate", "base"], "scaled",
             ["Percentage", "GenericScalar"], "GenericScalar"),
         _pn("arithmetic.add", ["scaled", "adjustment"], "combined",
             ["GenericScalar", "GenericScalar"], "GenericScalar"),
         _pn("comparison.greater_than", ["combined", "threshold"],
             "over_threshold", ["GenericScalar", "GenericScalar"], "Boolean")],
        _CHAIN, ["process", "mixture"]))
    w.append(_wf(
        "statistics.two_value_mean_threshold", "statistics",
        "average two readings and check whether the mean exceeds a limit",
        ["reading_a", "reading_b", "limit"],
        {"reading_a": "GenericScalar", "reading_b": "GenericScalar",
         "limit": "GenericScalar", "mean_ab": "GenericScalar",
         "exceeds": "Boolean"},
        "exceeds", "Boolean",
        [_pn("statistics.average_two", ["reading_a", "reading_b"], "mean_ab",
             ["GenericScalar", "GenericScalar"], "GenericScalar"),
         _pn("comparison.greater_than", ["mean_ab", "limit"], "exceeds",
             ["GenericScalar", "GenericScalar"], "Boolean")],
        _JOIN, ["survey", "series"]))
    # statistics capability maps to average_two - 2 params OK but only 2 nodes.
    # 4-call chain:
    w.append(_wf(
        "resource.branch_costs_then_total", "resource_allocation",
        "compute two branch costs from bases and rates, then total them",
        ["base_a", "rate_a", "base_b", "rate_b"],
        {"base_a": "Money", "rate_a": "Percentage", "base_b": "Money",
         "rate_b": "Percentage", "cost_a": "Money", "cost_b": "Money",
         "total_cost": "Money"},
        "total_cost", "Money",
        [_pn("arithmetic.percentage_of", ["rate_a", "base_a"], "cost_a",
             ["Percentage", "Money"], "Money"),
         _pn("arithmetic.percentage_of", ["rate_b", "base_b"], "cost_b",
             ["Percentage", "Money"], "Money"),
         _pn("arithmetic.add", ["cost_a", "cost_b"], "total_cost",
             ["Money", "Money"], "Money")],
        ("PARALLEL_THEN_MERGE", "FAN_IN_MULTIPLE", "MULTI_JOIN"),
        ["team", "project"]))
    w.append(_wf(
        "measurement.convert_scale_add", "measurement",
        "scale a measurement by a factor, apply a percent change, then add an offset",
        ["raw", "factor", "change_rate", "offset"],
        {"raw": "GenericScalar", "factor": "GenericScalar",
         "change_rate": "Percentage", "offset": "GenericScalar",
         "scaled": "GenericScalar", "adjusted": "GenericScalar",
         "final_value": "GenericScalar"},
        "final_value", "GenericScalar",
        [_pn("arithmetic.multiply", ["raw", "factor"], "scaled",
             ["GenericScalar", "GenericScalar"], "GenericScalar"),
         _pn("arithmetic.increase_by_percent", ["scaled", "change_rate"],
             "adjusted", ["GenericScalar", "Percentage"], "GenericScalar"),
         _pn("arithmetic.add", ["adjusted", "offset"], "final_value",
             ["GenericScalar", "GenericScalar"], "GenericScalar")],
        _CHAIN, ["sample", "panel"]))

    # --- 5-call ---
    w.append(_wf(
        "commerce.discount_tax_fee_compare", "commerce",
        "apply discount and tax, add fee, then check against a budget cap",
        ["base_price", "discount_rate", "tax_rate", "fee", "budget_cap"],
        {"base_price": "Money", "discount_rate": "Percentage", "tax_rate": "Percentage",
         "fee": "Money", "budget_cap": "Money", "discounted": "Money",
         "tax_amount": "Money", "pre_fee": "Money", "final_price": "Money",
         "over_budget": "Boolean"},
        "over_budget", "Boolean",
        [_pn("arithmetic.decrease_by_percent", ["base_price", "discount_rate"],
             "discounted", ["Money", "Percentage"], "Money"),
         _pn("arithmetic.percentage_of", ["tax_rate", "discounted"],
             "tax_amount", ["Percentage", "Money"], "Money"),
         _pn("arithmetic.add", ["discounted", "tax_amount"], "pre_fee",
             ["Money", "Money"], "Money"),
         _pn("arithmetic.add", ["pre_fee", "fee"], "final_price",
             ["Money", "Money"], "Money"),
         _pn("comparison.greater_than", ["final_price", "budget_cap"],
             "over_budget", ["Money", "Money"], "Boolean")],
        _RICH, ["checkout", "quote"]))
    w.append(_wf(
        "travel.distance_time_speed_check", "travel_distance",
        "derive average speed from distance and hours, then check a speed limit",
        ["distance", "hours", "speed_limit"],
        {"distance": "GenericScalar", "hours": "GenericScalar",
         "speed_limit": "GenericScalar", "speed": "GenericScalar",
         "over_limit": "Boolean"},
        "over_limit", "Boolean",
        [_pn("arithmetic.divide", ["distance", "hours"], "speed",
             ["GenericScalar", "GenericScalar"], "GenericScalar"),
         _pn("comparison.greater_than", ["speed", "speed_limit"],
             "over_limit", ["GenericScalar", "GenericScalar"], "Boolean")],
        _CHAIN, ["route", "journey"]))
    # pad travel to longer via multi-step
    w.append(_wf(
        "travel.leg_sum_then_speed", "travel_distance",
        "sum two legs, divide by hours to get speed, compare to limit",
        ["leg_a", "leg_b", "hours", "speed_limit"],
        {"leg_a": "GenericScalar", "leg_b": "GenericScalar", "hours": "GenericScalar",
         "speed_limit": "GenericScalar", "distance": "GenericScalar",
         "speed": "GenericScalar", "over_limit": "Boolean"},
        "over_limit", "Boolean",
        [_pn("arithmetic.add", ["leg_a", "leg_b"], "distance",
             ["GenericScalar", "GenericScalar"], "GenericScalar"),
         _pn("arithmetic.divide", ["distance", "hours"], "speed",
             ["GenericScalar", "GenericScalar"], "GenericScalar"),
         _pn("comparison.greater_than", ["speed", "speed_limit"],
             "over_limit", ["GenericScalar", "GenericScalar"], "Boolean")],
        _JOIN, ["route", "journey"]))

    # --- 6-call enrichment ---
    w.append(_wf(
        "quality.multi_metric_decision", "quality_control",
        "blend two metrics with weights, apply a tolerance percent, then gate",
        ["metric_a", "metric_b", "weight_a", "tolerance_rate", "gate_limit"],
        {"metric_a": "GenericScalar", "metric_b": "GenericScalar",
         "weight_a": "Percentage", "tolerance_rate": "Percentage",
         "gate_limit": "GenericScalar", "weighted_a": "GenericScalar",
         "weighted_b": "GenericScalar", "blend": "GenericScalar",
         "tolerant": "GenericScalar", "passes_gate": "Boolean"},
        "passes_gate", "Boolean",
        [_pn("arithmetic.percentage_of", ["weight_a", "metric_a"], "weighted_a",
             ["Percentage", "GenericScalar"], "GenericScalar"),
         _pn("arithmetic.subtract", ["metric_b", "weighted_a"], "weighted_b",
             ["GenericScalar", "GenericScalar"], "GenericScalar"),
         _pn("arithmetic.add", ["weighted_a", "weighted_b"], "blend",
             ["GenericScalar", "GenericScalar"], "GenericScalar"),
         _pn("arithmetic.increase_by_percent", ["blend", "tolerance_rate"],
             "tolerant", ["GenericScalar", "Percentage"], "GenericScalar"),
         _pn("comparison.greater_than", ["tolerant", "gate_limit"],
             "passes_gate", ["GenericScalar", "GenericScalar"], "Boolean")],
        _RICH, ["lot", "sample"]))
    w.append(_wf(
        "threshold.double_adjust_gate", "threshold_decision",
        "increase a score, decrease by a penalty rate, add a bonus, then gate",
        ["score", "boost_rate", "penalty_rate", "bonus", "gate"],
        {"score": "GenericScalar", "boost_rate": "Percentage",
         "penalty_rate": "Percentage", "bonus": "GenericScalar", "gate": "GenericScalar",
         "boosted": "GenericScalar", "penalized": "GenericScalar",
         "final_score": "GenericScalar", "passes": "Boolean"},
        "passes", "Boolean",
        [_pn("arithmetic.increase_by_percent", ["score", "boost_rate"],
             "boosted", ["GenericScalar", "Percentage"], "GenericScalar"),
         _pn("arithmetic.decrease_by_percent", ["boosted", "penalty_rate"],
             "penalized", ["GenericScalar", "Percentage"], "GenericScalar"),
         _pn("arithmetic.add", ["penalized", "bonus"], "final_score",
             ["GenericScalar", "GenericScalar"], "GenericScalar"),
         _pn("comparison.greater_than", ["final_score", "gate"], "passes",
             ["GenericScalar", "GenericScalar"], "Boolean")],
        _RICH, ["inspection", "gate"]))

    # Domain clones with distinct ids for coverage (still workflow-first plans)
    domain_clones = [
        ("data_summary", "summarize two values with a percent uplift then compare"),
        ("scheduling", "adjust workload with rates and compare to capacity"),
        ("geometry", "scale a length, adjust by percent, add offset"),
        ("energy", "scale usage, apply tariff percent, add fixed charge"),
        ("operations", "adjust a KPI by growth and fee then compare to target"),
        ("time_duration", "combine two durations, scale, compare to limit"),
        ("list_processing", "combine two scores, apply rate, compare threshold"),
        ("text_processing", "combine two counts, apply rate, compare limit"),
        ("file_path", "combine two sizes, apply rate, compare quota"),
        ("url_processing", "combine two request counts, apply rate, compare cap"),
        ("boolean_logic", "adjust a score and compare against two successive gates"),
        ("validation", "normalize two fields via rates and validate a limit"),
        ("classification", "score two features and decide a threshold class"),
        ("date_time", "combine two intervals, adjust, compare to window"),
        ("dictionary_processing", "combine two mapped values, adjust, compare"),
    ]
    for domain, goal in domain_clones:
        w.append(_wf(
            f"{domain}.scale_adjust_compare", domain, goal,
            ["v1", "v2", "rate", "limit"],
            {"v1": "GenericScalar", "v2": "GenericScalar", "rate": "Percentage",
             "limit": "GenericScalar", "sum_v": "GenericScalar",
             "adjusted": "GenericScalar", "ok": "Boolean"},
            "ok", "Boolean",
            [_pn("arithmetic.add", ["v1", "v2"], "sum_v",
                 ["GenericScalar", "GenericScalar"], "GenericScalar"),
             _pn("arithmetic.increase_by_percent", ["sum_v", "rate"],
                 "adjusted", ["GenericScalar", "Percentage"], "GenericScalar"),
             _pn("comparison.greater_than", ["adjusted", "limit"], "ok",
                 ["GenericScalar", "GenericScalar"], "Boolean")],
            _JOIN, [domain.replace("_", " "), "record"],
            coding=domain in ("list_processing", "text_processing", "file_path",
                              "url_processing", "dictionary_processing",
                              "boolean_logic", "validation")))

    # Extra 2-call variants for CORE density
    for domain in ("commerce", "personal_finance", "measurement", "rates_and_ratios",
                   "inventory", "resource_allocation", "statistics", "quality_control"):
        w.append(_wf(
            f"{domain}.grow_then_add", domain,
            f"increase a {domain.replace('_', ' ')} base by a percent and add a surcharge",
            ["base", "rate", "surcharge"],
            {"base": "Money", "rate": "Percentage", "surcharge": "Money",
             "grown": "Money", "total": "Money"},
            "total", "Money",
            [_pn("arithmetic.increase_by_percent", ["base", "rate"], "grown",
                 ["Money", "Percentage"], "Money"),
             _pn("arithmetic.add", ["grown", "surcharge"], "total",
                 ["Money", "Money"], "Money")],
            _CHAIN, [domain.replace("_", " "), "case"]))
        w.append(_wf(
            f"{domain}.shrink_then_compare", domain,
            f"decrease a {domain.replace('_', ' ')} value and test a floor",
            ["base", "rate", "floor"],
            {"base": "Money", "rate": "Percentage", "floor": "Money",
             "shrunk": "Money", "above_floor": "Boolean"},
            "above_floor", "Boolean",
            [_pn("arithmetic.decrease_by_percent", ["base", "rate"], "shrunk",
                 ["Money", "Percentage"], "Money"),
             _pn("comparison.greater_than", ["shrunk", "floor"], "above_floor",
                 ["Money", "Money"], "Boolean")],
            _CHAIN, [domain.replace("_", " "), "case"]))

    # Additional 4–6 call enrichment families for Nestful-like call profile
    for i, domain in enumerate((
            "commerce", "personal_finance", "measurement", "rates_and_ratios",
            "inventory", "resource_allocation", "statistics", "quality_control",
            "threshold_decision", "travel_distance", "energy", "operations")):
        w.append(_wf(
            f"{domain}.four_step_adjust_total", domain,
            f"scale, adjust by percent, add fee, then compare a cap for {domain.replace('_', ' ')}",
            ["base", "factor", "rate", "fee", "cap"],
            {"base": "Money", "factor": "GenericScalar", "rate": "Percentage",
             "fee": "Money", "cap": "Money", "scaled": "Money", "adjusted": "Money",
             "total": "Money", "over_cap": "Boolean"},
            "over_cap", "Boolean",
            [_pn("arithmetic.multiply", ["base", "factor"], "scaled",
                 ["Money", "GenericScalar"], "Money"),
             _pn("arithmetic.increase_by_percent", ["scaled", "rate"], "adjusted",
                 ["Money", "Percentage"], "Money"),
             _pn("arithmetic.add", ["adjusted", "fee"], "total",
                 ["Money", "Money"], "Money"),
             _pn("comparison.greater_than", ["total", "cap"], "over_cap",
                 ["Money", "Money"], "Boolean")],
            _RICH, [domain.replace("_", " "), "case"]))
        if i % 2 == 0:
            w.append(_wf(
                f"{domain}.six_step_pipeline", domain,
                f"multi-step {domain.replace('_', ' ')} pipeline with successive adjustments",
                ["a", "b", "r1", "r2", "fee", "limit"],
                {"a": "Money", "b": "Money", "r1": "Percentage", "r2": "Percentage",
                 "fee": "Money", "limit": "Money", "sum_ab": "Money", "s1": "Money",
                 "s2": "Money", "with_fee": "Money", "flag": "Boolean"},
                "flag", "Boolean",
                [_pn("arithmetic.add", ["a", "b"], "sum_ab",
                     ["Money", "Money"], "Money"),
                 _pn("arithmetic.increase_by_percent", ["sum_ab", "r1"], "s1",
                     ["Money", "Percentage"], "Money"),
                 _pn("arithmetic.decrease_by_percent", ["s1", "r2"], "s2",
                     ["Money", "Percentage"], "Money"),
                 _pn("arithmetic.add", ["s2", "fee"], "with_fee",
                     ["Money", "Money"], "Money"),
                 _pn("comparison.greater_than", ["with_fee", "limit"], "flag",
                     ["Money", "Money"], "Boolean")],
                _RICH, [domain.replace("_", " "), "pipeline"]))

    # Deduplicate by workflow_id
    by_id = {x.workflow_id: x for x in w}
    return list(by_id.values())


_REGISTRY: Tuple[WorkflowBlueprint, ...] | None = None


def get_workflows(*, reload: bool = False) -> List[WorkflowBlueprint]:
    global _REGISTRY
    if _REGISTRY is None or reload:
        _REGISTRY = tuple(build_default_workflows())
    return list(_REGISTRY)


def workflows_by_id() -> Dict[str, WorkflowBlueprint]:
    return {w.workflow_id: w for w in get_workflows()}


def export_registry(path: Path) -> Dict[str, Any]:
    rows = [w.as_dict() for w in get_workflows()]
    payload = {"schema_version": SCHEMA_VERSION, "n_workflows": len(rows),
               "domains": sorted({w.domain for w in get_workflows()}),
               "workflows": rows, "registry_hash": sha256_obj(rows)}
    write_json(path, payload)
    return payload
