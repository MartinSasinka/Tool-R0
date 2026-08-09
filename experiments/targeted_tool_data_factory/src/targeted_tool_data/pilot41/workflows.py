"""Semantic workflow grammar: domain → goal → roles → typed DAG constraints."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..repro import sha256_obj, write_json
from ..util import short_hash

SCHEMA_VERSION = "ttdf.workflow.v1"


@dataclass
class WorkflowFamily:
    workflow_id: str
    domain: str
    user_goal_template: str
    required_roles: List[str]
    target_role: str
    allowed_capability_sequence: List[str]
    allowed_structural_patterns: List[str]
    allowed_units: List[str] = field(default_factory=list)
    forbidden_transitions: List[str] = field(default_factory=list)
    semantic_constraints: List[str] = field(default_factory=list)
    difficulty_variants: List[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    entity_pool: List[str] = field(default_factory=list)
    fact_templates: List[str] = field(default_factory=list)
    min_calls: int = 2
    max_calls: int = 6
    natural_language_assets: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d


_CHAIN = ["LINEAR_CHAIN", "REPEATED_PRIMITIVE", "TYPE_TRANSITION_CHAIN"]
_JOIN = ["FAN_IN_SINGLE", "LINEAR_CHAIN", "REUSE_EARLY_OUTPUT"]
_RICH = ["FAN_IN_SINGLE", "FAN_OUT", "DIAMOND", "PARALLEL_THEN_MERGE",
         "REUSE_EARLY_OUTPUT", "LATE_REFERENCE", "MULTI_JOIN",
         "TWO_STAGE_AGGREGATION", "NESTED_AGGREGATION", "LINEAR_CHAIN"]


def _wf(wid, domain, goal, roles, target, caps, patterns, *,
        entities=None, units=None, forbidden=None, min_c=2, max_c=6,
        facts=None) -> WorkflowFamily:
    return WorkflowFamily(
        workflow_id=wid, domain=domain, user_goal_template=goal,
        required_roles=list(roles), target_role=target,
        allowed_capability_sequence=list(caps),
        allowed_structural_patterns=list(patterns),
        allowed_units=list(units or []),
        forbidden_transitions=list(forbidden or [
            "GenericScalar->DurationDays",
            "GenericScalar->TemperatureCelsius",
            "FreeText->Money",
            "Filename->DurationDays",
        ]),
        entity_pool=list(entities or []),
        fact_templates=list(facts or []),
        min_calls=min_c, max_calls=max_c,
    )


def build_default_workflows() -> List[WorkflowFamily]:
    """~40 auditovatelných workflow families across the required domains."""
    w: List[WorkflowFamily] = []
    # commerce
    w.append(_wf("commerce.discount_tax_total", "commerce",
                 "determine the final price after discount and tax",
                 ["base_price", "discount_rate", "tax_rate"], "final_price",
                 ["arithmetic.binary", "arithmetic.binary"],
                 _JOIN,
                 entities=["invoice", "cart", "SKU", "checkout"],
                 units=["EUR", "USD", "CZK"],
                 facts=["base price is {base_price}",
                        "discount is {discount_rate} percent",
                        "tax is {tax_rate} percent"]))
    w.append(_wf("commerce.bulk_unit_cost", "commerce",
                 "find the unit cost after a bulk purchase discount",
                 ["total_price", "quantity", "discount_rate"], "unit_cost",
                 ["arithmetic.binary", "arithmetic.binary"], _CHAIN,
                 entities=["pallet", "case pack", "wholesale lot"]))
    w.append(_wf("commerce.margin_check", "commerce",
                 "check whether the sale clears the required margin",
                 ["cost", "price", "min_margin"], "margin_ok",
                 ["arithmetic.binary", "comparison"], _CHAIN,
                 entities=["listing", "offer"]))
    # personal_finance
    w.append(_wf("personal_finance.savings_goal", "personal_finance",
                 "compute how much remains to reach a savings goal",
                 ["goal_amount", "saved_so_far", "monthly_add"], "remaining",
                 ["arithmetic.binary", "arithmetic.binary"], _JOIN,
                 entities=["emergency fund", "vacation fund"],
                 units=["EUR", "USD"]))
    w.append(_wf("personal_finance.loan_payment_share", "personal_finance",
                 "find what share of income a payment consumes",
                 ["payment", "income"], "share",
                 ["arithmetic.binary"], _CHAIN, units=["EUR"]))
    w.append(_wf("personal_finance.interest_growth", "personal_finance",
                 "project the balance after interest is applied",
                 ["principal", "rate"], "new_balance",
                 ["arithmetic.binary"], _CHAIN))
    # measurement
    w.append(_wf("measurement.length_area", "measurement",
                 "compute the area from two measured sides",
                 ["length", "width"], "area",
                 ["arithmetic.binary"], _CHAIN,
                 units=["m", "cm"], entities=["room", "plot", "panel"]))
    w.append(_wf("measurement.scale_convert", "measurement",
                 "rescale a measurement into a different unit factor",
                 ["value", "factor"], "converted",
                 ["arithmetic.binary"], _CHAIN))
    w.append(_wf("measurement.volume_box", "measurement",
                 "find the volume of a rectangular box",
                 ["length", "width", "height"], "volume",
                 ["arithmetic.binary", "arithmetic.binary"], _CHAIN))
    # time_duration
    w.append(_wf("time_duration.shift_overlap", "time_duration",
                 "compute total worked minutes across two shifts",
                 ["shift_a_minutes", "shift_b_minutes"], "total_minutes",
                 ["arithmetic.binary"], _JOIN,
                 units=["minutes"],
                 forbidden=["GenericScalar->DurationDays",
                            "FreeText->DurationMinutes"]))
    w.append(_wf("time_duration.deadline_buffer", "time_duration",
                 "find the remaining buffer before a deadline",
                 ["allotted_hours", "spent_hours"], "buffer_hours",
                 ["arithmetic.binary"], _CHAIN, units=["hours"]))
    w.append(_wf("time_duration.days_to_hours", "time_duration",
                 "convert a multi-day window into hours",
                 ["days", "hours_per_day"], "total_hours",
                 ["arithmetic.binary"], _CHAIN, units=["days", "hours"]))
    # travel_distance
    w.append(_wf("travel_distance.trip_cost", "travel_distance",
                 "estimate trip cost from distance and rate",
                 ["distance_km", "rate_per_km"], "trip_cost",
                 ["arithmetic.binary"], _CHAIN,
                 entities=["route", "detour"], units=["km", "EUR"]))
    w.append(_wf("travel_distance.round_trip", "travel_distance",
                 "compute round-trip distance and travel time",
                 ["one_way_km", "speed_kmh"], "hours",
                 ["arithmetic.binary", "arithmetic.binary"], _CHAIN))
    # inventory
    w.append(_wf("inventory.restock_need", "inventory",
                 "determine how many units to reorder",
                 ["on_hand", "reserved", "reorder_point"], "order_qty",
                 ["arithmetic.binary", "arithmetic.binary"], _JOIN,
                 entities=["SKU", "bin", "warehouse"]))
    w.append(_wf("inventory.shrinkage", "inventory",
                 "compute shrinkage against recorded stock",
                 ["recorded", "counted"], "shrink",
                 ["arithmetic.binary"], _CHAIN))
    # data_summary / statistics
    w.append(_wf("data_summary.mean_threshold", "data_summary",
                 "summarise a series and test it against a threshold",
                 ["values", "threshold"], "above_threshold",
                 ["statistics", "comparison"], _CHAIN,
                 entities=["sensor batch", "daily readings"]))
    w.append(_wf("statistics.range_spread", "statistics",
                 "report the spread between extremes of a series",
                 ["values"], "spread",
                 ["statistics", "arithmetic.binary"], _CHAIN))
    w.append(_wf("statistics.weighted_blend", "statistics",
                 "blend two rates with given weights",
                 ["rate_a", "rate_b", "weight_a"], "blended",
                 ["arithmetic.binary", "arithmetic.binary"], _JOIN))
    # threshold_decision
    w.append(_wf("threshold_decision.pass_fail", "threshold_decision",
                 "decide whether a measured value meets the limit",
                 ["measured", "limit"], "passes",
                 ["comparison"], _CHAIN,
                 entities=["QA sample", "inspection"]))
    w.append(_wf("threshold_decision.band_check", "threshold_decision",
                 "check that a value sits inside an allowed band",
                 ["value", "low", "high"], "in_band",
                 ["comparison", "boolean.logic"], _JOIN))
    # scheduling
    w.append(_wf("scheduling.slot_fit", "scheduling",
                 "see whether a task fits the remaining slot",
                 ["slot_minutes", "task_minutes"], "fits",
                 ["comparison"], _CHAIN))
    w.append(_wf("scheduling.total_load", "scheduling",
                 "sum task loads and compare to capacity",
                 ["load_a", "load_b", "capacity"], "overloaded",
                 ["arithmetic.binary", "comparison"], _JOIN))
    # text_processing
    w.append(_wf("text_processing.label_length", "text_processing",
                 "measure a label and decide if it exceeds a limit",
                 ["label", "max_len"], "too_long",
                 ["string.transform", "comparison"], _CHAIN,
                 entities=["product title", "ticket subject"],
                 forbidden=["Filename->Money", "URL->DurationDays"]))
    w.append(_wf("text_processing.concat_tag", "text_processing",
                 "build a tagged label from parts",
                 ["prefix", "code"], "tagged",
                 ["string.transform"], _CHAIN))
    # file_path
    w.append(_wf("file_path.extension_check", "file_path",
                 "confirm a filename uses an allowed extension",
                 ["filename", "allowed_ext"], "allowed",
                 ["string.parse", "comparison"], _CHAIN,
                 entities=["upload", "export file"],
                 forbidden=["Filename->DurationDays", "FileExtension->Money"]))
    w.append(_wf("file_path.stem_join", "file_path",
                 "compose a new filename from stem and extension",
                 ["stem", "extension"], "filename",
                 ["string.transform"], _CHAIN))
    # url_processing
    w.append(_wf("url_processing.domain_match", "url_processing",
                 "check whether a URL belongs to an allowed domain",
                 ["url", "allowed_domain"], "matches",
                 ["string.parse", "comparison"], _CHAIN,
                 forbidden=["URL->Money", "Domain->DurationHours"]))
    w.append(_wf("url_processing.host_length", "url_processing",
                 "measure the host part of a URL",
                 ["url"], "host_len",
                 ["string.parse", "string.transform"], _CHAIN))
    # list_processing
    w.append(_wf("list_processing.filter_count", "list_processing",
                 "count how many entries survive a filter",
                 ["values", "threshold"], "kept",
                 ["sequence.filter", "sequence.reduce"], _CHAIN))
    w.append(_wf("list_processing.index_lookup", "list_processing",
                 "read a ranked entry and adjust it",
                 ["values", "rank", "delta"], "adjusted",
                 ["sequence.index", "arithmetic.binary"], _CHAIN))
    # geometry
    w.append(_wf("geometry.triangle_perimeter", "geometry",
                 "compute the perimeter of a triangle",
                 ["side_a", "side_b", "side_c"], "perimeter",
                 ["arithmetic.binary", "arithmetic.binary"], _JOIN,
                 units=["m", "cm"]))
    w.append(_wf("geometry.circle_area", "geometry",
                 "find the area of a circle from its radius",
                 ["radius"], "area",
                 ["geometry"], _CHAIN, units=["m"]))
    # quality_control
    w.append(_wf("quality_control.defect_rate", "quality_control",
                 "compute the defect rate and test the quality gate",
                 ["defects", "inspected", "max_rate"], "gate_ok",
                 ["arithmetic.binary", "comparison"], _JOIN,
                 entities=["lot", "batch"]))
    w.append(_wf("quality_control.tolerance", "quality_control",
                 "check absolute deviation against tolerance",
                 ["measured", "nominal", "tolerance"], "within",
                 ["arithmetic.binary", "comparison"], _CHAIN))
    # resource_allocation
    w.append(_wf("resource_allocation.split_budget", "resource_allocation",
                 "allocate a budget across two teams by share",
                 ["budget", "share_a"], "team_a_amount",
                 ["arithmetic.binary"], _CHAIN, units=["EUR"]))
    w.append(_wf("resource_allocation.capacity_left", "resource_allocation",
                 "compute remaining capacity after assignments",
                 ["capacity", "used_a", "used_b"], "remaining",
                 ["arithmetic.binary", "arithmetic.binary"], _JOIN))
    # rates_and_ratios
    w.append(_wf("rates_and_ratios.efficiency", "rates_and_ratios",
                 "compute efficiency as output over input",
                 ["output", "input"], "efficiency",
                 ["arithmetic.binary"], _CHAIN))
    w.append(_wf("rates_and_ratios.mix_ratio", "rates_and_ratios",
                 "find the mix ratio between two components",
                 ["part_a", "part_b"], "ratio",
                 ["arithmetic.binary"], _CHAIN))
    w.append(_wf("rates_and_ratios.percent_change", "rates_and_ratios",
                 "report the percent change between two readings",
                 ["before", "after"], "pct_change",
                 ["arithmetic.binary", "arithmetic.binary"], _CHAIN))
    return w


_REGISTRY: Optional[List[WorkflowFamily]] = None


def get_workflows() -> List[WorkflowFamily]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = build_default_workflows()
    return _REGISTRY


def workflows_by_id() -> Dict[str, WorkflowFamily]:
    return {w.workflow_id: w for w in get_workflows()}


def workflows_for_domain(domain: str) -> List[WorkflowFamily]:
    return [w for w in get_workflows() if w.domain == domain]


def pick_workflow(rng, *, domain: Optional[str] = None,
                  n_calls: Optional[int] = None,
                  pattern: Optional[str] = None) -> WorkflowFamily:
    pool = get_workflows()
    if domain:
        pool = [w for w in pool if w.domain == domain] or pool
    if n_calls is not None:
        pool = [w for w in pool if w.min_calls <= n_calls <= w.max_calls] or pool
    if pattern:
        pool = [w for w in pool if pattern in w.allowed_structural_patterns] or pool
    return pool[rng.randrange(len(pool))]


def export_registry(path: Path) -> Dict[str, Any]:
    rows = [w.as_dict() for w in get_workflows()]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "n_workflows": len(rows),
        "domains": sorted({w["domain"] for w in rows}),
        "workflows": rows,
        "registry_hash": sha256_obj(rows)[:16],
    }
    write_json(path, payload)
    return payload


def registry_hash() -> str:
    return short_hash([w.as_dict() for w in get_workflows()])
