"""Banding: turning a derived quantity into a tier, a grade or a label.

The category sinks are deliberately the calibratable ones (``three_bands`` and
``ratio_band``), so the band distribution can be balanced the way boolean
answers are. The remaining plans keep the same subject matter but stop
somewhere else -- at a verdict, at a count of triggered rules, at the printed
label -- which is what stops the family from being nine copies of one shape.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── supplier tiering ────────────────────────────────────────────────
    supplier_roles = (
        R("on_time_deliveries", "count_items", "deliveries that arrived on time"),
        R("total_deliveries", "count_items", "deliveries in the review window"),
        R("dispute_rate", "percent_share",
          "share of the invoices that ended in a dispute"),
        R("low_cut", "cut_low", "boundary between the bronze and the silver tier"),
        R("high_cut", "cut_high", "boundary between the silver and the gold tier"),
        R("tier_floor", "threshold_percent",
          "reliability the supplier's current tier requires"),
        R("penalty_limit", "threshold_percent",
          "reliability points the disputes may cost"),
        R("order_values", "list_prices", "value of every order in the window"),
    )
    out.append(Blueprint(
        workflow_id="classify.supplier_tier",
        domain="classification",
        natural_user_goal="place a supplier in the tier its record justifies",
        target_description="the supplier tier or the review verdict",
        value_generator_id="classify.supplier",
        query_asset_family="supplier_record",
        hard_distractor_families=("rates", "classification"),
        boolean_balancing_strategy="threshold_band",
        entity_family="procurement",
        plans=(
            Plan("tier.v3", supplier_roles[:5],
                 (S("n1", "rates.share_percent",
                    ("on_time_deliveries", "total_deliveries"),
                    "raw reliability of the supplier"),
                  S("n2", "rates.decrease_by_percent", ("@n1", "dispute_rate")),
                  S("n3", "classification.three_bands",
                    ("@n2", "low_cut", "high_cut"))),
                 "n3", intent="reliability_tier"),
            Plan("tier.v6",
                 (supplier_roles[0], supplier_roles[1], supplier_roles[2],
                  supplier_roles[5], supplier_roles[6]),
                 (S("n1", "rates.share_percent",
                    ("on_time_deliveries", "total_deliveries"),
                    "raw reliability, read again by the penalty step"),
                  S("n2", "rates.decrease_by_percent", ("@n1", "dispute_rate")),
                  S("n3", "arithmetic.subtract", ("@n1", "@n2"),
                    "reliability points lost to disputes"),
                  S("n4", "comparison.at_least", ("@n2", "tier_floor")),
                  S("n5", "comparison.at_least", ("@n3", "penalty_limit")),
                  S("n6", "boolean.xor", ("@n4", "@n5"),
                    "the two signals contradict each other, so a human decides")),
                 "n6", intent="contradictory_record"),
            Plan("tier.v8",
                 (supplier_roles[0], supplier_roles[1], supplier_roles[2],
                  supplier_roles[7], supplier_roles[3], supplier_roles[4]),
                 (S("n1", "rates.share_percent",
                    ("on_time_deliveries", "total_deliveries"),
                    "raw reliability, read again six steps later"),
                  S("n2", "list.reduce_sum", ("order_values",)),
                  S("n3", "list.reduce_max", ("order_values",)),
                  S("n4", "rates.share_percent", ("@n3", "@n2"),
                    "how concentrated the spend is on one order"),
                  S("n5", "rates.decrease_by_percent", ("@n1", "dispute_rate")),
                  S("n6", "arithmetic.subtract", ("@n5", "@n4"),
                    "reliability once the concentration risk is priced in"),
                  S("n7", "rates.percent_change", ("@n1", "@n6"),
                    "how far the risks moved the raw figure"),
                  S("n8", "classification.three_bands",
                    ("@n7", "low_cut", "high_cut"))),
                 "n8", intent="risk_adjusted_tier"),
        )))

    # ── building energy rating ──────────────────────────────────────────
    energy_roles = (
        R("consumption_readings", "list_readings",
          "metered consumption of every week"),
        R("width_m", "length_m", "width of the floor plate"),
        R("depth_m", "length_m", "depth of the floor plate"),
        R("intensity_limit", "threshold_value",
          "consumption per square metre the rating allows"),
        R("balance_cut", "threshold_ratio",
          "ratio at which a consumption profile counts as flat"),
        R("peak_limit", "threshold_value", "peak intensity the rating allows"),
        R("occupants", "count_people", "people working in the building"),
        R("headcount_floor", "threshold_count",
          "occupancy the rating band assumes"),
    )
    out.append(Blueprint(
        workflow_id="classify.energy_band",
        domain="classification",
        natural_user_goal="rate a building on how evenly it consumes energy",
        target_description="the energy band or the rating verdict",
        value_generator_id="classify.energy",
        query_asset_family="building_meter",
        hard_distractor_families=("geometry", "list"),
        boolean_balancing_strategy="threshold_band",
        entity_family="facilities",
        plans=(
            Plan("band.v4", energy_roles[:4],
                 (S("n1", "list.reduce_sum", ("consumption_readings",)),
                  S("n2", "geometry.rectangle_area", ("width_m", "depth_m")),
                  S("n3", "arithmetic.divide", ("@n1", "@n2"),
                    "consumption per square metre"),
                  S("n4", "comparison.at_least", ("@n3", "intensity_limit"))),
                 "n4", intent="intensity_breach"),
            Plan("band.v7",
                 (energy_roles[0], energy_roles[1], energy_roles[2],
                  energy_roles[4]),
                 (S("n1", "list.reduce_sum", ("consumption_readings",)),
                  S("n2", "geometry.rectangle_area", ("width_m", "depth_m"),
                    "floor area, read again by the peak step"),
                  S("n3", "arithmetic.divide", ("@n1", "@n2")),
                  S("n4", "list.reduce_max", ("consumption_readings",)),
                  S("n5", "arithmetic.divide", ("@n4", "@n2")),
                  S("n6", "rates.ratio_of", ("@n3", "@n5"),
                    "how close the average week is to the worst one"),
                  S("n7", "classification.ratio_band", ("@n6", "balance_cut"))),
                 "n7", intent="profile_flatness_band"),
            Plan("band.v9",
                 (energy_roles[0], energy_roles[1], energy_roles[2],
                  energy_roles[3], energy_roles[5], energy_roles[6],
                  energy_roles[7]),
                 (S("n1", "list.reduce_sum", ("consumption_readings",)),
                  S("n2", "geometry.rectangle_area", ("width_m", "depth_m"),
                    "floor area, read again by the peak step"),
                  S("n3", "arithmetic.divide", ("@n1", "@n2")),
                  S("n4", "list.reduce_max", ("consumption_readings",)),
                  S("n5", "arithmetic.divide", ("@n4", "@n2")),
                  S("n6", "comparison.at_least", ("@n3", "intensity_limit")),
                  S("n7", "comparison.at_least", ("@n5", "peak_limit")),
                  S("n8", "comparison.at_least",
                    ("occupants", "headcount_floor")),
                  S("n9", "decision.majority", ("@n6", "@n7", "@n8"))),
                 "n9", intent="rating_downgrade"),
        )))

    # ── incident grading ────────────────────────────────────────────────
    incident_roles = (
        R("incident_score", "score_points",
          "severity score assigned to the incident"),
        R("downtime_hours", "duration_hours",
          "hours the service was unavailable"),
        R("affected_users", "count_people", "people who lost the service"),
        R("unit_word", "unit_word", "unit the report prints the loss in"),
        R("downtime_limit", "threshold_hours",
          "lost user-hours the policy tolerates"),
        R("effort_limit", "threshold_hours",
          "lost user-hours per severity point that are tolerated"),
        R("users_limit", "threshold_count",
          "affected people that force a post-mortem"),
        R("places", "places", "decimals the report prints"),
    )
    out.append(Blueprint(
        workflow_id="classify.incident_grade",
        domain="classification",
        natural_user_goal="grade an outage and write the line the report prints",
        target_description="the graded incident line or the number of rules hit",
        value_generator_id="classify.incident",
        query_asset_family="incident_report",
        hard_distractor_families=("format", "classification"),
        entity_family="support",
        plans=(
            Plan("grade.v2",
                 (incident_roles[1], incident_roles[2], incident_roles[3]),
                 (S("n1", "arithmetic.multiply",
                    ("downtime_hours", "affected_users"),
                    "user-hours the outage destroyed"),
                  S("n2", "format.with_unit", ("@n1", "unit_word"))),
                 "n2", intent="loss_with_unit"),
            Plan("grade.v6",
                 (incident_roles[1], incident_roles[2], incident_roles[0],
                  incident_roles[4], incident_roles[5], incident_roles[6]),
                 (S("n1", "arithmetic.multiply",
                    ("downtime_hours", "affected_users"),
                    "lost user-hours, read again by the absolute check"),
                  S("n2", "arithmetic.divide", ("@n1", "incident_score"),
                    "lost user-hours per point of severity"),
                  S("n3", "comparison.at_least", ("@n1", "downtime_limit")),
                  S("n4", "comparison.at_least", ("@n2", "effort_limit")),
                  S("n5", "comparison.at_least",
                    ("affected_users", "users_limit")),
                  S("n6", "decision.count_true", ("@n3", "@n4", "@n5"))),
                 "n6", intent="triggered_rule_count"),
            Plan("grade.v8",
                 (incident_roles[1], incident_roles[2], incident_roles[0],
                  incident_roles[7]),
                 (S("n1", "arithmetic.multiply",
                    ("downtime_hours", "affected_users"),
                    "lost user-hours, read again five steps later"),
                  S("n2", "classification.band", ("incident_score",)),
                  S("n3", "string.normalize_upper", ("@n2",)),
                  S("n4", "duration.convert_hours_minutes", ("@n1",)),
                  S("n5", "format.fixed", ("@n4", "places")),
                  S("n6", "arithmetic.divide", ("@n1", "incident_score")),
                  S("n7", "format.tag", ("@n3", "@n6"),
                    "grade and severity-weighted loss as one code"),
                  S("n8", "string.concat", ("@n7", "@n5"))),
                 "n8", intent="printed_incident_line"),
        )))

    return out
