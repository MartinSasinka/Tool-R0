"""Quality control: batch conformance, defect screening, inspection reporting.

The three families deliberately answer different questions with the same raw
material. Conformance plans reduce a measurement series to dispersion figures
and end in a calibrated predicate; the defect screen turns counts and rework
time into a rework-load band; the reporting family never produces a verdict at
all but a list, a label and a filtered scorecard, which is what keeps the
domain from collapsing into "compute a number, compare it".
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── batch conformance ───────────────────────────────────────────────
    batch_roles = (
        R("measurements", "list_readings", "measurements taken from the batch"),
        R("spread_limit", "threshold_value", "dispersion the process allows"),
        R("variation_limit", "threshold_value",
          "relative variation the process allows"),
        R("places", "places", "decimals the gauge reports"),
        R("deviation_limit", "threshold_value",
          "gap above the batch mean a single part may show"),
        R("ceiling", "threshold_value", "value no part may exceed"),
    )
    out.append(Blueprint(
        workflow_id="quality.batch_conformance",
        domain="quality_control",
        natural_user_goal="judge whether a production batch is under control",
        target_description="the conformance verdict",
        value_generator_id="quality.batch",
        query_asset_family="measurement_batch",
        hard_distractor_families=("statistics", "validation"),
        boolean_balancing_strategy="threshold_band",
        entity_family="manufacturing",
        plans=(
            Plan("conform.v2", batch_roles[:2],
                 (S("n1", "statistics.stdev", ("measurements",),
                    "dispersion of the batch"),
                  S("n2", "comparison.at_least", ("@n1", "spread_limit"))),
                 "n2", intent="dispersion_breach"),
            Plan("conform.v5", (batch_roles[0], batch_roles[2]),
                 (S("n1", "statistics.mean", ("measurements",)),
                  S("n2", "statistics.stdev", ("measurements",)),
                  S("n3", "arithmetic.divide", ("@n2", "@n1"),
                    "variation relative to the batch level"),
                  S("n4", "comparison.at_least", ("@n3", "variation_limit")),
                  S("n5", "boolean.not", ("@n4",), "the process is stable")),
                 "n5", intent="process_stability"),
            Plan("conform.v10",
                 (batch_roles[0], batch_roles[3], batch_roles[4],
                  batch_roles[2], batch_roles[5]),
                 (S("n1", "list.map_round", ("measurements", "places"),
                    "gauge-resolution series, read by four later steps"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.stdev", ("@n1",)),
                  S("n4", "list.reduce_max", ("@n1",)),
                  S("n5", "arithmetic.subtract", ("@n4", "@n2"),
                    "how far the worst part sits above the mean"),
                  S("n6", "arithmetic.divide", ("@n3", "@n2")),
                  S("n7", "comparison.at_least", ("@n5", "deviation_limit")),
                  S("n8", "comparison.at_least", ("@n6", "variation_limit")),
                  S("n9", "validation.list_limit", ("@n1", "ceiling")),
                  S("n10", "decision.all_of", ("@n7", "@n8", "@n9"))),
                 "n10", intent="full_conformance_review"),
        )))

    # ── defect screening ────────────────────────────────────────────────
    defect_roles = (
        R("defect_count", "count_small", "parts rejected in the lot"),
        R("inspected_count", "count_items", "parts inspected in the lot"),
        R("defect_limit", "threshold_percent", "defect rate the customer accepts"),
        R("rework_hours", "duration_hours", "hours spent reworking the lot"),
        R("rework_limit", "threshold_hours", "rework hours a single defect may cost"),
        R("sample_floor", "threshold_count", "parts the sampling plan requires"),
        R("baseline_percent", "percent_share", "defect rate of the previous lot"),
        R("low_cut", "cut_low", "boundary between a light and a normal rework load"),
        R("high_cut", "cut_high", "boundary between a normal and a heavy rework load"),
    )
    out.append(Blueprint(
        workflow_id="quality.defect_screen",
        domain="quality_control",
        natural_user_goal="screen an inspection lot for its defect and rework load",
        target_description="the lot verdict or the rework load band",
        value_generator_id="quality.defect",
        query_asset_family="inspection_lot",
        hard_distractor_families=("rates", "duration"),
        boolean_balancing_strategy="threshold_band",
        entity_family="manufacturing",
        plans=(
            Plan("screen.v3", defect_roles[:3],
                 (S("n1", "rates.share_percent",
                    ("defect_count", "inspected_count"), "defect rate of the lot"),
                  S("n2", "comparison.at_least", ("@n1", "defect_limit")),
                  S("n3", "boolean.not", ("@n2",), "the lot passes")),
                 "n3", intent="lot_pass"),
            Plan("screen.v6", defect_roles[:6],
                 (S("n1", "rates.share_percent",
                    ("defect_count", "inspected_count")),
                  S("n2", "comparison.at_least", ("@n1", "defect_limit")),
                  S("n3", "arithmetic.divide", ("rework_hours", "defect_count"),
                    "rework time each defect costs"),
                  S("n4", "comparison.at_least", ("@n3", "rework_limit")),
                  S("n5", "comparison.at_least",
                    ("inspected_count", "sample_floor")),
                  S("n6", "decision.majority", ("@n2", "@n4", "@n5"))),
                 "n6", intent="majority_lot_concern"),
            Plan("screen.v8",
                 (defect_roles[3], defect_roles[1], defect_roles[0],
                  defect_roles[6], defect_roles[7], defect_roles[8]),
                 (S("n1", "duration.convert_hours_minutes", ("rework_hours",),
                    "rework budget in minutes, read again two steps later"),
                  S("n2", "arithmetic.divide", ("@n1", "inspected_count")),
                  S("n3", "rates.share_percent",
                    ("defect_count", "inspected_count")),
                  S("n4", "arithmetic.divide", ("@n1", "defect_count")),
                  S("n5", "statistics.average_two", ("@n2", "@n4"),
                    "typical rework minutes the lot generates"),
                  S("n6", "rates.percent_change", ("baseline_percent", "@n3"),
                    "movement of the defect rate against the previous lot"),
                  S("n7", "rates.increase_by_percent", ("@n5", "@n6"),
                    "rework load once the trend is carried forward"),
                  S("n8", "classification.three_bands",
                    ("@n7", "low_cut", "high_cut"))),
                 "n8", intent="rework_load_band"),
        )))

    # ── inspection reporting ────────────────────────────────────────────
    report_roles = (
        R("lots", "record_list", "inspection lots recorded for the shift"),
        R("amount_field", "field_name", "numeric column the report reads"),
        R("label_field", "text_field_name", "text column the report reads"),
        R("site_scores", "mapping_amounts", "quality score held for every site"),
    )
    out.append(Blueprint(
        workflow_id="quality.inspection_records",
        domain="quality_control",
        natural_user_goal=("summarise the inspection lots of a shift and the "
                           "site scorecard behind them"),
        target_description="the shortfall series, the leading lot or the scorecard",
        value_generator_id="quality.records",
        query_asset_family="inspection_register",
        hard_distractor_families=("record", "dictionary"),
        entity_family="manufacturing",
        plans=(
            Plan("report.v4", report_roles[:2],
                 (S("n1", "record.project", ("lots", "amount_field"),
                    "the column itself, read again by the offset step"),
                  S("n2", "list.reduce_max", ("@n1",)),
                  S("n3", "arithmetic.negate", ("@n2",)),
                  S("n4", "list.map_offset", ("@n1", "@n3"),
                    "how far every lot sits below the best one")),
                 "n4", intent="shortfall_series"),
            Plan("report.v6", report_roles[:3],
                 (S("n1", "record.project", ("lots", "amount_field"),
                    "numeric column, read again four steps later"),
                  S("n2", "record.project_text", ("lots", "label_field")),
                  S("n3", "list.index_of_max", ("@n1",)),
                  S("n4", "list.index_text", ("@n2", "@n3"),
                    "label sitting at the winning position"),
                  S("n5", "list.reduce_max", ("@n1",)),
                  S("n6", "format.tag", ("@n4", "@n5"))),
                 "n6", intent="leading_lot_tag"),
            Plan("report.v7", report_roles[3:],
                 (S("n1", "dictionary.values", ("site_scores",),
                    "scores as a series, read twice"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.stdev", ("@n1",)),
                  S("n4", "dictionary.aggregate_max", ("site_scores",)),
                  S("n5", "arithmetic.subtract", ("@n4", "@n3"),
                    "one dispersion step below the best site"),
                  # the cut stays strictly under the peak, so the filter can
                  # never empty the scorecard
                  S("n6", "statistics.average_two", ("@n5", "@n2")),
                  S("n7", "dictionary.aggregate_filter", ("site_scores", "@n6"))),
                 "n7", intent="leading_site_scorecard"),
        )))

    return out
