"""Measurement workflows: reading series, calibration drift, instrument logs.

The family covers what a metrology desk does with raw readings: correct them by
a calibration offset, describe their dispersion, weigh a balance against a
certified reference, and turn a log into the figures a QC sheet needs. The
plans differ in how much of that chain the user states. The short variants ask
for one derived figure, so they stay flat; the long ones need the corrected
series twice, once for its centre and once for its spread, and that reuse is
what produces the diamonds and the aggregations built on aggregations.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── reading spread audit ────────────────────────────────────────────
    spread_roles = (
        R("readings", "list_readings", "readings the probe logged during the run"),
        R("calibration_offset", "temp_c",
          "offset the calibration sheet adds to every reading"),
        R("dispersion_limit", "threshold_percent",
          "relative dispersion the instrument is allowed to show"),
        R("mean_floor", "threshold_value", "lowest mean reading the run must reach"),
        R("spread_limit", "threshold_value",
          "largest reading-to-reading spread the run may show"),
    )
    out.append(Blueprint(
        workflow_id="measurement.reading_spread_audit",
        domain="measurement",
        natural_user_goal=("judge how stable a probe was during a run once the "
                           "calibration offset is taken into account"),
        target_description="the relative dispersion or the stability verdict",
        value_generator_id="measurement.sensor_series",
        query_asset_family="sensor_log",
        hard_distractor_families=("statistics", "list"),
        boolean_balancing_strategy="threshold_band",
        entity_family="lab",
        plans=(
            Plan("spread.v3", spread_roles[:1],
                 (S("n1", "statistics.mean", ("readings",), "centre of the run"),
                  S("n2", "statistics.stdev", ("readings",), "dispersion of the run"),
                  S("n3", "rates.share_percent", ("@n2", "@n1"),
                    "dispersion relative to the centre")),
                 "n3", intent="relative_dispersion"),
            Plan("spread.v6", spread_roles[:3],
                 (S("n1", "list.map_offset", ("readings", "calibration_offset"),
                    "corrected series, needed for both the centre and the spread"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.stdev", ("@n1",)),
                  S("n4", "arithmetic.add", ("@n2", "@n3"),
                    "upper control point of the corrected run"),
                  S("n5", "rates.share_percent", ("@n4", "@n2"),
                    "control point as a share of the corrected centre"),
                  S("n6", "comparison.at_least", ("@n5", "dispersion_limit"))),
                 "n6", intent="corrected_dispersion_verdict"),
            Plan("spread.v9", spread_roles,
                 (S("n1", "list.map_offset", ("readings", "calibration_offset")),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.stdev", ("@n1",)),
                  S("n4", "statistics.range", ("@n1",)),
                  S("n5", "rates.share_percent", ("@n3", "@n2")),
                  S("n6", "comparison.at_least", ("@n5", "dispersion_limit")),
                  S("n7", "comparison.at_least", ("@n2", "mean_floor")),
                  S("n8", "comparison.at_least", ("@n4", "spread_limit")),
                  S("n9", "decision.majority", ("@n6", "@n7", "@n8"),
                    "the run passes when most of the checks hold")),
                 "n9", intent="three_check_stability"),
        )))

    # ── calibration drift check ─────────────────────────────────────────
    drift_roles = (
        R("measured_mass", "mass_kg", "mass the bench balance reported"),
        R("reference_mass", "mass_kg", "mass of the certified reference weight"),
        R("repeat_count", "count_small", "times the weighing was repeated"),
        R("deviation_limit", "threshold_percent",
          "relative deviation the balance is allowed to show"),
        R("nominal_drift", "threshold_value",
          "drift the service sheet expects after the repeats"),
        R("drift_allowance", "tolerance_value",
          "allowance around the expected drift"),
    )
    out.append(Blueprint(
        workflow_id="measurement.calibration_drift_check",
        domain="measurement",
        natural_user_goal=("work out how far a balance sits from its reference "
                           "weight and whether that drift is still acceptable"),
        target_description="the relative drift or the calibration verdict",
        value_generator_id="measurement.balance_check",
        query_asset_family="calibration_record",
        hard_distractor_families=("arithmetic", "validation"),
        boolean_balancing_strategy="threshold_band",
        entity_family="engineering",
        plans=(
            Plan("drift.v2", drift_roles[:2],
                 (S("n1", "arithmetic.abs_difference",
                    ("measured_mass", "reference_mass"), "absolute drift"),
                  S("n2", "rates.share_percent", ("@n1", "reference_mass"),
                    "drift against the reference")),
                 "n2", intent="relative_drift"),
            Plan("drift.v4", drift_roles[:2] + drift_roles[3:4],
                 (S("n1", "arithmetic.abs_difference",
                    ("measured_mass", "reference_mass")),
                  S("n2", "statistics.average_two",
                    ("measured_mass", "reference_mass"), "working mass of the pair"),
                  S("n3", "rates.share_percent", ("@n1", "@n2")),
                  S("n4", "comparison.at_least", ("@n3", "deviation_limit"))),
                 "n4", intent="drift_verdict"),
            Plan("drift.v7", drift_roles,
                 (S("n1", "arithmetic.abs_difference",
                    ("measured_mass", "reference_mass"),
                    "single-weighing drift, read by both branches"),
                  S("n2", "statistics.average_two",
                    ("measured_mass", "reference_mass")),
                  S("n3", "rates.share_percent", ("@n1", "@n2")),
                  S("n4", "arithmetic.multiply", ("@n1", "repeat_count"),
                    "drift accumulated over the repeats"),
                  S("n5", "validation.tolerance",
                    ("@n4", "nominal_drift", "drift_allowance")),
                  S("n6", "comparison.at_least", ("@n3", "deviation_limit")),
                  S("n7", "boolean.and", ("@n5", "@n6"),
                    "accumulated drift and relative drift both have to hold")),
                 "n7", intent="two_condition_calibration"),
        )))

    # ── instrument log digest (records -> series, record, band) ─────────
    log_roles = (
        R("log_rows", "record_list", "rows of the instrument log"),
        R("reading_field", "field_name", "column of the log holding the reading"),
        R("places", "places", "decimals the tidied series keeps"),
        R("station_label", "text_label", "label of the measuring station"),
        R("cut_low", "cut_low", "exceedance below which the run counts as minor"),
        R("cut_high", "cut_high", "exceedance above which the run counts as severe"),
    )
    out.append(Blueprint(
        workflow_id="measurement.instrument_log_digest",
        domain="measurement",
        natural_user_goal=("turn an instrument log into what a QC sheet needs: a "
                           "tidy series, an action limit, a severity band"),
        target_description=("the tidied series, the station action limit or the "
                            "severity band"),
        value_generator_id="measurement.instrument_log",
        query_asset_family="instrument_log",
        hard_distractor_families=("record", "statistics"),
        entity_family="lab",
        plans=(
            Plan("log.v3", log_roles[:3],
                 (S("n1", "record.project", ("log_rows", "reading_field"),
                    "readings pulled out of the log"),
                  S("n2", "list.map_sort_asc", ("@n1",)),
                  S("n3", "list.map_round", ("@n2", "places"))),
                 "n3", intent="tidied_series"),
            Plan("log.v5", log_roles[:2] + log_roles[3:4],
                 (S("n1", "record.project", ("log_rows", "reading_field")),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.stdev", ("@n1",)),
                  S("n4", "arithmetic.add", ("@n2", "@n3"), "action limit"),
                  S("n5", "record.build", ("station_label", "@n4"),
                    "action limit filed under the station")),
                 "n5", intent="station_action_limit"),
            Plan("log.v8", log_roles[:2] + log_roles[4:],
                 (S("n1", "record.project", ("log_rows", "reading_field")),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.stdev", ("@n1",)),
                  S("n4", "list.reduce_max", ("@n1",), "worst reading in the log"),
                  S("n5", "arithmetic.add", ("@n2", "@n3"), "action limit"),
                  S("n6", "arithmetic.abs_difference", ("@n4", "@n5"),
                    "how far the worst reading sits from the limit"),
                  S("n7", "rates.share_percent", ("@n6", "@n2"),
                    "that exceedance against the centre of the log"),
                  S("n8", "classification.three_bands",
                    ("@n7", "cut_low", "cut_high"))),
                 "n8", intent="exceedance_band"),
        )))

    # ── batch conformance report (readings -> counts and text) ──────────
    report_roles = (
        R("readings", "list_readings", "readings taken across the batch"),
        R("unit_word", "unit_word", "unit the batch is measured in"),
        R("separator", "separator", "separator the report puts between figures"),
    )
    out.append(Blueprint(
        workflow_id="measurement.batch_conformance_report",
        domain="measurement",
        natural_user_goal=("report how a measured batch sits against the control "
                           "band its own readings imply"),
        target_description=("the printed action limit, the conforming count or the "
                            "report line"),
        value_generator_id="measurement.batch_series",
        query_asset_family="batch_sheet",
        hard_distractor_families=("format", "list"),
        entity_family="engineering",
        plans=(
            Plan("report.v4", report_roles[:2],
                 (S("n1", "statistics.mean", ("readings",)),
                  S("n2", "statistics.stdev", ("readings",)),
                  S("n3", "arithmetic.add", ("@n1", "@n2"), "upper action limit"),
                  S("n4", "format.with_unit", ("@n3", "unit_word"))),
                 "n4", intent="printed_action_limit"),
            Plan("report.v7", report_roles[:1],
                 (S("n1", "statistics.mean", ("readings",)),
                  S("n2", "statistics.stdev", ("readings",)),
                  S("n3", "arithmetic.add", ("@n1", "@n2"), "upper action limit"),
                  S("n4", "arithmetic.subtract", ("@n1", "@n2"),
                    "lower action limit"),
                  S("n5", "list.reduce_count_above", ("readings", "@n3"),
                    "breaches of the upper limit"),
                  S("n6", "list.reduce_count_above", ("readings", "@n4"),
                    "readings above the lower limit"),
                  S("n7", "rates.share_percent", ("@n5", "@n6"),
                    "share of the in-band readings that breached the upper limit")),
                 "n7", intent="escalation_share"),
            Plan("report.v10", report_roles,
                 (S("n1", "statistics.mean", ("readings",),
                    "centre of the batch, quoted again in the report line"),
                  S("n2", "statistics.stdev", ("readings",)),
                  S("n3", "arithmetic.add", ("@n1", "@n2"), "upper action limit"),
                  S("n4", "arithmetic.subtract", ("@n1", "@n2"),
                    "lower action limit"),
                  S("n5", "list.reduce_count_above", ("readings", "@n3")),
                  S("n6", "list.reduce_count_above", ("readings", "@n4")),
                  S("n7", "rates.share_percent", ("@n5", "@n6"),
                    "how much of the in-band traffic breached the upper limit"),
                  S("n8", "list.build", ("@n5", "@n7", "@n1")),
                  S("n9", "list.combine_join", ("@n8", "separator")),
                  S("n10", "string.concat", ("@n9", "unit_word"),
                    "report line closed with the unit of the batch")),
                 "n10", intent="control_band_report_line"),
        )))

    return out
