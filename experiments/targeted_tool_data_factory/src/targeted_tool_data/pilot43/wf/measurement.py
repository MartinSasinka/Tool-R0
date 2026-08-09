"""Measurement workflows: tank levels, batch weighing, temperature logs.

Physical quantities are carried in their own units the whole way through, so a
level in litres never reaches a mass parameter; where a different unit is
genuinely needed the plan says so with an explicit conversion step.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── tank level ──────────────────────────────────────────────────────
    tank_volume_l = R("tank_volume_l", "volume_l",
                      "litres the tank was holding at the start of the shift")
    drawn_ml = R("drawn_ml", "volume_ml", "millilitres drawn off during the shift")
    readings = R("readings", "list_readings",
                 "level readings taken through the shift")
    calibration_offset = R("calibration_offset", "generic_value",
                           "correction the calibration sheet asks for")
    top_count = R("top_count", "count_small", "readings the report shows")
    places = R("places", "places", "decimals the readings are rounded to")
    capacity_limit = R("capacity_limit", "threshold_value",
                       "litres that must still be left in the tank")
    reading_cap = R("reading_cap", "threshold_value",
                    "reading that counts as an overfill")
    spread_limit = R("spread_limit", "threshold_percent",
                     "how far above average the peak may sit")
    unit_word = R("unit_word", "unit_word", "unit the log is written in")

    out.append(Blueprint(
        workflow_id="measurement.tank_level_audit",
        domain="measurement",
        natural_user_goal=("check how much liquid is left in a tank after a "
                           "shift and whether the level log looks sound"),
        target_description="the remaining volume, the tidied log or the level verdict",
        value_generator_id="measurement.tank",
        query_asset_family="tank_log",
        hard_distractor_families=("unit_conversion", "list"),
        boolean_balancing_strategy="calibrate_tank_level_limit",
        entity_family="plant",
        plans=(
            Plan("tank.v3", (tank_volume_l, drawn_ml, capacity_limit),
                 (S("n1", "unit_conversion.volume_ml_l", ("drawn_ml",),
                    "the draw-off in the tank's own unit"),
                  S("n2", "arithmetic.subtract", ("tank_volume_l", "@n1")),
                  S("n3", "comparison.at_least", ("@n2", "capacity_limit"))),
                 "n3", intent="remaining_level_verdict"),
            Plan("tank.v4", (readings, calibration_offset, places, top_count),
                 (S("n1", "list.map_offset", ("readings",
                                              "calibration_offset"),
                    "readings after the calibration correction"),
                  S("n2", "list.map_round", ("@n1", "places")),
                  S("n3", "list.map_sort", ("@n2",)),
                  S("n4", "list.slice_first", ("@n3", "top_count"),
                    "the highest readings for the report")),
                 "n4", intent="top_corrected_readings"),
            Plan("tank.v5", (tank_volume_l, drawn_ml, unit_word),
                 (S("n1", "unit_conversion.volume_ml_l", ("drawn_ml",)),
                  S("n2", "arithmetic.subtract", ("tank_volume_l", "@n1")),
                  S("n3", "unit_conversion.volume_l_ml", ("@n2",)),
                  S("n4", "rounding.to_int", ("@n3",)),
                  S("n5", "format.with_unit", ("@n4", "unit_word"),
                    "the figure written on the tank card")),
                 "n5", intent="remaining_level_label"),
            Plan("tank.v7", (readings, reading_cap, spread_limit),
                 (S("n1", "statistics.mean", ("readings",),
                    "average level over the shift"),
                  S("n2", "list.reduce_max", ("readings",), "the peak level"),
                  S("n3", "arithmetic.subtract", ("@n2", "@n1"),
                    "how far the peak sat above the average"),
                  S("n4", "rates.share_percent", ("@n3", "@n1")),
                  S("n5", "validation.list_limit", ("readings", "reading_cap"),
                    "did any single reading breach the cap"),
                  S("n6", "comparison.at_least", ("@n4", "spread_limit")),
                  S("n7", "boolean.and", ("@n6", "@n5"))),
                 "n7", intent="level_stability_verdict"),
        )))

    # ── batch weighing ──────────────────────────────────────────────────
    sample_mass_g = R("sample_mass_g", "mass_g", "grams one sample weighs")
    sample_count = R("sample_count", "count_items", "samples packed in the crate")
    tare_mass_kg = R("tare_mass_kg", "mass_kg", "kilograms the empty crate weighs")
    masses = R("masses", "list_quantities",
               "kilograms recorded for each crate on the pallet")
    mass_limit = R("mass_limit", "threshold_value",
                   "kilograms a single crate may reach")
    mass_cap = R("mass_cap", "threshold_value",
                 "kilograms no crate on the pallet may exceed")
    net_share_floor = R("net_share_floor", "threshold_percent",
                        "share of the gross weight that should be goods")
    spread_floor = R("spread_floor", "threshold_percent",
                     "how close the average crate should be to the heaviest")

    out.append(Blueprint(
        workflow_id="measurement.crate_weighing",
        domain="measurement",
        natural_user_goal=("work out what a packed crate weighs and whether the "
                           "pallet it sits on is within its weight rules"),
        target_description="the crate weight or the weight-rule verdict",
        value_generator_id="measurement.weighing",
        query_asset_family="weighbridge_ticket",
        hard_distractor_families=("unit_conversion", "statistics"),
        boolean_balancing_strategy="calibrate_crate_mass_limit",
        entity_family="logistics",
        plans=(
            Plan("weigh.v3", (sample_mass_g, sample_count, tare_mass_kg),
                 (S("n1", "unit_conversion.mass_g_kg", ("sample_mass_g",)),
                  S("n2", "arithmetic.multiply", ("@n1", "sample_count"),
                    "weight of the goods in the crate"),
                  S("n3", "arithmetic.add", ("@n2", "tare_mass_kg"),
                    "weight the weighbridge will show")),
                 "n3", intent="gross_crate_weight"),
            Plan("weigh.v7", (sample_mass_g, sample_count, tare_mass_kg,
                              mass_limit, net_share_floor),
                 (S("n1", "unit_conversion.mass_g_kg", ("sample_mass_g",)),
                  S("n2", "arithmetic.multiply", ("@n1", "sample_count"),
                    "net weight, needed again for the packaging share"),
                  S("n3", "arithmetic.add", ("@n2", "tare_mass_kg")),
                  S("n4", "rates.share_percent", ("@n2", "@n3")),
                  S("n5", "comparison.at_least", ("@n4", "net_share_floor")),
                  S("n6", "comparison.at_least", ("@n3", "mass_limit")),
                  S("n7", "boolean.and", ("@n5", "@n6"))),
                 "n7", intent="crate_weight_verdict"),
            Plan("weigh.v9", (masses, mass_limit, mass_cap, spread_floor),
                 (S("n1", "list.reduce_sum", ("masses",),
                    "everything on the pallet"),
                  S("n2", "list.reduce_max", ("masses",), "the heaviest crate"),
                  S("n3", "arithmetic.subtract", ("@n1", "@n2"),
                    "weight of the pallet without its heaviest crate"),
                  S("n4", "statistics.mean", ("masses",)),
                  S("n5", "rates.share_percent", ("@n4", "@n2"),
                    "how close an average crate is to the heaviest"),
                  S("n6", "comparison.at_least", ("@n5", "spread_floor")),
                  S("n7", "comparison.at_least", ("@n3", "mass_limit")),
                  S("n8", "validation.list_limit", ("masses", "mass_cap")),
                  S("n9", "decision.all_of", ("@n6", "@n7", "@n8"),
                    "every pallet rule has to hold")),
                 "n9", intent="pallet_weight_compliance"),
        )))

    # ── temperature log ─────────────────────────────────────────────────
    probe_c = R("probe_c", "temp_c", "temperature the probe reported in Celsius")
    ambient_f = R("ambient_f", "temp_f",
                  "ambient temperature the site logs in Fahrenheit")
    temps = R("temps", "list_readings", "temperatures logged through the day")
    drift_percent = R("drift_percent", "percent_margin",
                      "swing either side of the mean the process allows")
    range_low = R("range_low", "range_low", "lowest drift the process tolerates")
    range_high = R("range_high", "range_high", "highest drift the process tolerates")
    spec_target = R("spec_target", "threshold_value",
                    "control band the process was set up for")
    allowance = R("allowance", "tolerance_value",
                  "how far the control band may drift")
    probe_label = R("probe_label", "text_label", "name of the probe")
    places2 = R("places", "places", "decimals the summary is written to")

    out.append(Blueprint(
        workflow_id="measurement.temperature_drift",
        domain="measurement",
        natural_user_goal=("find out how far a probe has drifted from the "
                           "conditions around it and whether the day's log is stable"),
        target_description="the drift, the stability verdict or the summary label",
        value_generator_id="measurement.temperature",
        query_asset_family="temperature_log",
        hard_distractor_families=("unit_conversion", "validation"),
        boolean_balancing_strategy="calibrate_drift_range",
        entity_family="plant",
        plans=(
            Plan("temp.v4", (probe_c, ambient_f, range_low, range_high),
                 (S("n1", "unit_conversion.temperature_f_c", ("ambient_f",),
                    "ambient reading in the probe's scale"),
                  S("n2", "arithmetic.abs_difference", ("probe_c", "@n1"),
                    "the drift"),
                  S("n3", "validation.in_range", ("@n2", "range_low",
                                                  "range_high")),
                  S("n4", "boolean.not", ("@n3",),
                    "true when the drift has left its window")),
                 "n4", intent="drift_out_of_window"),
            Plan("temp.v6", (temps, drift_percent, spec_target, allowance),
                 (S("n1", "statistics.mean", ("temps",),
                    "the level the control points are hung from"),
                  S("n2", "rates.increase_by_percent", ("@n1",
                                                        "drift_percent"),
                    "upper control point"),
                  S("n3", "rates.decrease_by_percent", ("@n1",
                                                        "drift_percent"),
                    "lower control point"),
                  S("n4", "arithmetic.subtract", ("@n2", "@n3"),
                    "width of the control band"),
                  S("n5", "validation.tolerance", ("@n4", "spec_target",
                                                   "allowance")),
                  S("n6", "boolean.not", ("@n5",),
                    "true when the band has drifted off its setting")),
                 "n6", intent="control_band_verdict"),
            Plan("temp.v7", (temps, places2, probe_label),
                 (S("n1", "statistics.mean", ("temps",)),
                  S("n2", "statistics.median", ("temps",)),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2"),
                    "how skewed the day was"),
                  S("n4", "rates.share_percent", ("@n3", "@n1")),
                  S("n5", "format.percent", ("@n4", "places")),
                  S("n6", "string.normalize_slug", ("probe_label",)),
                  S("n7", "string.concat", ("@n6", "@n5"),
                    "the entry that goes into the day's summary")),
                 "n7", intent="probe_skew_entry"),
        )))

    return out
