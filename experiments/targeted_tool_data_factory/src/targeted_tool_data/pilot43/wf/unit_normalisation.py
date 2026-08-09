"""Unit-handling workflows: mixed-unit consignments, shift spans, fills, scales.

Every plan in this family exists because the facts arrive in different units of
the same quantity, and the sum or the comparison the user wants is meaningless
until they are normalised. The conversion is therefore load-bearing rather than
decorative: hours can only be added to hours, a Celsius parameter only accepts
Celsius, and a date can only be shifted by whole days, so the plan has to say
which conversion it performs. The plans differ in how far the normalised figure
travels afterwards -- straight to an answer, back into the unit the paperwork
uses, or into two independent checks that are then combined.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── mixed-unit consignment ──────────────────────────────────────────
    mass_roles = (
        R("parcel_mass_kg", "mass_kg", "kilograms the palletised parcel weighs"),
        R("sample_mass_g", "mass_g", "grams the sample bag weighs"),
        R("crate_mass_kg", "mass_kg", "kilograms the outer crate weighs"),
        R("load_limit", "threshold_value",
          "kilograms the trolley is rated to carry"),
        R("share_limit", "threshold_percent",
          "share of the consignment the packed goods may represent"),
    )
    out.append(Blueprint(
        workflow_id="units.mixed_mass_consolidation",
        domain="unit_conversion",
        natural_user_goal=("total a consignment whose items are weighed partly in "
                           "grams and partly in kilograms"),
        target_description="the consolidated mass, its composition or the load verdict",
        value_generator_id="units.consignment_mass",
        query_asset_family="consignment_note",
        hard_distractor_families=("arithmetic", "unit_conversion"),
        boolean_balancing_strategy="threshold_band",
        entity_family="logistics",
        plans=(
            Plan("mass.v2", mass_roles[:2],
                 (S("n1", "unit_conversion.mass_g_kg", ("sample_mass_g",),
                    "the sample in the unit the pallet is weighed in"),
                  S("n2", "arithmetic.add", ("parcel_mass_kg", "@n1"),
                    "parcel and sample in one unit")),
                 "n2", intent="normalised_pair_mass"),
            Plan("mass.v4", mass_roles[:3],
                 (S("n1", "unit_conversion.mass_g_kg", ("sample_mass_g",),
                    "the sample in kilograms, read again for its share"),
                  S("n2", "arithmetic.add", ("parcel_mass_kg", "@n1")),
                  S("n3", "arithmetic.add", ("@n2", "crate_mass_kg"),
                    "the whole consignment"),
                  S("n4", "rates.share_percent", ("@n1", "@n3"),
                    "how much of the consignment the sample is")),
                 "n4", intent="sample_share"),
            Plan("mass.v7", mass_roles,
                 (S("n1", "unit_conversion.mass_g_kg", ("sample_mass_g",)),
                  S("n2", "arithmetic.add", ("parcel_mass_kg", "@n1"),
                    "packed goods, needed for the total and for its share"),
                  S("n3", "arithmetic.add", ("@n2", "crate_mass_kg")),
                  S("n4", "rates.share_percent", ("@n2", "@n3")),
                  S("n5", "comparison.at_least", ("@n3", "load_limit")),
                  S("n6", "comparison.at_least", ("@n4", "share_limit")),
                  S("n7", "boolean.and", ("@n5", "@n6"),
                    "the trolley rating and the packing rule both have to hold")),
                 "n7", intent="load_and_composition_verdict"),
        )))

    # ── shift duration normalisation ────────────────────────────────────
    span_roles = (
        R("travel_days", "duration_days", "days the crew spends travelling"),
        R("setup_hours", "duration_hours", "hours the setup on site takes"),
        R("handover_minutes", "duration_minutes", "minutes the handover takes"),
        R("project_weeks", "count_small", "weeks the build phase is planned for"),
        R("start_date", "date_iso", "date the crew starts"),
        R("shift_limit", "threshold_hours", "hours the assignment may not exceed"),
        R("share_limit", "threshold_percent",
          "share of the assignment the build phase may take"),
    )
    out.append(Blueprint(
        workflow_id="units.shift_duration_normalisation",
        domain="unit_conversion",
        natural_user_goal=("add up an assignment quoted in weeks, days, hours and "
                           "minutes and see what it comes to"),
        target_description=("the normalised span, the handover date or the "
                            "assignment verdict"),
        value_generator_id="units.crew_assignment",
        query_asset_family="crew_assignment",
        hard_distractor_families=("duration", "date"),
        boolean_balancing_strategy="threshold_band",
        entity_family="operations",
        plans=(
            Plan("span.v3", (span_roles[0], span_roles[1], span_roles[5]),
                 (S("n1", "duration.convert_days_hours", ("travel_days",),
                    "travel time in the unit the setup is quoted in"),
                  S("n2", "duration.sum", ("@n1", "setup_hours")),
                  S("n3", "comparison.at_least", ("@n2", "shift_limit"))),
                 "n3", intent="assignment_length_verdict"),
            Plan("span.v5", span_roles[:3],
                 (S("n1", "duration.convert_days_hours", ("travel_days",)),
                  S("n2", "duration.sum", ("@n1", "setup_hours")),
                  S("n3", "duration.convert_hours_minutes", ("@n2",),
                    "the planned span in the unit the handover uses"),
                  S("n4", "arithmetic.add", ("@n3", "handover_minutes"),
                    "the whole assignment in minutes"),
                  S("n5", "rates.share_percent", ("@n3", "@n4"),
                    "share of the assignment that is not handover")),
                 "n5", intent="planned_share_of_assignment"),
            Plan("span.v6", (span_roles[0], span_roles[1], span_roles[4]),
                 (S("n1", "duration.convert_days_hours", ("travel_days",)),
                  S("n2", "duration.sum", ("@n1", "setup_hours")),
                  S("n3", "duration.convert_hours_days", ("@n2",),
                    "back into days, which is what the calendar takes"),
                  # date arithmetic only accepts whole days
                  S("n4", "rounding.to_int", ("@n3",)),
                  S("n5", "date.add_duration", ("start_date", "@n4"),
                    "date the crew hands over"),
                  S("n6", "date.weekday", ("@n5",),
                    "which weekday that handover falls on")),
                 "n6", intent="handover_weekday"),
            Plan("span.v9", (span_roles[0], span_roles[1]) + span_roles[3:4]
                 + span_roles[5:],
                 (S("n1", "duration.convert_weeks_days", ("project_weeks",)),
                  S("n2", "duration.convert_days_hours", ("@n1",),
                    "build phase in hours, read again for its share"),
                  S("n3", "duration.convert_days_hours", ("travel_days",)),
                  S("n4", "duration.sum", ("@n2", "@n3")),
                  S("n5", "duration.sum", ("@n4", "setup_hours"),
                    "the whole assignment in hours"),
                  S("n6", "rates.share_percent", ("@n2", "@n5")),
                  S("n7", "comparison.at_least", ("@n5", "shift_limit")),
                  S("n8", "comparison.at_least", ("@n6", "share_limit")),
                  S("n9", "boolean.and", ("@n7", "@n8"))),
                 "n9", intent="assignment_length_and_share_verdict"),
        )))

    # ── bottling fill plan ──────────────────────────────────────────────
    fill_roles = (
        R("tank_volume_l", "volume_l", "litres of product the tank holds"),
        R("bottle_volume_ml", "volume_ml", "millilitres one bottle takes"),
        R("headspace_ml", "volume_ml", "millilitres of headspace left per bottle"),
        R("order_size", "threshold_count", "bottles the order calls for"),
        R("residue_limit", "threshold_value",
          "litres that may be left in the tank at the end"),
        R("cut_low", "cut_low", "utilisation below which the run counts as poor"),
        R("cut_high", "cut_high", "utilisation above which the run counts as good"),
    )
    out.append(Blueprint(
        workflow_id="units.bottling_fill_plan",
        domain="unit_conversion",
        natural_user_goal=("see how many bottles a tank fills when the tank is "
                           "measured in litres and the bottles in millilitres"),
        target_description=("the fill verdict, the tank utilisation band or the "
                            "residue check"),
        value_generator_id="units.bottling_run",
        query_asset_family="bottling_order",
        hard_distractor_families=("unit_conversion", "arithmetic"),
        boolean_balancing_strategy="threshold_band",
        entity_family="logistics",
        plans=(
            Plan("fill.v5", fill_roles[:4],
                 (S("n1", "unit_conversion.volume_l_ml", ("tank_volume_l",),
                    "the tank in the unit the bottles are measured in"),
                  S("n2", "arithmetic.add", ("bottle_volume_ml", "headspace_ml"),
                    "millilitres each bottle actually draws"),
                  S("n3", "rates.ratio_of", ("@n1", "@n2"),
                    "how many draws the tank holds"),
                  S("n4", "rounding.floor", ("@n3",), "whole bottles"),
                  S("n5", "comparison.at_least", ("@n4", "order_size"))),
                 "n5", intent="order_coverage_verdict"),
            Plan("fill.v7", fill_roles[:3] + fill_roles[5:],
                 (S("n1", "unit_conversion.volume_l_ml", ("tank_volume_l",),
                    "the tank in millilitres, read again for the utilisation"),
                  S("n2", "arithmetic.add", ("bottle_volume_ml", "headspace_ml")),
                  S("n3", "rates.ratio_of", ("@n1", "@n2")),
                  S("n4", "rounding.floor", ("@n3",), "whole bottles"),
                  S("n5", "arithmetic.multiply", ("@n2", "@n4"),
                    "millilitres the run actually draws"),
                  S("n6", "rates.share_percent", ("@n5", "@n1")),
                  S("n7", "classification.three_bands",
                    ("@n6", "cut_low", "cut_high"))),
                 "n7", intent="tank_utilisation_band"),
            Plan("fill.v10", fill_roles[:5],
                 (S("n1", "unit_conversion.volume_l_ml", ("tank_volume_l",)),
                  S("n2", "arithmetic.add", ("bottle_volume_ml", "headspace_ml")),
                  S("n3", "rates.ratio_of", ("@n1", "@n2")),
                  S("n4", "rounding.floor", ("@n3",),
                    "whole bottles, checked again against the order"),
                  S("n5", "arithmetic.multiply", ("@n2", "@n4")),
                  S("n6", "arithmetic.subtract", ("@n1", "@n5"),
                    "millilitres left in the tank"),
                  S("n7", "unit_conversion.volume_ml_l", ("@n6",),
                    "the residue in the unit the tank sheet uses"),
                  S("n8", "comparison.at_least", ("@n4", "order_size")),
                  S("n9", "comparison.at_least", ("@n7", "residue_limit")),
                  S("n10", "boolean.and", ("@n8", "@n9"))),
                 "n10", intent="fill_and_residue_verdict"),
        )))

    # ── temperature scale reconciliation ────────────────────────────────
    temp_roles = (
        R("probe_f", "temp_f", "temperature the imported probe reports"),
        R("ambient_c", "temp_c", "temperature the site log records"),
        R("places", "places", "decimals the reconciled figure is written to"),
        R("spread_limit", "threshold_value",
          "spread between the two readings the audit accepts"),
        R("gap_limit", "threshold_value",
          "gap between probe and site reading the audit accepts"),
    )
    out.append(Blueprint(
        workflow_id="units.temperature_scale_reconciliation",
        domain="unit_conversion",
        natural_user_goal=("reconcile a probe that reports Fahrenheit with a site "
                           "log kept in Celsius"),
        target_description=("the reconciled temperature, its printed form or the "
                            "audit verdict"),
        value_generator_id="units.temperature_pair",
        query_asset_family="probe_reconciliation",
        hard_distractor_families=("unit_conversion", "statistics"),
        boolean_balancing_strategy="threshold_band",
        entity_family="engineering",
        plans=(
            Plan("temp.v3", temp_roles[:2],
                 (S("n1", "unit_conversion.temperature_f_c", ("probe_f",),
                    "the probe on the scale the site log uses"),
                  S("n2", "statistics.average_two", ("@n1", "ambient_c")),
                  S("n3", "unit_conversion.temperature_c_f", ("@n2",),
                    "the reconciled figure back on the probe's scale")),
                 "n3", intent="reconciled_temperature"),
            Plan("temp.v6", temp_roles[:3],
                 (S("n1", "unit_conversion.temperature_f_c", ("probe_f",),
                    "the probe in Celsius, used for the mean and for the gap"),
                  S("n2", "statistics.average_two", ("@n1", "ambient_c")),
                  S("n3", "arithmetic.abs_difference", ("@n1", "ambient_c")),
                  S("n4", "arithmetic.add", ("@n2", "@n3"),
                    "worst case the two instruments allow"),
                  S("n5", "unit_conversion.temperature_c_f", ("@n4",)),
                  S("n6", "format.fixed", ("@n5", "places"))),
                 "n6", intent="printed_worst_case"),
            Plan("temp.v10", temp_roles[:2] + temp_roles[3:],
                 (S("n1", "unit_conversion.temperature_f_c", ("probe_f",)),
                  S("n2", "statistics.average_two", ("@n1", "ambient_c"),
                    "reconciled mean, converted again further down"),
                  S("n3", "arithmetic.abs_difference", ("@n1", "ambient_c"),
                    "instrument gap, tested only at the end"),
                  S("n4", "arithmetic.add", ("@n2", "@n3")),
                  S("n5", "unit_conversion.temperature_c_f", ("@n4",)),
                  S("n6", "unit_conversion.temperature_c_f", ("@n2",)),
                  S("n7", "arithmetic.abs_difference", ("@n5", "@n6"),
                    "the spread on the probe's own scale"),
                  S("n8", "comparison.at_least", ("@n7", "spread_limit")),
                  S("n9", "comparison.at_least", ("@n3", "gap_limit")),
                  S("n10", "boolean.and", ("@n8", "@n9"))),
                 "n10", intent="two_scale_audit_verdict"),
        )))

    return out
