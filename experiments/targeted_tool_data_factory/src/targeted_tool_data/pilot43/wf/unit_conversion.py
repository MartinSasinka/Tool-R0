"""Unit-conversion workflows: site distances, storage transfers, recipe scaling.

Every plan here exists because the numbers arrive in the wrong unit: a leg is
quoted in kilometres while the site plan is in metres, a file size in bytes
while the report wants kibibytes, a batch in litres while the portion is in
millilitres. The conversion is therefore a real step, not decoration.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── site length reconciliation ──────────────────────────────────────
    leg_km = R("leg_km", "length_km", "kilometres quoted for one leg of the run")
    site_length_m = R("site_length_m", "length_m",
                      "metres of track measured inside the site")
    segment_count = R("segment_count", "count_small",
                      "times the leg is repeated on the run")
    length_limit = R("length_limit", "threshold_value",
                     "kilometres the daily run may cover")
    share_limit = R("share_limit", "threshold_percent",
                    "share of the run that may sit on one leg")
    places = R("places", "places", "decimals the sheet is written to")

    out.append(Blueprint(
        workflow_id="unit_conversion.site_length_reconciliation",
        domain="unit_conversion",
        natural_user_goal=("add up a run that is partly quoted in kilometres and "
                           "partly measured in metres and see if it is too long"),
        target_description="the combined length, its share or the length verdict",
        value_generator_id="unit_conversion.site_length",
        query_asset_family="site_survey",
        hard_distractor_families=("unit_conversion", "arithmetic"),
        boolean_balancing_strategy="calibrate_run_length_limit",
        entity_family="rail_operations",
        plans=(
            Plan("len.v4", (leg_km, site_length_m, length_limit),
                 (S("n1", "unit_conversion.length_km_m", ("leg_km",),
                    "the quoted leg in metres"),
                  S("n2", "arithmetic.add", ("@n1", "site_length_m"),
                    "the whole run in metres"),
                  S("n3", "unit_conversion.length_m_km", ("@n2",),
                    "back into the unit the timetable uses"),
                  S("n4", "comparison.at_least", ("@n3", "length_limit"))),
                 "n4", intent="run_length_verdict"),
            Plan("len.v5", (leg_km, site_length_m, segment_count, places),
                 (S("n1", "unit_conversion.length_km_m", ("leg_km",),
                    "one leg in metres, needed again for its share"),
                  S("n2", "arithmetic.multiply", ("@n1", "segment_count")),
                  S("n3", "arithmetic.add", ("@n2", "site_length_m")),
                  S("n4", "rates.share_percent", ("@n1", "@n3")),
                  S("n5", "format.percent", ("@n4", "places"),
                    "the share as it appears on the sheet")),
                 "n5", intent="single_leg_share_label"),
            Plan("len.v8", (leg_km, site_length_m, segment_count, length_limit,
                            share_limit),
                 (S("n1", "unit_conversion.length_km_m", ("leg_km",),
                    "one leg in metres, read again much later"),
                  S("n2", "arithmetic.multiply", ("@n1", "segment_count")),
                  S("n3", "arithmetic.add", ("@n2", "site_length_m")),
                  S("n4", "unit_conversion.length_m_km", ("@n3",)),
                  S("n5", "comparison.at_least", ("@n4", "length_limit")),
                  S("n6", "rates.share_percent", ("@n1", "@n3")),
                  S("n7", "comparison.at_least", ("@n6", "share_limit")),
                  S("n8", "boolean.and", ("@n5", "@n7"),
                    "length and concentration both have to fail")),
                 "n8", intent="run_length_and_share_verdict"),
        )))

    # ── storage transfer ────────────────────────────────────────────────
    file_bytes = R("file_bytes", "bytes_size", "size of the export in bytes")
    transfer_minutes = R("transfer_minutes", "duration_minutes",
                         "minutes the transfer window lasts")
    chunk_count = R("chunk_count", "quantity_units",
                    "chunks the export is split into")
    size_limit = R("size_limit", "threshold_value",
                   "kibibytes the receiving system accepts in one go")
    rate_floor = R("rate_floor", "threshold_value",
                   "kibibytes per second the link has to sustain")
    unit_word = R("unit_word", "unit_word", "unit the throughput is quoted in")
    places2 = R("places", "places", "decimals the throughput is quoted to")

    out.append(Blueprint(
        workflow_id="unit_conversion.storage_transfer",
        domain="unit_conversion",
        natural_user_goal=("work out how fast an export has to move to get "
                           "through its transfer window"),
        target_description="the throughput, its label or the transfer verdict",
        value_generator_id="unit_conversion.transfer",
        query_asset_family="transfer_job",
        hard_distractor_families=("unit_conversion", "duration"),
        boolean_balancing_strategy="calibrate_transfer_rate_floor",
        entity_family="it_operations",
        plans=(
            Plan("store.v3", (file_bytes, chunk_count, size_limit),
                 (S("n1", "unit_conversion.bytes_kib", ("file_bytes",),
                    "the export in kibibytes"),
                  S("n2", "arithmetic.divide", ("@n1", "chunk_count"),
                    "size of one chunk"),
                  S("n3", "comparison.at_least", ("@n2", "size_limit"))),
                 "n3", intent="chunk_size_verdict"),
            Plan("store.v5", (file_bytes, transfer_minutes, places2, unit_word),
                 (S("n1", "unit_conversion.bytes_kib", ("file_bytes",)),
                  S("n2", "duration.convert_minutes_seconds",
                    ("transfer_minutes",),
                    "the window in seconds, worked out independently"),
                  S("n3", "arithmetic.divide", ("@n1", "@n2"),
                    "throughput the link must sustain"),
                  S("n4", "rounding.places", ("@n3", "places")),
                  S("n5", "format.with_unit", ("@n4", "unit_word"))),
                 "n5", intent="throughput_label"),
            Plan("store.v7", (file_bytes, transfer_minutes, chunk_count,
                              size_limit, rate_floor),
                 (S("n1", "unit_conversion.bytes_kib", ("file_bytes",),
                    "the export in kibibytes, tested again at the end"),
                  S("n2", "arithmetic.divide", ("@n1", "chunk_count")),
                  S("n3", "duration.convert_minutes_seconds",
                    ("transfer_minutes",)),
                  S("n4", "arithmetic.divide", ("@n2", "@n3"),
                    "throughput needed per chunk"),
                  S("n5", "comparison.at_least", ("@n4", "rate_floor")),
                  S("n6", "comparison.at_least", ("@n1", "size_limit")),
                  S("n7", "boolean.and", ("@n5", "@n6"))),
                 "n7", intent="transfer_feasibility_verdict"),
        )))

    # ── recipe scaling ──────────────────────────────────────────────────
    batch_volume_l = R("batch_volume_l", "volume_l",
                       "litres of mixture the batch produces")
    portion_ml = R("portion_ml", "volume_ml", "millilitres in one portion")
    ingredient_kg = R("ingredient_kg", "mass_kg",
                      "kilograms of the main ingredient used")
    waste_percent = R("waste_percent", "percent_discount",
                      "share of each portion lost in handling")
    portion_target = R("portion_target", "threshold_count",
                       "portions the kitchen has promised")
    gram_floor = R("gram_floor", "threshold_value",
                   "grams of the ingredient each portion must contain")

    out.append(Blueprint(
        workflow_id="unit_conversion.recipe_scaling",
        domain="unit_conversion",
        natural_user_goal=("see how many portions a batch of mixture yields and "
                           "how much of the main ingredient each one carries"),
        target_description="the portion count, the ingredient per portion or the yield verdict",
        value_generator_id="unit_conversion.recipe",
        query_asset_family="recipe_card",
        hard_distractor_families=("unit_conversion", "rates"),
        boolean_balancing_strategy="calibrate_portion_target",
        entity_family="catering",
        plans=(
            Plan("recipe.v3", (batch_volume_l, portion_ml),
                 (S("n1", "unit_conversion.volume_l_ml", ("batch_volume_l",),
                    "the batch in the portion's unit"),
                  S("n2", "arithmetic.divide", ("@n1", "portion_ml")),
                  S("n3", "rounding.to_int", ("@n2",),
                    "whole portions the batch yields")),
                 "n3", intent="portion_yield"),
            Plan("recipe.v6", (batch_volume_l, portion_ml, ingredient_kg,
                               waste_percent),
                 (S("n1", "unit_conversion.volume_l_ml", ("batch_volume_l",)),
                  S("n2", "arithmetic.divide", ("@n1", "portion_ml")),
                  S("n3", "rounding.to_int", ("@n2",), "whole portions"),
                  S("n4", "unit_conversion.mass_kg_g", ("ingredient_kg",),
                    "the ingredient in grams, worked out on its own"),
                  S("n5", "arithmetic.divide", ("@n4", "@n3"),
                    "grams of ingredient per portion"),
                  S("n6", "rates.decrease_by_percent", ("@n5",
                                                        "waste_percent"),
                    "what actually reaches the plate")),
                 "n6", intent="ingredient_per_portion"),
            Plan("recipe.v9", (batch_volume_l, portion_ml, ingredient_kg,
                               waste_percent, portion_target, gram_floor),
                 (S("n1", "unit_conversion.volume_l_ml", ("batch_volume_l",)),
                  S("n2", "arithmetic.divide", ("@n1", "portion_ml")),
                  S("n3", "rounding.to_int", ("@n2",),
                    "whole portions, checked again at the end"),
                  S("n4", "unit_conversion.mass_kg_g", ("ingredient_kg",)),
                  S("n5", "arithmetic.divide", ("@n4", "@n3")),
                  S("n6", "rates.decrease_by_percent", ("@n5",
                                                        "waste_percent")),
                  S("n7", "comparison.at_least", ("@n3", "portion_target")),
                  S("n8", "comparison.at_least", ("@n6", "gram_floor")),
                  S("n9", "boolean.and", ("@n7", "@n8"),
                    "the kitchen needs both the count and the portion size")),
                 "n9", intent="recipe_promise_verdict"),
        )))

    return out
