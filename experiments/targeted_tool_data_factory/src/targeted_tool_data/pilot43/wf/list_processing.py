"""List-processing workflows: reading series, pick waves, routes, price sheets.

Every plan here genuinely manipulates a sequence -- filtering against a
threshold that is itself computed from the series, slicing both ends, combining
two series position by position, locating a value by index -- instead of doing
arithmetic under a list-flavoured name. The long plans get their shape from a
series that is needed again several calls later, which is where the late
references and the reuse come from.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── meter reading series ────────────────────────────────────────────
    readings = R("readings", "list_readings",
                 "the meter readings logged during the shift")
    correction = R("correction_rate", "percent_growth",
                   "the correction the calibration sheet adds to every reading")
    checkpoint = R("checkpoint", "index_position",
                   "which reading in the series the supervisor asks about")
    window = R("window_size", "count_small",
               "how many readings the report keeps at each end of the series")
    min_flagged = R("min_flagged", "threshold_count",
                    "how many readings must sit above average before the shift "
                    "is flagged")
    total_floor = R("total_floor", "threshold_value",
                    "the total the corrected readings have to reach")
    spot_floor = R("spot_floor", "threshold_value",
                   "the value the questioned reading has to reach")
    out.append(Blueprint(
        workflow_id="list_processing.meter_series_review",
        domain="list_processing",
        natural_user_goal=("make sense of a shift's meter readings once the "
                           "calibration correction is taken into account"),
        target_description=("the corrected series, the reading asked about or "
                            "the verdict on the shift"),
        value_generator_id="list_processing.meter_series",
        query_asset_family="meter_log",
        hard_distractor_families=("list", "statistics"),
        boolean_balancing_strategy="calibrate_series_threshold",
        entity_family="field_service",
        plans=(
            Plan("series.v2", (readings, checkpoint),
                 (S("n1", "list.map_sort_asc", ("readings",),
                    "readings ordered from lowest to highest"),
                  S("n2", "list.index", ("@n1", "checkpoint"),
                    "the reading sitting at the asked-for rank")),
                 "n2", intent="ranked_reading"),
            Plan("series.v4", (readings, correction),
                 (S("n1", "list.map_percent", ("readings", "correction_rate"),
                    "the correction owed by each reading"),
                  S("n2", "list.combine_pairwise", ("readings", "@n1"),
                    "corrected series, needed by both branches"),
                  S("n3", "statistics.mean", ("@n2",),
                    "average of the corrected series"),
                  S("n4", "list.filter", ("@n2", "@n3"),
                    "the readings that sit on one side of the average")),
                 "n4", intent="off_average_readings"),
            Plan("series.v6", (readings, correction, min_flagged),
                 (S("n1", "list.map_percent", ("readings", "correction_rate")),
                  S("n2", "list.combine_pairwise", ("readings", "@n1")),
                  S("n3", "statistics.mean", ("@n2",)),
                  S("n4", "list.filter", ("@n2", "@n3")),
                  S("n5", "list.reduce_count", ("@n4",),
                    "how many readings that leaves"),
                  S("n6", "comparison.at_least", ("@n5", "min_flagged"))),
                 "n6", intent="off_average_verdict"),
            Plan("series.v8", (readings, window),
                 (S("n1", "list.map_sort_asc", ("readings",),
                    "ordered series, read again much later"),
                  S("n2", "list.slice_first", ("@n1", "window_size"),
                    "the quietest readings"),
                  S("n3", "list.slice_last", ("@n1", "window_size"),
                    "the busiest readings"),
                  S("n4", "list.reduce_sum", ("@n2",)),
                  S("n5", "list.reduce_sum", ("@n3",)),
                  S("n6", "rates.share_percent", ("@n4", "@n5"),
                    "the quiet end as a share of the busy end"),
                  S("n7", "list.map_percent", ("@n1", "@n6"),
                    "that share applied back to the whole ordered series"),
                  S("n8", "list.reduce_sum", ("@n7",))),
                 "n8", intent="quiet_end_weighted_total"),
            Plan("series.v10", (readings, checkpoint, total_floor, spot_floor),
                 (S("n1", "list.map_sort_asc", ("readings",)),
                  S("n2", "list.map_running_max", ("readings",),
                    "the peak seen so far at each point of the shift"),
                  S("n3", "list.combine_pairwise", ("@n1", "@n2"),
                    "ordered series added to the running peak"),
                  S("n4", "statistics.mean", ("@n3",)),
                  S("n5", "list.filter", ("@n3", "@n4")),
                  S("n6", "list.reduce_sum", ("@n5",)),
                  S("n7", "list.index", ("@n1", "checkpoint"),
                    "the ranked reading, read from the ordered series again"),
                  S("n8", "comparison.at_least", ("@n6", "total_floor")),
                  S("n9", "comparison.at_least", ("@n7", "spot_floor")),
                  S("n10", "boolean.and", ("@n8", "@n9"))),
                 "n10", intent="series_and_spot_verdict"),
        )))

    # ── pick wave ───────────────────────────────────────────────────────
    quantities = R("pick_quantities", "list_quantities",
                   "the quantity to pick on each line of the wave")
    places = R("rounding_places", "places",
               "how many decimals the picking sheet keeps")
    spread_limit = R("spread_limit", "threshold_value",
                     "the spread between the biggest and smallest line that "
                     "still counts as balanced")
    share_floor = R("share_floor", "threshold_percent",
                    "the share of the wave the heavy lines must carry")
    peak_cut = R("peak_cut", "threshold_ratio",
                 "how far above the average line the biggest one may sit "
                 "before the wave counts as lopsided")
    out.append(Blueprint(
        workflow_id="list_processing.pick_wave_balance",
        domain="list_processing",
        natural_user_goal=("check how evenly the quantities of a picking wave "
                           "are spread over its lines"),
        target_description="the balance of the wave or the lines that stand out",
        value_generator_id="list_processing.pick_wave",
        query_asset_family="pick_list",
        hard_distractor_families=("list", "comparison"),
        boolean_balancing_strategy="calibrate_wave_threshold",
        entity_family="warehouse",
        plans=(
            Plan("wave.v3", (quantities,),
                 (S("n1", "statistics.mean", ("pick_quantities",),
                    "the average line quantity"),
                  S("n2", "list.filter", ("pick_quantities", "@n1"),
                    "the lines on one side of that average"),
                  S("n3", "list.combine_append", ("@n2", "@n1"),
                    "the average appended as a reference row")),
                 "n3", intent="off_average_shortlist"),
            Plan("wave.v4", (quantities, peak_cut),
                 (S("n1", "list.reduce_max", ("pick_quantities",),
                    "the heaviest line in the wave"),
                  S("n2", "statistics.mean", ("pick_quantities",),
                    "what a line carries on average"),
                  S("n3", "rates.ratio_of", ("@n1", "@n2"),
                    "how many average lines the heaviest one is worth"),
                  S("n4", "classification.threshold", ("@n3", "peak_cut"))),
                 "n4", intent="wave_balance_class"),
            Plan("wave.v7", (quantities, places, spread_limit),
                 (S("n1", "list.map_round", ("pick_quantities",
                                             "rounding_places"),
                    "quantities at sheet precision, used four times"),
                  S("n2", "list.index_of_max", ("@n1",)),
                  S("n3", "list.index_of_min", ("@n1",)),
                  S("n4", "list.index", ("@n1", "@n2")),
                  S("n5", "list.index", ("@n1", "@n3")),
                  S("n6", "arithmetic.subtract", ("@n4", "@n5"),
                    "spread between the heaviest and lightest line"),
                  S("n7", "comparison.at_least", ("@n6", "spread_limit"))),
                 "n7", intent="line_spread_verdict"),
            Plan("wave.v9", (quantities, places, share_floor),
                 (S("n1", "list.map_round", ("pick_quantities",
                                             "rounding_places")),
                  S("n2", "list.map_running_max", ("@n1",)),
                  S("n3", "list.combine_pairwise", ("@n1", "@n2"),
                    "each line loaded with the peak seen so far"),
                  S("n4", "statistics.mean", ("@n3",)),
                  S("n5", "list.filter", ("@n3", "@n4")),
                  S("n6", "list.reduce_sum", ("@n5",)),
                  S("n7", "list.reduce_sum", ("@n3",),
                    "total of the loaded series, needed at the end"),
                  S("n8", "rates.share_percent", ("@n6", "@n7")),
                  S("n9", "comparison.at_least", ("@n8", "share_floor"))),
                 "n9", intent="heavy_line_share_verdict"),
        )))

    # ── route durations ─────────────────────────────────────────────────
    durations = R("leg_hours", "list_durations_h",
                  "how long each leg of the route takes")
    repeats = R("weekly_runs", "count_small",
                "how often the route is driven in a week")
    stop = R("stop_index", "index_position",
             "the stop the planner is asking about")
    top_count = R("longest_count", "count_small",
                  "how many of the longest legs the summary lists")
    joiner = R("separator", "separator",
               "the separator the planner wants between the values")
    leg_places = R("leg_places", "places",
                   "how many decimals the summary shows")
    share_target = R("share_target", "threshold_ratio",
                     "the share of the route that should be done by that stop")
    remaining_low = R("remaining_low", "range_low",
                      "the smallest acceptable amount of driving left")
    remaining_high = R("remaining_high", "range_high",
                       "the largest acceptable amount of driving left")
    done_floor = R("done_floor", "threshold_value",
                   "the driving time that must already be behind the driver")
    out.append(Blueprint(
        workflow_id="list_processing.route_duration_plan",
        domain="list_processing",
        natural_user_goal=("work out how a delivery route stands part way "
                           "through the day"),
        target_description=("the driving time reached, the progress band or the "
                            "summary of the longest legs"),
        value_generator_id="list_processing.route_legs",
        query_asset_family="route_sheet",
        hard_distractor_families=("list", "duration"),
        boolean_balancing_strategy="calibrate_route_threshold",
        entity_family="logistics",
        plans=(
            Plan("route.v4", (durations, top_count, leg_places, joiner),
                 (S("n1", "list.map_sort_asc", ("leg_hours",)),
                  S("n2", "list.slice_last", ("@n1", "longest_count"),
                    "the longest legs"),
                  S("n3", "list.map_round", ("@n2", "leg_places")),
                  S("n4", "list.combine_join", ("@n3", "separator"),
                    "the printable summary line")),
                 "n4", intent="longest_leg_summary"),
            Plan("route.v6", (durations, repeats, stop, share_target),
                 (S("n1", "list.map_scale", ("leg_hours", "weekly_runs"),
                    "driving time over a whole week"),
                  S("n2", "list.map_cumulative", ("@n1",),
                    "driving time accumulated up to each stop"),
                  S("n3", "list.index", ("@n2", "stop_index")),
                  S("n4", "list.reduce_sum", ("@n1",),
                    "the whole week's driving time"),
                  S("n5", "rates.ratio_of", ("@n3", "@n4")),
                  S("n6", "classification.ratio_band", ("@n5", "share_target"))),
                 "n6", intent="route_progress_band"),
            Plan("route.v8", (durations, repeats, stop, remaining_low,
                              remaining_high, done_floor),
                 (S("n1", "list.map_scale", ("leg_hours", "weekly_runs")),
                  S("n2", "list.map_cumulative", ("@n1",)),
                  S("n3", "list.index", ("@n2", "stop_index"),
                    "driving time already done, used twice"),
                  S("n4", "list.reduce_sum", ("@n1",)),
                  S("n5", "arithmetic.subtract", ("@n4", "@n3"),
                    "driving time still ahead"),
                  S("n6", "validation.in_range", ("@n5", "remaining_low",
                                                  "remaining_high")),
                  S("n7", "comparison.at_least", ("@n3", "done_floor")),
                  S("n8", "boolean.xor", ("@n6", "@n7"))),
                 "n8", intent="route_progress_verdict"),
        )))

    # ── production line stage factors ───────────────────────────────────
    factors = R("stage_factors", "list_readings",
                "the multiplier each stage of the line applies")
    stage_count = R("stage_count", "count_small",
                    "how many stages of the line are in scope")
    min_yield = R("min_yield", "threshold_value",
                  "the combined multiplier the line has to reach")
    low_cut = R("low_cut", "cut_low",
                "the throughput below which the line counts as slow")
    high_cut = R("high_cut", "cut_high",
                 "the throughput above which the line counts as fast")
    out.append(Blueprint(
        workflow_id="list_processing.stage_factor_chain",
        domain="list_processing",
        natural_user_goal=("work out what a chain of production stages does to "
                           "the amount that enters the line"),
        target_description="the combined effect of the stages on the throughput",
        value_generator_id="list_processing.stage_factors",
        query_asset_family="line_settings",
        hard_distractor_families=("list", "arithmetic"),
        boolean_balancing_strategy="calibrate_yield_threshold",
        entity_family="manufacturing",
        plans=(
            Plan("stage.v3", (factors, stage_count, min_yield),
                 (S("n1", "list.slice_first", ("stage_factors", "stage_count"),
                    "the stages that are actually in scope"),
                  S("n2", "list.reduce_product", ("@n1",),
                    "their combined multiplier"),
                  S("n3", "comparison.at_least", ("@n2", "min_yield"))),
                 "n3", intent="combined_multiplier_verdict"),
            Plan("stage.v4", (factors, stage_count),
                 (S("n1", "list.slice_last", ("stage_factors", "stage_count"),
                    "the closing stages of the line, read twice below"),
                  S("n2", "list.reduce_product", ("@n1",),
                    "their combined multiplier"),
                  S("n3", "list.reduce_max", ("@n1",),
                    "the strongest single stage"),
                  S("n4", "arithmetic.divide", ("@n2", "@n3"),
                    "the combined multiplier with that stage taken out")),
                 "n4", intent="multiplier_without_peak_stage"),
            Plan("stage.v5", (factors, stage_count, low_cut, high_cut),
                 (S("n1", "list.slice_first", ("stage_factors", "stage_count"),
                    "the stages in scope, read again below"),
                  S("n2", "list.reduce_product", ("@n1",)),
                  S("n3", "list.reduce_second_largest", ("@n1",),
                    "the second strongest stage"),
                  S("n4", "arithmetic.divide", ("@n2", "@n3"),
                    "the multiplier once that stage is discounted"),
                  S("n5", "classification.three_bands", ("@n4", "low_cut",
                                                         "high_cut"))),
                 "n5", intent="stage_chain_band"),
        )))

    # ── price sheet cleanup ─────────────────────────────────────────────
    line_prices = R("line_prices", "list_prices",
                    "the prices already on the sheet")
    unit_price = R("new_unit_price", "money_price",
                   "the price of the item being added")
    freight = R("freight_charge", "money_fee", "the freight charge to add")
    handling = R("handling_charge", "money_fee", "the handling charge to add")
    surcharge = R("surcharge", "money_fee",
                  "the surcharge added to every line of the sheet")
    ceiling = R("price_ceiling", "threshold_value",
                "the highest price the sheet is allowed to contain")
    sheet_top = R("sheet_top_count", "count_small",
                  "how many of the dearest lines the extract shows")
    sheet_places = R("sheet_places", "places",
                     "how many decimals the extract shows")
    sheet_joiner = R("sheet_separator", "separator",
                     "the separator used in the extract")
    band_low = R("band_low", "cut_low",
                 "the share below which the cheap lines dominate the sheet")
    band_high = R("band_high", "cut_high",
                  "the share above which the dear lines dominate the sheet")
    out.append(Blueprint(
        workflow_id="list_processing.price_sheet_cleanup",
        domain="list_processing",
        natural_user_goal=("tidy up a price sheet after new charges have been "
                           "added to it"),
        target_description=("the cleaned sheet, its extract or how top-heavy it "
                            "has become"),
        value_generator_id="list_processing.price_sheet",
        query_asset_family="price_sheet",
        hard_distractor_families=("list", "format"),
        boolean_balancing_strategy="calibrate_price_ceiling",
        entity_family="finance",
        plans=(
            Plan("sheet.v2", (line_prices, surcharge, ceiling),
                 (S("n1", "list.map_offset", ("line_prices", "surcharge"),
                    "every line with the surcharge added"),
                  S("n2", "validation.list_limit", ("@n1", "price_ceiling"))),
                 "n2", intent="ceiling_verdict"),
            Plan("sheet.v4", (line_prices, surcharge, sheet_top, sheet_places,
                              sheet_joiner),
                 (S("n1", "list.map_offset", ("line_prices", "surcharge")),
                  S("n2", "list.map_sort_asc", ("@n1",)),
                  S("n3", "list.slice_last", ("@n2", "sheet_top_count")),
                  S("n4", "list.map_round", ("@n3", "sheet_places")),
                  S("n5", "list.combine_join", ("@n4", "sheet_separator"),
                    "the printable extract")),
                 "n5", intent="dearest_lines_extract"),
            Plan("sheet.v5", (line_prices, unit_price, freight, handling),
                 (S("n1", "list.build", ("new_unit_price", "freight_charge",
                                         "handling_charge"),
                    "the three new charges as one list"),
                  S("n2", "list.combine_concat", ("line_prices", "@n1"),
                    "the sheet with the new charges appended"),
                  S("n3", "statistics.mean", ("@n2",)),
                  S("n4", "list.filter", ("@n2", "@n3")),
                  S("n5", "list.reduce_count", ("@n4",),
                    "how many lines sit on that side of the average")),
                 "n5", intent="lines_off_average"),
            Plan("sheet.v9", (line_prices, unit_price, freight, handling,
                              band_low, band_high),
                 (S("n1", "list.build", ("new_unit_price", "freight_charge",
                                         "handling_charge")),
                  S("n2", "list.combine_concat", ("line_prices", "@n1"),
                    "the full sheet, read again near the end"),
                  S("n3", "list.reduce_max", ("@n2",)),
                  S("n4", "list.index_of_value", ("@n2", "@n3"),
                    "where the dearest line sits"),
                  S("n5", "list.slice_first", ("@n2", "@n4"),
                    "everything up to and including it"),
                  S("n6", "list.reduce_sum", ("@n5",)),
                  S("n7", "list.reduce_sum", ("@n2",),
                    "the whole sheet's value"),
                  S("n8", "rates.share_percent", ("@n6", "@n7")),
                  S("n9", "classification.three_bands", ("@n8", "band_low",
                                                         "band_high"))),
                 "n9", intent="sheet_top_heaviness_band"),
        )))

    return out
