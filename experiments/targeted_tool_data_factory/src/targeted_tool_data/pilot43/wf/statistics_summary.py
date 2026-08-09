"""Statistical summary workflows: reading spread, line comparison, scorecards.

The long plans here are genuinely two-stage: a summary of the series is derived
first, the series is then re-read against that summary, and only the second
reading is reported. That is why the same reduction appears twice with different
provenance rather than once.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── reading spread ──────────────────────────────────────────────────
    readings = R("readings", "list_readings",
                 "readings the instrument logged this run")
    calibration_offset = R("calibration_offset", "generic_value",
                           "correction the calibration certificate applies")
    tolerance_percent = R("tolerance_percent", "percent_margin",
                          "allowance the method permits either side of the mean")
    places = R("places", "places", "decimals the summary is written to")
    band_low = R("band_low", "cut_low", "count that still looks like a clean run")
    band_high = R("band_high", "cut_high", "count that looks like a bad run")
    share_limit = R("share_limit", "threshold_percent",
                    "share of readings allowed to sit above the control line")

    out.append(Blueprint(
        workflow_id="statistics_summary.reading_spread",
        domain="statistics_summary",
        natural_user_goal=("understand how tightly a set of instrument readings "
                           "sits around its own average"),
        target_description="the spread, the count outside the band or the run verdict",
        value_generator_id="statistics_summary.spread",
        query_asset_family="instrument_run",
        hard_distractor_families=("statistics", "list"),
        boolean_balancing_strategy="calibrate_outlier_share",
        entity_family="laboratory",
        plans=(
            Plan("spread.v3", (readings,),
                 (S("n1", "statistics.mean", ("readings",)),
                  S("n2", "statistics.stdev", ("readings",),
                    "the scatter, measured independently of the mean"),
                  S("n3", "arithmetic.divide", ("@n2", "@n1"),
                    "scatter relative to the level")),
                 "n3", intent="relative_spread"),
            Plan("spread.v4", (readings, calibration_offset, places),
                 (S("n1", "list.map_offset", ("readings", "calibration_offset"),
                    "readings after calibration"),
                  S("n2", "list.map_round", ("@n1", "places")),
                  S("n3", "statistics.mean", ("@n2",)),
                  S("n4", "list.reduce_count_above", ("@n2", "@n3"),
                    "readings sitting above their own average")),
                 "n4", intent="above_average_count"),
            Plan("spread.v5", (readings, tolerance_percent, places),
                 (S("n1", "statistics.mean", ("readings",),
                    "the level the control lines are built around"),
                  S("n2", "rates.increase_by_percent", ("@n1",
                                                        "tolerance_percent"),
                    "upper control line"),
                  S("n3", "rates.decrease_by_percent", ("@n1",
                                                        "tolerance_percent"),
                    "lower control line"),
                  S("n4", "arithmetic.subtract", ("@n2", "@n3"),
                    "width of the control band"),
                  S("n5", "format.fixed", ("@n4", "places"),
                    "the width as the method sheet writes it")),
                 "n5", intent="control_band_width"),
            Plan("spread.v6", (readings, calibration_offset, band_low,
                               band_high),
                 (S("n1", "list.map_offset", ("readings",
                                              "calibration_offset")),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.stdev", ("@n1",)),
                  S("n4", "arithmetic.add", ("@n2", "@n3"),
                    "the line a reading has to clear to count as high"),
                  S("n5", "list.reduce_count_above", ("@n1", "@n4")),
                  S("n6", "classification.three_bands", ("@n5", "band_low",
                                                         "band_high"))),
                 "n6", intent="run_quality_band"),
            Plan("spread.v8", (readings, share_limit),
                 (S("n1", "statistics.mean", ("readings",),
                    "the level, used for both control lines"),
                  S("n2", "statistics.stdev", ("readings",)),
                  S("n3", "arithmetic.add", ("@n1", "@n2")),
                  S("n4", "arithmetic.subtract", ("@n1", "@n2")),
                  S("n5", "list.reduce_count_above", ("readings", "@n3"),
                    "readings above the upper line"),
                  S("n6", "list.reduce_count_above", ("readings", "@n4"),
                    "readings above the lower line, counted the same way"),
                  S("n7", "rates.share_percent", ("@n5", "@n6")),
                  S("n8", "comparison.at_least", ("@n7", "share_limit"))),
                 "n8", intent="outlier_share_verdict"),
        )))

    # ── line comparison ─────────────────────────────────────────────────
    line_a = R("line_a", "list_readings", "output recorded on the first line")
    line_b = R("line_b", "list_readings", "output recorded on the second line")
    gap_share_limit = R("gap_share_limit", "threshold_percent",
                        "gap between the lines the plant manager will accept")
    tolerance_value = R("tolerance_value", "tolerance_value",
                        "how far the two lines' scatter may differ")
    cut_low = R("cut_low", "cut_low", "difference that counts as comparable")
    cut_high = R("cut_high", "cut_high", "difference that counts as divergent")

    out.append(Blueprint(
        workflow_id="statistics_summary.line_comparison",
        domain="statistics_summary",
        natural_user_goal=("compare how two production lines performed over the "
                           "same period"),
        target_description="the gap between the lines, its band or the review verdict",
        value_generator_id="statistics_summary.comparison",
        query_asset_family="line_output_log",
        hard_distractor_families=("statistics", "rates"),
        boolean_balancing_strategy="calibrate_line_gap_share",
        entity_family="manufacturing",
        plans=(
            Plan("compare.v5", (line_a, line_b),
                 (S("n1", "statistics.mean", ("line_a",)),
                  S("n2", "statistics.mean", ("line_b",),
                    "the other line, summarised the same way"),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2"),
                    "how far apart the lines ran"),
                  S("n4", "statistics.average_two", ("@n1", "@n2"),
                    "what the plant did on average"),
                  S("n5", "rates.share_percent", ("@n3", "@n4"))),
                 "n5", intent="relative_line_gap"),
            Plan("compare.v8", (line_a, line_b, cut_low, cut_high),
                 (S("n1", "statistics.mean", ("line_a",)),
                  S("n2", "statistics.stdev", ("line_a",)),
                  S("n3", "arithmetic.divide", ("@n2", "@n1"),
                    "how erratic the first line was"),
                  S("n4", "statistics.mean", ("line_b",)),
                  S("n5", "statistics.stdev", ("line_b",)),
                  S("n6", "arithmetic.divide", ("@n5", "@n4"),
                    "the same reading for the second line"),
                  S("n7", "rates.percent_change", ("@n3", "@n6")),
                  S("n8", "classification.three_bands", ("@n7", "cut_low",
                                                         "cut_high"))),
                 "n8", intent="consistency_band"),
            Plan("compare.v10", (line_a, line_b, gap_share_limit,
                                 tolerance_value),
                 (S("n1", "statistics.mean", ("line_a",),
                    "first line's level, needed again much later"),
                  S("n2", "statistics.mean", ("line_b",)),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2")),
                  S("n4", "statistics.average_two", ("@n1", "@n2")),
                  S("n5", "rates.share_percent", ("@n3", "@n4")),
                  S("n6", "statistics.stdev", ("line_a",)),
                  S("n7", "statistics.stdev", ("line_b",)),
                  S("n8", "comparison.at_least", ("@n5", "gap_share_limit")),
                  S("n9", "validation.tolerance", ("@n6", "@n7",
                                                   "tolerance_value"),
                    "did the two lines scatter by a similar amount"),
                  S("n10", "boolean.and", ("@n8", "@n9"),
                    "a real divergence shows up in level and in scatter")),
                 "n10", intent="line_divergence_verdict"),
        )))

    # ── weighted scorecard ──────────────────────────────────────────────
    score_a = R("score_a", "score_points", "score the first assessor gave")
    score_b = R("score_b", "score_points", "score the second assessor gave")
    score_c = R("score_c", "score_points", "score the third assessor gave")
    weight_a = R("weight_a", "count_small",
                 "weight the first assessor's score carries")
    weight_b = R("weight_b", "count_small",
                 "weight the second assessor's score carries")
    weight_factor = R("weight_factor", "generic_value",
                      "factor the panel rescales every score by")
    band_cut = R("band_cut", "threshold_ratio",
                 "spread the panel treats as agreement")
    grade_low = R("grade_low", "cut_low", "score that still counts as weak")
    grade_high = R("grade_high", "cut_high", "score that counts as strong")
    places2 = R("places", "places", "decimals the headline is written to")

    out.append(Blueprint(
        workflow_id="statistics_summary.weighted_scorecard",
        domain="statistics_summary",
        natural_user_goal=("turn several assessors' scores into one figure and "
                           "see how much the weighting changed it"),
        target_description="the panel grade, the weighting effect or the headline",
        value_generator_id="statistics_summary.scorecard",
        query_asset_family="assessment_sheet",
        hard_distractor_families=("statistics", "list"),
        entity_family="assessment",
        plans=(
            Plan("score.v2", (score_a, weight_a, score_b, weight_b, grade_low,
                              grade_high),
                 (S("n1", "statistics.weighted_average", ("score_a", "weight_a",
                                                          "score_b",
                                                          "weight_b"),
                    "the panel's combined score"),
                  S("n2", "classification.three_bands", ("@n1", "grade_low",
                                                         "grade_high"))),
                 "n2", intent="panel_grade"),
            Plan("score.v4", (score_a, weight_a, score_b, weight_b, score_c),
                 (S("n1", "statistics.weighted_average", ("score_a", "weight_a",
                                                          "score_b",
                                                          "weight_b")),
                  S("n2", "statistics.mean_three", ("score_a", "score_b",
                                                    "score_c"),
                    "the same panel with every voice equal"),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2"),
                    "what the weighting is worth"),
                  S("n4", "rates.share_percent", ("@n3", "@n2"))),
                 "n4", intent="weighting_effect"),
            Plan("score.v6", (score_a, score_b, score_c, weight_factor,
                              band_cut),
                 (S("n1", "list.build", ("score_a", "score_b", "score_c"),
                    "the panel as a series"),
                  S("n2", "list.map_scale", ("@n1", "weight_factor")),
                  S("n3", "statistics.mean", ("@n2",)),
                  S("n4", "statistics.range", ("@n2",),
                    "how far apart the assessors sat"),
                  S("n5", "rates.ratio_of", ("@n4", "@n3")),
                  S("n6", "classification.ratio_band", ("@n5", "band_cut"))),
                 "n6", intent="panel_agreement_band"),
            Plan("score.v7", (score_a, score_b, score_c, places2),
                 (S("n1", "list.build", ("score_a", "score_b", "score_c"),
                    "the panel as a series, read three ways"),
                  S("n2", "list.reduce_max", ("@n1",)),
                  S("n3", "list.reduce_second_largest", ("@n1",)),
                  S("n4", "statistics.average_two", ("@n2", "@n3"),
                    "the two scores a headline would quote"),
                  S("n5", "statistics.mean", ("@n1",),
                    "the whole panel, for comparison"),
                  S("n6", "rates.percent_change", ("@n5", "@n4")),
                  S("n7", "format.percent", ("@n6", "places"),
                    "how flattering the headline is")),
                 "n7", intent="headline_flattery"),
        )))

    return out
