"""Quality-control workflows: defect sampling, dimension checks, supplier release.

Defect rates are always built as a share of the sample that produced them, so
two shifts can only be compared after each has been turned into its own rate --
the reason the comparison plans have two independent roots.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── defect sampling ─────────────────────────────────────────────────
    shift_a_defects = R("shift_a_defects", "count_items",
                        "faulty pieces found on the early shift")
    shift_a_units = R("shift_a_units", "quantity_units",
                      "pieces the early shift produced")
    shift_b_defects = R("shift_b_defects", "count_items",
                        "faulty pieces found on the late shift")
    shift_b_units = R("shift_b_units", "quantity_units",
                      "pieces the late shift produced")
    defect_limit = R("defect_limit", "threshold_percent",
                     "defect rate the customer's specification allows")
    gap_limit = R("gap_limit", "threshold_percent",
                  "difference between shifts that counts as a real change")
    tolerance_value = R("tolerance_value", "tolerance_value",
                        "how close the two shifts have to be to call them equal")
    band_low = R("band_low", "cut_low", "defect rate that still looks acceptable")
    band_high = R("band_high", "cut_high", "defect rate that looks out of control")
    places = R("places", "places", "decimals the certificate is written to")
    batch_label = R("batch_label", "text_label", "reference the batch is filed under")

    out.append(Blueprint(
        workflow_id="quality_control.defect_sampling",
        domain="quality_control",
        natural_user_goal=("see how faulty a shift's output was and whether the "
                           "other shift did any better"),
        target_description="the defect rate, its band or the specification verdict",
        value_generator_id="quality_control.defects",
        query_asset_family="inspection_sheet",
        hard_distractor_families=("rates", "classification"),
        boolean_balancing_strategy="calibrate_defect_rate_limit",
        entity_family="manufacturing",
        plans=(
            Plan("defect.v2", (shift_a_defects, shift_a_units, band_low,
                               band_high),
                 (S("n1", "rates.share_percent", ("shift_a_defects",
                                                  "shift_a_units"),
                    "the shift's defect rate"),
                  S("n2", "classification.three_bands", ("@n1", "band_low",
                                                         "band_high"))),
                 "n2", intent="defect_rate_band"),
            Plan("defect.v5", (shift_a_defects, shift_a_units, places,
                               batch_label),
                 (S("n1", "rates.share_percent", ("shift_a_defects",
                                                  "shift_a_units")),
                  S("n2", "format.percent", ("@n1", "places")),
                  S("n3", "string.normalize_slug", ("batch_label",),
                    "the batch reference in filing form"),
                  S("n4", "string.concat", ("@n3", "@n2")),
                  S("n5", "string.normalize_upper", ("@n4",),
                    "the line stamped on the certificate")),
                 "n5", intent="certificate_line"),
            Plan("defect.v6", (shift_a_defects, shift_a_units, shift_b_defects,
                               shift_b_units, defect_limit, gap_limit),
                 (S("n1", "rates.share_percent", ("shift_a_defects",
                                                  "shift_a_units"),
                    "early shift's rate, tested twice"),
                  S("n2", "rates.share_percent", ("shift_b_defects",
                                                  "shift_b_units"),
                    "late shift's rate, measured the same way"),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2")),
                  S("n4", "comparison.at_least", ("@n3", "gap_limit")),
                  S("n5", "comparison.at_least", ("@n1", "defect_limit")),
                  S("n6", "boolean.or", ("@n4", "@n5"),
                    "either a bad shift or a shift-to-shift change matters")),
                 "n6", intent="shift_review_verdict"),
            Plan("defect.v9", (shift_a_defects, shift_a_units, shift_b_defects,
                               shift_b_units, defect_limit, gap_limit,
                               tolerance_value),
                 (S("n1", "rates.share_percent", ("shift_a_defects",
                                                  "shift_a_units"),
                    "early shift's rate, needed again much later"),
                  S("n2", "rates.share_percent", ("shift_b_defects",
                                                  "shift_b_units")),
                  S("n3", "statistics.average_two", ("@n1", "@n2"),
                    "the day's rate"),
                  S("n4", "arithmetic.abs_difference", ("@n1", "@n2")),
                  S("n5", "rates.share_percent", ("@n4", "@n3"),
                    "the gap relative to the day"),
                  S("n6", "comparison.at_least", ("@n5", "gap_limit")),
                  S("n7", "comparison.at_least", ("@n3", "defect_limit")),
                  S("n8", "validation.tolerance", ("@n1", "@n2",
                                                   "tolerance_value"),
                    "did the two shifts really land in the same place"),
                  S("n9", "decision.majority", ("@n6", "@n7", "@n8"),
                    "most of the day's quality rules have to agree")),
                 "n9", intent="day_quality_verdict"),
        )))

    # ── dimension check ─────────────────────────────────────────────────
    measured_length_m = R("measured_length_m", "length_m",
                          "length the part actually measures")
    nominal_length_m = R("nominal_length_m", "length_m",
                         "length the drawing calls for")
    part_lengths = R("part_lengths", "list_readings",
                     "lengths measured across the sample of parts")
    tolerance_percent = R("tolerance_percent", "threshold_percent",
                          "deviation from the drawing the shop allows")
    range_low2 = R("range_low", "range_low", "smallest drift the shop accepts")
    range_high2 = R("range_high", "range_high", "largest drift the shop accepts")
    cut_low = R("cut_low", "cut_low", "share that still looks under control")
    cut_high = R("cut_high", "cut_high", "share that looks out of control")

    out.append(Blueprint(
        workflow_id="quality_control.dimension_check",
        domain="quality_control",
        natural_user_goal=("check whether machined parts are coming out the "
                           "size the drawing asks for"),
        target_description="the deviation, the share of oversized parts or the check verdict",
        value_generator_id="quality_control.dimensions",
        query_asset_family="dimension_report",
        hard_distractor_families=("validation", "statistics"),
        boolean_balancing_strategy="calibrate_dimension_window",
        entity_family="machine_shop",
        plans=(
            Plan("dim.v3", (measured_length_m, nominal_length_m,
                            tolerance_percent),
                 (S("n1", "arithmetic.abs_difference", ("measured_length_m",
                                                        "nominal_length_m"),
                    "how far the part is from the drawing"),
                  S("n2", "rates.share_percent", ("@n1", "nominal_length_m")),
                  S("n3", "comparison.at_least", ("@n2", "tolerance_percent"))),
                 "n3", intent="deviation_verdict"),
            Plan("dim.v5", (part_lengths, nominal_length_m, range_low2,
                            range_high2),
                 (S("n1", "statistics.mean", ("part_lengths",)),
                  S("n2", "statistics.stdev", ("part_lengths",),
                    "how consistent the run was"),
                  S("n3", "arithmetic.add", ("@n1", "@n2"),
                    "the size the worst of the run reaches"),
                  S("n4", "arithmetic.abs_difference", ("@n3",
                                                        "nominal_length_m")),
                  S("n5", "validation.in_range", ("@n4", "range_low",
                                                  "range_high"))),
                 "n5", intent="worst_case_window_verdict"),
            Plan("dim.v6", (part_lengths,),
                 (S("n1", "statistics.mean", ("part_lengths",)),
                  S("n2", "statistics.stdev", ("part_lengths",)),
                  S("n3", "arithmetic.add", ("@n1", "@n2"),
                    "the line an oversized part has to cross"),
                  S("n4", "list.reduce_count_above", ("part_lengths", "@n3")),
                  S("n5", "list.reduce_count", ("part_lengths",)),
                  S("n6", "rates.share_percent", ("@n4", "@n5"),
                    "share of the run that came out oversized")),
                 "n6", intent="oversized_share"),
            Plan("dim.v8", (part_lengths, cut_low, cut_high),
                 (S("n1", "statistics.mean", ("part_lengths",),
                    "the level both control lines are built from"),
                  S("n2", "statistics.stdev", ("part_lengths",)),
                  S("n3", "arithmetic.add", ("@n1", "@n2")),
                  S("n4", "arithmetic.subtract", ("@n1", "@n2")),
                  S("n5", "list.reduce_count_above", ("part_lengths", "@n3"),
                    "parts above the upper line"),
                  S("n6", "list.reduce_count_above", ("part_lengths", "@n4"),
                    "parts above the lower line, counted the same way"),
                  S("n7", "rates.share_percent", ("@n5", "@n6")),
                  S("n8", "classification.three_bands", ("@n7", "cut_low",
                                                         "cut_high"))),
                 "n8", intent="process_control_band"),
        )))

    # ── supplier release ────────────────────────────────────────────────
    goods_readings = R("goods_readings", "list_readings",
                       "readings taken on the delivered goods")
    calibration_offset = R("calibration_offset", "generic_value",
                           "correction the goods-in gauge needs")
    reading_cap = R("reading_cap", "threshold_value",
                    "level no delivered item may exceed")
    mean_limit = R("mean_limit", "threshold_value",
                   "average level the purchase order specified")
    outlier_share = R("outlier_share", "threshold_percent",
                      "share of the delivery allowed to read high")
    inspection_note = R("inspection_note", "text_note",
                        "note the goods-in inspector left")
    needle = R("needle", "needle_text", "word the inspector's note has to mention")
    supplier_label = R("supplier_label", "text_label", "name of the supplier")
    places2 = R("places", "places", "decimals the goods-in record uses")

    out.append(Blueprint(
        workflow_id="quality_control.supplier_release",
        domain="quality_control",
        natural_user_goal=("decide whether a delivery can be accepted into "
                           "stock on the evidence the inspector gathered"),
        target_description="the acceptance verdict or the goods-in summary line",
        value_generator_id="quality_control.supplier_release",
        query_asset_family="goods_in_record",
        hard_distractor_families=("validation", "list"),
        boolean_balancing_strategy="calibrate_goods_in_limits",
        entity_family="procurement",
        plans=(
            Plan("goods.v5", (goods_readings, reading_cap, mean_limit,
                              inspection_note, needle),
                 (S("n1", "statistics.mean", ("goods_readings",)),
                  S("n2", "comparison.at_least", ("@n1", "mean_limit")),
                  S("n3", "validation.list_limit", ("goods_readings",
                                                    "reading_cap")),
                  S("n4", "string.validate_contains", ("inspection_note",
                                                       "needle"),
                    "did the inspector record the check we asked for"),
                  S("n5", "decision.all_of", ("@n2", "@n3", "@n4"),
                    "paperwork and readings both have to hold up")),
                 "n5", intent="acceptance_verdict"),
            Plan("goods.v7", (goods_readings, calibration_offset, places2,
                              supplier_label),
                 (S("n1", "list.map_offset", ("goods_readings",
                                              "calibration_offset"),
                    "readings once the gauge correction is applied"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.median", ("@n1",),
                    "the middle of the delivery, for comparison"),
                  S("n4", "arithmetic.abs_difference", ("@n2", "@n3"),
                    "how lopsided the delivery was"),
                  S("n5", "rates.share_percent", ("@n4", "@n2")),
                  S("n6", "format.percent", ("@n5", "places")),
                  S("n7", "string.concat", ("supplier_label", "@n6"),
                    "the line filed against the supplier")),
                 "n7", intent="supplier_skew_line"),
            Plan("goods.v10", (goods_readings, calibration_offset, reading_cap,
                               outlier_share),
                 (S("n1", "list.map_offset", ("goods_readings",
                                              "calibration_offset"),
                    "corrected readings, re-read by every rule below"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.stdev", ("@n1",)),
                  S("n4", "arithmetic.add", ("@n2", "@n3"),
                    "the line a high reading has to cross"),
                  S("n5", "list.reduce_count_above", ("@n1", "@n4")),
                  S("n6", "list.reduce_count", ("@n1",)),
                  S("n7", "rates.share_percent", ("@n5", "@n6")),
                  S("n8", "comparison.at_least", ("@n7", "outlier_share")),
                  S("n9", "validation.list_limit", ("@n1", "reading_cap"),
                    "did every corrected reading stay under the cap"),
                  S("n10", "boolean.or", ("@n8", "@n9"),
                    "either reading of the delivery can stop it")),
                 "n10", intent="delivery_hold_verdict"),
        )))

    return out
