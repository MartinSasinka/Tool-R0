"""Validation workflows: does this batch, this count or this card pass?

Every plan ends on a check that can be calibrated against the value it is about
to judge, and the multi-condition plans either combine such checks or count how
many of them held before judging the count. The intermediate work is real
validation work: bounding a figure, testing a whole series against a ceiling,
measuring how far two independent counts have drifted apart.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── batch spec check ────────────────────────────────────────────────
    readings = R("batch_readings", "list_readings",
                 "the measurements taken across the batch")
    offset = R("calibration_offset", "generic_value",
               "the offset the calibration certificate adds to every measurement")
    nominal = R("nominal_value", "threshold_value",
                "the value the batch is supposed to hit")
    allowance = R("allowance", "tolerance_value",
                  "how far from nominal the batch may sit")
    ceiling = R("ceiling", "threshold_value",
                "the value no single measurement may exceed")
    control_low = R("control_low", "range_low",
                    "the bottom of the control band")
    control_high = R("control_high", "range_high",
                     "the top of the control band")
    alert_level = R("alert_level", "score_points",
                    "the level above which a measurement counts as an alert")
    max_alerts = R("max_alerts", "threshold_count",
                   "how many alerts the batch is allowed to raise")
    out.append(Blueprint(
        workflow_id="validation_rules.batch_spec_check",
        domain="validation_rules",
        natural_user_goal=("decide whether a batch of measurements is inside "
                           "specification"),
        target_description="the specification verdict for the batch",
        value_generator_id="validation_rules.batch_readings",
        query_asset_family="qa_record",
        hard_distractor_families=("validation", "statistics"),
        boolean_balancing_strategy="calibrate_spec_threshold",
        entity_family="quality",
        plans=(
            Plan("spec.v2", (readings, nominal, allowance),
                 (S("n1", "statistics.mean", ("batch_readings",),
                    "where the batch sits on average"),
                  S("n2", "validation.tolerance", ("@n1", "nominal_value",
                                                   "allowance"))),
                 "n2", intent="batch_on_nominal"),
            Plan("spec.v3", (readings, offset, ceiling),
                 (S("n1", "list.map_offset", ("batch_readings",
                                              "calibration_offset"),
                    "the corrected measurements"),
                  S("n2", "validation.list_limit", ("@n1", "ceiling")),
                  S("n3", "boolean.not", ("@n2",),
                    "the batch is escalated when the ceiling is breached")),
                 "n3", intent="ceiling_breach_escalation"),
            Plan("spec.v5", (readings, offset, nominal, allowance, ceiling),
                 (S("n1", "list.map_offset", ("batch_readings",
                                              "calibration_offset"),
                    "corrected measurements, judged twice"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "validation.tolerance", ("@n2", "nominal_value",
                                                   "allowance")),
                  S("n4", "validation.list_limit", ("@n1", "ceiling")),
                  S("n5", "boolean.and", ("@n3", "@n4"))),
                 "n5", intent="batch_release_verdict"),
            Plan("spec.v7", (readings, offset, control_low, control_high,
                             ceiling),
                 (S("n1", "list.map_offset", ("batch_readings",
                                              "calibration_offset")),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.stdev", ("@n1",)),
                  S("n4", "arithmetic.add", ("@n2", "@n3"),
                    "the upper control point of the batch"),
                  S("n5", "validation.in_range", ("@n4", "control_low",
                                                  "control_high")),
                  S("n6", "validation.list_limit", ("@n1", "ceiling")),
                  S("n7", "boolean.xor", ("@n5", "@n6"),
                    "exactly one of the two checks may fail")),
                 "n7", intent="control_point_disagreement"),
            Plan("spec.v9", (readings, offset, nominal, allowance, alert_level,
                             max_alerts, ceiling),
                 (S("n1", "list.map_offset", ("batch_readings",
                                              "calibration_offset"),
                    "corrected measurements, used three separate times"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.stdev", ("@n1",)),
                  S("n4", "arithmetic.add", ("@n2", "@n3")),
                  S("n5", "validation.tolerance", ("@n4", "nominal_value",
                                                   "allowance")),
                  S("n6", "list.reduce_count_above", ("@n1", "alert_level")),
                  S("n7", "comparison.at_least", ("@n6", "max_alerts")),
                  S("n8", "validation.list_limit", ("@n1", "ceiling")),
                  S("n9", "decision.majority", ("@n5", "@n7", "@n8"))),
                 "n9", intent="batch_majority_verdict"),
        )))

    # ── stock count reconciliation ──────────────────────────────────────
    counted = R("counted_lines", "list_quantities",
                "what the stock count found on each line")
    ledger = R("ledger", "mapping_counts",
               "what the ledger says is in stock for each product")
    gap_low = R("gap_low", "range_low",
                "the smallest discrepancy that is still worth recording")
    gap_high = R("gap_high", "range_high",
                 "the largest discrepancy the count may show")
    target_gap = R("target_gap", "threshold_percent",
                   "the discrepancy the process normally produces")
    drift_allowance = R("drift_allowance", "tolerance_value",
                        "how far from that the count may drift")
    line_floor = R("line_floor", "threshold_count",
                   "the average line size the count has to show")
    out.append(Blueprint(
        workflow_id="validation_rules.stock_count_reconciliation",
        domain="validation_rules",
        natural_user_goal=("reconcile a physical stock count against the ledger "
                           "and decide whether it can be signed off"),
        target_description="the reconciliation verdict",
        value_generator_id="validation_rules.stock_count",
        query_asset_family="count_sheet",
        hard_distractor_families=("dictionary", "validation"),
        boolean_balancing_strategy="calibrate_discrepancy_threshold",
        entity_family="warehouse",
        plans=(
            Plan("recon.v4", (counted, ledger, gap_low, gap_high),
                 (S("n1", "list.reduce_sum", ("counted_lines",),
                    "what the count found in total"),
                  S("n2", "dictionary.aggregate_sum", ("ledger",),
                    "what the ledger expects in total"),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2")),
                  S("n4", "validation.in_range", ("@n3", "gap_low",
                                                  "gap_high"))),
                 "n4", intent="discrepancy_in_band"),
            Plan("recon.v6", (counted, ledger, target_gap, drift_allowance),
                 (S("n1", "list.reduce_sum", ("counted_lines",)),
                  S("n2", "dictionary.aggregate_sum", ("ledger",),
                    "the ledger total, needed twice"),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2")),
                  S("n4", "rates.share_percent", ("@n3", "@n2")),
                  S("n5", "validation.tolerance", ("@n4", "target_gap",
                                                   "drift_allowance")),
                  S("n6", "boolean.not", ("@n5",),
                    "an unusual discrepancy is what triggers the recount")),
                 "n6", intent="recount_trigger"),
            Plan("recon.v9", (counted, ledger, target_gap, drift_allowance,
                              line_floor, gap_low, gap_high),
                 (S("n1", "list.reduce_sum", ("counted_lines",)),
                  S("n2", "dictionary.aggregate_sum", ("ledger",)),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2"),
                    "the discrepancy, judged again at the end"),
                  S("n4", "rates.share_percent", ("@n3", "@n2")),
                  S("n5", "validation.tolerance", ("@n4", "target_gap",
                                                   "drift_allowance")),
                  S("n6", "statistics.mean", ("counted_lines",),
                    "the average line the count produced"),
                  S("n7", "comparison.at_least", ("@n6", "line_floor")),
                  S("n8", "validation.in_range", ("@n3", "gap_low",
                                                  "gap_high")),
                  S("n9", "decision.all_of", ("@n5", "@n7", "@n8"))),
                 "n9", intent="sign_off_verdict"),
        )))

    # ── filed document check ────────────────────────────────────────────
    card = R("filed_entry", "record_row",
             "the entry as it was filed")
    num_field = R("value_field", "field_name",
                  "the numeric field the rules apply to")
    text_field = R("text_field", "text_field_name",
                   "the text field the rules apply to")
    needle = R("expected_fragment", "needle_text",
               "the wording the entry is supposed to mention")
    bound_low = R("bound_low", "range_low",
                  "the smallest figure the entry may carry")
    bound_high = R("bound_high", "range_high",
                   "the largest figure the entry may carry")
    trim_low = R("trim_low", "cut_low",
                 "the share below which the figure was heavily trimmed")
    trim_high = R("trim_high", "cut_high",
                  "the share above which the figure was left as filed")
    min_passes = R("min_passes", "threshold_count",
                   "how many of the filing rules have to pass")
    out.append(Blueprint(
        workflow_id="validation_rules.filed_entry_check",
        domain="validation_rules",
        natural_user_goal=("check a filed entry against the rules the register "
                           "applies"),
        target_description="the filing verdict or how far the entry was trimmed",
        value_generator_id="validation_rules.filed_entry",
        query_asset_family="register_entry",
        hard_distractor_families=("record", "string"),
        boolean_balancing_strategy="calibrate_entry_fragment",
        entity_family="administration",
        plans=(
            Plan("doc.v3", (card, text_field, needle),
                 (S("n1", "record.select_text", ("filed_entry", "text_field")),
                  S("n2", "string.normalize_lower", ("@n1",),
                    "compared without worrying about capitals"),
                  S("n3", "string.validate_contains", ("@n2",
                                                       "expected_fragment"))),
                 "n3", intent="entry_wording_check"),
            Plan("doc.v4", (card, num_field, bound_low, bound_high, trim_low,
                            trim_high),
                 (S("n1", "record.select", ("filed_entry", "value_field"),
                    "the figure as filed, needed twice"),
                  S("n2", "validation.clamp", ("@n1", "bound_low",
                                               "bound_high"),
                    "the figure the register will actually store"),
                  S("n3", "rates.share_percent", ("@n2", "@n1")),
                  S("n4", "classification.three_bands", ("@n3", "trim_low",
                                                         "trim_high"))),
                 "n4", intent="trimming_band"),
            Plan("doc.v7", (card, num_field, text_field, bound_low, bound_high,
                            min_passes),
                 (S("n1", "record.select", ("filed_entry", "value_field"),
                    "the filed figure, checked twice"),
                  S("n2", "validation.non_negative", ("@n1",)),
                  S("n3", "record.select_text", ("filed_entry", "text_field")),
                  S("n4", "string.validate_identifier", ("@n3",)),
                  S("n5", "validation.in_range", ("@n1", "bound_low",
                                                  "bound_high")),
                  S("n6", "decision.count_true", ("@n2", "@n4", "@n5"),
                    "how many filing rules the entry satisfies"),
                  S("n7", "comparison.at_least", ("@n6", "min_passes"))),
                 "n7", intent="entry_acceptance_verdict"),
        )))

    # ── despatch gate with several written rules ────────────────────────
    lines = R("batch_lines", "list_quantities",
              "the quantity written on each line of the batch")
    bays = R("bay_stock", "mapping_counts",
             "what the system says is standing in each bay")
    total_floor = R("total_floor", "threshold_count",
                    "the batch size the despatch rule asks for")
    line_ceiling = R("line_ceiling", "threshold_count",
                     "the largest single line the rule allows")
    bay_ceiling = R("bay_ceiling", "threshold_count",
                    "the amount above which a bay counts as overfull")
    span_low = R("span_low", "range_low",
                 "the smallest average line the rule accepts")
    span_high = R("span_high", "range_high",
                  "the largest average line the rule accepts")
    expected_total = R("expected_total", "threshold_count",
                       "the stock the bays are supposed to be holding")
    stock_allowance = R("stock_allowance", "tolerance_value",
                        "how far the bays may sit from that")
    rules_passed = R("rules_passed", "threshold_count",
                     "how many of the rules have to hold")
    top_bays = R("top_bays", "count_small",
                 "how many of the fullest bays the rule looks at")
    bay_share_floor = R("bay_share_floor", "threshold_percent",
                        "the share of the batch those bays have to cover")
    cover_cut = R("cover_cut", "threshold_ratio",
                  "the cover ratio at which the batch counts as well stocked")
    out.append(Blueprint(
        workflow_id="validation_rules.despatch_gate",
        domain="validation_rules",
        natural_user_goal=("decide whether a batch can go out under the rules "
                           "the despatch desk works to"),
        target_description="the despatch verdict or how well the batch is covered",
        value_generator_id="validation_rules.despatch_batch",
        query_asset_family="despatch_note",
        hard_distractor_families=("validation", "decision"),
        boolean_balancing_strategy="calibrate_despatch_threshold",
        entity_family="warehouse",
        plans=(
            Plan("gate.v4", (lines, bays, cover_cut),
                 (S("n1", "dictionary.aggregate_sum", ("bay_stock",),
                    "everything the bays are holding"),
                  S("n2", "list.reduce_sum", ("batch_lines",),
                    "what the batch asks for"),
                  S("n3", "rates.ratio_of", ("@n1", "@n2"),
                    "how many times over the bays cover the batch"),
                  S("n4", "classification.threshold", ("@n3", "cover_cut"))),
                 "n4", intent="batch_cover_class"),
            Plan("gate.v5", (lines, total_floor, line_ceiling),
                 (S("n1", "list.reduce_sum", ("batch_lines",)),
                  S("n2", "comparison.at_least", ("@n1", "total_floor"),
                    "the batch is big enough to go out"),
                  S("n3", "validation.list_limit", ("batch_lines",
                                                    "line_ceiling"),
                    "no single line breaks the picking limit"),
                  S("n4", "boolean.or", ("@n2", "@n3"),
                    "either rule on its own lets the batch through"),
                  S("n5", "boolean.not", ("@n4",),
                    "the batch is held when neither holds")),
                 "n5", intent="batch_hold_verdict"),
            Plan("gate.v6", (lines, bays, total_floor, bay_ceiling,
                             line_ceiling),
                 (S("n1", "list.reduce_sum", ("batch_lines",)),
                  S("n2", "comparison.at_least", ("@n1", "total_floor")),
                  S("n3", "dictionary.aggregate_max", ("bay_stock",),
                    "the fullest bay"),
                  S("n4", "comparison.greater", ("@n3", "bay_ceiling")),
                  S("n5", "validation.list_limit", ("batch_lines",
                                                    "line_ceiling")),
                  S("n6", "decision.any_of", ("@n2", "@n4", "@n5"),
                    "any one of the three reasons sends the batch for review")),
                 "n6", intent="batch_review_trigger"),
            Plan("gate.v7", (lines, bays, total_floor, span_low, span_high,
                             expected_total, stock_allowance),
                 (S("n1", "list.reduce_sum", ("batch_lines",)),
                  S("n2", "comparison.at_least", ("@n1", "total_floor")),
                  S("n3", "statistics.mean", ("batch_lines",),
                    "the average line in the batch"),
                  S("n4", "validation.in_range", ("@n3", "span_low",
                                                  "span_high")),
                  S("n5", "dictionary.aggregate_sum", ("bay_stock",)),
                  S("n6", "validation.tolerance", ("@n5", "expected_total",
                                                   "stock_allowance")),
                  S("n7", "decision.all_of", ("@n2", "@n4", "@n6"),
                    "every rule has to hold before the batch is released")),
                 "n7", intent="batch_release_verdict"),
            Plan("gate.v8", (lines, bays, span_low, span_high, rules_passed),
                 (S("n1", "dictionary.values", ("bay_stock",),
                    "the bay figures on their own"),
                  S("n2", "validation.list_positive", ("@n1",),
                    "no bay is showing a negative"),
                  S("n3", "list.reduce_sum", ("@n1",)),
                  S("n4", "list.reduce_sum", ("batch_lines",)),
                  S("n5", "comparison.at_least", ("@n3", "@n4")),
                  S("n6", "validation.in_range", ("@n4", "span_low",
                                                  "span_high")),
                  S("n7", "decision.count_true", ("@n2", "@n5", "@n6"),
                    "how many of the desk's rules the batch satisfies"),
                  S("n8", "comparison.at_least", ("@n7", "rules_passed"))),
                 "n8", intent="rules_passed_verdict"),
            Plan("gate.v10", (lines, bays, top_bays, bay_share_floor,
                              line_ceiling, total_floor),
                 (S("n1", "dictionary.values", ("bay_stock",)),
                  S("n2", "list.map_sort_asc", ("@n1",)),
                  S("n3", "list.slice_last", ("@n2", "top_bays"),
                    "the fullest bays"),
                  S("n4", "list.reduce_sum", ("@n3",)),
                  S("n5", "list.reduce_sum", ("batch_lines",),
                    "the batch total, judged again below"),
                  S("n6", "rates.share_percent", ("@n4", "@n5")),
                  S("n7", "comparison.at_least", ("@n6", "bay_share_floor"),
                    "those bays alone can cover the batch"),
                  S("n8", "validation.list_limit", ("batch_lines",
                                                    "line_ceiling")),
                  S("n9", "comparison.at_least", ("@n5", "total_floor")),
                  S("n10", "decision.majority", ("@n7", "@n8", "@n9"))),
                 "n10", intent="despatch_majority_verdict"),
        )))

    return out
