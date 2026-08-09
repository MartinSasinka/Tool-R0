"""Record workflows: reading rows out of a small table and summarising columns.

These plans treat a list of records the way a person treats a table export:
project a column, find the row carrying a label, read one field of it, roll a
column up, and hand back either a number or a small record of its own. The
lookups always take their key from the table itself, so the row that is asked
for is guaranteed to exist.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── site table review ───────────────────────────────────────────────
    rows = R("table_rows", "record_list",
             "the rows of the table, one per site")
    amount_field = R("value_column", "field_name",
                     "the numeric column the review is about")
    label_column = R("label_column", "text_field_name",
                     "the text column that names each row")
    row_slot = R("row_slot", "index_position",
                 "which of the listed names the question is about")
    joiner = R("separator", "separator",
               "the separator wanted between the names")
    row_target = R("row_target", "threshold_money",
                   "the level the average row has to reach")
    count_cut = R("row_cut", "count_items",
                  "the value a row has to beat to count as a large row")
    share_floor = R("share_floor", "threshold_percent",
                    "the share of the column the large rows must carry")
    count_floor = R("count_floor", "threshold_count",
                    "how many large rows the table must contain")
    out.append(Blueprint(
        workflow_id="record_processing.site_table_review",
        domain="record_processing",
        natural_user_goal=("go through a small site table and see what its rows "
                           "add up to"),
        target_description="the figure asked for or the verdict on the table",
        value_generator_id="record_processing.site_table",
        query_asset_family="table_export",
        hard_distractor_families=("record", "list"),
        boolean_balancing_strategy="calibrate_table_threshold",
        entity_family="operations",
        plans=(
            Plan("table.v2", (rows, amount_field, row_target),
                 (S("n1", "record.aggregate_mean", ("table_rows",
                                                    "value_column"),
                    "the average of that column"),
                  S("n2", "comparison.at_least", ("@n1", "row_target"))),
                 "n2", intent="average_row_verdict"),
            Plan("table.v3", (rows, label_column, joiner),
                 (S("n1", "record.project_text", ("table_rows",
                                                  "label_column"),
                    "the names held in that column"),
                  S("n2", "list.map_sort_text", ("@n1",)),
                  S("n3", "list.combine_join_text", ("@n2", "separator"))),
                 "n3", intent="ordered_name_line"),
            Plan("table.v4", (rows, amount_field),
                 (S("n1", "record.project", ("table_rows", "value_column"),
                    "that column as a plain series"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "list.filter", ("@n1", "@n2")),
                  S("n4", "list.reduce_sum", ("@n3",),
                    "what the rows on that side of the average add up to")),
                 "n4", intent="off_average_column_total"),
            Plan("table.v5", (rows, label_column, amount_field, row_slot),
                 (S("n1", "record.project_text", ("table_rows",
                                                  "label_column")),
                  S("n2", "list.index_text", ("@n1", "row_slot"),
                    "the name being asked about, kept for the answer"),
                  S("n3", "record.lookup", ("table_rows", "label_column",
                                            "@n2"),
                    "the row carrying that name"),
                  S("n4", "record.select", ("@n3", "value_column")),
                  S("n5", "record.build", ("@n2", "@n4"),
                    "that row cut down to the name and the figure")),
                 "n5", intent="single_row_extract"),
            Plan("table.v6", (rows, label_column, amount_field, row_slot),
                 (S("n1", "record.project_text", ("table_rows",
                                                  "label_column")),
                  S("n2", "list.index_text", ("@n1", "row_slot")),
                  S("n3", "record.lookup", ("table_rows", "label_column",
                                            "@n2")),
                  S("n4", "record.select", ("@n3", "value_column"),
                    "the figure that row carries"),
                  S("n5", "record.aggregate_count", ("table_rows",
                                                     "value_column", "@n4"),
                    "how many rows reach at least that figure"),
                  S("n6", "record.aggregate_size", ("table_rows",)),
                  S("n7", "rates.share_percent", ("@n5", "@n6"),
                    "the share of the table sitting at or above that row")),
                 "n7", intent="row_standing_share"),
            Plan("table.v7", (rows, label_column, amount_field, row_slot),
                 (S("n1", "record.project_text", ("table_rows",
                                                  "label_column")),
                  S("n2", "list.index_text", ("@n1", "row_slot"),
                    "the name being asked about, needed again at the end"),
                  S("n3", "record.lookup", ("table_rows", "label_column",
                                            "@n2"),
                    "the row carrying that name"),
                  S("n4", "record.select", ("@n3", "value_column")),
                  S("n5", "record.aggregate_mean", ("table_rows",
                                                    "value_column")),
                  S("n6", "arithmetic.subtract", ("@n4", "@n5"),
                    "how far that row sits from the average"),
                  S("n7", "record.build", ("@n2", "@n6"),
                    "the name with its deviation")),
                 "n7", intent="row_deviation_record"),
            Plan("table.v10", (rows, amount_field, count_cut, share_floor,
                               count_floor),
                 (S("n1", "record.project", ("table_rows", "value_column")),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "list.filter", ("@n1", "@n2")),
                  S("n4", "list.reduce_sum", ("@n3",)),
                  S("n5", "record.aggregate_sum", ("table_rows",
                                                   "value_column"),
                    "the whole column"),
                  S("n6", "rates.share_percent", ("@n4", "@n5")),
                  S("n7", "record.aggregate_count", ("table_rows",
                                                     "value_column", "row_cut"),
                    "how many rows beat the large-row level"),
                  S("n8", "comparison.at_least", ("@n6", "share_floor")),
                  S("n9", "comparison.at_least", ("@n7", "count_floor")),
                  S("n10", "boolean.and", ("@n8", "@n9"))),
                 "n10", intent="table_concentration_verdict"),
        )))

    # ── single job card ─────────────────────────────────────────────────
    card = R("job_card", "record_row",
             "the job card as it was filed")
    card_number = R("card_value_field", "field_name",
                    "the numeric field of the card that matters here")
    card_text = R("card_text_field", "text_field_name",
                  "the text field of the card that matters here")
    expected_start = R("expected_start", "prefix_text",
                       "how the entry on the card is supposed to start")
    uplift = R("revision_uplift", "percent_growth",
               "the uplift the revision applies to the figure on the card")
    low_bound = R("low_bound", "range_low",
                  "the smallest figure the card may carry")
    high_bound = R("high_bound", "range_high",
                   "the largest figure the card may carry")
    out.append(Blueprint(
        workflow_id="record_processing.job_card_summary",
        domain="record_processing",
        natural_user_goal=("turn a filed job card into the tidy entry the "
                           "system expects"),
        target_description="the tidied entry or the check on the filed card",
        value_generator_id="record_processing.job_card",
        query_asset_family="job_card",
        hard_distractor_families=("record", "string"),
        boolean_balancing_strategy="calibrate_card_prefix",
        entity_family="field_service",
        plans=(
            Plan("job.v2", (card, card_text, expected_start),
                 (S("n1", "record.select_text", ("job_card",
                                                 "card_text_field")),
                  S("n2", "string.validate_prefix", ("@n1", "expected_start"))),
                 "n2", intent="card_entry_prefix_check"),
            Plan("job.v3", (card, card_number, card_text),
                 (S("n1", "record.select", ("job_card", "card_value_field")),
                  S("n2", "record.select_text", ("job_card", "card_text_field")),
                  S("n3", "record.build", ("@n2", "@n1"),
                    "the two fields kept, everything else dropped")),
                 "n3", intent="trimmed_card_record"),
            Plan("job.v5", (card, card_number, card_text, uplift),
                 (S("n1", "record.select", ("job_card", "card_value_field")),
                  S("n2", "rates.increase_by_percent", ("@n1",
                                                        "revision_uplift")),
                  S("n3", "record.select_text", ("job_card",
                                                 "card_text_field")),
                  S("n4", "string.normalize_title", ("@n3",)),
                  S("n5", "record.build", ("@n4", "@n2"),
                    "the revised card entry")),
                 "n5", intent="revised_card_record"),
            Plan("job.v6", (card, card_number, card_text, low_bound,
                            high_bound),
                 (S("n1", "record.select_text", ("job_card",
                                                 "card_text_field")),
                  S("n2", "string.normalize_slug", ("@n1",)),
                  S("n3", "record.select", ("job_card", "card_value_field")),
                  S("n4", "validation.clamp", ("@n3", "low_bound",
                                               "high_bound"),
                    "the figure pulled back inside the allowed bounds"),
                  S("n5", "format.number_text", ("@n4",)),
                  S("n6", "string.concat", ("@n2", "@n5"),
                    "the reference the system files it under")),
                 "n6", intent="card_reference_text"),
        )))

    # ── column audit ────────────────────────────────────────────────────
    audit_rows = R("audit_rows", "record_list",
                   "the rows that came out of the export")
    audit_field = R("audit_column", "field_name",
                    "the numeric column being audited")
    ceiling = R("column_ceiling", "threshold_value",
                "the highest value the column is allowed to contain")
    min_rows = R("min_rows", "threshold_count",
                 "how many rows the export must contain to be usable")
    min_passes = R("min_passes", "threshold_count",
                   "how many of the checks have to pass")
    gap_low = R("gap_low", "cut_low",
                "the gap below which the column is called flat")
    gap_high = R("gap_high", "cut_high",
                 "the gap above which the column is called skewed")
    peak_target = R("peak_target", "threshold_ratio",
                    "the share of the column the run of peaks should reach")
    mean_low = R("mean_low", "range_low",
                 "the smallest acceptable column average")
    mean_high = R("mean_high", "range_high",
                  "the largest acceptable column average")
    out.append(Blueprint(
        workflow_id="record_processing.column_audit",
        domain="record_processing",
        natural_user_goal=("audit one column of an export before it is loaded "
                           "into the system"),
        target_description="the audit verdict or the shape of the column",
        value_generator_id="record_processing.export_rows",
        query_asset_family="table_export",
        hard_distractor_families=("record", "validation"),
        boolean_balancing_strategy="calibrate_audit_threshold",
        entity_family="operations",
        plans=(
            Plan("audit.v4", (audit_rows, audit_field, gap_low, gap_high),
                 (S("n1", "record.aggregate_max", ("audit_rows",
                                                   "audit_column")),
                  S("n2", "record.aggregate_mean", ("audit_rows",
                                                    "audit_column")),
                  S("n3", "arithmetic.subtract", ("@n1", "@n2"),
                    "how far the biggest row sits above the average"),
                  S("n4", "classification.three_bands", ("@n3", "gap_low",
                                                         "gap_high"))),
                 "n4", intent="column_skew_band"),
            Plan("audit.v7", (audit_rows, audit_field, ceiling, min_rows,
                              min_passes),
                 (S("n1", "record.project", ("audit_rows", "audit_column"),
                    "the column as a series, checked twice"),
                  S("n2", "validation.list_positive", ("@n1",)),
                  S("n3", "validation.list_limit", ("@n1", "column_ceiling")),
                  S("n4", "record.aggregate_size", ("audit_rows",)),
                  S("n5", "comparison.at_least", ("@n4", "min_rows")),
                  S("n6", "decision.count_true", ("@n2", "@n3", "@n5"),
                    "how many of the three checks passed"),
                  S("n7", "comparison.at_least", ("@n6", "min_passes"))),
                 "n7", intent="export_ready_verdict"),
            Plan("audit.v8", (audit_rows, audit_field, peak_target),
                 (S("n1", "record.project", ("audit_rows", "audit_column")),
                  S("n2", "list.map_running_max", ("@n1",),
                    "the best row seen so far at each position"),
                  S("n3", "list.combine_pairwise", ("@n1", "@n2")),
                  S("n4", "list.reduce_sum", ("@n3",)),
                  S("n5", "record.aggregate_sum", ("audit_rows",
                                                   "audit_column")),
                  S("n6", "rates.ratio_of", ("@n4", "@n5")),
                  S("n7", "classification.ratio_band", ("@n6", "peak_target")),
                  S("n8", "record.build", ("@n7", "@n5"),
                    "the verdict together with the column total")),
                 "n8", intent="column_shape_record"),
            Plan("audit.v9", (audit_rows, audit_field, min_rows, mean_low,
                              mean_high, min_passes),
                 (S("n1", "record.aggregate_mean", ("audit_rows",
                                                    "audit_column"),
                    "the column average, needed again much later"),
                  S("n2", "record.aggregate_max", ("audit_rows",
                                                   "audit_column")),
                  S("n3", "arithmetic.subtract", ("@n1", "@n2")),
                  S("n4", "validation.non_negative", ("@n3",)),
                  S("n5", "record.aggregate_size", ("audit_rows",)),
                  S("n6", "comparison.at_least", ("@n5", "min_rows")),
                  S("n7", "validation.in_range", ("@n1", "mean_low",
                                                  "mean_high")),
                  S("n8", "decision.count_true", ("@n4", "@n6", "@n7")),
                  S("n9", "comparison.at_least", ("@n8", "min_passes"))),
                 "n9", intent="column_audit_verdict"),
        )))

    return out
