"""Record workflows: row inspection, column summaries, site ledgers, composition.

These plans treat a record list as a table: they project a column, look a row up
by the value of one of its own text fields, read a field out of the row they
found and build a new record from what they read. The lookup value is always
taken from the table itself (project the text column, pick a position), so
``record.lookup`` can never miss, and the composition family keeps its answers
as records so the module contributes object sinks rather than more floats.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── single row inspection ───────────────────────────────────────────
    row = R("row", "record_row", "the row the clerk has open")
    amount_field = R("amount_field", "field_name",
                     "which numeric column of the row is meant")
    text_field = R("text_field", "text_field_name",
                   "which text column of the row is meant")
    unit = R("unit", "unit_word", "the unit the column is reported in")
    places = R("places", "places", "how many decimals the report keeps")
    out.append(Blueprint(
        workflow_id="record.row_field_inspection",
        domain="record_processing",
        natural_user_goal="report what a single table row actually says",
        target_description="the printed line for the row",
        value_generator_id="record.single_row",
        query_asset_family="table_row",
        hard_distractor_families=("record", "format"),
        entity_family="operations",
        plans=(
            Plan("row.v2", (row, amount_field, unit),
                 (S("n1", "record.select", ("row", "amount_field")),
                  S("n2", "format.with_unit", ("@n1", "unit"))),
                 "n2", intent="row_value_with_unit"),
            Plan("row.v4", (row, amount_field, text_field),
                 (S("n1", "record.select", ("row", "amount_field")),
                  S("n2", "record.select_text", ("row", "text_field")),
                  S("n3", "format.tag", ("@n2", "@n1")),
                  S("n4", "string.normalize_title", ("@n3",))),
                 "n4", intent="row_reference_line"),
            Plan("row.v7", (row, amount_field, text_field, places),
                 (S("n1", "record.select", ("row", "amount_field")),
                  S("n2", "record.select_text", ("row", "text_field"),
                    "the row's label, read by two branches"),
                  S("n3", "string.normalize_slug", ("@n2",)),
                  S("n4", "string.count_length", ("@n2",)),
                  S("n5", "rates.share_percent", ("@n1", "@n4"),
                    "the value measured against the label length"),
                  S("n6", "format.percent", ("@n5", "places")),
                  S("n7", "string.concat", ("@n3", "@n6"))),
                 "n7", intent="row_density_line"),
        )))

    # ── column summary over a record list ───────────────────────────────
    rows = R("rows", "record_list", "the rows the extract contains")
    col_amount = R("col_amount", "field_name",
                   "which numeric column the summary is about")
    col_text = R("col_text", "text_field_name",
                 "which text column names the rows")
    col_slot = R("col_slot", "index_position",
                 "which row of the extract is highlighted")
    mean_floor = R("mean_floor", "threshold_value",
                   "the column average the extract has to reach")
    even_cut = R("even_cut", "cut_low",
                 "a share of above-average rows that still counts as even")
    skewed_cut = R("skewed_cut", "cut_high",
                   "the share of above-average rows that counts as skewed")
    out.append(Blueprint(
        workflow_id="record.table_column_summary",
        domain="record_processing",
        natural_user_goal=("summarise one column of a table extract and say "
                           "how its rows sit around the column average"),
        target_description=("the column verdict, its evenness band or the "
                            "highlighted row's line"),
        value_generator_id="record.column_extract",
        query_asset_family="table_extract",
        hard_distractor_families=("record", "statistics"),
        boolean_balancing_strategy="threshold_band",
        entity_family="logistics",
        plans=(
            Plan("column.v3", (rows, col_amount, mean_floor),
                 (S("n1", "record.project", ("rows", "col_amount"),
                    "the column pulled out as a series"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "comparison.at_least", ("@n2", "mean_floor"))),
                 "n3", intent="column_average_verdict"),
            Plan("column.v6", (rows, col_amount, even_cut, skewed_cut),
                 (S("n1", "record.project", ("rows", "col_amount")),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "record.aggregate_count",
                    ("rows", "col_amount", "@n2"),
                    "how many rows beat their own column average"),
                  S("n4", "record.aggregate_size", ("rows",)),
                  S("n5", "rates.share_percent", ("@n3", "@n4")),
                  S("n6", "classification.three_bands",
                    ("@n5", "even_cut", "skewed_cut"))),
                 "n6", intent="column_evenness_band"),
            Plan("column.v9", (rows, col_amount, col_text, col_slot),
                 (S("n1", "record.project", ("rows", "col_amount"),
                    "the column, totalled again four calls later"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "list.filter", ("@n1", "@n2")),
                  S("n4", "list.reduce_sum", ("@n3",)),
                  S("n5", "list.reduce_sum", ("@n1",)),
                  S("n6", "rates.share_percent", ("@n4", "@n5"),
                    "what share of the column the selected rows carry"),
                  S("n7", "record.project_text", ("rows", "col_text")),
                  S("n8", "list.index_text", ("@n7", "col_slot")),
                  S("n9", "format.tag", ("@n8", "@n6"))),
                 "n9", intent="highlighted_row_share_line"),
        )))

    # ── site ledger comparison ──────────────────────────────────────────
    ledger = R("ledger", "record_list", "the rows of the site ledger")
    ledger_amount = R("ledger_amount", "field_name",
                      "which numeric column the ledger is read on")
    ledger_site = R("ledger_site", "text_field_name",
                    "which text column identifies the site")
    ledger_slot = R("ledger_slot", "index_position",
                    "which site of the ledger is being questioned")
    ledger_share = R("ledger_share", "threshold_percent",
                     "the share of the ledger the quiet rows may carry")
    out.append(Blueprint(
        workflow_id="record.site_ledger_comparison",
        domain="record_processing",
        natural_user_goal=("place one site of a ledger against what the whole "
                           "ledger does"),
        target_description="the site's standing in the ledger",
        value_generator_id="record.site_ledger",
        query_asset_family="site_ledger",
        hard_distractor_families=("record", "comparison"),
        boolean_balancing_strategy="threshold_band",
        entity_family="logistics",
        plans=(
            Plan("ledger.v6", (ledger, ledger_amount, ledger_site, ledger_slot),
                 (S("n1", "record.project_text", ("ledger", "ledger_site"),
                    "the sites the ledger names"),
                  S("n2", "list.index_text", ("@n1", "ledger_slot")),
                  S("n3", "record.lookup",
                    ("ledger", "ledger_site", "@n2"),
                    "the row belonging to that site"),
                  S("n4", "record.select", ("@n3", "ledger_amount")),
                  S("n5", "record.aggregate_mean",
                    ("ledger", "ledger_amount")),
                  S("n6", "rates.share_percent", ("@n4", "@n5"))),
                 "n6", intent="site_against_ledger_average"),
            Plan("ledger.v8", (ledger, ledger_amount, ledger_share),
                 (S("n1", "record.project", ("ledger", "ledger_amount"),
                    "the ledger column, totalled again four calls later"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "list.filter", ("@n1", "@n2")),
                  S("n4", "list.reduce_sum", ("@n3",)),
                  S("n5", "list.reduce_sum", ("@n1",)),
                  S("n6", "arithmetic.subtract", ("@n5", "@n4"),
                    "what the rows outside the selection carry"),
                  S("n7", "rates.share_percent", ("@n6", "@n5")),
                  S("n8", "comparison.at_least", ("@n7", "ledger_share"))),
                 "n8", intent="ledger_tail_verdict"),
            Plan("ledger.v10",
                 (ledger, ledger_amount, ledger_site, ledger_slot),
                 (S("n1", "record.project_text", ("ledger", "ledger_site")),
                  S("n2", "list.index_text", ("@n1", "ledger_slot")),
                  S("n3", "record.lookup", ("ledger", "ledger_site", "@n2")),
                  S("n4", "record.select", ("@n3", "ledger_amount"),
                    "the questioned site's amount, used four calls later"),
                  S("n5", "record.aggregate_sum", ("ledger", "ledger_amount"),
                    "the ledger total, used by both shares"),
                  S("n6", "record.aggregate_max", ("ledger", "ledger_amount")),
                  S("n7", "record.aggregate_size", ("ledger",)),
                  S("n8", "rates.share_percent", ("@n4", "@n5")),
                  S("n9", "rates.share_percent", ("@n6", "@n5")),
                  S("n10", "statistics.mean_three", ("@n8", "@n9", "@n7"))),
                 "n10", intent="site_leader_and_size_profile"),
        )))

    # ── record composition (object answers) ─────────────────────────────
    entries = R("entries", "record_list", "the entries the extract contains")
    entry_row = R("entry_row", "record_row", "the entry the clerk has open")
    entry_amount = R("entry_amount", "field_name",
                     "which numeric column the new entry carries")
    entry_label = R("entry_label", "text_field_name",
                    "which text column names the new entry")
    entry_slot = R("entry_slot", "index_position",
                   "which entry of the extract is being rewritten")
    out.append(Blueprint(
        workflow_id="record.entry_composition",
        domain="record_processing",
        natural_user_goal="write the summary entry a table extract boils down to",
        target_description="the composed entry",
        value_generator_id="record.entry_extract",
        query_asset_family="entry_extract",
        hard_distractor_families=("record", "string"),
        entity_family="operations",
        plans=(
            Plan("entry.v3", (entry_row, entry_amount, entry_label),
                 (S("n1", "record.select_text", ("entry_row", "entry_label")),
                  S("n2", "record.select", ("entry_row", "entry_amount")),
                  S("n3", "record.build", ("@n1", "@n2"))),
                 "n3", intent="row_restated_as_entry"),
            Plan("entry.v4", (entries, entry_amount, entry_label, entry_slot),
                 (S("n1", "record.project_text", ("entries", "entry_label")),
                  S("n2", "list.index_text", ("@n1", "entry_slot")),
                  S("n3", "record.aggregate_mean",
                    ("entries", "entry_amount")),
                  S("n4", "record.build", ("@n2", "@n3"))),
                 "n4", intent="named_average_entry"),
            Plan("entry.v5", (entries, entry_amount, entry_label, entry_slot),
                 (S("n1", "record.project_text", ("entries", "entry_label")),
                  S("n2", "list.index_text", ("@n1", "entry_slot"),
                    "the name, reused by the lookup and by the new entry"),
                  S("n3", "record.lookup",
                    ("entries", "entry_label", "@n2")),
                  S("n4", "record.select", ("@n3", "entry_amount")),
                  S("n5", "record.build", ("@n2", "@n4"))),
                 "n5", intent="looked_up_entry_rebuilt"),
        )))

    return out
