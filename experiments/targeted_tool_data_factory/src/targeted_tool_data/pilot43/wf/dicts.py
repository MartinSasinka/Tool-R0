"""Mapping workflows: rate tables, amount rebalancing, concentration, key reports.

The family exists to bind the whole ``dictionary.*`` surface -- read a key,
write a key, drop a key, list the keys, aggregate the values -- rather than to
put a table-shaped name on an arithmetic chain. Keys are never invented: a plan
that needs one either lists the keys and picks a position or asks the table
which key carries its largest value, so a lookup can never miss. The plans of a
blueprint differ by how far the table travels: the short ones read it once, the
long ones filter it and then keep working inside the filtered table.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── rate table lookup ───────────────────────────────────────────────
    rate_table = R("rate_table", "mapping_rates",
                   "the rate the price list holds for each department")
    slot = R("slot", "index_position",
             "which department of the alphabetical list is asked about")
    share_limit = R("share_limit", "threshold_percent",
                    "the share of the remaining rates that still passes review")
    out.append(Blueprint(
        workflow_id="dict.rate_table_lookup",
        domain="dictionary_processing",
        natural_user_goal=("read one department's rate out of the rate table "
                           "and place it against the rest of the table"),
        target_description="the looked-up rate or its standing in the table",
        value_generator_id="dict.rate_table",
        query_asset_family="rate_table",
        hard_distractor_families=("dictionary", "rates"),
        boolean_balancing_strategy="threshold_band",
        entity_family="finance",
        plans=(
            Plan("ratetable.v3", (rate_table, slot),
                 (S("n1", "dictionary.keys", ("rate_table",),
                    "the departments the table covers, in order"),
                  S("n2", "list.index_text", ("@n1", "slot")),
                  S("n3", "dictionary.lookup", ("rate_table", "@n2"))),
                 "n3", intent="rate_for_department"),
            Plan("ratetable.v5", (rate_table, slot),
                 (S("n1", "dictionary.keys", ("rate_table",)),
                  S("n2", "list.index_text", ("@n1", "slot")),
                  S("n3", "dictionary.lookup", ("rate_table", "@n2")),
                  S("n4", "dictionary.aggregate_sum", ("rate_table",)),
                  S("n5", "rates.share_percent", ("@n3", "@n4"))),
                 "n5", intent="rate_share_of_table"),
            Plan("ratetable.v8", (rate_table, slot, share_limit),
                 (S("n1", "dictionary.keys", ("rate_table",)),
                  S("n2", "list.index_text", ("@n1", "slot")),
                  S("n3", "dictionary.lookup", ("rate_table", "@n2"),
                    "the rate asked about, weighed up four calls later"),
                  S("n4", "dictionary.aggregate_sum", ("rate_table",)),
                  S("n5", "dictionary.aggregate_max", ("rate_table",)),
                  S("n6", "arithmetic.subtract", ("@n4", "@n5"),
                    "the table without its leading department"),
                  S("n7", "rates.share_percent", ("@n3", "@n6")),
                  S("n8", "comparison.at_least", ("@n7", "share_limit"))),
                 "n8", intent="rate_standing_verdict"),
        )))

    # ── amount table rebalancing (mapping answers) ──────────────────────
    amount_table = R("amount_table", "mapping_amounts",
                     "the amount booked against each site")
    new_site = R("new_site", "text_label",
                 "the site the new entry is booked against")
    amount_slot = R("amount_slot", "index_position",
                    "which site of the alphabetical list the entry copies")
    out.append(Blueprint(
        workflow_id="dict.amount_table_rebalance",
        domain="dictionary_processing",
        natural_user_goal="write a further entry into the site amount table",
        target_description="the rebalanced amount table",
        value_generator_id="dict.amount_table",
        query_asset_family="amount_table",
        hard_distractor_families=("dictionary", "arithmetic"),
        entity_family="operations",
        plans=(
            Plan("rebalance.v2", (amount_table, new_site),
                 (S("n1", "dictionary.aggregate_max", ("amount_table",)),
                  S("n2", "dictionary.update",
                    ("amount_table", "new_site", "@n1"))),
                 "n2", intent="match_the_leading_site"),
            Plan("rebalance.v4", (amount_table, new_site, amount_slot),
                 (S("n1", "dictionary.keys", ("amount_table",)),
                  S("n2", "list.index_text", ("@n1", "amount_slot")),
                  S("n3", "dictionary.lookup", ("amount_table", "@n2")),
                  S("n4", "dictionary.update",
                    ("amount_table", "new_site", "@n3"))),
                 "n4", intent="copy_one_site"),
            Plan("rebalance.v6", (amount_table,),
                 (S("n1", "dictionary.values", ("amount_table",)),
                  S("n2", "statistics.mean", ("@n1",),
                    "the table average, used as its own cut"),
                  S("n3", "dictionary.aggregate_filter",
                    ("amount_table", "@n2"),
                    "the above-average part, worked on by three branches"),
                  S("n4", "dictionary.aggregate_argmax", ("@n3",)),
                  S("n5", "dictionary.aggregate_sum", ("@n3",)),
                  S("n6", "dictionary.update", ("@n3", "@n4", "@n5"),
                    "the leading site rewritten as the whole above-average total")),
                 "n6", intent="consolidate_leading_site"),
        )))

    # ── count table concentration ───────────────────────────────────────
    count_table = R("count_table", "mapping_counts",
                    "the number of items held for each product")
    low_cut = R("low_cut", "cut_low",
                "a concentration that still counts as evenly spread")
    high_cut = R("high_cut", "cut_high",
                 "the concentration that counts as dominated by one product")
    count_limit = R("count_limit", "threshold_percent",
                    "the share of products that may sit above the average")
    out.append(Blueprint(
        workflow_id="dict.count_table_concentration",
        domain="dictionary_processing",
        natural_user_goal=("judge how evenly the counts of a stock table are "
                           "spread over its products"),
        target_description="the concentration band or verdict for the table",
        value_generator_id="dict.count_table",
        query_asset_family="stock_table",
        hard_distractor_families=("dictionary", "statistics"),
        boolean_balancing_strategy="threshold_band",
        entity_family="inventory",
        plans=(
            Plan("concentration.v4", (count_table, low_cut, high_cut),
                 (S("n1", "dictionary.aggregate_max", ("count_table",)),
                  S("n2", "dictionary.aggregate_sum", ("count_table",)),
                  S("n3", "rates.share_percent", ("@n1", "@n2"),
                    "the leading product's share of the table"),
                  S("n4", "classification.three_bands",
                    ("@n3", "low_cut", "high_cut"))),
                 "n4", intent="table_concentration_band"),
            Plan("concentration.v7", (count_table, count_limit),
                 (S("n1", "dictionary.values", ("count_table",)),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "dictionary.aggregate_filter",
                    ("count_table", "@n2")),
                  S("n4", "dictionary.keys_count", ("@n3",)),
                  S("n5", "dictionary.keys_count", ("count_table",)),
                  S("n6", "rates.share_percent", ("@n4", "@n5")),
                  S("n7", "comparison.at_least", ("@n6", "count_limit"))),
                 "n7", intent="above_average_share_verdict"),
            Plan("concentration.v9", (count_table,),
                 (S("n1", "dictionary.values", ("count_table",),
                    "the counts as a series, read by two statistics"),
                  S("n2", "statistics.mean", ("@n1",),
                    "the average, needed again six calls later"),
                  S("n3", "statistics.stdev", ("@n1",)),
                  S("n4", "dictionary.aggregate_filter",
                    ("count_table", "@n2")),
                  S("n5", "dictionary.aggregate_sum", ("@n4",)),
                  S("n6", "dictionary.aggregate_sum", ("count_table",)),
                  S("n7", "rates.share_percent", ("@n5", "@n6")),
                  S("n8", "arithmetic.add", ("@n2", "@n3"),
                    "one standard deviation above the average"),
                  S("n9", "arithmetic.divide", ("@n7", "@n8"))),
                 "n9", intent="concentration_per_dispersion"),
        )))

    # ── key report (mapping -> text / list) ─────────────────────────────
    report_table = R("report_table", "mapping_rates",
                     "the rate the schedule holds for each department")
    glue = R("glue", "separator", "the separator the report is printed with")
    report_slot = R("report_slot", "index_position",
                    "which department of the report is highlighted")
    out.append(Blueprint(
        workflow_id="dict.table_key_report",
        domain="dictionary_processing",
        natural_user_goal="print the key report of a rate schedule",
        target_description="the report line or the remaining schedule",
        value_generator_id="dict.report_table",
        query_asset_family="rate_schedule",
        hard_distractor_families=("dictionary", "string"),
        entity_family="operations",
        plans=(
            Plan("keyreport.v3", (report_table, glue),
                 (S("n1", "dictionary.keys", ("report_table",)),
                  S("n2", "list.combine_join_text", ("@n1", "glue")),
                  S("n3", "string.normalize_upper", ("@n2",))),
                 "n3", intent="key_line"),
            Plan("keyreport.v5", (report_table, glue, report_slot),
                 (S("n1", "dictionary.keys", ("report_table",),
                    "the key list, joined again three calls later"),
                  S("n2", "list.index_text", ("@n1", "report_slot")),
                  S("n3", "dictionary.lookup", ("report_table", "@n2")),
                  S("n4", "list.combine_join_text", ("@n1", "glue")),
                  S("n5", "format.tag", ("@n4", "@n3"))),
                 "n5", intent="key_line_with_rate"),
            Plan("keyreport.v6", (report_table,),
                 (S("n1", "dictionary.aggregate_argmax", ("report_table",)),
                  S("n2", "dictionary.update_remove", ("report_table", "@n1"),
                    "the schedule without its leading department"),
                  S("n3", "dictionary.aggregate_argmax", ("@n2",)),
                  S("n4", "dictionary.update_remove", ("@n2", "@n3")),
                  S("n5", "dictionary.values", ("@n4",)),
                  S("n6", "list.map_sort_asc", ("@n5",))),
                 "n6", intent="remaining_rates_ordered"),
        )))

    return out
