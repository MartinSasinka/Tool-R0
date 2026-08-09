"""Lookup workflows: resolving a key against a table, a row and a position.

Resolution is spread over three op families -- ``dictionary.lookup`` for a keyed
table, ``record.lookup`` for a row found by one of its own fields, and
``list.index_of_value`` / ``list.index_text`` for a position in a series -- and
the point of this family is to make plans that use them together: resolve a
name in one structure, then carry the resolved value into another. The key or
position is always produced by the structure being searched, never sampled, so
a resolution step can never be handed a key the table does not hold. The plans
of a blueprint differ by how many structures the resolution has to cross.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── rate card resolution (table + row) ──────────────────────────────
    rate_card = R("rate_card", "mapping_rates",
                  "the rate the card holds for each department")
    ledger_rows = R("ledger_rows", "record_list",
                    "the rows the booking extract contains")
    site_field = R("site_field", "text_field_name",
                   "which text column of the extract names the row")
    amount_field = R("amount_field", "field_name",
                     "which numeric column of the extract is charged")
    card_slot = R("card_slot", "index_position",
                  "which entry of the alphabetical list is being resolved")
    out.append(Blueprint(
        workflow_id="lookup.rate_card_resolution",
        domain="lookup",
        natural_user_goal=("resolve a booking against the rate card that "
                           "applies to it"),
        target_description="the resolved charge line or entry",
        value_generator_id="lookup.rate_card",
        query_asset_family="rate_card",
        hard_distractor_families=("dictionary", "record"),
        entity_family="finance",
        plans=(
            Plan("card.v3", (rate_card,),
                 (S("n1", "dictionary.aggregate_argmax", ("rate_card",),
                    "the department the card charges most for"),
                  S("n2", "dictionary.lookup", ("rate_card", "@n1")),
                  S("n3", "format.tag", ("@n1", "@n2"))),
                 "n3", intent="leading_card_entry"),
            Plan("card.v6", (rate_card, ledger_rows, site_field, card_slot),
                 (S("n1", "dictionary.keys", ("rate_card",)),
                  S("n2", "list.index_text", ("@n1", "card_slot")),
                  S("n3", "dictionary.lookup", ("rate_card", "@n2"),
                    "the rate resolved on the card"),
                  S("n4", "record.project_text",
                    ("ledger_rows", "site_field")),
                  S("n5", "list.index_text", ("@n4", "card_slot")),
                  S("n6", "record.build", ("@n5", "@n3"),
                    "the row name carrying the resolved rate")),
                 "n6", intent="row_charged_at_card_rate"),
            Plan("card.v8",
                 (rate_card, ledger_rows, site_field, amount_field, card_slot),
                 (S("n1", "record.project_text",
                    ("ledger_rows", "site_field")),
                  S("n2", "list.index_text", ("@n1", "card_slot"),
                    "the row name, tagged again six calls later"),
                  S("n3", "record.lookup",
                    ("ledger_rows", "site_field", "@n2")),
                  S("n4", "record.select", ("@n3", "amount_field")),
                  S("n5", "dictionary.aggregate_argmax", ("rate_card",)),
                  S("n6", "dictionary.lookup", ("rate_card", "@n5")),
                  S("n7", "arithmetic.multiply", ("@n4", "@n6"),
                    "the row's amount charged at the card's leading rate"),
                  S("n8", "format.tag", ("@n2", "@n7"))),
                 "n8", intent="resolved_charge_line"),
        )))

    # ── position search in a series / label page ────────────────────────
    readings = R("readings", "list_readings",
                 "the readings the logger recorded in the order they arrived")
    page_labels = R("page_labels", "text_list_labels",
                    "the labels the index page lists")
    page_slot = R("page_slot", "index_position",
                  "which entry of the ordered page is being searched for")
    stem_width = R("stem_width", "count_small",
                   "how many leading characters a label family shares")
    out.append(Blueprint(
        workflow_id="lookup.catalogue_position_search",
        domain="lookup",
        natural_user_goal=("find where a value or a label sits and read the "
                           "series up to that point"),
        target_description="what the searched-for position turns out to hold",
        value_generator_id="lookup.index_page",
        query_asset_family="index_page",
        hard_distractor_families=("list", "string"),
        entity_family="field_service",
        plans=(
            Plan("search.v4", (readings,),
                 (S("n1", "list.reduce_max", ("readings",)),
                  S("n2", "list.index_of_value", ("readings", "@n1"),
                    "when the peak reading arrived"),
                  S("n3", "list.slice_first", ("readings", "@n2"),
                    "everything logged up to and including the peak"),
                  S("n4", "list.reduce_sum", ("@n3",))),
                 "n4", intent="total_up_to_the_peak"),
            Plan("search.v5", (page_labels, page_slot, stem_width),
                 (S("n1", "list.map_sort_text", ("page_labels",)),
                  S("n2", "list.index_text", ("@n1", "page_slot")),
                  S("n3", "string.truncate", ("@n2", "stem_width"),
                    "the family stem of that label"),
                  S("n4", "list.filter_prefix", ("page_labels", "@n3")),
                  S("n5", "list.reduce_count_text", ("@n4",))),
                 "n5", intent="labels_in_the_same_family"),
            Plan("search.v7", (readings, page_slot),
                 (S("n1", "list.map_sort_asc", ("readings",),
                    "the ranked series, totalled again five calls later"),
                  S("n2", "list.index", ("@n1", "page_slot")),
                  S("n3", "list.index_of_value", ("readings", "@n2"),
                    "where the ranked reading was originally logged"),
                  S("n4", "list.slice_first", ("readings", "@n3")),
                  S("n5", "list.reduce_sum", ("@n4",)),
                  S("n6", "list.reduce_sum", ("@n1",)),
                  S("n7", "rates.share_percent", ("@n5", "@n6"))),
                 "n7", intent="share_logged_before_the_rank"),
        )))

    # ── joining a keyed table with a record table ───────────────────────
    join_card = R("join_card", "mapping_rates",
                  "the rate the schedule holds for each department")
    join_rows = R("join_rows", "record_list",
                  "the rows the extract contains")
    join_site = R("join_site", "text_field_name",
                  "which text column of the extract names the row")
    join_amount = R("join_amount", "field_name",
                    "which numeric column of the extract is joined on")
    join_slot = R("join_slot", "index_position",
                  "which entry of either table is being joined")
    join_floor = R("join_floor", "threshold_value",
                   "how much of an average row the schedule has to cover "
                   "before the join is accepted")
    out.append(Blueprint(
        workflow_id="lookup.reference_table_join",
        domain="lookup",
        natural_user_goal=("join a booking extract onto the rate schedule and "
                           "see what the joined table says"),
        target_description="the joined schedule or the verdict on the join",
        value_generator_id="lookup.reference_join",
        query_asset_family="reference_join",
        hard_distractor_families=("dictionary", "record"),
        boolean_balancing_strategy="threshold_band",
        entity_family="procurement",
        plans=(
            Plan("join.v6", (join_card, join_rows, join_amount, join_floor),
                 (S("n1", "record.aggregate_sum",
                    ("join_rows", "join_amount")),
                  S("n2", "dictionary.aggregate_sum", ("join_card",)),
                  S("n3", "rates.share_percent", ("@n2", "@n1"),
                    "how much of the extract the schedule covers"),
                  S("n4", "record.aggregate_mean",
                    ("join_rows", "join_amount")),
                  S("n5", "rates.percent_of", ("@n3", "@n4"),
                    "that cover applied to an average row"),
                  S("n6", "comparison.at_least", ("@n5", "join_floor"))),
                 "n6", intent="covered_rows_verdict"),
            Plan("join.v9",
                 (join_card, join_rows, join_site, join_amount, join_slot),
                 (S("n1", "record.project_text", ("join_rows", "join_site")),
                  S("n2", "list.index_text", ("@n1", "join_slot"),
                    "the row name, written into the schedule five calls later"),
                  S("n3", "record.lookup", ("join_rows", "join_site", "@n2")),
                  S("n4", "record.select", ("@n3", "join_amount")),
                  S("n5", "dictionary.keys", ("join_card",)),
                  S("n6", "list.index_text", ("@n5", "join_slot")),
                  S("n7", "dictionary.update", ("join_card", "@n2", "@n4"),
                    "the extract's amount written into the schedule"),
                  S("n8", "dictionary.update_remove", ("@n7", "@n6"),
                    "the department that entry replaces"),
                  S("n9", "dictionary.aggregate_sum", ("@n8",))),
                 "n9", intent="joined_schedule_total"),
            Plan("join.v10",
                 (join_card, join_rows, join_site, join_amount, join_slot),
                 (S("n1", "record.project_text", ("join_rows", "join_site")),
                  S("n2", "list.index_text", ("@n1", "join_slot"),
                    "the row name, keyed into the schedule eight calls later"),
                  S("n3", "record.aggregate_mean",
                    ("join_rows", "join_amount")),
                  S("n4", "dictionary.values", ("join_card",)),
                  S("n5", "statistics.mean", ("@n4",)),
                  S("n6", "dictionary.aggregate_filter", ("join_card", "@n5"),
                    "the above-average part of the schedule, read three times"),
                  S("n7", "dictionary.aggregate_argmax", ("@n6",)),
                  S("n8", "dictionary.lookup", ("@n6", "@n7")),
                  S("n9", "arithmetic.multiply", ("@n3", "@n8")),
                  S("n10", "dictionary.update", ("@n6", "@n2", "@n9"))),
                 "n10", intent="schedule_with_joined_row"),
        )))

    return out
