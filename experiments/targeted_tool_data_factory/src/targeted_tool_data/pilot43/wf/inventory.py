"""Inventory workflows: replenishment, the stock ledger, building a pick list.

A pick is always taken as a share of what is actually on the shelf, so a plan
never asks for more units than exist. The ledger plans read the depot as records
and the pick list reads it as a mapping, so both list-shaped and mapping-shaped
stock data appear alongside the arithmetic instead of being replaced by it.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── replenishment ───────────────────────────────────────────────────
    stock = R("stock_on_hand", "quantity_stock", "how many units the shelf holds")
    pick_rate = R("reserved_share", "percent_share",
                  "the share of the shelf already reserved for orders")
    reorder = R("reorder_point", "threshold_count",
                "the level at which we have to reorder")
    label = R("item_label", "text_label", "what the item is called")
    places = R("places", "places", "how many decimals the stock report shows")
    rows = R("depot_rows", "record_list", "one row per depot holding this item")
    field = R("stock_field", "field_name", "the column the depot rows are read from")
    share_cap = R("concentration_cap", "threshold_percent",
                  "the share one depot may hold before we call it concentrated")

    out.append(Blueprint(
        workflow_id="inventory.replenishment_check",
        domain="inventory",
        natural_user_goal=("see whether the shelf still covers us once the "
                           "reserved units are taken out"),
        target_description="the remaining stock or the reorder verdict",
        value_generator_id="inventory.shelf",
        query_asset_family="stock_position",
        hard_distractor_families=("rates", "record"),
        boolean_balancing_strategy="calibrate_reorder_point",
        entity_family="warehouse",
        plans=(
            Plan("inv.v3", (stock, pick_rate, reorder),
                 (S("n1", "rates.percent_of", ("reserved_share", "stock_on_hand"),
                    "the reserved units can never exceed the shelf"),
                  S("n2", "arithmetic.subtract", ("stock_on_hand", "@n1")),
                  S("n3", "comparison.at_least", ("@n2", "reorder_point"))),
                 "n3", intent="still_above_reorder_point"),
            Plan("inv.v5", (stock, pick_rate, places, label),
                 (S("n1", "rates.percent_of", ("reserved_share", "stock_on_hand")),
                  S("n2", "arithmetic.subtract", ("stock_on_hand", "@n1")),
                  S("n3", "rates.share_percent", ("@n2", "stock_on_hand")),
                  S("n4", "format.percent", ("@n3", "places")),
                  S("n5", "string.concat", ("item_label", "@n4"))),
                 "n5", intent="free_stock_line"),
            Plan("inv.v9", (rows, field, share_cap, reorder),
                 (S("n1", "record.project", ("depot_rows", "stock_field")),
                  S("n2", "list.reduce_sum", ("@n1",)),
                  S("n3", "record.aggregate_max", ("depot_rows", "stock_field"),
                    "the largest single holding, read straight off the rows"),
                  S("n4", "rates.share_percent", ("@n3", "@n2")),
                  S("n5", "record.aggregate_size", ("depot_rows",)),
                  S("n6", "rates.ratio_of", ("@n2", "@n5"),
                    "what an average depot in the network holds"),
                  S("n7", "comparison.at_least", ("@n4", "concentration_cap")),
                  S("n8", "comparison.at_least", ("@n6", "reorder_point")),
                  S("n9", "boolean.or", ("@n7", "@n8"))),
                 "n9", intent="network_stock_verdict"),
        )))

    # ── stock ledger ────────────────────────────────────────────────────
    led_rows = R("ledger_rows", "record_list", "the rows of the stock ledger")
    led_text = R("group_field", "text_field_name", "the column the ledger groups by")
    led_field = R("amount_field", "field_name", "the column the amounts sit in")
    led_slot = R("slot", "index_position", "which group we are asked about")
    led_sep = R("separator", "separator", "the delimiter the ledger export uses")
    led_places = R("places", "places", "how many decimals the ledger shows")
    led_floor = R("amount_floor", "threshold_value",
                  "the amount a group has to reach to matter")

    out.append(Blueprint(
        workflow_id="inventory.stock_ledger",
        domain="inventory",
        natural_user_goal=("look up what one group in the stock ledger holds and "
                           "how it compares with the rest"),
        target_description="the ledger group, its holding or its share",
        value_generator_id="inventory.ledger",
        query_asset_family="stock_ledger",
        hard_distractor_families=("record", "list"),
        boolean_balancing_strategy="calibrate_amount_floor",
        entity_family="warehouse",
        plans=(
            Plan("led.v2", (led_rows, led_text),
                 (S("n1", "record.project_text", ("ledger_rows", "group_field")),
                  S("n2", "list.map_sort_text", ("@n1",))),
                 "n2", intent="ledger_groups"),
            Plan("led.v5", (led_rows, led_text, led_field, led_slot, led_floor),
                 (S("n1", "record.project_text", ("ledger_rows", "group_field")),
                  S("n2", "list.index_text", ("@n1", "slot")),
                  S("n3", "record.lookup", ("ledger_rows", "group_field", "@n2")),
                  S("n4", "record.select", ("@n3", "amount_field")),
                  S("n5", "comparison.at_least", ("@n4", "amount_floor"))),
                 "n5", intent="group_worth_reporting"),
            Plan("led.v8", (led_rows, led_text, led_field, led_slot, led_places),
                 (S("n1", "record.project_text", ("ledger_rows", "group_field")),
                  S("n2", "list.map_sort_text", ("@n1",)),
                  S("n3", "list.index_text", ("@n2", "slot")),
                  S("n4", "record.lookup", ("ledger_rows", "group_field", "@n3")),
                  S("n5", "record.select", ("@n4", "amount_field")),
                  S("n6", "record.aggregate_sum", ("ledger_rows", "amount_field"),
                    "the whole ledger, summed on its own branch"),
                  S("n7", "rates.share_percent", ("@n5", "@n6")),
                  S("n8", "format.percent", ("@n7", "places"))),
                 "n8", intent="group_share_of_ledger"),
        )))

    # ── pick list ───────────────────────────────────────────────────────
    counts = R("shelf_counts", "mapping_counts", "how many units each item has")
    pik_sep = R("separator", "separator", "the delimiter the pick list is printed with")
    pik_unit = R("unit", "unit_word", "the unit the pick list is written in")
    pik_slot = R("slot", "index_position", "which item on the list we are picking")
    pik_rate = R("pick_share", "percent_share",
                 "the share of the shelf this order takes")
    pik_places = R("places", "places", "how many decimals the pick list shows")
    pik_cap = R("concentration_cap", "threshold_percent",
                "the share one item may make up before the list is lopsided")

    out.append(Blueprint(
        workflow_id="inventory.pick_list",
        domain="inventory",
        natural_user_goal=("put together the pick list for a depot order without "
                           "taking more than the shelf holds"),
        target_description="the pick list entry or how the depot is left",
        value_generator_id="inventory.pick_run",
        query_asset_family="pick_list",
        hard_distractor_families=("dictionary", "rates"),
        boolean_balancing_strategy="calibrate_concentration_cap",
        entity_family="warehouse",
        plans=(
            Plan("pik.v3", (counts, pik_sep),
                 (S("n1", "dictionary.keys", ("shelf_counts",)),
                  S("n2", "list.combine_join_text", ("@n1", "separator"),
                    "the pick list as the handheld prints it"),
                  S("n3", "string.split_count", ("@n2", "separator"))),
                 "n3", intent="lines_on_the_pick_list"),
            Plan("pik.v4", (counts, pik_unit),
                 (S("n1", "dictionary.aggregate_argmax", ("shelf_counts",)),
                  S("n2", "dictionary.lookup", ("shelf_counts", "@n1")),
                  S("n3", "format.with_unit", ("@n2", "unit")),
                  S("n4", "string.concat", ("@n1", "@n3"))),
                 "n4", intent="biggest_holding_line"),
            Plan("pik.v6", (counts, pik_cap),
                 (S("n1", "dictionary.values", ("shelf_counts",)),
                  S("n2", "list.reduce_sum", ("@n1",)),
                  S("n3", "dictionary.aggregate_max", ("shelf_counts",),
                    "the largest holding, read on its own branch"),
                  S("n4", "rates.share_percent", ("@n3", "@n2")),
                  S("n5", "comparison.at_least", ("@n4", "concentration_cap")),
                  S("n6", "boolean.not", ("@n5",))),
                 "n6", intent="depot_is_balanced"),
            Plan("pik.v9", (counts, pik_slot, pik_rate, pik_places),
                 (S("n1", "dictionary.keys", ("shelf_counts",)),
                  S("n2", "list.index_text", ("@n1", "slot"),
                    "the item we are picking, named again at the very end"),
                  S("n3", "dictionary.lookup", ("shelf_counts", "@n2")),
                  S("n4", "rates.percent_of", ("pick_share", "@n3"),
                    "the pick is a share of what is there"),
                  S("n5", "arithmetic.subtract", ("@n3", "@n4")),
                  S("n6", "dictionary.aggregate_sum", ("shelf_counts",)),
                  S("n7", "rates.share_percent", ("@n5", "@n6")),
                  S("n8", "format.percent", ("@n7", "places")),
                  S("n9", "string.concat", ("@n2", "@n8"))),
                 "n9", intent="post_pick_position"),
        )))

    return out
