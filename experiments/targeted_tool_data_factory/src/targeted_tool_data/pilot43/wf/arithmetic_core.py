"""Arithmetic-core workflows: packing runs, crew workload, reconciliation, stock.

These are the plans whose structure comes from plain quantity arithmetic, so the
interesting wiring has to come from genuine reuse: the padded item count is
needed both for the whole-box division and for the remainder, and the projected
workload is needed both per person and as a calendar span.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── packing run ─────────────────────────────────────────────────────
    total_items = R("total_items", "quantity_units", "items waiting in the batch")
    extra_items = R("extra_items", "quantity_units",
                    "items added to the batch after it was booked")
    per_box = R("per_box", "count_small", "items that fit in one box")
    pallet_capacity = R("pallet_capacity", "count_small",
                        "boxes that fit on one pallet")
    box_fee = R("box_fee", "money_fee", "handling fee charged for each box")
    pallet_fee = R("pallet_fee", "money_fee", "dispatch fee charged per pallet")
    fee_budget = R("fee_budget", "threshold_money",
                   "handling budget agreed for the batch")
    loose_limit = R("loose_limit", "threshold_count",
                    "loose items the packer is willing to tolerate")

    out.append(Blueprint(
        workflow_id="arithmetic.packing_run",
        domain="arithmetic_core",
        natural_user_goal=("work out how a batch of goods fills its boxes and "
                           "what the handling ends up costing"),
        target_description="the box count, the handling cost or the budget verdict",
        value_generator_id="arithmetic.packing",
        query_asset_family="packing_slip",
        hard_distractor_families=("arithmetic", "rounding"),
        boolean_balancing_strategy="calibrate_handling_budget",
        entity_family="warehouse",
        plans=(
            Plan("pack.v3", (total_items, extra_items, per_box),
                 (S("n1", "arithmetic.add", ("total_items", "extra_items"),
                    "items actually going out"),
                  S("n2", "arithmetic.divide", ("@n1", "per_box"),
                    "boxes the batch needs"),
                  S("n3", "rounding.to_int", ("@n2",), "whole boxes")),
                 "n3", intent="whole_box_count"),
            Plan("pack.v5", (total_items, extra_items, per_box, box_fee,
                             fee_budget),
                 (S("n1", "arithmetic.add", ("total_items", "extra_items")),
                  S("n2", "arithmetic.divide", ("@n1", "per_box")),
                  S("n3", "rounding.to_int", ("@n2",)),
                  S("n4", "arithmetic.multiply", ("box_fee", "@n3"),
                    "handling cost of the boxes"),
                  S("n5", "comparison.at_least", ("@n4", "fee_budget"),
                    "budget verdict")),
                 "n5", intent="handling_budget_verdict"),
            Plan("pack.v6", (total_items, extra_items, per_box, pallet_capacity,
                             pallet_fee),
                 (S("n1", "arithmetic.add", ("total_items", "extra_items")),
                  S("n2", "arithmetic.divide", ("@n1", "per_box")),
                  S("n3", "rounding.to_int", ("@n2",), "whole boxes"),
                  S("n4", "arithmetic.divide", ("@n3", "pallet_capacity")),
                  S("n5", "rounding.to_int", ("@n4",), "whole pallets"),
                  S("n6", "arithmetic.multiply", ("pallet_fee", "@n5"),
                    "dispatch cost")),
                 "n6", intent="pallet_dispatch_cost"),
            Plan("pack.v8", (total_items, extra_items, per_box, box_fee,
                             fee_budget, loose_limit),
                 (S("n1", "arithmetic.add", ("total_items", "extra_items"),
                    "padded item count, needed twice"),
                  S("n2", "arithmetic.divide", ("@n1", "per_box")),
                  S("n3", "rounding.to_int", ("@n2",)),
                  S("n4", "arithmetic.multiply", ("box_fee", "@n3")),
                  S("n5", "arithmetic.modulo", ("@n1", "per_box"),
                    "items left over after the last full box"),
                  S("n6", "comparison.at_least", ("@n4", "fee_budget")),
                  S("n7", "comparison.greater", ("@n5", "loose_limit")),
                  S("n8", "boolean.and", ("@n6", "@n7"),
                    "cost and leftovers both matter")),
                 "n8", intent="packing_two_condition_verdict"),
        )))

    # ── crew workload ───────────────────────────────────────────────────
    task_count = R("task_count", "count_items", "tasks queued for the crew")
    hours_per_task = R("hours_per_task", "duration_hours",
                       "hours one task takes to finish")
    handover_hours = R("handover_hours", "duration_hours",
                       "hours lost to handover at the end of the job")
    crew_size = R("crew_size", "count_people", "people working on the job")
    shift_limit = R("shift_limit", "threshold_hours",
                    "hours a single shift is allowed to run")
    day_limit = R("day_limit", "threshold_value",
                  "days the job may take from start to finish")
    unit_word = R("unit_word", "unit_word", "unit the report is written in")
    crew_label = R("crew_label", "text_label", "name of the team doing the work")

    out.append(Blueprint(
        workflow_id="arithmetic.crew_workload",
        domain="arithmetic_core",
        natural_user_goal=("find out how long a queue of jobs keeps a team busy "
                           "and whether that still fits one shift"),
        target_description="the workload per person or the shift verdict",
        value_generator_id="arithmetic.crew",
        query_asset_family="shift_roster",
        hard_distractor_families=("duration", "arithmetic"),
        boolean_balancing_strategy="calibrate_shift_hour_limit",
        entity_family="field_service",
        plans=(
            Plan("crew.v3", (task_count, hours_per_task, crew_size),
                 (S("n1", "arithmetic.multiply", ("hours_per_task", "task_count"),
                    "total hours of work"),
                  S("n2", "arithmetic.divide", ("@n1", "crew_size"),
                    "hours each person carries"),
                  S("n3", "duration.convert_hours_minutes", ("@n2",),
                    "the same span in minutes")),
                 "n3", intent="minutes_per_person"),
            Plan("crew.v6", (task_count, hours_per_task, crew_size, unit_word,
                             crew_label),
                 (S("n1", "arithmetic.multiply", ("hours_per_task", "task_count")),
                  S("n2", "arithmetic.divide", ("@n1", "crew_size")),
                  S("n3", "duration.convert_hours_minutes", ("@n2",)),
                  S("n4", "rounding.to_int", ("@n3",)),
                  S("n5", "format.with_unit", ("@n4", "unit_word")),
                  S("n6", "string.concat", ("crew_label", "@n5"),
                    "the line that goes on the roster")),
                 "n6", intent="roster_line"),
            Plan("crew.v7", (task_count, hours_per_task, handover_hours,
                             crew_size, shift_limit, day_limit),
                 (S("n1", "arithmetic.multiply", ("hours_per_task", "task_count"),
                    "total hours, needed twice"),
                  S("n2", "arithmetic.divide", ("@n1", "crew_size")),
                  S("n3", "duration.sum", ("@n2", "handover_hours")),
                  S("n4", "comparison.at_least", ("@n3", "shift_limit")),
                  S("n5", "duration.convert_hours_days", ("@n1",),
                    "the whole job as a calendar span"),
                  S("n6", "comparison.at_least", ("@n5", "day_limit")),
                  S("n7", "boolean.and", ("@n4", "@n6"))),
                 "n7", intent="shift_and_calendar_verdict"),
        )))

    # ── two-line reconciliation ─────────────────────────────────────────
    first_rate = R("first_rate", "money_price", "unit rate the first team booked")
    first_units = R("first_units", "quantity_units", "units the first team booked")
    second_rate = R("second_rate", "money_price",
                    "unit rate the second team booked")
    second_units = R("second_units", "quantity_units",
                     "units the second team booked")
    third_rate = R("third_rate", "money_price", "unit rate in the ledger extract")
    third_units = R("third_units", "quantity_units",
                    "units recorded in the ledger extract")
    uplift_rate = R("uplift_rate", "percent_margin",
                    "surcharge applied while the books were being closed")
    gap_limit = R("gap_limit", "threshold_money", "gap the reviewer will accept")
    share_limit = R("share_limit", "threshold_percent",
                    "share of the average the gap may represent")

    out.append(Blueprint(
        workflow_id="arithmetic.two_line_reconciliation",
        domain="arithmetic_core",
        natural_user_goal=("see how far two separately compiled totals for the "
                           "same delivery sit apart"),
        target_description="the size of the disagreement or the review verdict",
        value_generator_id="arithmetic.reconciliation",
        query_asset_family="ledger_extract",
        hard_distractor_families=("arithmetic", "statistics"),
        boolean_balancing_strategy="calibrate_reconciliation_gap",
        entity_family="finance",
        plans=(
            Plan("recon.v3", (first_rate, first_units, second_rate, second_units),
                 (S("n1", "arithmetic.multiply", ("first_rate", "first_units"),
                    "first team's total"),
                  S("n2", "arithmetic.multiply", ("second_rate", "second_units"),
                    "second team's total, worked out independently"),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2"),
                    "the disagreement")),
                 "n3", intent="line_gap"),
            Plan("recon.v6", (first_rate, first_units, second_rate, second_units,
                              third_rate, third_units, uplift_rate),
                 (S("n1", "arithmetic.multiply", ("first_rate", "first_units")),
                  S("n2", "arithmetic.multiply", ("second_rate", "second_units")),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2")),
                  S("n4", "rates.increase_by_percent", ("@n3", "uplift_rate")),
                  S("n5", "arithmetic.multiply", ("third_rate", "third_units"),
                    "the ledger's own figure"),
                  S("n6", "arithmetic.abs_difference", ("@n4", "@n5"),
                    "how far the surcharged gap sits from the ledger")),
                 "n6", intent="ledger_residual"),
            Plan("recon.v8", (first_rate, first_units, second_rate, second_units,
                              gap_limit, share_limit),
                 (S("n1", "arithmetic.multiply", ("first_rate", "first_units")),
                  S("n2", "arithmetic.multiply", ("second_rate", "second_units")),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2")),
                  S("n4", "statistics.average_two", ("@n1", "@n2"),
                    "what the two teams agree on on average"),
                  S("n5", "rates.share_percent", ("@n3", "@n4")),
                  S("n6", "comparison.at_least", ("@n5", "share_limit")),
                  S("n7", "comparison.at_least", ("@n3", "gap_limit")),
                  S("n8", "boolean.or", ("@n6", "@n7"),
                    "either reading of the gap can fail the review")),
                 "n8", intent="reconciliation_verdict"),
        )))

    # ── stock replenishment ─────────────────────────────────────────────
    shelf_capacity = R("shelf_capacity", "quantity_stock",
                       "units the shelf holds when full")
    current_stock = R("current_stock", "quantity_stock",
                      "units sitting on the shelf now")
    incoming_units = R("incoming_units", "quantity_units",
                       "units already on their way to the shop")
    weekly_sales = R("weekly_sales", "count_items", "units sold in a normal week")
    case_size = R("case_size", "count_small", "units in one supplier case")
    refill_target = R("refill_target", "threshold_count",
                      "cases the manager expects to order")
    loose_limit2 = R("loose_limit", "threshold_count",
                     "single units the manager will accept ordering")
    cover_target = R("cover_target", "threshold_value",
                     "weeks of cover the shop wants to hold")
    product_label = R("product_label", "text_label", "name of the product")
    places = R("places", "places", "decimals the report is written to")

    out.append(Blueprint(
        workflow_id="arithmetic.stock_replenishment",
        domain="arithmetic_core",
        natural_user_goal=("decide how much of a product to reorder so the shelf "
                           "is full again without over-ordering"),
        target_description="the replenishment size, its label or the order verdict",
        value_generator_id="arithmetic.replenishment",
        query_asset_family="stock_card",
        hard_distractor_families=("arithmetic", "format"),
        boolean_balancing_strategy="calibrate_reorder_thresholds",
        entity_family="retail",
        plans=(
            Plan("stock.v5", (shelf_capacity, current_stock, case_size,
                              refill_target, loose_limit2),
                 (S("n1", "arithmetic.subtract", ("shelf_capacity",
                                                  "current_stock"),
                    "the gap on the shelf, needed twice"),
                  S("n2", "arithmetic.divide", ("@n1", "case_size")),
                  S("n3", "comparison.at_least", ("@n2", "refill_target")),
                  S("n4", "comparison.greater", ("@n1", "loose_limit")),
                  S("n5", "boolean.and", ("@n3", "@n4"))),
                 "n5", intent="reorder_verdict"),
            Plan("stock.v6", (shelf_capacity, current_stock, weekly_sales,
                              places, product_label),
                 (S("n1", "arithmetic.subtract", ("shelf_capacity",
                                                  "current_stock")),
                  S("n2", "arithmetic.divide", ("@n1", "weekly_sales"),
                    "weeks of cover the gap represents"),
                  S("n3", "rounding.places", ("@n2", "places")),
                  S("n4", "format.number_text", ("@n3",)),
                  S("n5", "string.concat", ("product_label", "@n4")),
                  S("n6", "string.normalize_upper", ("@n5",),
                    "the tag printed on the pick list")),
                 "n6", intent="cover_tag"),
            Plan("stock.v9", (shelf_capacity, current_stock, incoming_units,
                              weekly_sales, case_size, refill_target,
                              loose_limit2, cover_target),
                 (S("n1", "arithmetic.add", ("current_stock", "incoming_units"),
                    "stock once the delivery lands"),
                  S("n2", "arithmetic.subtract", ("shelf_capacity", "@n1"),
                    "remaining gap, read three ways"),
                  S("n3", "arithmetic.divide", ("@n2", "case_size")),
                  S("n4", "arithmetic.modulo", ("@n2", "case_size")),
                  S("n5", "arithmetic.divide", ("@n2", "weekly_sales")),
                  S("n6", "comparison.at_least", ("@n3", "refill_target")),
                  S("n7", "comparison.greater", ("@n4", "loose_limit")),
                  S("n8", "comparison.at_least", ("@n5", "cover_target")),
                  S("n9", "decision.majority", ("@n6", "@n7", "@n8"),
                    "most of the reorder rules must agree")),
                 "n9", intent="replenishment_majority"),
        )))

    return out
