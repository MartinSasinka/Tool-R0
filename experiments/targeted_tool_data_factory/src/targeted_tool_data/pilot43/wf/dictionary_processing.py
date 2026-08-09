"""Dictionary workflows: rate cards, stock ledgers and spend tables.

The unit of work here is a keyed table: read a key, replace an entry, drop an
entry, keep the rows above a level, count what is left. Several plans return a
table rather than a number, which is the answer shape the dataset is short of,
and the thresholds that decide which rows survive are computed from the table
itself so a filter can never empty it.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── rate card maintenance ───────────────────────────────────────────
    rate_card = R("rate_card", "mapping_rates",
                  "the hourly rate held for each team")
    card_key = R("team_slot", "index_position",
                 "which team of the alphabetical list is being asked about")
    out.append(Blueprint(
        workflow_id="dictionary_processing.rate_card_maintenance",
        domain="dictionary_processing",
        natural_user_goal=("keep the team rate card up to date and see what it "
                           "looks like afterwards"),
        target_description="the rate asked for or the card after the edit",
        value_generator_id="dictionary_processing.rate_card",
        query_asset_family="rate_card",
        hard_distractor_families=("dictionary", "lookup"),
        entity_family="operations",
        plans=(
            Plan("card.v2", (rate_card,),
                 (S("n1", "dictionary.aggregate_argmax", ("rate_card",),
                    "the team charging the most"),
                  S("n2", "dictionary.update_remove", ("rate_card", "@n1"),
                    "the card with that team taken out")),
                 "n2", intent="card_without_top_team"),
            Plan("card.v3", (rate_card, card_key),
                 (S("n1", "dictionary.keys", ("rate_card",),
                    "the teams in alphabetical order"),
                  S("n2", "list.index_text", ("@n1", "team_slot")),
                  S("n3", "dictionary.lookup", ("rate_card", "@n2"))),
                 "n3", intent="rate_of_listed_team"),
            Plan("card.v4", (rate_card,),
                 (S("n1", "dictionary.values", ("rate_card",)),
                  S("n2", "statistics.mean", ("@n1",),
                    "the average rate on the card"),
                  S("n3", "dictionary.aggregate_filter", ("rate_card", "@n2"),
                    "the teams charging above it"),
                  S("n4", "dictionary.keys_count", ("@n3",))),
                 "n4", intent="teams_above_average"),
            Plan("card.v9", (rate_card, card_key),
                 (S("n1", "dictionary.keys", ("rate_card",)),
                  S("n2", "list.index_text", ("@n1", "team_slot"),
                    "the team in question, needed again for the edit"),
                  S("n3", "dictionary.lookup", ("rate_card", "@n2")),
                  S("n4", "dictionary.aggregate_sum", ("rate_card",),
                    "everything the card adds up to"),
                  S("n5", "rates.share_percent", ("@n3", "@n4")),
                  S("n6", "dictionary.update", ("rate_card", "@n2", "@n5"),
                    "that team's entry replaced by its share of the card"),
                  S("n7", "dictionary.values", ("@n6",)),
                  S("n8", "statistics.mean", ("@n7",)),
                  S("n9", "dictionary.aggregate_filter", ("@n6", "@n8"),
                    "the entries of the rewritten card that stay above average")),
                 "n9", intent="rewritten_card_top_entries"),
        )))

    # ── stock ledger ────────────────────────────────────────────────────
    ledger = R("stock_ledger", "mapping_counts",
               "how many units of each product the ledger holds")
    new_item = R("delivered_item", "text_label",
                 "the product that has just been delivered")
    delivered = R("delivered_units", "count_items",
                  "how many units arrived with that delivery")
    pack_size = R("pack_size", "threshold_count",
                  "how many units make up one full pack")
    reorder = R("reorder_point", "threshold_count",
                "the stock level that triggers a reorder")
    min_busy = R("min_busy_lines", "threshold_count",
                 "how many products must be above average to call the ledger busy")
    uplift = R("restock_uplift", "percent_growth",
               "the uplift the restocking policy adds to the busiest product")
    ledger_slot = R("ledger_slot", "index_position",
                    "which product of the alphabetical ledger is being checked")
    spot_floor = R("spot_floor", "threshold_count",
                   "the stock that product must still have")
    mean_floor = R("mean_floor", "threshold_count",
                   "the average stock the ledger must keep")
    min_items = R("min_items", "threshold_count",
                  "how many products the ledger must list")
    stock_ceiling = R("stock_ceiling", "threshold_count",
                      "the level above which a product counts as overstocked")
    holding_floor = R("holding_floor", "threshold_count",
                      "the total holding the site is meant to keep")
    out.append(Blueprint(
        workflow_id="dictionary_processing.stock_ledger_adjustment",
        domain="dictionary_processing",
        natural_user_goal=("bring the stock ledger up to date after a delivery "
                           "and see where it stands"),
        target_description="the adjusted ledger or the verdict on stock levels",
        value_generator_id="dictionary_processing.stock_ledger",
        query_asset_family="stock_ledger",
        hard_distractor_families=("dictionary", "list"),
        boolean_balancing_strategy="calibrate_stock_threshold",
        entity_family="warehouse",
        plans=(
            Plan("ledger.v3", (ledger, reorder),
                 (S("n1", "dictionary.aggregate_argmax", ("stock_ledger",),
                    "the best stocked product"),
                  S("n2", "dictionary.lookup", ("stock_ledger", "@n1")),
                  S("n3", "comparison.at_least", ("@n2", "reorder_point"))),
                 "n3", intent="top_stock_verdict"),
            Plan("ledger.v4", (ledger, new_item, delivered, pack_size),
                 (S("n1", "dictionary.update", ("stock_ledger", "delivered_item",
                                                "delivered_units"),
                    "the ledger with the delivery booked in"),
                  S("n2", "dictionary.values", ("@n1",)),
                  S("n3", "list.reduce_sum", ("@n2",)),
                  S("n4", "boolean.divisible", ("@n3", "pack_size"))),
                 "n4", intent="whole_packs_verdict"),
            Plan("ledger.v6", (ledger, new_item, delivered, min_busy),
                 (S("n1", "dictionary.update", ("stock_ledger", "delivered_item",
                                                "delivered_units"),
                    "the booked-in ledger, read twice"),
                  S("n2", "dictionary.values", ("@n1",)),
                  S("n3", "statistics.mean", ("@n2",)),
                  S("n4", "dictionary.aggregate_filter", ("@n1", "@n3")),
                  S("n5", "dictionary.keys_count", ("@n4",)),
                  S("n6", "comparison.at_least", ("@n5", "min_busy_lines"))),
                 "n6", intent="busy_ledger_verdict"),
            Plan("ledger.v7", (ledger, uplift),
                 (S("n1", "dictionary.values", ("stock_ledger",)),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "dictionary.aggregate_filter", ("stock_ledger", "@n2"),
                    "the well stocked products, needed twice more"),
                  S("n4", "dictionary.aggregate_argmax", ("@n3",)),
                  S("n5", "dictionary.lookup", ("@n3", "@n4")),
                  S("n6", "rates.increase_by_percent", ("@n5",
                                                        "restock_uplift")),
                  S("n7", "dictionary.update", ("@n3", "@n4", "@n6"),
                    "that shortlist with the leader restocked")),
                 "n7", intent="restocked_shortlist"),
            Plan("ledger.v8", (ledger, new_item, delivered, stock_ceiling,
                               holding_floor, min_items),
                 (S("n1", "dictionary.update", ("stock_ledger", "delivered_item",
                                                "delivered_units"),
                    "the ledger with the delivery booked in, checked three ways"),
                  S("n2", "dictionary.values", ("@n1",)),
                  S("n3", "validation.list_limit", ("@n2", "stock_ceiling"),
                    "nothing has been left overstocked"),
                  S("n4", "dictionary.aggregate_sum", ("@n1",)),
                  S("n5", "comparison.at_least", ("@n4", "holding_floor"),
                    "the site still holds enough altogether"),
                  S("n6", "dictionary.keys_count", ("@n1",)),
                  S("n7", "comparison.at_least", ("@n6", "min_items"),
                    "the ledger still lists enough products"),
                  S("n8", "decision.any_of", ("@n3", "@n5", "@n7"),
                    "one passing check is enough to leave the ledger alone")),
                 "n8", intent="ledger_any_check_passes"),
            Plan("ledger.v10", (ledger, ledger_slot, spot_floor, mean_floor,
                                min_items),
                 (S("n1", "dictionary.keys", ("stock_ledger",)),
                  S("n2", "list.index_text", ("@n1", "ledger_slot")),
                  S("n3", "dictionary.lookup", ("stock_ledger", "@n2")),
                  S("n4", "dictionary.values", ("stock_ledger",)),
                  S("n5", "statistics.mean", ("@n4",)),
                  S("n6", "comparison.at_least", ("@n3", "spot_floor")),
                  S("n7", "comparison.at_least", ("@n5", "mean_floor")),
                  S("n8", "dictionary.keys_count", ("stock_ledger",)),
                  S("n9", "comparison.at_least", ("@n8", "min_items")),
                  S("n10", "decision.majority", ("@n6", "@n7", "@n9"))),
                 "n10", intent="ledger_health_verdict"),
        )))

    # ── spend by site ───────────────────────────────────────────────────
    spend = R("site_spend", "mapping_amounts",
              "what each depot has spent this quarter")
    site_uplift = R("forecast_uplift", "percent_growth",
                    "the uplift the forecast applies to the biggest spender")
    budget = R("quarter_budget", "threshold_money",
               "the budget the remaining depots have to stay inside")
    band_low = R("band_low", "cut_low",
                 "the spend below which a depot counts as quiet")
    band_high = R("band_high", "cut_high",
                  "the spend above which a depot counts as busy")
    group_size = R("group_size", "count_small",
                   "how many depots the comparison looks at on each side")
    concentration = R("concentration_target", "threshold_ratio",
                      "the concentration the finance team is prepared to accept")
    out.append(Blueprint(
        workflow_id="dictionary_processing.site_spend_table",
        domain="dictionary_processing",
        natural_user_goal=("understand how quarterly spend is distributed over "
                           "the depots"),
        target_description="the spend band, the adjusted table or its concentration",
        value_generator_id="dictionary_processing.site_spend",
        query_asset_family="spend_table",
        hard_distractor_families=("dictionary", "rates"),
        boolean_balancing_strategy="calibrate_budget_threshold",
        entity_family="finance",
        plans=(
            Plan("spend.v2", (spend, band_low, band_high),
                 (S("n1", "dictionary.aggregate_max", ("site_spend",),
                    "the largest depot spend"),
                  S("n2", "classification.three_bands", ("@n1", "band_low",
                                                         "band_high"))),
                 "n2", intent="largest_spend_band"),
            Plan("spend.v4", (spend, budget),
                 (S("n1", "dictionary.aggregate_argmax", ("site_spend",),
                    "the depot that spent the most"),
                  S("n2", "dictionary.update_remove", ("site_spend", "@n1"),
                    "the table without it"),
                  S("n3", "dictionary.aggregate_sum", ("@n2",)),
                  S("n4", "comparison.at_least", ("@n3", "quarter_budget"))),
                 "n4", intent="rest_of_table_verdict"),
            Plan("spend.v6", (spend, site_uplift),
                 (S("n1", "dictionary.aggregate_argmax", ("site_spend",),
                    "the biggest spender, needed three times"),
                  S("n2", "dictionary.lookup", ("site_spend", "@n1")),
                  S("n3", "rates.increase_by_percent", ("@n2",
                                                        "forecast_uplift")),
                  S("n4", "dictionary.update", ("site_spend", "@n1", "@n3"),
                    "the table with the forecast written into it"),
                  S("n5", "dictionary.aggregate_sum", ("@n4",)),
                  S("n6", "record.build", ("@n1", "@n5"),
                    "the headline: which depot and what the table now totals")),
                 "n6", intent="forecast_headline_record"),
            Plan("spend.v10", (spend, group_size, concentration),
                 (S("n1", "dictionary.values", ("site_spend",)),
                  S("n2", "list.map_sort_asc", ("@n1",),
                    "the spends in order, used at both ends"),
                  S("n3", "list.slice_last", ("@n2", "group_size")),
                  S("n4", "list.reduce_sum", ("@n3",)),
                  S("n5", "list.slice_first", ("@n2", "group_size")),
                  S("n6", "list.reduce_sum", ("@n5",)),
                  S("n7", "arithmetic.subtract", ("@n4", "@n6"),
                    "how much more the busy depots spend than the quiet ones"),
                  S("n8", "dictionary.aggregate_sum", ("site_spend",),
                    "the quarter's whole spend"),
                  S("n9", "rates.ratio_of", ("@n7", "@n8")),
                  S("n10", "classification.ratio_band", ("@n9",
                                                         "concentration_target"))),
                 "n10", intent="spend_concentration_band"),
        )))

    return out
