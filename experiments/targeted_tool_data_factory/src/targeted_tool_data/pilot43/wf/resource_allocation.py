"""Allocation workflows: crew assignment, budget splits, capacity planning.

Allocation is done over the data that describes the resource -- rows of a
workload table, the machines on the floor -- so the plans combine record and
list capabilities with the arithmetic rather than doing arithmetic on two bare
numbers and calling it an allocation.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── crew assignment ─────────────────────────────────────────────────
    rows = R("workload_rows", "record_list", "one row per site with its workload")
    field = R("workload_field", "field_name", "the column the workload sits in")
    text_field = R("site_field", "text_field_name", "the column that names the site")
    crews = R("crew_count", "count_people", "how many people we can send out")
    unit = R("unit", "unit_word", "the unit the workload is measured in")
    cap = R("per_crew_cap", "threshold_value", "the workload one crew can absorb")
    share_cap = R("concentration_cap", "threshold_percent",
                  "the share one site may take before the work is lopsided")

    out.append(Blueprint(
        workflow_id="resources.crew_assignment",
        domain="resources",
        natural_user_goal=("work out how the field work spreads over the crews we "
                           "can actually send"),
        target_description="the load per crew or the verdict on the spread",
        value_generator_id="resources.workload",
        query_asset_family="workload_table",
        hard_distractor_families=("record", "list"),
        boolean_balancing_strategy="calibrate_crew_cap_and_concentration",
        entity_family="field service",
        plans=(
            Plan("crw.v3", (rows, field, crews, unit),
                 (S("n1", "record.aggregate_sum", ("workload_rows", "workload_field")),
                  S("n2", "rates.ratio_of", ("@n1", "crew_count")),
                  S("n3", "format.with_unit", ("@n2", "unit"))),
                 "n3", intent="load_per_crew"),
            Plan("crw.v5", (rows, field, cap),
                 (S("n1", "record.project", ("workload_rows", "workload_field"),
                    "the workload column, reduced two different ways"),
                  S("n2", "list.reduce_sum", ("@n1",)),
                  S("n3", "list.reduce_max", ("@n1",)),
                  S("n4", "arithmetic.subtract", ("@n2", "@n3")),
                  S("n5", "comparison.at_least", ("@n4", "per_crew_cap"))),
                 "n5", intent="rest_of_the_network_still_heavy"),
            Plan("crw.v10", (rows, field, text_field, cap, share_cap),
                 (S("n1", "record.project", ("workload_rows", "workload_field")),
                  S("n2", "list.reduce_sum", ("@n1",),
                    "the total workload, needed by two later calls"),
                  S("n3", "list.reduce_max", ("@n1",)),
                  S("n4", "rates.share_percent", ("@n3", "@n2")),
                  S("n5", "record.project_text", ("workload_rows", "site_field"),
                    "the sites are counted on their own branch"),
                  S("n6", "list.reduce_count_text", ("@n5",)),
                  S("n7", "rates.ratio_of", ("@n2", "@n6"),
                    "the workload an average site carries"),
                  S("n8", "comparison.at_least", ("@n7", "per_crew_cap")),
                  S("n9", "comparison.at_least", ("@n4", "concentration_cap")),
                  S("n10", "boolean.and", ("@n8", "@n9"))),
                 "n10", intent="heavy_and_concentrated"),
        )))

    # ── budget split ────────────────────────────────────────────────────
    budget = R("budget", "money_budget", "the budget the programme was given")
    rate = R("reserve_share", "percent_share", "the share held back as a reserve")
    teams = R("team_count", "count_people", "how many teams share what is left")
    currency = R("currency", "currency_code", "the currency the budget is held in")
    bud_places = R("places", "places", "how many decimals the split shows")
    bud_label = R("programme", "text_label", "what the programme is called")

    out.append(Blueprint(
        workflow_id="resources.budget_split",
        domain="resources",
        natural_user_goal=("split what is left of the programme budget over the "
                           "teams once the reserve is held back"),
        target_description="the reserve, the split or the line the sheet shows",
        value_generator_id="resources.budget",
        query_asset_family="budget_split",
        hard_distractor_families=("rates", "format"),
        entity_family="finance",
        plans=(
            Plan("bud.v2", (budget, rate, currency),
                 (S("n1", "rates.percent_of", ("reserve_share", "budget")),
                  S("n2", "format.currency", ("@n1", "currency"))),
                 "n2", intent="reserve_amount"),
            Plan("bud.v3", (budget, rate, currency),
                 (S("n1", "rates.percent_of", ("reserve_share", "budget")),
                  S("n2", "arithmetic.subtract", ("budget", "@n1")),
                  S("n3", "format.currency", ("@n2", "currency"))),
                 "n3", intent="available_budget"),
            Plan("bud.v5", (budget, rate, teams, currency, bud_label),
                 (S("n1", "rates.percent_of", ("reserve_share", "budget")),
                  S("n2", "arithmetic.subtract", ("budget", "@n1")),
                  S("n3", "arithmetic.divide", ("@n2", "team_count")),
                  S("n4", "format.currency", ("@n3", "currency")),
                  S("n5", "string.concat", ("programme", "@n4"))),
                 "n5", intent="budget_line_per_team"),
            Plan("bud.v7", (budget, rate, teams, currency, bud_places),
                 (S("n1", "rates.percent_of", ("reserve_share", "budget"),
                    "the reserve, reported again three calls later"),
                  S("n2", "arithmetic.subtract", ("budget", "@n1")),
                  S("n3", "arithmetic.divide", ("@n2", "team_count")),
                  S("n4", "rates.share_percent", ("@n1", "budget")),
                  S("n5", "format.percent", ("@n4", "places")),
                  S("n6", "format.currency", ("@n3", "currency")),
                  S("n7", "string.concat", ("@n6", "@n5"))),
                 "n7", intent="split_with_reserve_share"),
        )))

    # ── capacity planning ───────────────────────────────────────────────
    machines = R("line_capacities", "list_quantities",
                 "what each production line can take")
    demand = R("committed_units", "quantity_units", "units already committed")
    top_n = R("lead_lines", "count_small", "how many lines carry the bulk of the work")
    limit = R("spare_target", "threshold_value", "the spare capacity we want to keep")
    cap_share = R("concentration_cap", "threshold_percent",
                  "the share the lead lines may cover before we are exposed")
    cap_places = R("places", "places", "how many decimals the capacity plan shows")

    out.append(Blueprint(
        workflow_id="resources.capacity_planning",
        domain="resources",
        natural_user_goal=("see whether the production lines still have room for "
                           "what we have committed to"),
        target_description="the spare capacity or how concentrated it is",
        value_generator_id="resources.capacity",
        query_asset_family="capacity_plan",
        hard_distractor_families=("list", "rates"),
        boolean_balancing_strategy="calibrate_spare_and_concentration",
        entity_family="fabrication",
        plans=(
            Plan("cap.v3", (machines, demand, limit),
                 (S("n1", "list.reduce_sum", ("line_capacities",)),
                  S("n2", "arithmetic.subtract", ("@n1", "committed_units")),
                  S("n3", "comparison.at_least", ("@n2", "spare_target"))),
                 "n3", intent="enough_spare_capacity"),
            Plan("cap.v6", (machines, top_n, cap_places),
                 (S("n1", "list.map_sort", ("line_capacities",)),
                  S("n2", "list.slice_first", ("@n1", "lead_lines")),
                  S("n3", "list.reduce_sum", ("@n2",)),
                  S("n4", "list.reduce_sum", ("line_capacities",),
                    "the whole floor, summed independently"),
                  S("n5", "rates.share_percent", ("@n3", "@n4")),
                  S("n6", "format.percent", ("@n5", "places"))),
                 "n6", intent="share_carried_by_lead_lines"),
            Plan("cap.v9", (machines, top_n, limit, cap_share),
                 (S("n1", "list.map_sort", ("line_capacities",)),
                  S("n2", "list.slice_first", ("@n1", "lead_lines")),
                  S("n3", "list.reduce_sum", ("@n2",),
                    "capacity of the lead lines, used twice"),
                  S("n4", "list.reduce_sum", ("line_capacities",)),
                  S("n5", "rates.share_percent", ("@n3", "@n4")),
                  S("n6", "arithmetic.subtract", ("@n4", "@n3")),
                  S("n7", "comparison.at_least", ("@n6", "spare_target")),
                  S("n8", "comparison.at_least", ("@n5", "concentration_cap")),
                  S("n9", "boolean.xor", ("@n7", "@n8"))),
                 "n9", intent="capacity_exposure_verdict"),
        )))

    return out
