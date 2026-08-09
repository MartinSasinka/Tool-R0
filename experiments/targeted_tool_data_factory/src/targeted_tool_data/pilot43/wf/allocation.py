"""Resource allocation workflows: budget splits, team capacity, overtime cost.

Everything here divides one pot -- money, hours or capacity -- over teams and
shifts, so the interesting quantity is almost always a *share*: what one team
receives out of what is left after the reserve, how far the largest team sits
above the average, how much of the wage bill is overtime. The short plans stop
at the split; the long ones need the undivided pot again after the split, which
is where the reuse and the second join come from.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── budget split ────────────────────────────────────────────────────
    split_roles = (
        R("total_budget", "money_budget", "budget available for the programme"),
        R("core_share", "percent_share",
          "share of the budget committed to core work"),
        R("team_count", "count_people", "teams the budget is split between"),
        R("reserve_percent", "percent_margin",
          "share held back as a contingency reserve"),
        R("overhead_fee", "money_fee", "flat overhead charged on the programme"),
        R("min_per_team", "threshold_money", "amount every team must receive"),
    )
    out.append(Blueprint(
        workflow_id="allocation.budget_split",
        domain="resource_allocation",
        natural_user_goal=("split a programme budget across teams once the "
                           "reserve and the overhead are taken out"),
        target_description="the amount per team or the funding verdict",
        value_generator_id="allocation.budget",
        query_asset_family="programme_budget",
        hard_distractor_families=("rates", "arithmetic"),
        boolean_balancing_strategy="threshold_band",
        entity_family="programme_office",
        plans=(
            Plan("split.v2", split_roles[:3],
                 (S("n1", "rates.percent_of", ("core_share", "total_budget")),
                  S("n2", "arithmetic.divide", ("@n1", "team_count"),
                    "core funding per team")),
                 "n2", intent="core_per_team"),
            Plan("split.v4", split_roles[:1] + split_roles[2:6],
                 (S("n1", "rates.decrease_by_percent",
                    ("total_budget", "reserve_percent")),
                  S("n2", "arithmetic.subtract", ("@n1", "overhead_fee")),
                  S("n3", "arithmetic.divide", ("@n2", "team_count")),
                  S("n4", "comparison.at_least", ("@n3", "min_per_team"))),
                 "n4", intent="minimum_funding_verdict"),
            Plan("split.v6", split_roles[:5],
                 (S("n1", "rates.decrease_by_percent",
                    ("total_budget", "reserve_percent"),
                    "allocatable pot, still needed as the denominator"),
                  S("n2", "rates.percent_of", ("core_share", "@n1")),
                  S("n3", "arithmetic.subtract", ("@n1", "@n2"),
                    "what is left for discretionary work"),
                  S("n4", "arithmetic.divide", ("@n3", "team_count")),
                  S("n5", "arithmetic.add", ("@n4", "overhead_fee")),
                  S("n6", "rates.share_percent", ("@n5", "@n1"),
                    "share of the pot one team ends up costing")),
                 "n6", intent="team_cost_share"),
        )))

    # ── team capacity ───────────────────────────────────────────────────
    capacity_roles = (
        R("team_hours", "mapping_rates", "weekly hours each team can supply"),
        R("low_cut", "cut_low", "imbalance that still counts as even"),
        R("high_cut", "cut_high", "imbalance that counts as severe"),
    )
    out.append(Blueprint(
        workflow_id="allocation.team_capacity",
        domain="resource_allocation",
        natural_user_goal=("see how evenly the available hours are spread "
                           "across the teams"),
        target_description="the capacity concentration, its band or the share "
                           "of overloaded teams",
        value_generator_id="allocation.capacity",
        query_asset_family="capacity_table",
        hard_distractor_families=("dictionary", "statistics"),
        entity_family="programme_office",
        plans=(
            Plan("capacity.v3", capacity_roles[:1],
                 (S("n1", "dictionary.aggregate_sum", ("team_hours",)),
                  S("n2", "dictionary.aggregate_max", ("team_hours",)),
                  S("n3", "rates.share_percent", ("@n2", "@n1"),
                    "share of all hours held by the largest team")),
                 "n3", intent="largest_team_share"),
            Plan("capacity.v5", capacity_roles,
                 (S("n1", "dictionary.values", ("team_hours",),
                    "hours per team, reduced two ways"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "list.reduce_max", ("@n1",)),
                  S("n4", "rates.share_percent", ("@n3", "@n2")),
                  S("n5", "classification.three_bands",
                    ("@n4", "low_cut", "high_cut"))),
                 "n5", intent="imbalance_band"),
            Plan("capacity.v7", capacity_roles[:1],
                 (S("n1", "dictionary.values", ("team_hours",),
                    "hours per team, also counted against the ceiling"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.stdev", ("@n1",)),
                  S("n4", "arithmetic.add", ("@n2", "@n3"),
                    "capacity ceiling one standard deviation above the mean"),
                  S("n5", "list.reduce_count_above", ("@n1", "@n4")),
                  S("n6", "dictionary.keys_count", ("team_hours",)),
                  S("n7", "rates.share_percent", ("@n5", "@n6"),
                    "share of teams carrying more than the ceiling")),
                 "n7", intent="overloaded_team_share"),
        )))

    # ── overtime allocation ─────────────────────────────────────────────
    overtime_roles = (
        R("planned_hours", "list_durations_h", "hours planned for each shift"),
        R("overtime_percent", "percent_margin",
          "overtime added on top of every planned shift"),
        R("shift_rates", "list_prices", "hourly rate paid on each shift"),
        R("places", "places", "decimals the wage bill is reported with"),
        R("roster_code", "identifier_code", "code of the roster being priced"),
        R("overtime_limit", "threshold_percent",
          "share of the wage bill overtime may reach"),
    )
    out.append(Blueprint(
        workflow_id="allocation.overtime_cost",
        domain="resource_allocation",
        natural_user_goal=("price the overtime a shift roster generates and "
                           "check how much of the wage bill it eats"),
        target_description="the overtime cost, the wage bill note or the "
                           "overtime verdict",
        value_generator_id="allocation.overtime",
        query_asset_family="shift_roster",
        hard_distractor_families=("list", "statistics"),
        boolean_balancing_strategy="threshold_band",
        entity_family="workforce",
        plans=(
            Plan("overtime.v4", overtime_roles[:3],
                 (S("n1", "list.map_percent",
                    ("planned_hours", "overtime_percent")),
                  S("n2", "list.reduce_sum", ("@n1",)),
                  S("n3", "statistics.mean", ("shift_rates",)),
                  S("n4", "arithmetic.multiply", ("@n3", "@n2"),
                    "cost of the overtime hours")),
                 "n4", intent="overtime_cost"),
            Plan("overtime.v6", overtime_roles[:5],
                 (S("n1", "list.reduce_sum", ("planned_hours",)),
                  S("n2", "statistics.mean", ("shift_rates",)),
                  S("n3", "arithmetic.multiply", ("@n2", "@n1")),
                  S("n4", "rates.increase_by_percent",
                    ("@n3", "overtime_percent")),
                  S("n5", "format.fixed", ("@n4", "places")),
                  S("n6", "string.concat", ("roster_code", "@n5"),
                    "roster code with its wage bill")),
                 "n6", intent="wage_bill_note"),
            Plan("overtime.v9", overtime_roles[:3] + overtime_roles[5:6],
                 (S("n1", "list.reduce_sum", ("planned_hours",)),
                  S("n2", "list.map_percent",
                    ("planned_hours", "overtime_percent")),
                  S("n3", "list.reduce_sum", ("@n2",),
                    "overtime hours, priced separately further down"),
                  S("n4", "arithmetic.add", ("@n1", "@n3")),
                  S("n5", "statistics.mean", ("shift_rates",)),
                  S("n6", "arithmetic.multiply", ("@n5", "@n4"),
                    "full wage bill"),
                  S("n7", "arithmetic.multiply", ("@n5", "@n3"),
                    "overtime part of the wage bill"),
                  S("n8", "rates.share_percent", ("@n7", "@n6")),
                  S("n9", "comparison.at_least", ("@n8", "overtime_limit"))),
                 "n9", intent="overtime_share_verdict"),
        )))

    return out
