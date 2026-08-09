"""Travel workflows: itinerary durations, route cost, refuelling on a route.

Three families that each keep one unit system honest. Durations only ever meet
durations (hours stay hours until an explicit converter turns them into minutes
or days), distances only meet distances, and money enters a distance chain only
through a per-kilometre rate. The long plans differ from the short ones by real
reuse -- the driving time before delays is needed again to express what share of
the journey is actually spent driving, and the route without the detour is
priced a second time to isolate what the detour costs.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── itinerary duration ──────────────────────────────────────────────
    itinerary_roles = (
        R("drive_one", "duration_hours", "driving time of the first leg"),
        R("drive_two", "duration_hours", "driving time of the second leg"),
        R("delay_percent", "percent_growth",
          "delay the traffic forecast adds to the driving time"),
        R("break_hours", "duration_hours", "scheduled break at the midpoint"),
        R("rest_share", "ratio_target",
          "rest time the rules require per hour driven"),
        R("max_hours", "threshold_hours", "duty time the itinerary must fit in"),
        R("rest_limit", "threshold_hours", "rest time that must be reached"),
    )
    out.append(Blueprint(
        workflow_id="travel.itinerary_duration",
        domain="travel",
        natural_user_goal=("work out how long a two-leg journey really takes "
                           "once delays, breaks and rest rules are counted"),
        target_description="the itinerary duration or the duty-time verdict",
        value_generator_id="travel.itinerary",
        query_asset_family="journey_plan",
        hard_distractor_families=("duration", "rates"),
        boolean_balancing_strategy="threshold_band",
        entity_family="fleet",
        plans=(
            Plan("itinerary.v2", itinerary_roles[:2],
                 (S("n1", "duration.sum", ("drive_one", "drive_two"),
                    "total driving time"),
                  S("n2", "duration.convert_hours_minutes", ("@n1",),
                    "driving time in minutes")),
                 "n2", intent="driving_minutes"),
            Plan("itinerary.v4", itinerary_roles[:4],
                 (S("n1", "duration.sum", ("drive_one", "drive_two"),
                    "pure driving time, needed again at the end"),
                  S("n2", "rates.increase_by_percent", ("@n1", "delay_percent")),
                  S("n3", "duration.sum", ("@n2", "break_hours")),
                  S("n4", "rates.share_percent", ("@n1", "@n3"),
                    "share of the itinerary spent driving")),
                 "n4", intent="driving_share"),
            Plan("itinerary.v6", itinerary_roles[:6],
                 (S("n1", "duration.sum", ("drive_one", "drive_two")),
                  S("n2", "rates.increase_by_percent", ("@n1", "delay_percent")),
                  S("n3", "duration.scale", ("@n1", "rest_share"),
                    "rest earned by the driving time"),
                  S("n4", "duration.sum", ("@n2", "@n3")),
                  S("n5", "duration.sum", ("@n4", "break_hours")),
                  S("n6", "comparison.at_least", ("@n5", "max_hours"))),
                 "n6", intent="duty_time_verdict"),
            Plan("itinerary.v8", itinerary_roles,
                 (S("n1", "duration.sum", ("drive_one", "drive_two")),
                  S("n2", "rates.increase_by_percent", ("@n1", "delay_percent")),
                  S("n3", "duration.scale", ("@n1", "rest_share"),
                    "rest requirement, checked separately later"),
                  S("n4", "duration.sum", ("@n2", "@n3")),
                  S("n5", "duration.sum", ("@n4", "break_hours")),
                  S("n6", "comparison.at_least", ("@n5", "max_hours")),
                  S("n7", "comparison.at_least", ("@n3", "rest_limit")),
                  S("n8", "boolean.and", ("@n6", "@n7"),
                    "duty time and rest rule must both hold")),
                 "n8", intent="duty_and_rest_verdict"),
        )))

    # ── route cost ──────────────────────────────────────────────────────
    route_roles = (
        R("leg_distances", "list_readings", "distance of every leg in km"),
        R("cost_per_km", "money_price", "vehicle cost per kilometre"),
        R("tax_rate", "percent_tax", "tax charged on the transport service"),
        R("toll_fee", "money_fee", "tolls paid on the route"),
        R("passengers", "count_people", "people sharing the vehicle"),
        R("currency", "currency_code", "currency the fare is quoted in"),
    )
    out.append(Blueprint(
        workflow_id="travel.route_cost",
        domain="travel",
        natural_user_goal=("price a multi-leg route and see how much of the "
                           "cost the longest leg is responsible for"),
        target_description="the route cost, the per-head fare or the leg share",
        value_generator_id="travel.route",
        query_asset_family="route_sheet",
        hard_distractor_families=("list", "arithmetic"),
        entity_family="fleet",
        plans=(
            Plan("route.v3", route_roles[:3],
                 (S("n1", "list.reduce_sum", ("leg_distances",),
                    "kilometres over all legs"),
                  S("n2", "arithmetic.multiply", ("@n1", "cost_per_km")),
                  S("n3", "rates.apply_tax", ("@n2", "tax_rate"))),
                 "n3", intent="route_cost_with_tax"),
            Plan("route.v5", route_roles[:2] + route_roles[3:6],
                 (S("n1", "list.reduce_sum", ("leg_distances",)),
                  S("n2", "arithmetic.multiply", ("@n1", "cost_per_km")),
                  S("n3", "arithmetic.add", ("@n2", "toll_fee")),
                  S("n4", "arithmetic.divide", ("@n3", "passengers")),
                  S("n5", "format.currency", ("@n4", "currency"))),
                 "n5", intent="fare_per_head_label"),
            Plan("route.v7", route_roles[:2] + route_roles[3:4],
                 (S("n1", "list.reduce_sum", ("leg_distances",),
                    "kilometres over all legs, priced twice"),
                  S("n2", "list.reduce_max", ("leg_distances",),
                    "the longest single leg"),
                  S("n3", "arithmetic.subtract", ("@n1", "@n2"),
                    "kilometres outside the longest leg"),
                  S("n4", "arithmetic.multiply", ("@n1", "cost_per_km")),
                  S("n5", "arithmetic.multiply", ("@n3", "cost_per_km")),
                  S("n6", "arithmetic.add", ("@n4", "toll_fee")),
                  S("n7", "rates.share_percent", ("@n5", "@n6"),
                    "share of the bill not caused by the longest leg")),
                 "n7", intent="short_leg_cost_share"),
        )))

    # ── refuelling on a route ───────────────────────────────────────────
    refuel_roles = (
        R("outbound_km", "length_km", "distance of the outbound trip"),
        R("return_km", "length_km", "distance of the return trip"),
        R("detour_km", "length_km", "extra distance of the planned detour"),
        R("range_km", "length_km", "distance the vehicle covers on one tank"),
        R("refill_cost", "money_price", "cost of filling the tank once"),
        R("low_cut", "cut_low", "fuel spend that still counts as cheap"),
        R("high_cut", "cut_high", "fuel spend that counts as expensive"),
        R("extra_budget", "threshold_money",
          "extra fuel spend the detour may cost"),
    )
    out.append(Blueprint(
        workflow_id="travel.refuelling_plan",
        domain="travel",
        natural_user_goal=("work out what refuelling a round trip costs and "
                           "whether a detour is still affordable"),
        target_description="the tank fills, the fuel spend band or the detour verdict",
        value_generator_id="travel.refuelling",
        query_asset_family="trip_fuel_plan",
        hard_distractor_families=("rates", "classification"),
        boolean_balancing_strategy="threshold_band",
        entity_family="fleet",
        plans=(
            Plan("refuel.v3", refuel_roles[:4],
                 (S("n1", "arithmetic.add", ("outbound_km", "return_km")),
                  S("n2", "arithmetic.add", ("@n1", "detour_km")),
                  S("n3", "rates.ratio_of", ("@n2", "range_km"),
                    "tank fills the route needs")),
                 "n3", intent="tank_fills_needed"),
            Plan("refuel.v5", refuel_roles[:7],
                 (S("n1", "arithmetic.add", ("outbound_km", "return_km")),
                  S("n2", "arithmetic.add", ("@n1", "detour_km")),
                  S("n3", "rates.ratio_of", ("@n2", "range_km")),
                  S("n4", "arithmetic.multiply", ("refill_cost", "@n3")),
                  S("n5", "classification.three_bands",
                    ("@n4", "low_cut", "high_cut"))),
                 "n5", intent="fuel_spend_band"),
            Plan("refuel.v7", refuel_roles[:5] + refuel_roles[7:8],
                 (S("n1", "arithmetic.add", ("outbound_km", "return_km"),
                    "round trip without the detour, priced again below"),
                  S("n2", "arithmetic.add", ("@n1", "detour_km")),
                  S("n3", "rates.ratio_of", ("@n2", "range_km")),
                  S("n4", "rates.ratio_of", ("@n1", "range_km")),
                  S("n5", "arithmetic.subtract", ("@n3", "@n4"),
                    "extra tank fills caused by the detour"),
                  S("n6", "arithmetic.multiply", ("refill_cost", "@n5")),
                  S("n7", "comparison.at_least", ("@n6", "extra_budget"))),
                 "n7", intent="detour_affordability"),
        )))

    return out
