"""Scheduling workflows: shift plans, appointment slots, maintenance calendars.

Hours, minutes, seconds and days never mix without an explicit conversion, so a
plan that starts from a shift length and ends on a calendar date has to walk the
whole chain: hours to days, days rounded to whole days, whole days added to a
date, and only then a weekday.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── shift plan ──────────────────────────────────────────────────────
    shift = R("shift_hours", "duration_hours", "how long one shift runs")
    crews = R("crew_count", "count_small", "how many crews are rostered")
    breaks = R("break_minutes", "duration_minutes", "the break every crew is owed")
    unit = R("unit", "unit_word", "the unit the roster is written in")
    target = R("hours_target", "threshold_hours", "the hours the roster has to deliver")
    places = R("places", "places", "how many decimals the roster shows")
    overtime = R("overtime_hours", "duration_hours", "hours agreed on top of the shift")
    start = R("roster_start", "date_iso", "the day the roster starts")
    deadline = R("completion_date", "date_deadline",
                 "the day the work has to be finished")
    minutes_cap = R("minutes_cap", "threshold_value",
                    "the working minutes the agreement caps at")

    out.append(Blueprint(
        workflow_id="scheduling.shift_plan",
        domain="scheduling",
        natural_user_goal=("work out what the rostered crews actually deliver and "
                           "whether that fits the agreement"),
        target_description="the rostered effort or the verdict on the roster",
        value_generator_id="scheduling.roster",
        query_asset_family="crew_roster",
        hard_distractor_families=("duration", "comparison"),
        boolean_balancing_strategy="calibrate_hours_and_minutes_caps",
        entity_family="field service",
        plans=(
            Plan("shf.v2", (shift, crews, target),
                 (S("n1", "duration.scale", ("shift_hours", "crew_count")),
                  S("n2", "comparison.at_least", ("@n1", "hours_target"))),
                 "n2", intent="roster_meets_target"),
            Plan("shf.v4", (shift, crews, unit),
                 (S("n1", "duration.scale", ("shift_hours", "crew_count")),
                  S("n2", "duration.convert_hours_minutes", ("@n1",)),
                  S("n3", "duration.convert_minutes_seconds", ("@n2",)),
                  S("n4", "format.with_unit", ("@n3", "unit"))),
                 "n4", intent="rostered_seconds"),
            Plan("shf.v6", (shift, crews, breaks, places),
                 (S("n1", "duration.convert_minutes_seconds", ("break_minutes",),
                    "the break, converted on its own branch"),
                  S("n2", "duration.scale", ("shift_hours", "crew_count")),
                  S("n3", "duration.convert_hours_minutes", ("@n2",)),
                  S("n4", "duration.convert_minutes_seconds", ("@n3",)),
                  S("n5", "arithmetic.subtract", ("@n4", "@n1")),
                  S("n6", "format.fixed", ("@n5", "places"))),
                 "n6", intent="net_working_seconds"),
            Plan("shf.v9", (shift, overtime, crews, start, deadline, minutes_cap),
                 (S("n1", "duration.sum", ("shift_hours", "overtime_hours")),
                  S("n2", "duration.scale", ("@n1", "crew_count"),
                    "the rostered effort, needed again five calls later"),
                  S("n3", "duration.convert_hours_days", ("@n2",)),
                  S("n4", "rounding.to_int", ("@n3",)),
                  S("n5", "date.add_duration", ("roster_start", "@n4")),
                  S("n6", "date.compare", ("@n5", "completion_date")),
                  S("n7", "duration.convert_hours_minutes", ("@n2",)),
                  S("n8", "comparison.at_least", ("@n7", "minutes_cap")),
                  S("n9", "boolean.and", ("@n6", "@n8"))),
                 "n9", intent="roster_finishes_within_cap"),
        )))

    # ── appointment slots ───────────────────────────────────────────────
    window = R("window_days", "duration_days", "how many days the booking window covers")
    open_hours = R("open_hours", "duration_hours", "opening hours on top of the window")
    slot = R("slot_minutes", "duration_minutes", "how long one appointment takes")
    staff = R("staff_count", "count_people", "how many people take appointments")
    slot_target = R("slot_target", "threshold_count",
                    "how many appointments we have to be able to offer")
    slot_unit = R("unit", "unit_word", "the unit the schedule is written in")
    slot_places = R("places", "places", "how many decimals the schedule shows")

    out.append(Blueprint(
        workflow_id="scheduling.appointment_slots",
        domain="scheduling",
        natural_user_goal=("find out how many appointments we can actually offer "
                           "in the booking window"),
        target_description="the number of slots or how they spread over the team",
        value_generator_id="scheduling.booking_window",
        query_asset_family="booking_window",
        hard_distractor_families=("duration", "arithmetic"),
        boolean_balancing_strategy="calibrate_slot_target",
        entity_family="support",
        plans=(
            Plan("slt.v3", (open_hours, slot, slot_target),
                 (S("n1", "duration.convert_hours_minutes", ("open_hours",)),
                  S("n2", "rates.ratio_of", ("@n1", "slot_minutes"),
                    "how many appointments fit into the opening hours"),
                  S("n3", "comparison.at_least", ("@n2", "slot_target"))),
                 "n3", intent="enough_slots"),
            Plan("slt.v5", (open_hours, slot, staff, slot_unit),
                 (S("n1", "duration.convert_hours_minutes", ("open_hours",)),
                  S("n2", "rates.ratio_of", ("@n1", "slot_minutes")),
                  S("n3", "rates.ratio_of", ("@n2", "staff_count")),
                  S("n4", "arithmetic.multiply", ("@n3", "slot_minutes")),
                  S("n5", "format.with_unit", ("@n4", "unit"))),
                 "n5", intent="booked_time_per_person"),
            Plan("slt.v7", (window, open_hours, slot, staff, slot_places),
                 (S("n1", "duration.convert_days_hours", ("window_days",)),
                  S("n2", "duration.sum", ("@n1", "open_hours")),
                  S("n3", "duration.convert_hours_minutes", ("@n2",)),
                  S("n4", "rates.ratio_of", ("@n3", "slot_minutes"),
                    "all the slots there are, used twice"),
                  S("n5", "rates.ratio_of", ("@n4", "staff_count")),
                  S("n6", "rates.share_percent", ("@n5", "@n4")),
                  S("n7", "format.percent", ("@n6", "places"))),
                 "n7", intent="share_one_person_covers"),
        )))

    # ── maintenance calendar ────────────────────────────────────────────
    last = R("last_service", "date_iso", "when the machine was last serviced")
    interval = R("service_interval", "duration_days", "how often it has to be serviced")
    grace = R("grace_hours", "duration_hours", "the grace the contract allows")
    audit = R("inspection_date", "date_deadline", "when the inspector comes")
    teams = R("team_factor", "count_small", "how many maintenance teams we can send")
    weekday_floor = R("weekday_floor", "threshold_count",
                      "the weekday from which a service is inconvenient")
    hours_target = R("hours_target", "threshold_hours",
                     "the hours of slack the contract requires")

    out.append(Blueprint(
        workflow_id="scheduling.maintenance_calendar",
        domain="scheduling",
        natural_user_goal=("plan the next service so the machine is covered when "
                           "the inspector turns up"),
        target_description="the service slack or where the service lands",
        value_generator_id="scheduling.service_plan",
        query_asset_family="service_plan",
        hard_distractor_families=("date", "duration"),
        boolean_balancing_strategy="calibrate_service_slack",
        entity_family="fabrication",
        plans=(
            Plan("svc.v4", (last, interval, audit, grace),
                 (S("n1", "date.add_duration", ("last_service", "service_interval")),
                  S("n2", "date.difference", ("inspection_date", "@n1")),
                  S("n3", "duration.convert_days_hours", ("@n2",)),
                  S("n4", "duration.sum", ("@n3", "grace_hours"))),
                 "n4", intent="hours_of_cover"),
            Plan("svc.v6", (last, interval, audit, hours_target),
                 (S("n1", "date.add_duration", ("last_service", "service_interval"),
                    "the due date, checked two different ways"),
                  S("n2", "date.compare", ("@n1", "inspection_date")),
                  S("n3", "date.difference", ("inspection_date", "@n1")),
                  S("n4", "duration.convert_days_hours", ("@n3",)),
                  S("n5", "comparison.at_least", ("@n4", "hours_target")),
                  S("n6", "boolean.xor", ("@n2", "@n5"))),
                 "n6", intent="one_of_the_two_holds"),
            Plan("svc.v10", (last, interval, audit, grace, teams, weekday_floor),
                 (S("n1", "date.add_duration", ("last_service", "service_interval"),
                    "the nominal due date, shifted again seven calls later"),
                  S("n2", "date.difference", ("inspection_date", "@n1")),
                  S("n3", "duration.convert_days_hours", ("@n2",)),
                  S("n4", "duration.sum", ("@n3", "grace_hours")),
                  S("n5", "duration.scale", ("@n4", "team_factor")),
                  S("n6", "duration.convert_hours_days", ("@n5",)),
                  S("n7", "rounding.to_int", ("@n6",)),
                  S("n8", "date.add_duration", ("@n1", "@n7")),
                  S("n9", "date.weekday", ("@n8",)),
                  S("n10", "comparison.at_least", ("@n9", "weekday_floor"))),
                 "n10", intent="service_lands_late_in_week"),
        )))

    return out
