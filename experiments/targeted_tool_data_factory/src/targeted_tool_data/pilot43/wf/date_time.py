"""Date and duration workflows: delivery windows, effort rescheduling, review cycles.

Dates and durations stay distinct here: a gap between two dates is a number of
days, days only become hours through an explicit conversion, and a duration only
becomes a date again after it has been rounded to whole days. That chain --
date to duration to number to date to weekday -- is what makes these plans
change value kind several times over.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── delivery window ─────────────────────────────────────────────────
    start = R("dispatch_date", "date_iso", "the day the goods leave the depot")
    lead = R("transit_days", "duration_days", "how long the carrier says it takes")
    deadline = R("promised_date", "date_deadline", "the date promised to the customer")
    unit = R("unit", "unit_word", "the unit the report writes the slack in")
    buffer_days = R("buffer_days", "threshold_count",
                    "how many spare days the planner wants to keep")
    overtime = R("overtime_hours", "duration_hours",
                 "hours the depot can add at the weekend")

    out.append(Blueprint(
        workflow_id="dates.delivery_window",
        domain="dates",
        natural_user_goal=("find out whether a shipment still makes the date we "
                           "promised, and how much room is left"),
        target_description="the arrival verdict or the remaining slack",
        value_generator_id="dates.delivery",
        query_asset_family="delivery_promise",
        hard_distractor_families=("date", "duration"),
        boolean_balancing_strategy="calibrate_promised_date",
        entity_family="logistics",
        plans=(
            Plan("dw.v2", (start, lead, deadline),
                 (S("n1", "date.add_duration", ("dispatch_date", "transit_days")),
                  S("n2", "date.compare", ("@n1", "promised_date"))),
                 "n2", intent="arrives_in_time"),
            Plan("dw.v3", (start, deadline, unit),
                 (S("n1", "date.difference", ("promised_date", "dispatch_date")),
                  S("n2", "duration.convert_days_hours", ("@n1",)),
                  S("n3", "format.with_unit", ("@n2", "unit"))),
                 "n3", intent="window_in_hours"),
            Plan("dw.v5", (start, lead, deadline, buffer_days),
                 (S("n1", "date.add_duration", ("dispatch_date", "transit_days"),
                    "the arrival date, used by both checks"),
                  S("n2", "date.compare", ("@n1", "promised_date")),
                  S("n3", "date.difference", ("promised_date", "@n1")),
                  S("n4", "comparison.at_least", ("@n3", "buffer_days")),
                  S("n5", "boolean.and", ("@n2", "@n4"))),
                 "n5", intent="in_time_with_buffer"),
            Plan("dw.v6", (start, lead, deadline, overtime, unit),
                 (S("n1", "date.add_duration", ("dispatch_date", "transit_days")),
                  S("n2", "date.difference", ("promised_date", "@n1")),
                  S("n3", "duration.convert_days_hours", ("@n2",)),
                  S("n4", "duration.sum", ("@n3", "overtime_hours")),
                  S("n5", "duration.convert_hours_minutes", ("@n4",)),
                  S("n6", "format.with_unit", ("@n5", "unit"))),
                 "n6", intent="slack_in_minutes"),
        )))

    # ── effort rescheduling ─────────────────────────────────────────────
    eff_start = R("start_date", "date_iso", "the day the crew starts on site")
    eff_deadline = R("handover_date", "date_deadline",
                     "the day the site has to be handed over")
    crews = R("crew_factor", "count_small",
              "how many times the standard crew we can field")
    extra = R("setup_hours", "duration_hours", "hours lost to setting the site up")
    places = R("places", "places", "how many decimals the plan shows")
    weekday_min = R("weekday_floor", "threshold_count",
                    "the weekday from which handover is awkward")

    out.append(Blueprint(
        workflow_id="dates.effort_reschedule",
        domain="dates",
        natural_user_goal=("work out what the site plan really looks like once "
                           "the crew size and the setup time are taken in"),
        target_description="the reworked effort or the day the work lands on",
        value_generator_id="dates.site_plan",
        query_asset_family="site_plan",
        hard_distractor_families=("duration", "date"),
        boolean_balancing_strategy="calibrate_weekday_floor",
        entity_family="field service",
        plans=(
            Plan("eff.v4", (eff_start, eff_deadline, crews, extra),
                 (S("n1", "date.difference", ("handover_date", "start_date")),
                  S("n2", "duration.convert_days_hours", ("@n1",)),
                  S("n3", "duration.scale", ("@n2", "crew_factor")),
                  S("n4", "duration.sum", ("@n3", "setup_hours"))),
                 "n4", intent="crew_hours"),
            Plan("eff.v6", (eff_start, eff_deadline, crews, extra, places),
                 (S("n1", "date.difference", ("handover_date", "start_date")),
                  S("n2", "duration.convert_days_hours", ("@n1",)),
                  S("n3", "duration.scale", ("@n2", "crew_factor")),
                  S("n4", "duration.sum", ("@n3", "setup_hours")),
                  S("n5", "duration.convert_hours_days", ("@n4",)),
                  S("n6", "format.fixed", ("@n5", "places"))),
                 "n6", intent="crew_days_written_out"),
            Plan("eff.v9", (eff_start, eff_deadline, crews, extra, weekday_min),
                 (S("n1", "date.difference", ("handover_date", "start_date")),
                  S("n2", "duration.convert_days_hours", ("@n1",)),
                  S("n3", "duration.scale", ("@n2", "crew_factor")),
                  S("n4", "duration.sum", ("@n3", "setup_hours")),
                  S("n5", "duration.convert_hours_days", ("@n4",)),
                  S("n6", "rounding.to_int", ("@n5",),
                    "a date can only be moved by whole days"),
                  S("n7", "date.add_duration", ("start_date", "@n6")),
                  S("n8", "date.weekday", ("@n7",)),
                  S("n9", "comparison.at_least", ("@n8", "weekday_floor"))),
                 "n9", intent="lands_late_in_the_week"),
        )))

    # ── review cycle ────────────────────────────────────────────────────
    rev_start = R("first_review", "date_iso", "the day the first review was held")
    period = R("review_period", "duration_days", "how often the review repeats")
    cycles = R("cycles", "count_small", "how many reviews ahead we are looking")
    rev_deadline = R("audit_date", "date_deadline", "the day the audit takes place")
    gap_target = R("gap_target", "threshold_count",
                   "the gap the auditor wants between review and audit")
    code = R("review_code", "identifier_code", "the code the review series runs under")
    rev_unit = R("unit", "unit_word", "the unit the slack is written in")

    out.append(Blueprint(
        workflow_id="dates.review_cycle",
        domain="dates",
        natural_user_goal=("see where the next few recurring reviews land relative "
                           "to the audit"),
        target_description="the review date, its bucket or the gap to the audit",
        value_generator_id="dates.review_series",
        query_asset_family="review_series",
        hard_distractor_families=("date", "arithmetic"),
        boolean_balancing_strategy="calibrate_audit_gap",
        entity_family="quality",
        plans=(
            Plan("rev.v3", (rev_start, period, cycles),
                 (S("n1", "arithmetic.multiply", ("review_period", "cycles")),
                  S("n2", "date.add_duration", ("first_review", "@n1")),
                  S("n3", "date.month", ("@n2",))),
                 "n3", intent="review_month"),
            Plan("rev.v4", (rev_start, period, cycles, code),
                 (S("n1", "arithmetic.multiply", ("review_period", "cycles")),
                  S("n2", "date.add_duration", ("first_review", "@n1")),
                  S("n3", "date.quarter", ("@n2",)),
                  S("n4", "format.tag", ("review_code", "@n3"))),
                 "n4", intent="review_quarter_code"),
            Plan("rev.v5", (rev_start, period, cycles),
                 (S("n1", "arithmetic.multiply", ("review_period", "cycles")),
                  S("n2", "date.add_duration", ("first_review", "@n1")),
                  S("n3", "date.month", ("@n2",)),
                  S("n4", "date.month", ("first_review",),
                    "the same reading taken from the original date"),
                  S("n5", "arithmetic.abs_difference", ("@n3", "@n4"))),
                 "n5", intent="months_moved"),
            Plan("rev.v6", (rev_start, period, cycles, rev_deadline, gap_target),
                 (S("n1", "arithmetic.multiply", ("review_period", "cycles")),
                  S("n2", "date.add_duration", ("first_review", "@n1"),
                    "the review date, checked two different ways"),
                  S("n3", "date.compare", ("@n2", "audit_date")),
                  S("n4", "date.difference", ("audit_date", "@n2")),
                  S("n5", "comparison.at_least", ("@n4", "gap_target")),
                  S("n6", "boolean.or", ("@n3", "@n5"))),
                 "n6", intent="review_before_audit"),
            Plan("rev.v8", (rev_start, period, cycles, rev_deadline, rev_unit),
                 (S("n1", "arithmetic.multiply", ("review_period", "cycles")),
                  S("n2", "date.add_duration", ("first_review", "@n1")),
                  S("n3", "date.quarter", ("@n2",),
                    "the bucket, only needed by the last call"),
                  S("n4", "date.difference", ("audit_date", "@n2")),
                  S("n5", "duration.convert_days_hours", ("@n4",)),
                  S("n6", "duration.convert_hours_minutes", ("@n5",)),
                  S("n7", "format.with_unit", ("@n6", "unit")),
                  S("n8", "format.tag", ("@n7", "@n3"))),
                 "n8", intent="review_slack_entry"),
        )))

    return out
