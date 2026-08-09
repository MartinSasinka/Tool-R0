"""Duration workflows in which the unit conversion is the actual work.

The registry converts days to hours, hours to minutes and minutes to seconds
(and seconds back to minutes), and every converter is typed, so a span stated
in minutes simply cannot be added to a span stated in hours until the hour span
has been converted. Each plan here is built around that constraint rather than
routed around it: the break in minutes reaches the roster only through
``duration.convert_hours_minutes``, the downtime in days reaches the repair
budget only through ``duration.convert_days_hours``.

Verdicts in the hour domain compare against ``threshold_hours``. The minute and
second domains have no calibrated hint of their own, so those predicates use
``threshold_value`` and state the unit in the role description instead of
quietly treating the number as dimensionless.

The plans differ in how many unit domains they have to cross: the short ones
stay in one, the long ones carry the same roster through hours, minutes and
seconds and have to keep the two representations apart while doing it.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── shift ledger (hours meets minutes) ──────────────────────────────
    shift_roles = (
        R("shift_hours", "duration_hours", "length of the rostered shift in hours"),
        R("overtime_hours", "duration_hours", "overtime booked on top, in hours"),
        R("break_minutes", "duration_minutes", "unpaid break in minutes"),
        R("handover_minutes", "duration_minutes", "handover at the end, in minutes"),
        R("cycle_count", "count_small", "how many shifts the roster cycle holds"),
        R("paid_limit", "threshold_value", "cap on paid time per shift, in minutes"),
        R("shift_limit", "threshold_hours", "cap on the rostered shift, in hours"),
    )
    out.append(Blueprint(
        workflow_id="duration.shift_ledger",
        domain="date_time",
        natural_user_goal=("settle how much of a rostered shift is actually paid "
                           "once the breaks are taken off"),
        target_description="the paid span or the verdict on the roster",
        value_generator_id="duration.shift_ledger",
        query_asset_family="shift_ledger",
        hard_distractor_families=("duration", "arithmetic"),
        boolean_balancing_strategy="threshold_band",
        entity_family="hr",
        plans=(
            Plan("shift.v3", (shift_roles[0], shift_roles[2], shift_roles[3]),
                 (S("n1", "duration.convert_hours_minutes", ("shift_hours",),
                    "the shift in minutes, so the break can be taken off it"),
                  S("n2", "arithmetic.subtract", ("@n1", "break_minutes")),
                  S("n3", "arithmetic.add", ("@n2", "handover_minutes"),
                    "paid minutes on the clock")),
                 "n3", intent="paid_minutes"),
            Plan("shift.v6", (shift_roles[0], shift_roles[1], shift_roles[2],
                              shift_roles[5], shift_roles[6]),
                 (S("n1", "duration.sum", ("shift_hours", "overtime_hours"),
                    "rostered hours, read again four steps later"),
                  S("n2", "duration.convert_hours_minutes", ("@n1",)),
                  S("n3", "arithmetic.subtract", ("@n2", "break_minutes"),
                    "paid minutes"),
                  S("n4", "comparison.at_least", ("@n3", "paid_limit")),
                  S("n5", "comparison.at_least", ("@n1", "shift_limit")),
                  S("n6", "boolean.and", ("@n4", "@n5"),
                    "the roster breaks both caps at once")),
                 "n6", intent="roster_cap_verdict"),
            Plan("shift.v9", shift_roles[:5],
                 (S("n1", "duration.sum", ("shift_hours", "overtime_hours"),
                    "rostered hours per shift, read again at n5"),
                  S("n2", "duration.convert_hours_minutes", ("@n1",)),
                  S("n3", "arithmetic.add", ("break_minutes", "handover_minutes"),
                    "off-task minutes"),
                  S("n4", "arithmetic.subtract", ("@n2", "@n3"),
                    "on-task minutes"),
                  S("n5", "duration.scale", ("@n1", "cycle_count"),
                    "rostered hours over the whole cycle"),
                  S("n6", "duration.convert_minutes_seconds", ("@n4",),
                    "on-task seconds in one shift"),
                  S("n7", "duration.convert_hours_minutes", ("@n5",)),
                  S("n8", "duration.convert_minutes_seconds", ("@n7",),
                    "the cycle in seconds"),
                  S("n9", "arithmetic.subtract", ("@n8", "@n6"),
                    "cycle seconds not covered by one shift on task")),
                 "n9", intent="cycle_seconds_uncovered"),
        )))

    # ── transfer window (seconds meets minutes) ─────────────────────────
    transfer_roles = (
        R("upload_seconds", "duration_seconds", "seconds spent uploading the batch"),
        R("verify_seconds", "duration_seconds", "seconds spent verifying it"),
        R("queue_minutes", "duration_minutes", "minutes the job waits in the queue"),
        R("retry_count", "count_small", "how many attempts the transfer takes"),
        R("low_cut", "cut_low", "overhead below which the window is comfortable"),
        R("high_cut", "cut_high", "overhead above which the window is at risk"),
    )
    out.append(Blueprint(
        workflow_id="duration.transfer_window",
        domain="date_time",
        natural_user_goal=("see what a nightly transfer really costs once the "
                           "queue and the retries are counted"),
        target_description="the transfer spans, the retry overhead or its band",
        value_generator_id="duration.transfer_window",
        query_asset_family="transfer_job",
        hard_distractor_families=("duration", "list"),
        entity_family="data",
        plans=(
            Plan("transfer.v4", transfer_roles[:3],
                 (S("n1", "duration.convert_seconds_minutes", ("upload_seconds",),
                    "upload in minutes, so it can sit beside the queue wait"),
                  S("n2", "duration.convert_seconds_minutes", ("verify_seconds",)),
                  S("n3", "list.build", ("@n1", "@n2", "queue_minutes"),
                    "the three spans, all in minutes"),
                  S("n4", "list.map_sort_asc", ("@n3",))),
                 "n4", intent="spans_in_minutes"),
            Plan("transfer.v5", (transfer_roles[0], transfer_roles[1],
                                 transfer_roles[3]),
                 (S("n1", "arithmetic.add", ("upload_seconds", "verify_seconds"),
                    "seconds for one clean attempt, read again at n4"),
                  S("n2", "arithmetic.multiply", ("@n1", "retry_count"),
                    "seconds once every attempt is counted"),
                  S("n3", "duration.convert_seconds_minutes", ("@n2",)),
                  S("n4", "duration.convert_seconds_minutes", ("@n1",)),
                  S("n5", "arithmetic.subtract", ("@n3", "@n4"),
                    "minutes the retries add")),
                 "n5", intent="retry_overhead_minutes"),
            Plan("transfer.v7", transfer_roles,
                 (S("n1", "arithmetic.add", ("upload_seconds", "verify_seconds"),
                    "seconds for one clean attempt, read again at n5"),
                  S("n2", "arithmetic.multiply", ("@n1", "retry_count")),
                  S("n3", "duration.convert_seconds_minutes", ("@n2",),
                    "all attempts in minutes"),
                  S("n4", "arithmetic.add", ("@n3", "queue_minutes"),
                    "door-to-door minutes"),
                  S("n5", "duration.convert_seconds_minutes", ("@n1",),
                    "a clean attempt in minutes"),
                  S("n6", "arithmetic.subtract", ("@n4", "@n5"),
                    "minutes on top of a clean run"),
                  S("n7", "classification.three_bands",
                    ("@n6", "low_cut", "high_cut"))),
                 "n7", intent="overhead_band"),
        )))

    # ── maintenance window (days meets hours) ───────────────────────────
    maintenance_roles = (
        R("downtime_days", "duration_days", "days the line is taken out of service"),
        R("repair_hours", "duration_hours", "hours the repair itself needs"),
        R("inspection_hours", "duration_hours", "hours the inspection needs"),
        R("crew_factor", "count_small", "how many passes the crew has to make"),
        R("window_limit", "threshold_hours", "slack the planner wants left, in hours"),
        R("backlog_limit", "threshold_hours",
          "slack at the slower pace below which work is deferred, in hours"),
    )
    out.append(Blueprint(
        workflow_id="duration.maintenance_window",
        domain="date_time",
        natural_user_goal=("check whether a maintenance window is long enough for "
                           "the work booked into it"),
        target_description="the slack in the window or the verdict on it",
        value_generator_id="duration.maintenance_window",
        query_asset_family="maintenance_window",
        hard_distractor_families=("duration", "comparison"),
        boolean_balancing_strategy="threshold_band",
        entity_family="engineering",
        plans=(
            Plan("maint.v3", maintenance_roles[:3],
                 (S("n1", "duration.convert_days_hours", ("downtime_days",),
                    "the window in hours, so the booked work can be taken off it"),
                  S("n2", "duration.sum", ("repair_hours", "inspection_hours")),
                  S("n3", "arithmetic.subtract", ("@n1", "@n2"),
                    "hours of slack left in the window")),
                 "n3", intent="window_slack"),
            Plan("maint.v5", (maintenance_roles[0], maintenance_roles[1],
                              maintenance_roles[2], maintenance_roles[4]),
                 (S("n1", "duration.convert_days_hours", ("downtime_days",)),
                  S("n2", "duration.sum", ("repair_hours", "inspection_hours")),
                  S("n3", "arithmetic.subtract", ("@n1", "@n2")),
                  S("n4", "comparison.at_least", ("@n3", "window_limit")),
                  S("n5", "boolean.not", ("@n4",),
                    "the window is too tight to book")),
                 "n5", intent="window_too_tight"),
            Plan("maint.v8", maintenance_roles,
                 (S("n1", "duration.convert_days_hours", ("downtime_days",),
                    "the window in hours, read again at n5"),
                  S("n2", "duration.sum", ("repair_hours", "inspection_hours"),
                    "booked hours, read again at n4"),
                  S("n3", "arithmetic.subtract", ("@n1", "@n2"),
                    "slack at the planned pace"),
                  S("n4", "duration.scale", ("@n2", "crew_factor"),
                    "booked hours once every pass is counted"),
                  S("n5", "arithmetic.subtract", ("@n1", "@n4"),
                    "slack at the slower pace"),
                  S("n6", "comparison.at_least", ("@n3", "window_limit")),
                  S("n7", "comparison.at_least", ("@n5", "backlog_limit")),
                  S("n8", "boolean.and", ("@n6", "@n7"),
                    "both paces still fit the window")),
                 "n8", intent="two_pace_window_verdict"),
        )))

    # ── travel leg (hours to minutes to seconds, and back to a date) ────
    travel_roles = (
        R("leg_hours", "duration_hours", "hours one leg of the route takes"),
        R("layover_minutes", "duration_minutes", "layover between legs, in minutes"),
        R("segment_count", "count_small", "how many legs the route has"),
        R("departure_date", "date_iso", "day the route starts"),
        R("places", "places", "decimal places the itinerary prints"),
    )
    out.append(Blueprint(
        workflow_id="duration.travel_leg",
        domain="date_time",
        natural_user_goal="write up how long a multi-leg route takes end to end",
        target_description="the per-leg span or the printed itinerary line",
        value_generator_id="duration.travel_leg",
        query_asset_family="route_itinerary",
        hard_distractor_families=("duration", "format"),
        entity_family="logistics",
        plans=(
            Plan("travel.v5", travel_roles[:3],
                 (S("n1", "duration.convert_hours_minutes", ("leg_hours",),
                    "the leg in minutes, so the layover can be added"),
                  S("n2", "arithmetic.add", ("@n1", "layover_minutes")),
                  S("n3", "duration.convert_minutes_seconds", ("@n2",)),
                  S("n4", "arithmetic.divide", ("@n3", "segment_count")),
                  S("n5", "rounding.to_int", ("@n4",),
                    "whole seconds attributable to each leg")),
                 "n5", intent="seconds_per_leg"),
            Plan("travel.v6", (travel_roles[0], travel_roles[1], travel_roles[4]),
                 (S("n1", "duration.convert_hours_minutes", ("leg_hours",)),
                  S("n2", "arithmetic.add", ("@n1", "layover_minutes"),
                    "door-to-door minutes for one leg"),
                  S("n3", "format.fixed", ("@n2", "places"),
                    "the printed span, read again by the tag"),
                  S("n4", "string.parse_number", ("@n3",),
                    "the printed span read back off the sheet"),
                  S("n5", "arithmetic.subtract", ("@n4", "layover_minutes"),
                    "the driving part of the printed span"),
                  S("n6", "format.tag", ("@n3", "@n5"),
                    "printed span tagged with its driving part")),
                 "n6", intent="itinerary_tag"),
            Plan("travel.v8", travel_roles,
                 (S("n1", "duration.scale", ("leg_hours", "segment_count"),
                    "hours for the whole route, read again at n5"),
                  S("n2", "duration.convert_hours_days", ("@n1",)),
                  S("n3", "rounding.to_int", ("@n2",),
                    "whole days, the only form a date can be shifted by"),
                  S("n4", "date.add_duration", ("departure_date", "@n3"),
                    "day the route ends"),
                  S("n5", "duration.convert_hours_minutes", ("@n1",)),
                  S("n6", "arithmetic.add", ("@n5", "layover_minutes"),
                    "door-to-door minutes"),
                  S("n7", "format.fixed", ("@n6", "places")),
                  S("n8", "string.concat", ("@n4", "@n7"),
                    "arrival date carrying the total span")),
                 "n8", intent="route_arrival_line"),
        )))

    return out
