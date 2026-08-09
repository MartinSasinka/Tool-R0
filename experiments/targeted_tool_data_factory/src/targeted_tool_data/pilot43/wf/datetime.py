"""Calendar workflows: spans between real dates, shifted dates, deadline verdicts.

Every plan starts from ISO dates and keeps whatever unit it derives from them.
``date.difference`` produces days, so a day count only ever meets another day
count or the explicit ``duration.convert_days_hours`` converter before it is
judged against an hour limit; a weekday, month or quarter index stays an index
and is only ever collected, compared or banded, never folded into a duration.

The plans differ in how much of the calendar they have to reconstruct. The
short ones just measure a span. The long ones rebuild a promised date out of
several independent day counts and then judge it, which is where the reuse of
an early span, the late references back to it and the second join come from.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── delivery window ─────────────────────────────────────────────────
    window_roles = (
        R("dispatch_date", "date_iso", "day the shipment leaves the supplier"),
        R("arrival_date", "date_iso", "day it reaches the depot"),
        R("buffer_days", "duration_days", "safety buffer the planner adds, in days"),
        R("handling_days", "duration_days", "days of handling at the depot"),
        R("transit_limit", "threshold_hours",
          "longest door-to-door transit the contract allows, in hours"),
        R("review_limit", "threshold_hours",
          "transit above which the leg goes to review, in hours"),
    )
    out.append(Blueprint(
        workflow_id="datetime.delivery_window",
        domain="date_time",
        natural_user_goal=("work out how long a shipment leg really takes and "
                           "when the goods can be promised"),
        target_description="the transit span, the promised date or the review verdict",
        value_generator_id="datetime.delivery_window",
        query_asset_family="shipment_leg",
        hard_distractor_families=("duration", "arithmetic"),
        boolean_balancing_strategy="threshold_band",
        entity_family="logistics",
        plans=(
            Plan("window.v2", window_roles[:2],
                 (S("n1", "date.difference", ("arrival_date", "dispatch_date"),
                    "transit measured in days"),
                  S("n2", "duration.convert_days_hours", ("@n1",),
                    "the same span in hours")),
                 "n2", intent="transit_hours"),
            Plan("window.v5", window_roles[:4],
                 (S("n1", "date.difference", ("arrival_date", "dispatch_date"),
                    "transit days, needed by both branches"),
                  S("n2", "arithmetic.add", ("@n1", "buffer_days"),
                    "transit with the safety buffer"),
                  S("n3", "arithmetic.add", ("@n1", "handling_days"),
                    "transit with depot handling"),
                  S("n4", "comparison.max", ("@n2", "@n3"),
                    "whichever allowance is larger"),
                  S("n5", "date.add_duration", ("dispatch_date", "@n4"),
                    "date the goods can be promised")),
                 "n5", intent="promised_date"),
            Plan("window.v9", window_roles,
                 (S("n1", "date.difference", ("arrival_date", "dispatch_date"),
                    "transit days, read again five steps later"),
                  S("n2", "duration.convert_days_hours", ("@n1",),
                    "transit hours"),
                  S("n3", "duration.convert_days_hours", ("handling_days",),
                    "handling hours; days cannot be summed with hours directly"),
                  S("n4", "duration.sum", ("@n2", "@n3"),
                    "door-to-door hours"),
                  S("n5", "arithmetic.add", ("@n1", "buffer_days"),
                    "transit days with the buffer"),
                  S("n6", "duration.convert_days_hours", ("@n5",),
                    "buffered transit in hours"),
                  S("n7", "comparison.at_least", ("@n4", "transit_limit")),
                  S("n8", "comparison.at_least", ("@n6", "review_limit")),
                  S("n9", "boolean.or", ("@n7", "@n8"),
                    "either reading sends the leg to review")),
                 "n9", intent="transit_review_verdict"),
        )))

    # ── inspection calendar ─────────────────────────────────────────────
    inspection_roles = (
        R("last_inspection", "date_iso", "date of the last inspection"),
        R("review_date", "date_iso", "date the review board sits"),
        R("interval_days", "duration_days", "inspection interval in days"),
        R("grace_days", "duration_days", "grace period after the due date, in days"),
        R("low_cut", "cut_low", "slack below which the site counts as tight"),
        R("high_cut", "cut_high", "slack above which the site counts as comfortable"),
    )
    out.append(Blueprint(
        workflow_id="datetime.inspection_calendar",
        domain="date_time",
        natural_user_goal=("place an inspection in the calendar and see how much "
                           "slack is left before the review"),
        target_description="the calendar positions, their order or the slack band",
        value_generator_id="datetime.inspection_calendar",
        query_asset_family="inspection_plan",
        hard_distractor_families=("date", "classification"),
        entity_family="engineering",
        plans=(
            Plan("inspect.v3", inspection_roles[:2],
                 (S("n1", "date.month", ("last_inspection",)),
                  S("n2", "date.month", ("review_date",)),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2"),
                    "how many month numbers apart the two sit")),
                 "n3", intent="month_gap"),
            Plan("inspect.v6", inspection_roles[:3],
                 (S("n1", "date.add_duration", ("last_inspection", "interval_days"),
                    "date the next inspection falls due, read by two branches"),
                  S("n2", "date.weekday", ("@n1",)),
                  S("n3", "date.quarter", ("@n1",)),
                  S("n4", "date.month", ("review_date",)),
                  S("n5", "list.build", ("@n2", "@n3", "@n4"),
                    "the three calendar positions"),
                  S("n6", "list.map_sort_asc", ("@n5",),
                    "positions in ascending order")),
                 "n6", intent="calendar_positions"),
            Plan("inspect.v8", inspection_roles,
                 (S("n1", "date.add_duration", ("last_inspection", "interval_days"),
                    "due date, reused by the grace branch"),
                  S("n2", "date.difference", ("review_date", "@n1"),
                    "days between the due date and the review"),
                  S("n3", "duration.convert_days_hours", ("@n2",),
                    "that slack in hours"),
                  S("n4", "date.add_duration", ("@n1", "grace_days"),
                    "last date the grace period still covers"),
                  S("n5", "date.difference", ("review_date", "@n4"),
                    "days between the end of the grace period and the review"),
                  S("n6", "duration.convert_days_hours", ("@n5",),
                    "that slack in hours"),
                  S("n7", "statistics.average_two", ("@n3", "@n6"),
                    "typical slack in hours"),
                  S("n8", "classification.three_bands",
                    ("@n7", "low_cut", "high_cut"))),
                 "n8", intent="slack_band"),
        )))

    # ── service pledge ──────────────────────────────────────────────────
    pledge_roles = (
        R("ticket_opened", "date_iso", "day the ticket was raised"),
        R("work_started", "date_iso", "day an engineer picked it up"),
        R("resolution_date", "date_iso", "day the fault was cleared"),
        R("pledged_date", "date_deadline", "date the contract pledges to the customer"),
        R("buffer_days", "duration_days", "days of paperwork after the fix"),
        R("response_limit", "threshold_hours",
          "hours the contract allows before work must start"),
        R("resolution_limit", "threshold_hours",
          "hours the contract allows before the fault must be cleared"),
        R("working_limit", "threshold_count",
          "days of actual work above which the ticket is escalated"),
    )
    out.append(Blueprint(
        workflow_id="datetime.service_pledge",
        domain="date_time",
        natural_user_goal=("decide whether a service ticket honoured what was "
                           "pledged to the customer"),
        target_description="whether the pledge was met",
        value_generator_id="datetime.service_pledge",
        query_asset_family="service_ticket",
        hard_distractor_families=("date", "comparison"),
        boolean_balancing_strategy="threshold_band",
        entity_family="field_service",
        plans=(
            Plan("pledge.v4", pledge_roles[:4],
                 (S("n1", "date.difference", ("work_started", "ticket_opened"),
                    "days lost before work started"),
                  S("n2", "date.add_duration", ("resolution_date", "@n1"),
                    "when a rerun of the same ticket would close"),
                  S("n3", "date.compare", ("@n2", "pledged_date")),
                  S("n4", "boolean.not", ("@n3",),
                    "a rerun would miss the pledge")),
                 "n4", intent="rerun_would_miss"),
            Plan("pledge.v7", (pledge_roles[0], pledge_roles[1], pledge_roles[2],
                               pledge_roles[6], pledge_roles[7]),
                 (S("n1", "date.difference", ("resolution_date", "ticket_opened"),
                    "total days the ticket was open, read again at n5"),
                  S("n2", "duration.convert_days_hours", ("@n1",)),
                  S("n3", "date.difference", ("work_started", "ticket_opened"),
                    "days the ticket waited in the queue"),
                  S("n4", "comparison.at_least", ("@n2", "resolution_limit")),
                  S("n5", "arithmetic.subtract", ("@n1", "@n3"),
                    "days an engineer actually worked on it"),
                  S("n6", "comparison.at_least", ("@n5", "working_limit")),
                  S("n7", "boolean.and", ("@n4", "@n6"))),
                 "n7", intent="slow_and_labour_heavy"),
            Plan("pledge.v10", (pledge_roles[0], pledge_roles[1], pledge_roles[2],
                                pledge_roles[3], pledge_roles[4], pledge_roles[5],
                                pledge_roles[6]),
                 (S("n1", "date.difference", ("work_started", "ticket_opened"),
                    "response delay in days"),
                  S("n2", "date.difference", ("resolution_date", "ticket_opened"),
                    "total days open, read again at n7"),
                  S("n3", "duration.convert_days_hours", ("@n1",)),
                  S("n4", "duration.convert_days_hours", ("@n2",)),
                  S("n5", "comparison.at_least", ("@n3", "response_limit")),
                  S("n6", "comparison.at_least", ("@n4", "resolution_limit")),
                  S("n7", "arithmetic.add", ("@n2", "buffer_days"),
                    "days until the paperwork is filed too"),
                  S("n8", "date.add_duration", ("ticket_opened", "@n7"),
                    "date the ticket is finally closed out"),
                  S("n9", "date.compare", ("@n8", "pledged_date")),
                  S("n10", "decision.all_of", ("@n5", "@n6", "@n9"),
                    "every clause of the pledge")),
                 "n10", intent="full_pledge_audit"),
        )))

    # ── cycle report (dates and numbers become text) ─────────────────────
    cycle_roles = (
        R("cycle_start", "date_iso", "first day of the billing cycle"),
        R("cycle_end", "date_iso", "last day of the billing cycle"),
        R("extension_days", "duration_days", "days the cycle is extended by"),
        R("cycle_name", "text_label", "name the cycle is filed under"),
        R("places", "places", "decimal places the report uses"),
        R("unit_word", "unit_word", "unit the report prints"),
    )
    out.append(Blueprint(
        workflow_id="datetime.cycle_report",
        domain="date_time",
        natural_user_goal="write the printable line for a billing cycle",
        target_description="the formatted cycle line",
        value_generator_id="datetime.cycle_report",
        query_asset_family="billing_cycle",
        hard_distractor_families=("format", "string"),
        entity_family="finance",
        plans=(
            Plan("cycle.v4", (cycle_roles[0], cycle_roles[1], cycle_roles[4],
                              cycle_roles[5]),
                 (S("n1", "date.difference", ("cycle_end", "cycle_start"),
                    "cycle length in days"),
                  S("n2", "duration.convert_days_hours", ("@n1",),
                    "cycle length in hours"),
                  S("n3", "format.fixed", ("@n2", "places")),
                  S("n4", "string.concat", ("@n3", "unit_word"))),
                 "n4", intent="cycle_length_text"),
            Plan("cycle.v6", cycle_roles[:4],
                 (S("n1", "date.difference", ("cycle_end", "cycle_start"),
                    "cycle length in days, read again at n4"),
                  S("n2", "arithmetic.add", ("@n1", "extension_days")),
                  S("n3", "date.add_duration", ("cycle_start", "@n2"),
                    "day the extended cycle closes"),
                  S("n4", "duration.convert_days_hours", ("@n1",)),
                  S("n5", "string.concat", ("cycle_name", "@n3")),
                  S("n6", "format.tag", ("@n5", "@n4"),
                    "cycle label carrying its length in hours")),
                 "n6", intent="extended_cycle_label"),
            Plan("cycle.v7", (cycle_roles[0], cycle_roles[1], cycle_roles[2],
                              cycle_roles[4], cycle_roles[5]),
                 (S("n1", "date.difference", ("cycle_end", "cycle_start"),
                    "cycle length in days"),
                  S("n2", "duration.convert_days_hours", ("@n1",)),
                  S("n3", "format.fixed", ("@n2", "places")),
                  S("n4", "arithmetic.add", ("@n1", "extension_days")),
                  S("n5", "date.add_duration", ("cycle_start", "@n4"),
                    "day the extended cycle closes"),
                  S("n6", "string.concat", ("@n3", "unit_word")),
                  S("n7", "string.concat", ("@n6", "@n5"))),
                 "n7", intent="cycle_line_with_end_date"),
        )))

    # ── visit audit ─────────────────────────────────────────────────────
    visit_roles = (
        R("first_visit", "date_iso", "date of the first site visit"),
        R("second_visit", "date_iso", "date of the second site visit"),
        R("third_visit", "date_iso", "date of the third site visit"),
        R("report_date", "date_iso", "date the audit report is written"),
        R("grace_days", "duration_days", "days the auditor allows on top"),
    )
    out.append(Blueprint(
        workflow_id="datetime.visit_audit",
        domain="date_time",
        natural_user_goal="see how evenly a site was visited before the audit",
        target_description="the spacing of the visits or how far it drifts",
        value_generator_id="datetime.visit_audit",
        query_asset_family="visit_log",
        hard_distractor_families=("statistics", "date"),
        entity_family="operations",
        plans=(
            Plan("visit.v3", visit_roles[:3],
                 (S("n1", "date.difference", ("second_visit", "first_visit")),
                  S("n2", "date.difference", ("third_visit", "second_visit")),
                  S("n3", "statistics.average_two", ("@n1", "@n2"),
                    "typical gap between visits, in days")),
                 "n3", intent="typical_visit_gap"),
            Plan("visit.v6", visit_roles,
                 (S("n1", "date.difference", ("second_visit", "first_visit")),
                  S("n2", "date.difference", ("third_visit", "second_visit")),
                  S("n3", "date.difference", ("report_date", "third_visit"),
                    "days since the last visit"),
                  S("n4", "comparison.max", ("@n1", "@n2"),
                    "longest gap between two visits"),
                  S("n5", "arithmetic.add", ("@n4", "@n3")),
                  S("n6", "arithmetic.add", ("@n5", "grace_days"),
                    "days the auditor will treat as uncovered")),
                 "n6", intent="uncovered_days"),
            Plan("visit.v9", visit_roles,
                 (S("n1", "date.difference", ("second_visit", "first_visit")),
                  S("n2", "date.difference", ("third_visit", "second_visit")),
                  S("n3", "date.difference", ("report_date", "third_visit"),
                    "days since the last visit, read again at n7"),
                  S("n4", "list.build", ("@n1", "@n2", "@n3"),
                    "the three gaps, all in days"),
                  S("n5", "statistics.mean", ("@n4",)),
                  S("n6", "list.reduce_max", ("@n4",)),
                  S("n7", "arithmetic.add", ("@n3", "grace_days")),
                  S("n8", "arithmetic.subtract", ("@n6", "@n5"),
                    "how far the worst gap sits above the mean"),
                  S("n9", "arithmetic.add", ("@n8", "@n7"),
                    "drift plus the days still uncovered")),
                 "n9", intent="visit_drift"),
        )))

    return out
