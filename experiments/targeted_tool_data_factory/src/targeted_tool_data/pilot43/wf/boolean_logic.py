"""Multi-condition logic: conjunction, disjunction, exclusion and majority.

The point of this family is the combinator, not the predicate, so every
condition fed into an ``and`` / ``or`` / ``xor`` / ``majority`` node is computed
from a genuinely different quantity: elapsed hours against jobs per head,
dispersion against outlier counts, an arrival date against an average speed.
Each predicate still gets its comparison constant from a calibrated
``threshold_*`` role, which is what lets the combined verdict be balanced.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── shift readiness ─────────────────────────────────────────────────
    shift_roles = (
        R("shift_hours", "duration_hours", "length of the shift"),
        R("handover_hours", "duration_hours", "handover time on top of the shift"),
        R("shift_limit", "threshold_hours", "hours a shift may not exceed"),
        R("open_tickets", "quantity_units", "jobs still open on the board"),
        R("staff_on_duty", "count_people", "people rostered for the shift"),
        R("required_staff", "threshold_count", "people the roster requires"),
        R("workload_limit", "threshold_value", "jobs one person can carry"),
        R("ticket_limit", "threshold_count", "jobs the board may hold"),
    )
    out.append(Blueprint(
        workflow_id="logic.shift_readiness",
        domain="boolean_logic",
        natural_user_goal="decide whether a shift can start as rostered",
        target_description="the readiness verdict",
        value_generator_id="logic.shift",
        query_asset_family="shift_roster",
        hard_distractor_families=("boolean", "duration"),
        boolean_balancing_strategy="threshold_band",
        entity_family="field service",
        plans=(
            Plan("ready.v3", shift_roles[:3],
                 (S("n1", "duration.sum", ("shift_hours", "handover_hours")),
                  S("n2", "comparison.at_least", ("@n1", "shift_limit")),
                  S("n3", "boolean.not", ("@n2",),
                    "the shift stays inside the permitted length")),
                 "n3", intent="shift_length_legal"),
            Plan("ready.v6", shift_roles[:7],
                 (S("n1", "duration.sum", ("shift_hours", "handover_hours")),
                  S("n2", "comparison.at_least", ("@n1", "shift_limit")),
                  S("n3", "arithmetic.divide", ("open_tickets", "staff_on_duty"),
                    "jobs each person would carry"),
                  S("n4", "comparison.at_least", ("@n3", "workload_limit")),
                  S("n5", "comparison.at_least",
                    ("staff_on_duty", "required_staff")),
                  S("n6", "decision.majority", ("@n2", "@n4", "@n5"))),
                 "n6", intent="majority_readiness"),
            Plan("ready.v9", shift_roles,
                 (S("n1", "duration.sum", ("shift_hours", "handover_hours"),
                    "rostered time, read again by the length check"),
                  S("n2", "arithmetic.divide", ("open_tickets", "staff_on_duty")),
                  S("n3", "arithmetic.multiply", ("@n2", "@n1"),
                    "job-hours each person would absorb"),
                  S("n4", "comparison.at_least", ("@n1", "shift_limit")),
                  S("n5", "comparison.at_least", ("@n3", "workload_limit")),
                  S("n6", "comparison.at_least",
                    ("staff_on_duty", "required_staff")),
                  S("n7", "comparison.at_least", ("open_tickets", "ticket_limit")),
                  S("n8", "boolean.xor", ("@n6", "@n7"),
                    "staffing and backlog point in opposite directions"),
                  S("n9", "decision.any_of", ("@n4", "@n5", "@n8"))),
                 "n9", intent="shift_escalation_logic"),
        )))

    # ── probe arbitration ───────────────────────────────────────────────
    sensor_roles = (
        R("readings", "list_readings", "readings the probe logged"),
        R("mean_floor", "threshold_value", "level the probe must reach on average"),
        R("ceiling", "threshold_value", "reading no sample may exceed"),
        R("drift_limit", "threshold_value", "relative drift the probe may show"),
        R("outlier_limit", "threshold_count",
          "readings above the mean that are still tolerated"),
        R("spread_limit", "threshold_value",
          "gap between the peak and the mean that is still tolerated"),
    )
    out.append(Blueprint(
        workflow_id="logic.sensor_arbitration",
        domain="boolean_logic",
        natural_user_goal=("work out whether the symptoms of a drifting probe "
                           "agree with each other"),
        target_description="the arbitration verdict",
        value_generator_id="logic.sensor",
        query_asset_family="probe_log",
        hard_distractor_families=("statistics", "boolean"),
        boolean_balancing_strategy="threshold_band",
        entity_family="operations",
        plans=(
            Plan("arbitrate.v4", sensor_roles[:3],
                 (S("n1", "statistics.mean", ("readings",)),
                  S("n2", "validation.list_limit", ("readings", "ceiling")),
                  S("n3", "comparison.at_least", ("@n1", "mean_floor")),
                  S("n4", "boolean.and", ("@n2", "@n3"),
                    "no sample is out of range and the level is reached")),
                 "n4", intent="range_and_level"),
            Plan("arbitrate.v7",
                 (sensor_roles[0], sensor_roles[3], sensor_roles[4]),
                 (S("n1", "statistics.mean", ("readings",),
                    "the reference level, read again three steps later"),
                  S("n2", "statistics.stdev", ("readings",)),
                  S("n3", "arithmetic.divide", ("@n2", "@n1")),
                  S("n4", "list.reduce_count_above", ("readings", "@n1")),
                  S("n5", "comparison.at_least", ("@n3", "drift_limit")),
                  S("n6", "comparison.at_least", ("@n4", "outlier_limit")),
                  S("n7", "boolean.xor", ("@n5", "@n6"),
                    "dispersion and outlier count disagree")),
                 "n7", intent="symptoms_disagree"),
            Plan("arbitrate.v10",
                 (sensor_roles[0], sensor_roles[3], sensor_roles[4],
                  sensor_roles[5]),
                 (S("n1", "statistics.mean", ("readings",),
                    "the reference level, read by three later steps"),
                  S("n2", "statistics.stdev", ("readings",)),
                  S("n3", "list.reduce_max", ("readings",)),
                  S("n4", "arithmetic.divide", ("@n2", "@n1")),
                  S("n5", "arithmetic.subtract", ("@n3", "@n1")),
                  S("n6", "list.reduce_count_above", ("readings", "@n1")),
                  S("n7", "comparison.at_least", ("@n4", "drift_limit")),
                  S("n8", "comparison.at_least", ("@n5", "spread_limit")),
                  S("n9", "comparison.at_least", ("@n6", "outlier_limit")),
                  S("n10", "decision.all_of", ("@n7", "@n8", "@n9"),
                    "every symptom has to be present to condemn the probe")),
                 "n10", intent="all_symptoms_present"),
        )))

    # ── dispatch override ───────────────────────────────────────────────
    dispatch_roles = (
        R("distance_km", "length_km", "distance of the run"),
        R("driver_hours", "duration_hours", "hours the driver still has"),
        R("speed_floor", "threshold_value",
          "average speed the schedule is built on"),
        R("order_date", "date_iso", "date the order was placed"),
        R("transit_days", "duration_days", "days the carrier needs"),
        R("promised_date", "date_deadline", "date promised to the customer"),
        R("distance_limit", "threshold_value", "distance one driver may cover"),
        R("hours_limit", "threshold_hours", "hours the driver may still work"),
        R("late_week_index", "threshold_count",
          "weekday from which the receiving depot is closed"),
    )
    out.append(Blueprint(
        workflow_id="logic.dispatch_override",
        domain="boolean_logic",
        natural_user_goal="decide whether a run may be dispatched as planned",
        target_description="the dispatch verdict or the number of blocking rules",
        value_generator_id="logic.dispatch",
        query_asset_family="dispatch_run",
        hard_distractor_families=("date", "decision"),
        boolean_balancing_strategy="threshold_band",
        entity_family="logistics",
        plans=(
            Plan("dispatch.v2", dispatch_roles[:3],
                 (S("n1", "arithmetic.divide", ("distance_km", "driver_hours"),
                    "average speed the run demands"),
                  S("n2", "comparison.at_least", ("@n1", "speed_floor"))),
                 "n2", intent="speed_feasible"),
            Plan("dispatch.v5",
                 (dispatch_roles[3], dispatch_roles[4], dispatch_roles[5],
                  dispatch_roles[0], dispatch_roles[6], dispatch_roles[1],
                  dispatch_roles[7]),
                 (S("n1", "date.add_duration", ("order_date", "transit_days"),
                    "date the goods would arrive"),
                  S("n2", "date.compare", ("@n1", "promised_date")),
                  S("n3", "comparison.at_least",
                    ("distance_km", "distance_limit")),
                  S("n4", "comparison.at_least", ("driver_hours", "hours_limit")),
                  S("n5", "decision.count_true", ("@n2", "@n3", "@n4"),
                    "how many of the three rules the run satisfies")),
                 "n5", intent="satisfied_rule_count"),
            Plan("dispatch.v8",
                 (dispatch_roles[3], dispatch_roles[4], dispatch_roles[5],
                  dispatch_roles[8], dispatch_roles[0], dispatch_roles[1],
                  dispatch_roles[2]),
                 (S("n1", "date.add_duration", ("order_date", "transit_days"),
                    "arrival date, read again by the weekday check"),
                  S("n2", "date.compare", ("@n1", "promised_date")),
                  S("n3", "date.weekday", ("@n1",)),
                  S("n4", "comparison.at_least", ("@n3", "late_week_index")),
                  S("n5", "arithmetic.divide", ("distance_km", "driver_hours")),
                  S("n6", "comparison.at_least", ("@n5", "speed_floor")),
                  S("n7", "boolean.or", ("@n4", "@n6"),
                    "either the depot is shut or the run is too fast to be real"),
                  S("n8", "boolean.and", ("@n2", "@n7"))),
                 "n8", intent="arrival_and_feasibility"),
        )))

    return out
