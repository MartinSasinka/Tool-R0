"""Threshold-decision workflows: service credits, batch release, budget gates.

Each blueprint ends in a rule someone actually applies, so the interesting part
is how many independent readings of the same quantity the rule needs: an
overrun matters both as a duration and as money, a spend matters both in total
and per line.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── service credit gate ─────────────────────────────────────────────
    promised_hours = R("promised_hours", "duration_hours",
                       "hours the contract promised the job would take")
    actual_hours = R("actual_hours", "duration_hours",
                     "hours the job actually took")
    penalty_rate = R("penalty_rate", "money_price",
                     "credit the contract owes for each hour of overrun")
    overrun_limit = R("overrun_limit", "threshold_hours",
                      "overrun the customer agreed to tolerate")
    minute_limit = R("minute_limit", "threshold_value",
                     "minutes of overrun that trigger a report")
    credit_limit = R("credit_limit", "threshold_money",
                     "credit above which the account manager gets involved")
    share_limit = R("share_limit", "threshold_percent",
                    "share of the promised time the overrun may add")
    currency = R("currency", "currency_code", "currency the credit is issued in")

    out.append(Blueprint(
        workflow_id="threshold_decision.service_credit_gate",
        domain="threshold_decision",
        natural_user_goal=("work out whether a job ran late enough to owe the "
                           "customer a credit"),
        target_description="the overrun, the credit due or the credit verdict",
        value_generator_id="threshold_decision.service_credit",
        query_asset_family="service_report",
        hard_distractor_families=("duration", "comparison"),
        boolean_balancing_strategy="calibrate_overrun_limits",
        entity_family="field_service",
        plans=(
            Plan("credit.v3", (promised_hours, actual_hours, penalty_rate,
                               currency),
                 (S("n1", "arithmetic.subtract", ("actual_hours",
                                                  "promised_hours"),
                    "how far the job ran over"),
                  S("n2", "arithmetic.multiply", ("penalty_rate", "@n1"),
                    "the credit the overrun earns"),
                  S("n3", "format.currency", ("@n2", "currency"))),
                 "n3", intent="credit_note_line"),
            Plan("credit.v5", (promised_hours, actual_hours, overrun_limit,
                               share_limit),
                 (S("n1", "arithmetic.subtract", ("actual_hours",
                                                  "promised_hours"),
                    "the overrun, judged two different ways"),
                  S("n2", "rates.share_percent", ("@n1", "promised_hours")),
                  S("n3", "comparison.at_least", ("@n2", "share_limit")),
                  S("n4", "comparison.greater", ("@n1", "overrun_limit")),
                  S("n5", "boolean.and", ("@n3", "@n4"),
                    "the overrun has to be big in both senses")),
                 "n5", intent="credit_due_verdict"),
            Plan("credit.v8", (promised_hours, actual_hours, penalty_rate,
                               minute_limit, credit_limit, share_limit),
                 (S("n1", "arithmetic.subtract", ("actual_hours",
                                                  "promised_hours"),
                    "the overrun, read three ways before the rule fires"),
                  S("n2", "duration.convert_hours_minutes", ("@n1",)),
                  S("n3", "comparison.at_least", ("@n2", "minute_limit")),
                  S("n4", "arithmetic.multiply", ("penalty_rate", "@n1")),
                  S("n5", "comparison.at_least", ("@n4", "credit_limit")),
                  S("n6", "rates.share_percent", ("@n1", "promised_hours")),
                  S("n7", "comparison.at_least", ("@n6", "share_limit")),
                  S("n8", "decision.majority", ("@n3", "@n5", "@n7"),
                    "most of the escalation rules have to fire")),
                 "n8", intent="escalation_verdict"),
        )))

    # ── batch release gate ──────────────────────────────────────────────
    batch_readings = R("batch_readings", "list_readings",
                       "readings taken across the batch")
    calibration_offset = R("calibration_offset", "generic_value",
                           "correction the calibration sheet applies")
    range_low = R("range_low", "range_low", "lowest level the spec allows")
    range_high = R("range_high", "range_high", "highest level the spec allows")
    drift_limit = R("drift_limit", "threshold_value",
                    "drift outside the spec the reviewer will accept")
    reading_cap = R("reading_cap", "threshold_value",
                    "level no single reading may exceed")
    spread_limit = R("spread_limit", "threshold_value",
                     "spread across the batch the process allows")
    outlier_share = R("outlier_share", "threshold_percent",
                      "share of readings allowed to sit high")

    out.append(Blueprint(
        workflow_id="threshold_decision.batch_release_gate",
        domain="threshold_decision",
        natural_user_goal=("decide whether a batch can be released on the "
                           "strength of the readings taken across it"),
        target_description="the drift from spec or the release verdict",
        value_generator_id="threshold_decision.batch_release",
        query_asset_family="batch_record",
        hard_distractor_families=("validation", "statistics"),
        boolean_balancing_strategy="calibrate_spec_window",
        entity_family="manufacturing",
        plans=(
            Plan("release.v3", (batch_readings, range_low, range_high),
                 (S("n1", "statistics.mean", ("batch_readings",),
                    "the level the batch settled at"),
                  S("n2", "validation.in_range", ("@n1", "range_low",
                                                  "range_high")),
                  S("n3", "boolean.not", ("@n2",),
                    "true when the batch has left its window")),
                 "n3", intent="out_of_spec_verdict"),
            Plan("release.v4", (batch_readings, range_low, range_high,
                                drift_limit),
                 (S("n1", "statistics.mean", ("batch_readings",),
                    "the level, compared with its own clamped version"),
                  S("n2", "validation.clamp", ("@n1", "range_low",
                                               "range_high")),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2"),
                    "how far outside the spec the batch sits"),
                  S("n4", "comparison.greater", ("@n3", "drift_limit"))),
                 "n4", intent="drift_verdict"),
            Plan("release.v6", (batch_readings, range_low, range_high,
                                reading_cap, spread_limit),
                 (S("n1", "statistics.mean", ("batch_readings",)),
                  S("n2", "statistics.range", ("batch_readings",),
                    "how far the batch wandered"),
                  S("n3", "validation.in_range", ("@n1", "range_low",
                                                  "range_high")),
                  S("n4", "validation.list_limit", ("batch_readings",
                                                    "reading_cap")),
                  S("n5", "comparison.at_least", ("@n2", "spread_limit")),
                  S("n6", "decision.all_of", ("@n3", "@n4", "@n5"),
                    "the batch has to satisfy every release rule")),
                 "n6", intent="release_verdict"),
            Plan("release.v9", (batch_readings, calibration_offset,
                                outlier_share),
                 (S("n1", "list.map_offset", ("batch_readings",
                                              "calibration_offset"),
                    "readings after calibration, re-read several times"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.stdev", ("@n1",)),
                  S("n4", "arithmetic.add", ("@n2", "@n3"),
                    "the line above which a reading counts as high"),
                  S("n5", "list.reduce_count_above", ("@n1", "@n4")),
                  S("n6", "list.reduce_count", ("@n1",),
                    "how many readings there were in total"),
                  S("n7", "rates.share_percent", ("@n5", "@n6")),
                  S("n8", "comparison.at_least", ("@n7", "outlier_share")),
                  S("n9", "boolean.not", ("@n8",),
                    "true when few enough readings ran high")),
                 "n9", intent="batch_clean_verdict"),
        )))

    # ── budget gate ─────────────────────────────────────────────────────
    spend_lines = R("spend_lines", "list_prices",
                    "amount on each line of the requisition")
    tax_rate = R("tax_rate", "percent_tax", "tax added to the requisition")
    budget_limit = R("budget_limit", "threshold_money",
                     "amount the delegated authority covers")
    line_limit = R("line_limit", "threshold_value",
                   "average line value that needs a second signature")
    single_line_limit = R("single_line_limit", "threshold_money",
                          "value a single line may reach unchallenged")
    concentration_limit = R("concentration_limit", "threshold_percent",
                            "share of the requisition one line may represent")
    cut_low = R("cut_low", "cut_low", "spend that still counts as routine")
    cut_high = R("cut_high", "cut_high", "spend that counts as major")

    out.append(Blueprint(
        workflow_id="threshold_decision.requisition_budget_gate",
        domain="threshold_decision",
        natural_user_goal=("find out whether a requisition can be signed off "
                           "locally or has to go up the chain"),
        target_description="the requisition total, its band or the approval verdict",
        value_generator_id="threshold_decision.requisition",
        query_asset_family="requisition_form",
        hard_distractor_families=("list", "rates"),
        boolean_balancing_strategy="calibrate_authority_limits",
        entity_family="procurement",
        plans=(
            Plan("req.v3", (spend_lines, tax_rate, cut_low, cut_high),
                 (S("n1", "list.reduce_sum", ("spend_lines",),
                    "everything on the requisition"),
                  S("n2", "rates.apply_tax", ("@n1", "tax_rate")),
                  S("n3", "classification.three_bands", ("@n2", "cut_low",
                                                         "cut_high"))),
                 "n3", intent="requisition_band"),
            Plan("req.v4", (spend_lines, single_line_limit),
                 (S("n1", "list.reduce_sum", ("spend_lines",)),
                  S("n2", "list.reduce_max", ("spend_lines",),
                    "the largest single line"),
                  S("n3", "arithmetic.subtract", ("@n1", "@n2"),
                    "what the rest of the requisition comes to"),
                  S("n4", "comparison.at_least", ("@n3",
                                                  "single_line_limit"))),
                 "n4", intent="residual_spend_verdict"),
            Plan("req.v7", (spend_lines, tax_rate, budget_limit, line_limit,
                            single_line_limit),
                 (S("n1", "list.reduce_sum", ("spend_lines",),
                    "the requisition total"),
                  S("n2", "rates.apply_tax", ("@n1", "tax_rate")),
                  S("n3", "comparison.at_least", ("@n2", "budget_limit")),
                  S("n4", "statistics.mean", ("spend_lines",),
                    "what a typical line is worth"),
                  S("n5", "comparison.at_least", ("@n4", "line_limit")),
                  S("n6", "validation.list_limit", ("spend_lines",
                                                    "single_line_limit"),
                    "did every individual line stay inside its own limit"),
                  S("n7", "decision.count_true", ("@n3", "@n5", "@n6"),
                    "how many of the escalation rules the requisition trips")),
                 "n7", intent="escalation_rule_count"),
            Plan("req.v10", (spend_lines, tax_rate, budget_limit, line_limit,
                             concentration_limit),
                 (S("n1", "list.reduce_sum", ("spend_lines",),
                    "the total, read again by three different rules"),
                  S("n2", "list.reduce_max", ("spend_lines",)),
                  S("n3", "statistics.mean", ("spend_lines",),
                    "average line value"),
                  S("n4", "rates.share_percent", ("@n2", "@n1"),
                    "how concentrated the requisition is"),
                  S("n5", "rates.apply_tax", ("@n1", "tax_rate")),
                  S("n6", "comparison.at_least", ("@n5", "budget_limit")),
                  S("n7", "comparison.at_least", ("@n4",
                                                  "concentration_limit")),
                  S("n8", "comparison.at_least", ("@n3", "line_limit")),
                  S("n9", "boolean.and", ("@n7", "@n8"),
                    "a fat average line inside a concentrated requisition"),
                  S("n10", "boolean.or", ("@n6", "@n9"),
                    "the total alone can also force it upstairs")),
                 "n10", intent="escalation_needed"),
        )))

    return out
