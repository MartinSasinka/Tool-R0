"""Threshold decisions: the point where measured quantities become a verdict.

Every predicate compares a derived quantity against a calibrated ``threshold_*``
role, never against a second sampled fact, so the True/False share of the
family can be balanced after execution. The plans differ in how much evidence
the verdict rests on: the short ones compare one derived quantity, the long
ones weigh elapsed effort, backlog, severity, load concentration and committed
spend against their own limits before combining the partial verdicts.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── support ticket escalation ───────────────────────────────────────
    escalation_roles = (
        R("wait_hours", "duration_hours", "hours the ticket waited in the queue"),
        R("handling_hours", "duration_hours",
          "hours already spent working on the ticket"),
        R("sla_hours", "threshold_hours", "hours the service level allows"),
        R("backlog_items", "count_items", "tickets still open in the queue"),
        R("backlog_limit", "threshold_count", "backlog the team can absorb"),
        R("severity", "score_points", "severity score of the ticket"),
        R("severity_floor", "threshold_value",
          "severity from which an escalation is automatic"),
        R("per_item_limit", "threshold_hours",
          "effort per open ticket the team can carry"),
    )
    out.append(Blueprint(
        workflow_id="decision.escalation_gate",
        domain="threshold_decision",
        natural_user_goal="decide whether a support ticket has to be escalated",
        target_description="the escalation verdict",
        value_generator_id="decision.escalation",
        query_asset_family="support_ticket",
        hard_distractor_families=("duration", "comparison"),
        boolean_balancing_strategy="threshold_band",
        entity_family="support",
        plans=(
            Plan("escalate.v2", escalation_roles[:3],
                 (S("n1", "duration.sum", ("wait_hours", "handling_hours"),
                    "total time the ticket has consumed"),
                  S("n2", "comparison.at_least", ("@n1", "sla_hours"),
                    "service level already breached?")),
                 "n2", intent="sla_breach"),
            Plan("escalate.v5", escalation_roles[:7],
                 (S("n1", "duration.sum", ("wait_hours", "handling_hours")),
                  S("n2", "comparison.at_least", ("@n1", "sla_hours")),
                  S("n3", "comparison.at_least",
                    ("backlog_items", "backlog_limit")),
                  S("n4", "comparison.at_least", ("severity", "severity_floor")),
                  S("n5", "decision.majority", ("@n2", "@n3", "@n4"),
                    "two of the three independent symptoms are enough")),
                 "n5", intent="majority_escalation"),
            Plan("escalate.v8", escalation_roles,
                 (S("n1", "duration.sum", ("wait_hours", "handling_hours"),
                    "elapsed effort, read again five steps later"),
                  S("n2", "arithmetic.divide", ("@n1", "backlog_items"),
                    "effort the team carries per open ticket"),
                  S("n3", "comparison.at_least", ("@n2", "per_item_limit")),
                  S("n4", "comparison.at_least",
                    ("backlog_items", "backlog_limit")),
                  S("n5", "comparison.at_least", ("severity", "severity_floor")),
                  S("n6", "comparison.at_least", ("@n1", "sla_hours")),
                  S("n7", "decision.majority", ("@n3", "@n4", "@n5")),
                  S("n8", "boolean.and", ("@n7", "@n6"),
                    "the queue is strained and this ticket is already late")),
                 "n8", intent="strained_queue_escalation"),
        )))

    # ── trailer load release ────────────────────────────────────────────
    capacity_roles = (
        R("pallet_mass", "mass_kg", "weight of a single pallet"),
        R("pallet_count", "count_items", "pallets in the consignment"),
        R("tare_mass", "mass_kg", "weight of the empty trailer bed"),
        R("deck_capacity", "threshold_value", "weight the deck may carry"),
        R("volume_each", "volume_l", "volume one pallet occupies"),
        R("hold_capacity", "threshold_value", "volume the hold may take"),
        R("density_limit", "threshold_value",
          "kilograms per litre the hold is rated for"),
        R("pallet_weights", "list_quantities",
          "weight of every pallet on the manifest"),
        R("concentration_limit", "threshold_percent",
          "share of the load a single pallet may represent"),
    )
    out.append(Blueprint(
        workflow_id="decision.capacity_release",
        domain="threshold_decision",
        natural_user_goal="decide whether a loaded trailer may be released",
        target_description="the load release verdict",
        value_generator_id="decision.capacity",
        query_asset_family="trailer_manifest",
        hard_distractor_families=("arithmetic", "list"),
        boolean_balancing_strategy="threshold_band",
        entity_family="logistics",
        plans=(
            Plan("release.v3", capacity_roles[:4],
                 (S("n1", "arithmetic.multiply", ("pallet_mass", "pallet_count"),
                    "weight of the palletised goods"),
                  S("n2", "arithmetic.add", ("@n1", "tare_mass")),
                  S("n3", "comparison.at_least", ("@n2", "deck_capacity"),
                    "deck rating exceeded?")),
                 "n3", intent="deck_overload"),
            Plan("release.v7",
                 (capacity_roles[0], capacity_roles[1], capacity_roles[3],
                  capacity_roles[4], capacity_roles[5], capacity_roles[6]),
                 (S("n1", "arithmetic.multiply", ("pallet_mass", "pallet_count"),
                    "total weight, read again by the density step"),
                  S("n2", "arithmetic.multiply", ("volume_each", "pallet_count"),
                    "total volume, read again by the density step"),
                  S("n3", "comparison.at_least", ("@n1", "deck_capacity")),
                  S("n4", "comparison.at_least", ("@n2", "hold_capacity")),
                  S("n5", "arithmetic.divide", ("@n1", "@n2"),
                    "density of the consignment"),
                  S("n6", "comparison.at_least", ("@n5", "density_limit")),
                  S("n7", "decision.majority", ("@n3", "@n4", "@n6"))),
                 "n7", intent="weight_volume_density_verdict"),
            Plan("release.v8",
                 (capacity_roles[7], capacity_roles[3], capacity_roles[4],
                  capacity_roles[1], capacity_roles[5], capacity_roles[8]),
                 (S("n1", "list.reduce_sum", ("pallet_weights",),
                    "total manifest weight, read again four steps later"),
                  S("n2", "list.reduce_max", ("pallet_weights",)),
                  S("n3", "rates.share_percent", ("@n2", "@n1"),
                    "share the heaviest pallet carries"),
                  S("n4", "arithmetic.multiply", ("volume_each", "pallet_count")),
                  S("n5", "comparison.at_least", ("@n1", "deck_capacity")),
                  S("n6", "comparison.at_least", ("@n4", "hold_capacity")),
                  S("n7", "comparison.at_least",
                    ("@n3", "concentration_limit")),
                  S("n8", "decision.any_of", ("@n5", "@n6", "@n7"),
                    "any one breach blocks the release")),
                 "n8", intent="manifest_release_verdict"),
        )))

    # ── out-of-plan spend approval ──────────────────────────────────────
    budget_roles = (
        R("spend_amount", "money_price", "price of the item being requested"),
        R("unit_count", "quantity_units", "units on the request"),
        R("remaining_budget", "money_budget", "budget left for the period"),
        R("approval_ceiling", "threshold_money",
          "amount above which a signature is required"),
        R("spend_share_limit", "threshold_percent",
          "share of the budget one request may take"),
        R("signoff_count", "count_people", "managers who have already signed"),
        R("quorum", "threshold_count", "signatures the policy requires"),
        R("prior_spends", "list_prices", "amounts already committed this period"),
        R("run_rate_limit", "threshold_money",
          "average commitment the period is planned around"),
    )
    out.append(Blueprint(
        workflow_id="decision.budget_override",
        domain="threshold_decision",
        natural_user_goal=("decide whether an out-of-plan purchase needs a "
                           "budget override"),
        target_description="the override verdict",
        value_generator_id="decision.budget",
        query_asset_family="purchase_request",
        hard_distractor_families=("rates", "statistics"),
        boolean_balancing_strategy="threshold_band",
        entity_family="finance",
        plans=(
            Plan("override.v4",
                 (budget_roles[0], budget_roles[1], budget_roles[2],
                  budget_roles[4]),
                 (S("n1", "arithmetic.multiply", ("spend_amount", "unit_count")),
                  S("n2", "rates.share_percent", ("@n1", "remaining_budget")),
                  S("n3", "comparison.at_least", ("@n2", "spend_share_limit")),
                  S("n4", "boolean.not", ("@n3",),
                    "the request still fits inside the delegated share")),
                 "n4", intent="within_delegated_share"),
            Plan("override.v6", budget_roles[:7],
                 (S("n1", "arithmetic.multiply", ("spend_amount", "unit_count"),
                    "request value, read again by the ceiling check"),
                  S("n2", "rates.share_percent", ("@n1", "remaining_budget")),
                  S("n3", "comparison.at_least", ("@n1", "approval_ceiling")),
                  S("n4", "comparison.at_least", ("@n2", "spend_share_limit")),
                  S("n5", "comparison.at_least", ("signoff_count", "quorum")),
                  S("n6", "decision.majority", ("@n3", "@n4", "@n5"))),
                 "n6", intent="majority_override"),
            Plan("override.v9",
                 (budget_roles[0], budget_roles[1], budget_roles[7],
                  budget_roles[2], budget_roles[4], budget_roles[8],
                  budget_roles[3]),
                 (S("n1", "arithmetic.multiply", ("spend_amount", "unit_count"),
                    "request value, read again seven steps later"),
                  S("n2", "list.reduce_sum", ("prior_spends",)),
                  S("n3", "statistics.mean", ("prior_spends",)),
                  S("n4", "arithmetic.add", ("@n1", "@n2"),
                    "everything committed once this request is booked"),
                  S("n5", "rates.share_percent", ("@n4", "remaining_budget")),
                  S("n6", "comparison.at_least", ("@n5", "spend_share_limit")),
                  S("n7", "comparison.at_least", ("@n3", "run_rate_limit")),
                  S("n8", "comparison.at_least", ("@n1", "approval_ceiling")),
                  S("n9", "decision.any_of", ("@n6", "@n7", "@n8"))),
                 "n9", intent="committed_spend_override"),
        )))

    return out
