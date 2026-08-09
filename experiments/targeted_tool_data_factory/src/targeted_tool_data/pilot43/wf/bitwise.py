"""Bitwise workflows about masks people actually keep in a table.

The registry does carry ``bitwise.and``/``or``/``xor``/``shift``, so these plans
bind them directly instead of imitating them with integer arithmetic. Every
mask is a whole number an operator would read off a permission table, a
feature-flag record or a capacity register, and every step is a question
someone asks about such a table: which permissions survive the policy, which
flags the override actually changed, how much of a doubled slot register is
still free. Nothing here is an abstract bit puzzle, and no mask is ever mixed
with a physical quantity.

The plans differ in how many registers have to be reconciled: two masks and a
request for the short ones, a policy mask reused against both an escalation and
a regional register for the long ones.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── permission masks ────────────────────────────────────────────────
    permission_roles = (
        R("role_mask", "count_items", "permission bits the role grants"),
        R("policy_mask", "count_items", "permission bits the security policy allows"),
        R("requested_mask", "count_items", "permission bits the request asks for"),
        R("escalation_mask", "count_items",
          "permission bits an on-call escalation adds"),
        R("gap_limit", "threshold_count",
          "mismatch between request and grant that still passes review"),
        R("overlap_limit", "threshold_count",
          "size of the escalated mask that still passes review"),
    )
    out.append(Blueprint(
        workflow_id="bitwise.permission_mask",
        domain="bitwise_realistic",
        natural_user_goal=("work out which permissions a role really ends up with "
                           "once the policy and the request are applied"),
        target_description="the effective permission mask or the review verdict",
        value_generator_id="bitwise.permission_mask",
        query_asset_family="permission_table",
        hard_distractor_families=("bitwise", "arithmetic"),
        boolean_balancing_strategy="threshold_band",
        entity_family="operations",
        plans=(
            Plan("perm.v2", permission_roles[:3],
                 (S("n1", "bitwise.and", ("role_mask", "policy_mask"),
                    "permissions the policy leaves standing"),
                  S("n2", "bitwise.and", ("@n1", "requested_mask"),
                    "permissions the request can actually use")),
                 "n2", intent="usable_permissions"),
            Plan("perm.v4", permission_roles[:3],
                 (S("n1", "bitwise.and", ("role_mask", "policy_mask"),
                    "effective permissions, read by both branches"),
                  S("n2", "bitwise.xor", ("@n1", "requested_mask"),
                    "bits where grant and request disagree"),
                  S("n3", "bitwise.or", ("@n1", "requested_mask"),
                    "bits either side names"),
                  S("n4", "arithmetic.subtract", ("@n3", "@n2"),
                    "bits both sides agree on")),
                 "n4", intent="agreed_permissions"),
            Plan("perm.v6", permission_roles,
                 (S("n1", "bitwise.and", ("role_mask", "policy_mask"),
                    "effective permissions, read again three steps later"),
                  S("n2", "bitwise.xor", ("@n1", "requested_mask"),
                    "bits the request would have to be told about"),
                  S("n3", "comparison.at_least", ("@n2", "gap_limit")),
                  S("n4", "bitwise.or", ("@n1", "escalation_mask"),
                    "permissions once the escalation is added"),
                  S("n5", "comparison.at_least", ("@n4", "overlap_limit")),
                  S("n6", "boolean.and", ("@n3", "@n5"),
                    "the request needs a human reviewer")),
                 "n6", intent="permission_review_verdict"),
        )))

    # ── feature flags ───────────────────────────────────────────────────
    flag_roles = (
        R("enabled_mask", "count_items", "feature bits switched on for the tenant"),
        R("default_mask", "count_items", "feature bits the product ships with"),
        R("override_mask", "count_items", "feature bits the support override sets"),
        R("region_mask", "count_items", "feature bits the region permits"),
        R("tier_shift", "count_small",
          "how many tiers up the flag block is moved in the register"),
        R("low_cut", "cut_low", "drift below which a tenant counts as stock"),
        R("high_cut", "cut_high", "drift above which a tenant counts as bespoke"),
    )
    out.append(Blueprint(
        workflow_id="bitwise.feature_flags",
        domain="bitwise_realistic",
        natural_user_goal=("see how far a tenant's feature flags have drifted from "
                           "what the product ships with"),
        target_description="the drift band, the flag label or the size of the drift",
        value_generator_id="bitwise.feature_flags",
        query_asset_family="feature_register",
        hard_distractor_families=("bitwise", "format"),
        entity_family="data",
        plans=(
            Plan("flags.v3", (flag_roles[0], flag_roles[1], flag_roles[4],
                              flag_roles[5], flag_roles[6]),
                 (S("n1", "bitwise.xor", ("enabled_mask", "default_mask"),
                    "flags that differ from the shipped defaults"),
                  S("n2", "bitwise.shift", ("@n1", "tier_shift"),
                    "the same flags in the tier block of the register"),
                  S("n3", "classification.three_bands",
                    ("@n2", "low_cut", "high_cut"))),
                 "n3", intent="drift_band"),
            Plan("flags.v5", (flag_roles[0], flag_roles[1], flag_roles[2]),
                 (S("n1", "bitwise.or", ("enabled_mask", "override_mask"),
                    "flags after the support override, read by both branches"),
                  S("n2", "bitwise.and", ("@n1", "default_mask"),
                    "flags that were on by default anyway"),
                  S("n3", "format.number_text", ("@n2",)),
                  S("n4", "bitwise.xor", ("@n1", "default_mask"),
                    "flags the override genuinely changed"),
                  S("n5", "format.tag", ("@n3", "@n4"),
                    "default flags tagged with what the override changed")),
                 "n5", intent="override_label"),
            Plan("flags.v8", flag_roles[:5],
                 (S("n1", "bitwise.or", ("enabled_mask", "override_mask"),
                    "flags after the override, read again four steps later"),
                  S("n2", "bitwise.and", ("@n1", "default_mask")),
                  S("n3", "bitwise.shift", ("@n2", "tier_shift"),
                    "those flags moved into the tier block"),
                  S("n4", "bitwise.and", ("region_mask", "default_mask"),
                    "defaults the region permits"),
                  S("n5", "bitwise.xor", ("@n1", "region_mask"),
                    "flags the region and the tenant disagree on"),
                  S("n6", "bitwise.or", ("@n3", "@n4"),
                    "everything the register should hold"),
                  S("n7", "arithmetic.abs_difference", ("@n5", "@n6")),
                  S("n8", "arithmetic.digit_sum", ("@n7",),
                    "checksum the register audit prints")),
                 "n8", intent="register_checksum"),
        )))

    # ── capacity register ───────────────────────────────────────────────
    capacity_roles = (
        R("base_slots", "count_items", "slots one block of the register holds"),
        R("doubling_steps", "count_small", "how many times the pool has been doubled"),
        R("reserved_mask", "count_items", "slot bits reserved for maintenance"),
        R("active_mask", "count_items", "slot bits currently in use"),
        R("region_mask", "count_items", "slot bits the region may address"),
        R("slot_limit", "threshold_count", "occupied slots that still count as healthy"),
        R("spare_limit", "threshold_count", "spare slots that still count as healthy"),
    )
    out.append(Blueprint(
        workflow_id="bitwise.capacity_register",
        domain="bitwise_realistic",
        natural_user_goal=("work out how much of a doubled slot register is still "
                           "free once reservations are taken into account"),
        target_description="the free capacity, its checks or the register summary",
        value_generator_id="bitwise.capacity_register",
        query_asset_family="capacity_register",
        hard_distractor_families=("bitwise", "statistics"),
        boolean_balancing_strategy="threshold_band",
        entity_family="engineering",
        plans=(
            Plan("cap.v4", capacity_roles[:3],
                 (S("n1", "bitwise.shift", ("base_slots", "doubling_steps"),
                    "slots after the doublings, read by both branches"),
                  S("n2", "bitwise.and", ("@n1", "reserved_mask"),
                    "slots the maintenance reservation holds"),
                  S("n3", "arithmetic.subtract", ("@n1", "@n2"),
                    "slots left for work"),
                  S("n4", "arithmetic.divide", ("@n3", "@n1"),
                    "share of the register still free")),
                 "n4", intent="free_share_of_register"),
            Plan("cap.v7", (capacity_roles[0], capacity_roles[1], capacity_roles[2],
                            capacity_roles[3], capacity_roles[5], capacity_roles[6]),
                 (S("n1", "bitwise.shift", ("base_slots", "doubling_steps"),
                    "slots after the doublings, read again three steps later"),
                  S("n2", "bitwise.and", ("@n1", "reserved_mask")),
                  S("n3", "bitwise.or", ("@n2", "active_mask"),
                    "slots that are not free"),
                  S("n4", "bitwise.xor", ("@n1", "active_mask"),
                    "slots whose state differs from the pool"),
                  S("n5", "comparison.at_least", ("@n3", "slot_limit")),
                  S("n6", "comparison.at_least", ("@n4", "spare_limit")),
                  S("n7", "boolean.xor", ("@n5", "@n6"),
                    "exactly one of the two capacity checks trips")),
                 "n7", intent="capacity_check_disagreement"),
            Plan("cap.v9", capacity_roles[:5],
                 (S("n1", "bitwise.shift", ("base_slots", "doubling_steps"),
                    "slots after the doublings, read again five steps later"),
                  S("n2", "bitwise.and", ("@n1", "reserved_mask"),
                    "reserved slots, read by both branches"),
                  S("n3", "bitwise.or", ("@n2", "active_mask"),
                    "reserved or in use"),
                  S("n4", "bitwise.xor", ("@n2", "region_mask"),
                    "reserved slots the region does not address"),
                  S("n5", "list.build", ("@n3", "@n4", "active_mask"),
                    "the three register readings"),
                  S("n6", "bitwise.and", ("@n1", "region_mask"),
                    "slots the region may actually address"),
                  S("n7", "list.reduce_max", ("@n5",)),
                  S("n8", "statistics.mean", ("@n5",)),
                  S("n9", "arithmetic.sum_three", ("@n6", "@n7", "@n8"),
                    "the figure the register summary reports")),
                 "n9", intent="register_summary"),
        )))

    return out
