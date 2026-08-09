"""Bitwise workflows grounded in things people actually store as bit masks.

Permission masks, feature-flag words and sensor status words are the three
places where a normal user really does ask "which bits survive": what a role
still has once the policy mask is applied, which flags changed between two
releases, which fault bits a device is reporting. The masks are ordinary small
integers, and the packing plans shift a channel id up before folding the flags
in, which is exactly how such a word is assembled in practice.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── permission masks ────────────────────────────────────────────────
    current_mask = R("current_mask", "count_items",
                     "the permission mask the role holds today")
    extra_mask = R("requested_mask", "count_people",
                   "the permissions the change request adds")
    policy_mask = R("policy_mask", "count_items",
                    "the permissions the policy allows at all")
    channel_id = R("channel_id", "count_small",
                   "the id of the channel the word belongs to")
    shift_places = R("channel_shift", "count_small",
                     "how many bits the channel id is stored above the flags")
    status_flags = R("status_flags", "count_items",
                     "the flag bits that are set on the account")
    baseline = R("baseline_mask", "threshold_count",
                 "the access level the role has to keep")
    drop_limit = R("drop_limit", "threshold_count",
                   "how much of the requested access may be refused")
    out.append(Blueprint(
        workflow_id="bitwise_realistic.permission_mask_review",
        domain="bitwise_realistic",
        natural_user_goal=("work out what access a role really ends up with "
                           "once the policy is applied to the request"),
        target_description="the effective access or the verdict on the request",
        value_generator_id="bitwise_realistic.permission_masks",
        query_asset_family="access_request",
        hard_distractor_families=("bitwise", "comparison"),
        boolean_balancing_strategy="calibrate_access_threshold",
        entity_family="it_operations",
        plans=(
            Plan("perm.v2", (current_mask, extra_mask, baseline),
                 (S("n1", "bitwise.or", ("current_mask", "requested_mask"),
                    "everything the role would hold after the change"),
                  S("n2", "comparison.at_least", ("@n1", "baseline_mask"))),
                 "n2", intent="requested_access_verdict"),
            Plan("perm.v4", (channel_id, shift_places, status_flags,
                             policy_mask, baseline),
                 (S("n1", "bitwise.shift", ("channel_id", "channel_shift"),
                    "the channel id moved into its own bits"),
                  S("n2", "bitwise.or", ("@n1", "status_flags"),
                    "the packed account word"),
                  S("n3", "bitwise.and", ("@n2", "policy_mask"),
                    "what the policy leaves of it"),
                  S("n4", "comparison.at_least", ("@n3", "baseline_mask"))),
                 "n4", intent="packed_word_access_verdict"),
            Plan("perm.v5", (current_mask, extra_mask, policy_mask, channel_id,
                             shift_places, status_flags),
                 (S("n1", "bitwise.or", ("current_mask", "requested_mask"),
                    "everything the role would hold after the change"),
                  S("n2", "bitwise.shift", ("channel_id", "channel_shift"),
                    "the channel id moved into its own bits"),
                  S("n3", "bitwise.and", ("@n1", "policy_mask"),
                    "the access the policy actually allows"),
                  S("n4", "bitwise.or", ("@n2", "status_flags"),
                    "the account word as the directory stores it"),
                  S("n5", "bitwise.xor", ("@n3", "@n4"),
                    "the bits where the two disagree")),
                 "n5", intent="access_word_disagreement"),
            Plan("perm.v6", (channel_id, shift_places, status_flags,
                             policy_mask, drop_limit),
                 (S("n1", "bitwise.shift", ("channel_id", "channel_shift")),
                  S("n2", "bitwise.or", ("@n1", "status_flags"),
                    "the packed word, compared with itself after filtering"),
                  S("n3", "bitwise.and", ("@n2", "policy_mask")),
                  S("n4", "bitwise.xor", ("@n2", "@n3"),
                    "the bits the policy took away"),
                  S("n5", "comparison.at_least", ("@n4", "drop_limit")),
                  S("n6", "boolean.not", ("@n5",),
                    "the request is accepted while little enough is refused")),
                 "n6", intent="refused_bits_verdict"),
            Plan("perm.v7", (channel_id, shift_places, status_flags,
                             policy_mask, baseline, drop_limit),
                 (S("n1", "bitwise.shift", ("channel_id", "channel_shift")),
                  S("n2", "bitwise.or", ("@n1", "status_flags"),
                    "the packed account word, read on both branches"),
                  S("n3", "bitwise.and", ("@n2", "policy_mask"),
                    "what the policy leaves of it"),
                  S("n4", "comparison.at_least", ("@n3", "baseline_mask"),
                    "the role keeps the access it needs"),
                  S("n5", "bitwise.xor", ("@n2", "@n3"),
                    "the bits the policy took away"),
                  S("n6", "comparison.at_least", ("@n5", "drop_limit"),
                    "a lot was refused"),
                  S("n7", "boolean.or", ("@n4", "@n6"),
                    "either outcome is worth writing on the ticket")),
                 "n7", intent="access_ticket_note_verdict"),
            Plan("perm.v8", (channel_id, shift_places, status_flags,
                             policy_mask),
                 (S("n1", "bitwise.shift", ("channel_id", "channel_shift")),
                  S("n2", "bitwise.or", ("@n1", "status_flags")),
                  S("n3", "bitwise.and", ("@n2", "policy_mask")),
                  S("n4", "bitwise.xor", ("@n2", "@n3")),
                  S("n5", "list.build", ("@n2", "@n3", "@n4"),
                    "requested, granted and refused side by side"),
                  S("n6", "list.reduce_sum", ("@n5",)),
                  S("n7", "list.reduce_max", ("@n5",)),
                  S("n8", "rates.share_percent", ("@n7", "@n6"),
                    "how much of the picture the largest of the three is")),
                 "n8", intent="access_word_breakdown"),
        )))

    # ── feature flags ───────────────────────────────────────────────────
    enabled_flags = R("enabled_flags", "count_items",
                      "the feature flags switched on in this release")
    previous_flags = R("previous_flags", "count_items",
                       "the feature flags that were on in the previous release")
    cohort_mask = R("cohort_mask", "count_people",
                    "the flags that matter for this user cohort")
    forced_flags = R("forced_flags", "count_people",
                     "the flags the rollout switches on regardless")
    alert_floor = R("alert_floor", "threshold_count",
                    "how much change is worth alerting the team about")
    flag_slot = R("flag_slot", "index_position",
                  "which of the three words the release note quotes")
    quote_floor = R("quote_floor", "threshold_count",
                    "the value that word has to reach")
    churn_low = R("churn_low", "cut_low",
                  "the churn below which the release is called quiet")
    churn_high = R("churn_high", "cut_high",
                   "the churn above which the release is called risky")
    out.append(Blueprint(
        workflow_id="bitwise_realistic.feature_flag_rollout",
        domain="bitwise_realistic",
        natural_user_goal=("see which feature flags actually changed for a "
                           "cohort between two releases"),
        target_description="the change in the flags or how risky it looks",
        value_generator_id="bitwise_realistic.feature_flags",
        query_asset_family="release_note",
        hard_distractor_families=("bitwise", "list"),
        boolean_balancing_strategy="calibrate_flag_threshold",
        entity_family="it_operations",
        plans=(
            Plan("flag.v4", (enabled_flags, previous_flags, forced_flags,
                             cohort_mask, alert_floor),
                 (S("n1", "bitwise.xor", ("enabled_flags", "previous_flags"),
                    "the flags that changed between the releases"),
                  S("n2", "bitwise.xor", ("@n1", "forced_flags"),
                    "the same comparison once the forced flags are toggled in"),
                  S("n3", "bitwise.and", ("@n2", "cohort_mask"),
                    "the ones this cohort can see"),
                  S("n4", "comparison.at_least", ("@n3", "alert_floor"))),
                 "n4", intent="cohort_change_alert"),
            Plan("flag.v5", (enabled_flags, previous_flags, cohort_mask,
                             forced_flags),
                 (S("n1", "bitwise.xor", ("enabled_flags", "previous_flags")),
                  S("n2", "bitwise.and", ("@n1", "cohort_mask")),
                  S("n3", "bitwise.or", ("@n2", "forced_flags"),
                    "with the flags the rollout forces on"),
                  S("n4", "list.build", ("@n1", "@n2", "@n3"),
                    "the three words the release note lists"),
                  S("n5", "list.reduce_distinct", ("@n4",),
                    "how many genuinely different words that is")),
                 "n5", intent="distinct_release_words"),
            Plan("flag.v7", (enabled_flags, previous_flags, cohort_mask,
                             forced_flags, flag_slot, quote_floor),
                 (S("n1", "bitwise.xor", ("enabled_flags", "previous_flags")),
                  S("n2", "bitwise.and", ("@n1", "cohort_mask")),
                  S("n3", "bitwise.or", ("@n2", "forced_flags")),
                  S("n4", "list.build", ("@n1", "@n2", "@n3")),
                  S("n5", "list.map_sort_asc", ("@n4",)),
                  S("n6", "list.index", ("@n5", "flag_slot")),
                  S("n7", "comparison.at_least", ("@n6", "quote_floor"))),
                 "n7", intent="quoted_word_verdict"),
            Plan("flag.v8", (enabled_flags, previous_flags, cohort_mask,
                             forced_flags, churn_low, churn_high),
                 (S("n1", "bitwise.xor", ("enabled_flags", "previous_flags")),
                  S("n2", "bitwise.and", ("@n1", "cohort_mask")),
                  S("n3", "bitwise.or", ("@n2", "forced_flags")),
                  S("n4", "list.build", ("@n1", "@n2", "@n3"),
                    "the three words, summarised two ways"),
                  S("n5", "list.reduce_max", ("@n4",)),
                  S("n6", "list.reduce_sum", ("@n4",)),
                  S("n7", "rates.share_percent", ("@n5", "@n6")),
                  S("n8", "classification.three_bands", ("@n7", "churn_low",
                                                         "churn_high"))),
                 "n8", intent="release_churn_band"),
        )))

    # ── sensor status words ─────────────────────────────────────────────
    status_word = R("status_word", "count_items",
                    "the status word the sensor last reported")
    fault_mask = R("fault_mask", "count_people",
                   "the bits that mean a fault on this sensor")
    shift_places = R("report_shift", "count_small",
                     "how far the fault bits move when the report is packed")
    channel = R("sensor_channel", "count_small",
                "the channel the sensor reports on")
    alarm_floor = R("alarm_floor", "threshold_count",
                    "the fault level that raises an alarm")
    group_size = R("group_size", "threshold_count",
                   "how many sensors share one report slot")
    word_low = R("word_low", "range_low",
                 "the smallest report word the gateway accepts")
    word_high = R("word_high", "range_high",
                  "the largest report word the gateway accepts")
    read_slot = R("read_slot", "index_position",
                  "which of the three words the technician reads")
    out.append(Blueprint(
        workflow_id="bitwise_realistic.sensor_status_word",
        domain="bitwise_realistic",
        natural_user_goal=("read a sensor's status word and work out what the "
                           "gateway will do with it"),
        target_description="the fault picture or the gateway verdict",
        value_generator_id="bitwise_realistic.sensor_status",
        query_asset_family="device_report",
        hard_distractor_families=("bitwise", "validation"),
        boolean_balancing_strategy="calibrate_alarm_threshold",
        entity_family="field_service",
        plans=(
            Plan("sensor.v4", (status_word, fault_mask, shift_places, channel,
                               group_size),
                 (S("n1", "bitwise.and", ("status_word", "fault_mask")),
                  S("n2", "bitwise.shift", ("@n1", "report_shift"),
                    "the fault bits moved into the report field"),
                  S("n3", "bitwise.or", ("@n2", "sensor_channel"),
                    "the packed report word"),
                  S("n4", "boolean.divisible", ("@n3", "group_size"),
                    "whether that word lands on a slot boundary")),
                 "n4", intent="report_slot_verdict"),
            Plan("sensor.v6", (status_word, fault_mask, shift_places, word_low,
                               word_high),
                 (S("n1", "bitwise.and", ("status_word", "fault_mask"),
                    "the fault bits, used on both branches"),
                  S("n2", "bitwise.xor", ("status_word", "@n1"),
                    "everything the sensor reports that is not a fault"),
                  S("n3", "bitwise.shift", ("@n1", "report_shift")),
                  S("n4", "bitwise.or", ("@n3", "@n2"),
                    "the repacked report word"),
                  S("n5", "validation.in_range", ("@n4", "word_low",
                                                  "word_high")),
                  S("n6", "boolean.not", ("@n5",),
                    "the gateway rejects the word when it falls outside")),
                 "n6", intent="gateway_rejection_verdict"),
            Plan("sensor.v7", (status_word, fault_mask, shift_places,
                               alarm_floor),
                 (S("n1", "bitwise.and", ("status_word", "fault_mask"),
                    "the fault bits, folded back in at the end"),
                  S("n2", "bitwise.shift", ("@n1", "report_shift")),
                  S("n3", "bitwise.xor", ("status_word", "@n1"),
                    "everything the sensor reports that is not a fault"),
                  S("n4", "bitwise.or", ("@n2", "@n3"),
                    "the repacked report word"),
                  S("n5", "bitwise.shift", ("@n4", "report_shift"),
                    "that word moved into the gateway frame"),
                  S("n6", "bitwise.or", ("@n5", "@n1"),
                    "the frame with the raw fault bits kept alongside"),
                  S("n7", "comparison.at_least", ("@n6", "alarm_floor"))),
                 "n7", intent="gateway_frame_alarm"),
            Plan("sensor.v9", (status_word, fault_mask, shift_places,
                               read_slot),
                 (S("n1", "bitwise.and", ("status_word", "fault_mask")),
                  S("n2", "bitwise.xor", ("status_word", "@n1")),
                  S("n3", "bitwise.shift", ("@n1", "report_shift")),
                  S("n4", "bitwise.or", ("@n3", "@n2")),
                  S("n5", "list.build", ("@n1", "@n2", "@n4"),
                    "fault bits, clean bits and the packed word"),
                  S("n6", "list.map_running_max", ("@n5",)),
                  S("n7", "list.index", ("@n6", "read_slot")),
                  S("n8", "list.reduce_sum", ("@n5",)),
                  S("n9", "rates.share_percent", ("@n7", "@n8"),
                    "how much of the report that reading accounts for")),
                 "n9", intent="report_word_share"),
        )))

    return out
