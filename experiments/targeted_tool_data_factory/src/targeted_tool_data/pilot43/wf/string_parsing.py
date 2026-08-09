"""String-parsing workflows: delimited records, asset codes, register rewrites.

The parsing here is real: a delimited line is split, reordered and rejoined, the
digits of an asset code become a number, and a register is masked by replacing a
field that was located with the delimiters themselves. Because the delimiter is
whatever the exporting system used, plans that need a field always build the
delimited text from the list they were given first.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── delimited job record ────────────────────────────────────────────
    record = R("record", "text_record", "the job line as the exporter wrote it")
    sep = R("separator", "separator", "the delimiter that exporter uses")
    field_target = R("field_target", "threshold_count",
                     "how many fields a complete line has to carry")

    out.append(Blueprint(
        workflow_id="string_parsing.delimited_record",
        domain="string_parsing",
        natural_user_goal=("make sense of a delimited job line that came out of "
                           "the old export"),
        target_description="the parsed line, its fields or the number it carries",
        value_generator_id="string_parsing.job_line",
        query_asset_family="export_line",
        hard_distractor_families=("string", "list"),
        boolean_balancing_strategy="calibrate_field_count_threshold",
        entity_family="dispatch",
        plans=(
            Plan("rec.v2", (record, sep, field_target),
                 (S("n1", "string.split_count", ("record", "separator")),
                  S("n2", "comparison.at_least", ("@n1", "field_target"))),
                 "n2", intent="line_complete"),
            Plan("rec.v3", (record, sep),
                 (S("n1", "string.normalize_lower", ("record",)),
                  S("n2", "string.split", ("@n1", "separator")),
                  S("n3", "list.map_sort_text", ("@n2",))),
                 "n3", intent="ordered_fields"),
            Plan("rec.v6", (record, sep),
                 (S("n1", "string.normalize_lower", ("record",)),
                  S("n2", "string.split", ("@n1", "separator")),
                  S("n3", "list.map_sort_text", ("@n2",)),
                  S("n4", "list.combine_join_text", ("@n3", "separator")),
                  S("n5", "string.extract_digits", ("@n4",),
                    "the only numeric field in the line"),
                  S("n6", "string.parse_number", ("@n5",))),
                 "n6", intent="numeric_field"),
        )))

    # ── code register ───────────────────────────────────────────────────
    codes = R("codes", "text_list_codes", "the codes currently on the register")
    reg_sep = R("separator", "separator", "the delimiter the register is printed with")
    slot = R("slot", "index_position", "which entry of the register we start from")
    width = R("width", "count_small",
              "how many leading characters make up the family prefix")
    matches = R("match_target", "threshold_count",
                "how many related codes we expect to find")
    places = R("places", "places", "how many decimals the summary shows")

    out.append(Blueprint(
        workflow_id="string_parsing.code_registry",
        domain="string_parsing",
        natural_user_goal=("find out which asset codes on the register belong to "
                           "the same family as the one I picked"),
        target_description="the related codes or how much of the register they cover",
        value_generator_id="string_parsing.code_register",
        query_asset_family="asset_register",
        hard_distractor_families=("list", "string"),
        boolean_balancing_strategy="calibrate_match_count_threshold",
        entity_family="facilities",
        plans=(
            Plan("code.v3", (codes, slot, width),
                 (S("n1", "list.index_text", ("codes", "slot")),
                  S("n2", "string.truncate", ("@n1", "width"),
                    "the family prefix of the picked code"),
                  S("n3", "list.filter_prefix", ("codes", "@n2"))),
                 "n3", intent="related_codes"),
            Plan("code.v5", (codes, slot, width, matches),
                 (S("n1", "list.index_text", ("codes", "slot")),
                  S("n2", "string.truncate", ("@n1", "width")),
                  S("n3", "list.filter_prefix", ("codes", "@n2")),
                  S("n4", "list.reduce_count_text", ("@n3",)),
                  S("n5", "comparison.at_least", ("@n4", "match_target"))),
                 "n5", intent="family_large_enough"),
            Plan("code.v10", (codes, slot, width, reg_sep, places),
                 (S("n1", "list.map_sort_text", ("codes",),
                    "the ordered register, read by three later calls"),
                  S("n2", "list.index_text", ("@n1", "slot")),
                  S("n3", "string.truncate", ("@n2", "width")),
                  S("n4", "list.filter_prefix", ("@n1", "@n3")),
                  S("n5", "list.combine_join_text", ("@n4", "separator")),
                  S("n6", "list.combine_join_text", ("@n1", "separator")),
                  S("n7", "string.count_length", ("@n5",)),
                  S("n8", "string.count_length", ("@n6",)),
                  S("n9", "rates.share_percent", ("@n7", "@n8")),
                  S("n10", "format.percent", ("@n9", "places"))),
                 "n10", intent="family_share_of_register"),
        )))

    # ── identifier checks ───────────────────────────────────────────────
    code = R("code", "identifier_code", "the asset code printed on the plate")
    start = R("expected_start", "prefix_text", "the project the code should belong to")
    min_len = R("min_length", "threshold_count",
                "the shortest code the register accepts")
    divisor = R("batch_size", "threshold_count", "the batch size codes are issued in")
    needle = R("fragment", "needle_text", "the fragment a valid code has to carry")
    lo = R("serial_low", "range_low", "lowest serial number still in service")
    hi = R("serial_high", "range_high", "highest serial number still in service")

    out.append(Blueprint(
        workflow_id="string_parsing.identifier_check",
        domain="string_parsing",
        natural_user_goal=("check an asset code on a plate against the rules the "
                           "register applies"),
        target_description="the verdict on the code or how many rules it passes",
        value_generator_id="string_parsing.asset_code",
        query_asset_family="asset_plate",
        hard_distractor_families=("string", "validation"),
        boolean_balancing_strategy="calibrate_prefix_and_serial",
        entity_family="facilities",
        plans=(
            Plan("idc.v2", (code, start),
                 (S("n1", "string.normalize_lower", ("code",)),
                  S("n2", "string.validate_prefix", ("@n1", "expected_start"))),
                 "n2", intent="belongs_to_project"),
            Plan("idc.v3", (code, divisor),
                 (S("n1", "string.extract_digits", ("code",)),
                  S("n2", "string.parse_number", ("@n1",)),
                  S("n3", "boolean.divisible", ("@n2", "batch_size"))),
                 "n3", intent="whole_batch"),
            Plan("idc.v6", (code, min_len, needle),
                 (S("n1", "string.normalize_slug", ("code",),
                    "the register stores codes as slugs"),
                  S("n2", "string.validate_identifier", ("@n1",)),
                  S("n3", "string.count_length", ("@n1",)),
                  S("n4", "comparison.at_least", ("@n3", "min_length")),
                  S("n5", "string.validate_contains", ("@n1", "fragment")),
                  S("n6", "decision.count_true", ("@n2", "@n4", "@n5"))),
                 "n6", intent="rules_passed"),
            Plan("idc.v7", (code, lo, hi, min_len),
                 (S("n1", "string.validate_identifier", ("code",)),
                  S("n2", "string.extract_digits", ("code",)),
                  S("n3", "string.parse_number", ("@n2",)),
                  S("n4", "validation.in_range", ("@n3", "serial_low", "serial_high")),
                  S("n5", "string.count_length", ("code",)),
                  S("n6", "comparison.at_least", ("@n5", "min_length")),
                  S("n7", "decision.count_true", ("@n1", "@n4", "@n6"))),
                 "n7", intent="register_rules_passed"),
        )))

    # ── masking a register ──────────────────────────────────────────────
    rw_codes = R("codes", "text_list_codes", "the codes in the shared register")
    rw_sep = R("separator", "separator", "the delimiter the register is written with")
    tag = R("mask_tag", "prefix_text", "the tag that replaces a code when it is masked")

    out.append(Blueprint(
        workflow_id="string_parsing.register_masking",
        domain="string_parsing",
        natural_user_goal=("mask one entry of a shared code register before the "
                           "list leaves the building"),
        target_description="the masked register or how much of it is masked",
        value_generator_id="string_parsing.masked_register",
        query_asset_family="shared_register",
        hard_distractor_families=("string", "list"),
        entity_family="quality",
        plans=(
            Plan("msk.v3", (rw_codes, rw_sep, tag),
                 (S("n1", "list.combine_join_text", ("codes", "separator")),
                  S("n2", "string.extract_between",
                    ("@n1", "separator", "separator"),
                    "the entry sitting between the first two delimiters"),
                  S("n3", "string.replace", ("@n1", "@n2", "mask_tag"))),
                 "n3", intent="masked_register"),
            Plan("msk.v6", (rw_codes, rw_sep, tag),
                 (S("n1", "list.combine_join_text", ("codes", "separator")),
                  S("n2", "string.extract_between",
                    ("@n1", "separator", "separator")),
                  S("n3", "string.replace", ("@n1", "@n2", "mask_tag")),
                  S("n4", "string.split", ("@n3", "separator")),
                  S("n5", "list.filter_prefix", ("@n4", "mask_tag")),
                  S("n6", "list.reduce_count_text", ("@n5",))),
                 "n6", intent="masked_entry_count"),
            Plan("msk.v9", (rw_codes, rw_sep, tag),
                 (S("n1", "list.map_sort_text", ("codes",),
                    "the ordered register, counted again seven calls later"),
                  S("n2", "list.combine_join_text", ("@n1", "separator")),
                  S("n3", "string.extract_between",
                    ("@n2", "separator", "separator")),
                  S("n4", "string.replace", ("@n2", "@n3", "mask_tag")),
                  S("n5", "string.split", ("@n4", "separator")),
                  S("n6", "list.filter_prefix", ("@n5", "mask_tag")),
                  S("n7", "list.reduce_count_text", ("@n6",)),
                  S("n8", "list.reduce_count_text", ("@n1",)),
                  S("n9", "rates.share_percent", ("@n7", "@n8"))),
                 "n9", intent="masked_share"),
        )))

    return out
