"""String-parsing workflows: reading values back out of delimited lines and codes.

Every family here follows the same honest route -- split a delimited line, take
one field, keep only its digits, turn those digits into a number, compute with
that number and render the result -- and they differ in where they stop and how
the branches rejoin. ``parsing.delimited_field_value`` prices the parsed field,
``parsing.asset_code_value`` parses a stencilled code, ``parsing.record_shape_check``
never leaves the shape of the line itself, and ``parsing.field_value_merge``
reads two fields and brings them back together. The delimiter the line is split
on is always the delimiter it was written with, which is why the field index is
safe to state in the query.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── delimited field value ───────────────────────────────────────────
    part_codes = R("part_codes", "text_list_codes",
                   "the part codes the picking line is written from")
    field_sep = R("field_sep", "separator",
                  "the character the picking line puts between fields")
    field_index = R("field_index", "index_position",
                    "which field of the line the clerk reads")
    unit_price = R("unit_price", "money_price",
                   "price charged for one unit of the part")
    currency = R("currency", "currency_code",
                 "currency the price list is kept in")

    out.append(Blueprint(
        workflow_id="parsing.delimited_field_value",
        domain="string_parsing",
        natural_user_goal=("read one field out of a delimited picking line and "
                           "work out what it is worth"),
        target_description="the parsed field, its value or the priced line",
        value_generator_id="parsing.picking_line",
        query_asset_family="delimited_picking_line",
        hard_distractor_families=("string", "list"),
        entity_family="logistics",
        plans=(
            Plan("field.v4", (part_codes, field_sep, field_index),
                 (S("n1", "list.combine_join_text", ("part_codes", "field_sep"),
                    "the picking line as the scanner writes it"),
                  S("n2", "string.split_take", ("@n1", "field_sep", "field_index")),
                  S("n3", "string.extract_digits", ("@n2",)),
                  S("n4", "string.parse_number", ("@n3",))),
                 "n4", intent="field_as_number"),
            Plan("field.v6", (part_codes, field_sep, field_index, unit_price,
                              currency),
                 (S("n1", "list.combine_join_text", ("part_codes", "field_sep")),
                  S("n2", "string.split_take", ("@n1", "field_sep", "field_index")),
                  S("n3", "string.extract_digits", ("@n2",)),
                  S("n4", "string.parse_number", ("@n3",)),
                  S("n5", "arithmetic.multiply", ("@n4", "unit_price")),
                  S("n6", "format.currency", ("@n5", "currency"))),
                 "n6", intent="priced_field"),
            Plan("field.v9", (part_codes, field_sep, field_index, unit_price,
                              currency),
                 (S("n1", "list.combine_join_text", ("part_codes", "field_sep"),
                    "the picking line, read again four calls later"),
                  S("n2", "string.split_take", ("@n1", "field_sep", "field_index"),
                    "the field itself, needed again at the very end"),
                  S("n3", "string.extract_digits", ("@n2",)),
                  S("n4", "string.parse_number", ("@n3",)),
                  S("n5", "string.split_count", ("@n1", "field_sep"),
                    "how many fields the line carries"),
                  S("n6", "arithmetic.multiply", ("@n4", "unit_price")),
                  S("n7", "arithmetic.divide", ("@n6", "@n5"),
                    "value carried per field of the line"),
                  S("n8", "format.currency", ("@n7", "currency")),
                  S("n9", "string.concat", ("@n2", "@n8"),
                    "the field and the money it stands for")),
                 "n9", intent="field_value_per_field"),
        )))

    # ── asset code value ────────────────────────────────────────────────
    asset_code = R("asset_code", "identifier_code",
                   "the asset code stencilled on the unit")
    code_price = R("unit_price", "money_price",
                   "price the register holds for one unit")
    code_currency = R("currency", "currency_code",
                      "currency the register is kept in")
    code_width = R("column_width", "count_items",
                   "the width the register column is printed at")
    code_fill = R("fill_char", "separator",
                  "the character the register pads a short entry with")
    low_cut = R("low_cut", "cut_low", "value below which an asset counts as minor")
    high_cut = R("high_cut", "cut_high", "value above which an asset counts as major")

    out.append(Blueprint(
        workflow_id="parsing.asset_code_value",
        domain="string_parsing",
        natural_user_goal=("get the serial number out of a stencilled asset code "
                           "and work out what the asset is booked at"),
        target_description="the booked value, its band or the register entry",
        value_generator_id="parsing.asset_register",
        query_asset_family="asset_code_entry",
        hard_distractor_families=("string", "arithmetic"),
        entity_family="facilities",
        plans=(
            Plan("code.v3", (asset_code, code_price),
                 (S("n1", "string.extract_digits", ("asset_code",),
                    "the serial hidden in the stencilled code"),
                  S("n2", "string.parse_number", ("@n1",)),
                  S("n3", "arithmetic.multiply", ("@n2", "unit_price"))),
                 "n3", intent="booked_value"),
            Plan("code.v5", (asset_code, code_price, low_cut, high_cut),
                 (S("n1", "string.normalize_lower", ("asset_code",)),
                  S("n2", "string.extract_digits", ("@n1",)),
                  S("n3", "string.parse_number", ("@n2",)),
                  S("n4", "arithmetic.multiply", ("@n3", "unit_price")),
                  S("n5", "classification.three_bands",
                    ("@n4", "low_cut", "high_cut"))),
                 "n5", intent="asset_value_band"),
            Plan("code.v7", (asset_code, code_currency, code_width, code_fill),
                 (S("n1", "string.normalize_upper", ("asset_code",),
                    "the register form, read by two branches"),
                  S("n2", "string.count_length", ("@n1",),
                    "entry width, stamped on four calls later"),
                  S("n3", "string.extract_digits", ("@n1",)),
                  S("n4", "string.parse_number", ("@n3",)),
                  S("n5", "format.currency", ("@n4", "currency")),
                  S("n6", "format.tag", ("@n5", "@n2")),
                  S("n7", "format.pad", ("@n6", "column_width", "fill_char"))),
                 "n7", intent="register_entry_line"),
        )))

    # ── record shape check ──────────────────────────────────────────────
    record_line = R("record_line", "text_record",
                    "the delimited line exactly as the feed delivered it")
    shape_sep = R("field_sep", "separator",
                  "the character the feed claims to delimit fields with")
    keyword = R("keyword", "needle_text", "the marker the reviewer looks for")
    hit_floor = R("hit_floor", "threshold_count",
                  "how often the marker has to appear")
    char_floor = R("char_floor", "threshold_count",
                   "how many characters a complete line has to reach")

    out.append(Blueprint(
        workflow_id="parsing.record_shape_check",
        domain="string_parsing",
        natural_user_goal=("check that a line arriving in the feed has the shape "
                           "the importer expects"),
        target_description="the field count, the field list or the import verdict",
        value_generator_id="parsing.feed_line",
        query_asset_family="feed_record_line",
        hard_distractor_families=("string", "comparison"),
        boolean_balancing_strategy="threshold_band",
        entity_family="support",
        plans=(
            Plan("shape.v2", (record_line, shape_sep),
                 (S("n1", "string.normalize_lower", ("record_line",)),
                  S("n2", "string.split_count", ("@n1", "field_sep"),
                    "how many fields the line really splits into")),
                 "n2", intent="field_count"),
            Plan("shape.v4", (record_line, shape_sep),
                 (S("n1", "string.normalize_lower", ("record_line",)),
                  S("n2", "string.normalize_whitespace", ("@n1",)),
                  S("n3", "string.split", ("@n2", "field_sep")),
                  S("n4", "list.map_sort_text", ("@n3",))),
                 "n4", intent="sorted_fields"),
            Plan("shape.v6", (record_line, keyword, hit_floor, char_floor),
                 (S("n1", "string.normalize_lower", ("record_line",),
                    "the comparable form, read by two measurements"),
                  S("n2", "string.count_substring", ("@n1", "keyword")),
                  S("n3", "string.count_length", ("@n1",)),
                  S("n4", "comparison.at_least", ("@n2", "hit_floor")),
                  S("n5", "comparison.at_least", ("@n3", "char_floor")),
                  S("n6", "boolean.and", ("@n4", "@n5"))),
                 "n6", intent="line_importable"),
        )))

    # ── two fields brought back together ────────────────────────────────
    merge_codes = R("part_codes", "text_list_codes",
                    "the codes the transfer line is written from")
    merge_sep = R("field_sep", "separator",
                  "the character the transfer line uses between fields")
    first_index = R("first_index", "index_position",
                    "which field carries the quantity")
    second_index = R("second_index", "index_position",
                     "which field carries the destination code")
    value_floor = R("value_floor", "threshold_value",
                    "the parsed quantity a valid transfer has to reach")
    field_floor = R("field_floor", "threshold_count",
                    "how many fields a complete transfer line has")
    code_start = R("code_start", "prefix_text",
                   "how a correctly issued code has to begin")
    merge_width = R("column_width", "count_items",
                    "the width the transfer sheet prints merged fields at")
    merge_fill = R("fill_char", "separator",
                   "the character the transfer sheet pads a short entry with")

    out.append(Blueprint(
        workflow_id="parsing.field_value_merge",
        domain="string_parsing",
        natural_user_goal=("read two fields out of a transfer line and decide "
                           "what they say together"),
        target_description="the merged fields, their ratio or the transfer verdict",
        value_generator_id="parsing.transfer_line",
        query_asset_family="transfer_record_line",
        hard_distractor_families=("string", "decision"),
        boolean_balancing_strategy="threshold_band",
        entity_family="dispatch",
        plans=(
            Plan("merge.v5", (merge_codes, merge_sep, first_index, second_index,
                              merge_width, merge_fill),
                 (S("n1", "list.combine_join_text", ("part_codes", "field_sep"),
                    "the transfer line, read by both field branches"),
                  S("n2", "string.split_take", ("@n1", "field_sep", "first_index")),
                  S("n3", "string.split_take", ("@n1", "field_sep", "second_index")),
                  S("n4", "string.concat", ("@n2", "@n3")),
                  S("n5", "format.pad", ("@n4", "column_width", "fill_char"))),
                 "n5", intent="merged_fields"),
            Plan("merge.v7", (merge_codes, merge_sep, first_index, second_index),
                 (S("n1", "list.combine_join_text", ("part_codes", "field_sep")),
                  S("n2", "string.split_take", ("@n1", "field_sep", "first_index")),
                  S("n3", "string.split_take", ("@n1", "field_sep", "second_index")),
                  S("n4", "string.extract_digits", ("@n2",)),
                  S("n5", "string.parse_number", ("@n4",)),
                  S("n6", "string.count_length", ("@n3",)),
                  S("n7", "rates.ratio_of", ("@n5", "@n6"),
                    "quantity carried per character of the destination code")),
                 "n7", intent="quantity_per_code_character"),
            Plan("merge.v9", (merge_codes, merge_sep, first_index, value_floor,
                              field_floor, code_start),
                 (S("n1", "list.combine_join_text", ("part_codes", "field_sep"),
                    "the transfer line, read again four calls later"),
                  S("n2", "string.split_take", ("@n1", "field_sep", "first_index"),
                    "the quantity field, checked again six calls later"),
                  S("n3", "string.extract_digits", ("@n2",)),
                  S("n4", "string.parse_number", ("@n3",)),
                  S("n5", "string.split_count", ("@n1", "field_sep")),
                  S("n6", "comparison.at_least", ("@n4", "value_floor")),
                  S("n7", "comparison.at_least", ("@n5", "field_floor")),
                  S("n8", "string.validate_prefix", ("@n2", "code_start")),
                  S("n9", "decision.majority", ("@n6", "@n7", "@n8"))),
                 "n9", intent="transfer_line_acceptable"),
        )))

    return out
