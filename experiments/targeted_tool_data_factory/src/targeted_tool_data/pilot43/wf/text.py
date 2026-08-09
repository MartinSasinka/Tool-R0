"""Text-shaping workflows: headlines, label sets, character budgets, casing audits.

The four families differ in what the text *becomes*. A headline stays text and
only borrows a number to size its column; a label set is a text list that
becomes one delimited line and a text list again; a character budget turns text
into counts and rates before it turns into a verdict; a casing audit measures
the same note twice, raw and slugged, and merges the two measurements. Every
number in this module is produced by a ``string.*`` call, so the long plans are
genuine text -> number -> text transitions rather than arithmetic wearing a
text-sounding name.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── headline composition ────────────────────────────────────────────
    item_label = R("item_label", "text_label", "the item the headline announces")
    quantity = R("quantity", "quantity_units", "how many units the headline states")
    unit_name = R("unit_name", "unit_word", "the unit the quantity is counted in")
    head_note = R("raw_note", "text_note",
                  "the note the headline has to be built from")
    fill_char = R("fill_char", "separator",
                  "the character the printer pads a short line with")
    head_width = R("column_width", "count_items",
                   "the width of the column the headline is printed in")

    out.append(Blueprint(
        workflow_id="text.headline_composition",
        domain="text_processing",
        natural_user_goal=("compose the headline that goes on top of a printed "
                           "job sheet"),
        target_description="the composed headline line",
        value_generator_id="text.headline",
        query_asset_family="print_headline",
        hard_distractor_families=("string", "format"),
        entity_family="fabrication",
        plans=(
            Plan("head.v3", (item_label, quantity, unit_name),
                 (S("n1", "string.normalize_title", ("item_label",),
                    "the item written the way a heading is written"),
                  S("n2", "format.with_unit", ("quantity", "unit_name"),
                    "the quantity rendered with its unit"),
                  S("n3", "string.concat", ("@n1", "@n2"),
                    "headline built from two independent pieces")),
                 "n3", intent="headline_from_item_and_quantity"),
            Plan("head.v5", (item_label, head_width, fill_char),
                 (S("n1", "string.normalize_slug", ("item_label",),
                    "the machine-readable form, read by two branches"),
                  S("n2", "string.count_length", ("@n1",)),
                  S("n3", "string.normalize_upper", ("@n1",)),
                  S("n4", "format.tag", ("@n3", "@n2"),
                    "the slug stamped with the width it needs"),
                  S("n5", "format.pad", ("@n4", "column_width", "fill_char"))),
                 "n5", intent="slug_headline_for_column"),
            Plan("head.v7", (head_note, head_width, fill_char),
                 (S("n1", "string.normalize_whitespace", ("raw_note",),
                    "the tidied note, read by two branches"),
                  S("n2", "string.count_words", ("@n1",),
                    "word count, stamped on four calls later"),
                  S("n3", "string.normalize_title", ("@n1",)),
                  S("n4", "string.normalize_slug", ("@n3",)),
                  S("n5", "string.normalize_upper", ("@n4",)),
                  S("n6", "format.tag", ("@n5", "@n2")),
                  S("n7", "format.pad", ("@n6", "column_width", "fill_char"),
                    "the headline padded out to the printed column")),
                 "n7", intent="headline_padded_for_column"),
        )))

    # ── label set rewrite ───────────────────────────────────────────────
    labels = R("labels", "text_list_labels", "the labels queued for the print run")
    codes = R("codes", "text_list_codes", "the part codes queued alongside them")
    joiner = R("joiner", "separator",
               "the character the print file puts between entries")
    alt_joiner = R("alt_joiner", "separator",
                   "the character the downstream system expects instead")

    out.append(Blueprint(
        workflow_id="text.label_set_rewrite",
        domain="text_processing",
        natural_user_goal=("rewrite a print file so the downstream system can "
                           "read the label entries back out of it"),
        target_description="the rewritten entry list or the print line",
        value_generator_id="text.print_file",
        query_asset_family="label_print_file",
        hard_distractor_families=("list", "string"),
        entity_family="fabrication",
        plans=(
            Plan("rewrite.v3", (labels, joiner),
                 (S("n1", "list.combine_join_text", ("labels", "joiner"),
                    "the labels written out as one print line"),
                  S("n2", "string.normalize_upper", ("@n1",)),
                  S("n3", "string.split", ("@n2", "joiner"),
                    "the entries read back out of the rewritten line")),
                 "n3", intent="uppercased_entries"),
            Plan("rewrite.v5", (labels, joiner),
                 (S("n1", "list.map_sort_text", ("labels",)),
                  S("n2", "list.combine_join_text", ("@n1", "joiner")),
                  S("n3", "list.reduce_count_text", ("labels",),
                    "how many entries the file should contain"),
                  S("n4", "string.normalize_upper", ("@n2",)),
                  S("n5", "format.tag", ("@n4", "@n3"),
                    "print line stamped with its entry count")),
                 "n5", intent="stamped_print_line"),
            Plan("rewrite.v8", (labels, codes, joiner, alt_joiner),
                 (S("n1", "list.map_sort_text", ("labels",)),
                  S("n2", "list.combine_join_text", ("@n1", "joiner")),
                  S("n3", "string.normalize_upper", ("@n2",)),
                  S("n4", "list.combine_join_text", ("codes", "joiner")),
                  S("n5", "string.normalize_lower", ("@n4",)),
                  S("n6", "string.concat", ("@n3", "@n5"),
                    "the two halves of the file written as one line"),
                  S("n7", "string.replace", ("@n6", "joiner", "alt_joiner"),
                    "re-delimit for the downstream system"),
                  S("n8", "string.split", ("@n7", "alt_joiner"))),
                 "n8", intent="redelimited_entries"),
        )))

    # ── character budget ────────────────────────────────────────────────
    budget_note = R("raw_note", "text_note",
                    "the note that has to fit the reporting budget")
    keyword = R("keyword", "needle_text", "the term the budget is measured against")
    places = R("places", "places", "how many decimals the report shows")
    density_floor = R("density_floor", "threshold_value",
                      "the keyword share the note has to reach")
    ratio_floor = R("ratio_floor", "threshold_ratio",
                    "the characters-per-word ratio a dense note has to reach")

    out.append(Blueprint(
        workflow_id="text.character_budget",
        domain="text_processing",
        natural_user_goal=("work out how much of a note is spent on the term we "
                           "care about and whether it is dense enough to keep"),
        target_description="the note's size, its keyword share or the density verdict",
        value_generator_id="text.budget_note",
        query_asset_family="reporting_note",
        hard_distractor_families=("string", "rates"),
        boolean_balancing_strategy="threshold_band",
        entity_family="quality",
        plans=(
            Plan("budget.v2", (budget_note,),
                 (S("n1", "string.normalize_whitespace", ("raw_note",)),
                  S("n2", "string.count_length", ("@n1",),
                    "characters the note really costs")),
                 "n2", intent="note_size"),
            Plan("budget.v6", (budget_note, keyword, places),
                 (S("n1", "string.normalize_lower", ("raw_note",)),
                  S("n2", "string.normalize_whitespace", ("@n1",),
                    "the comparable form, read by two branches"),
                  S("n3", "string.count_substring", ("@n2", "keyword")),
                  S("n4", "string.count_words", ("@n2",)),
                  S("n5", "rates.share_percent", ("@n3", "@n4")),
                  S("n6", "format.percent", ("@n5", "places"))),
                 "n6", intent="keyword_share_label"),
            Plan("budget.v10", (budget_note, keyword, density_floor, ratio_floor),
                 (S("n1", "string.normalize_lower", ("raw_note",)),
                  S("n2", "string.normalize_whitespace", ("@n1",),
                    "the comparable form, read by three measurements"),
                  S("n3", "string.count_words", ("@n2",),
                    "word count, needed by both rates"),
                  S("n4", "string.count_substring", ("@n2", "keyword")),
                  S("n5", "string.count_length", ("@n2",)),
                  S("n6", "rates.share_percent", ("@n4", "@n3")),
                  S("n7", "rates.ratio_of", ("@n5", "@n3"),
                    "characters per word, four calls after the word count"),
                  S("n8", "comparison.at_least", ("@n6", "density_floor")),
                  S("n9", "comparison.at_least", ("@n7", "ratio_floor")),
                  S("n10", "boolean.and", ("@n8", "@n9"))),
                 "n10", intent="note_dense_enough"),
        )))

    # ── code casing audit ───────────────────────────────────────────────
    asset_code = R("asset_code", "identifier_code",
                   "the asset code stencilled on the unit")
    audit_note = R("raw_note", "text_note", "the note as the typist left it")
    audit_places = R("places", "places", "how many decimals the audit prints")
    column_width = R("column_width", "count_items",
                     "the width of the column the audit line is printed in")
    audit_fill = R("fill_char", "separator",
                   "the character the audit pads a short line with")
    drop_floor = R("drop_floor", "threshold_percent",
                   "how wide a slugged word is allowed to be, in percent")

    out.append(Blueprint(
        workflow_id="text.code_casing_audit",
        domain="text_processing",
        natural_user_goal=("check what casing and slugging do to a code or a "
                           "note before it is filed"),
        target_description="the audited code line or the slugging verdict",
        value_generator_id="text.casing_audit",
        query_asset_family="asset_register_entry",
        hard_distractor_families=("string", "format"),
        boolean_balancing_strategy="threshold_band",
        entity_family="quality",
        plans=(
            Plan("casing.v4", (asset_code,),
                 (S("n1", "string.normalize_lower", ("asset_code",)),
                  S("n2", "string.normalize_slug", ("@n1",),
                    "the filed form, read by two branches"),
                  S("n3", "string.count_length", ("@n2",)),
                  S("n4", "format.tag", ("@n2", "@n3"))),
                 "n4", intent="filed_code_with_length"),
            Plan("casing.v7", (audit_note, drop_floor),
                 (S("n1", "string.normalize_whitespace", ("raw_note",),
                    "the tidied note, read by two branches"),
                  S("n2", "string.count_words", ("@n1",)),
                  S("n3", "string.normalize_slug", ("@n1",)),
                  S("n4", "string.count_length", ("@n3",)),
                  S("n5", "rates.ratio_of", ("@n4", "@n2"),
                    "slug characters per word of the note"),
                  S("n6", "rates.ratio_to_percent", ("@n5",)),
                  S("n7", "comparison.at_least", ("@n6", "drop_floor"))),
                 "n7", intent="slug_words_wide_enough"),
            Plan("casing.v9", (audit_note, audit_places, column_width, audit_fill),
                 (S("n1", "string.normalize_whitespace", ("raw_note",),
                    "the tidied note, read by three branches"),
                  S("n2", "string.normalize_slug", ("@n1",)),
                  S("n3", "string.normalize_upper", ("@n2",)),
                  S("n4", "string.count_length", ("@n1",)),
                  S("n5", "string.count_words", ("@n1",),
                    "word count, read four calls after the tidy-up"),
                  S("n6", "rates.ratio_of", ("@n4", "@n5")),
                  S("n7", "format.fixed", ("@n6", "places")),
                  S("n8", "string.concat", ("@n3", "@n7"),
                    "the filed code and the measurement it carries"),
                  S("n9", "format.pad", ("@n8", "column_width", "fill_char"))),
                 "n9", intent="audit_line_for_column"),
        )))

    return out
