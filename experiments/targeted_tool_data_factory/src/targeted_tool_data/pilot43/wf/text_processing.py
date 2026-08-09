"""Text workflows: cleaning up dictated notes, comparing them, screening them.

Every plan here really manipulates text -- whitespace, casing, slugs, word and
character counts -- and the longer plans get their shape from reuse of the
cleaned note rather than from a pattern label. Numbers only appear once the
text has produced them (a word count, a character count), which is what makes
``text -> text -> count -> boolean`` a genuine value-kind transition.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── shift note clean-up ─────────────────────────────────────────────
    note = R("note", "text_note", "the note exactly as it was typed into the log")
    label_width = R("label_width", "count_small",
                    "how many characters the short label may keep")
    word_target = R("word_target", "threshold_count",
                    "how many words a usable note has to contain")
    keyword = R("keyword", "needle_text", "the word the reviewer looks for")

    out.append(Blueprint(
        workflow_id="text.note_cleanup",
        domain="text",
        natural_user_goal=("tidy up a note somebody typed into the shift log and "
                           "turn it into a short label we can print"),
        target_description="the cleaned note, its short label or the quality verdict",
        value_generator_id="text.shift_note",
        query_asset_family="shift_log_note",
        hard_distractor_families=("string", "format"),
        boolean_balancing_strategy="calibrate_word_count_threshold",
        entity_family="operations",
        plans=(
            Plan("note.v2", (note,),
                 (S("n1", "string.normalize_whitespace", ("note",),
                    "collapse the stray spacing the typist left behind"),
                  S("n2", "string.normalize_title", ("@n1",),
                    "write the tidied note the way a heading is written")),
                 "n2", intent="tidied_note"),
            Plan("note.v3", (note, word_target),
                 (S("n1", "string.normalize_whitespace", ("note",)),
                  S("n2", "string.count_words", ("@n1",),
                    "how many real words survive the clean-up"),
                  S("n3", "comparison.at_least", ("@n2", "word_target"))),
                 "n3", intent="note_long_enough"),
            Plan("note.v5", (note, label_width),
                 (S("n1", "string.normalize_whitespace", ("note",)),
                  S("n2", "string.normalize_title", ("@n1",),
                    "the display form, needed by both branches"),
                  S("n3", "string.count_length", ("@n2",)),
                  S("n4", "string.truncate", ("@n2", "label_width")),
                  S("n5", "format.tag", ("@n4", "@n3"),
                    "short label carrying the full length")),
                 "n5", intent="printable_label"),
            Plan("note.v6", (note, keyword, word_target),
                 (S("n1", "string.normalize_whitespace", ("note",)),
                  S("n2", "string.normalize_lower", ("@n1",)),
                  S("n3", "string.count_words", ("@n1",),
                    "second consumer of the cleaned note"),
                  S("n4", "string.validate_contains", ("@n2", "keyword")),
                  S("n5", "comparison.at_least", ("@n3", "word_target")),
                  S("n6", "boolean.and", ("@n4", "@n5"))),
                 "n6", intent="note_usable"),
            Plan("note.v7", (note, label_width),
                 (S("n1", "string.normalize_whitespace", ("note",)),
                  S("n2", "string.normalize_title", ("@n1",)),
                  S("n3", "string.normalize_slug", ("@n2",)),
                  S("n4", "string.count_words", ("@n1",),
                    "reuse of the cleaned note four calls later"),
                  S("n5", "string.truncate", ("@n3", "label_width")),
                  S("n6", "format.tag", ("@n5", "@n4")),
                  S("n7", "string.count_length", ("@n6",))),
                 "n7", intent="label_width_needed"),
        )))

    # ── comparing two notes ─────────────────────────────────────────────
    note_a = R("note_a", "text_note", "the note written by the first crew")
    note_b = R("note_b", "text_note", "the note written by the second crew")
    places = R("places", "places", "how many decimals the report shows")
    gap_low = R("gap_low", "cut_low", "a difference this size is still small")
    gap_high = R("gap_high", "cut_high", "a difference this size is large")

    out.append(Blueprint(
        workflow_id="text.note_comparison",
        domain="text",
        natural_user_goal=("see how differently the two crews wrote up the same "
                           "job in their handover notes"),
        target_description="the difference between the two write-ups",
        value_generator_id="text.handover_pair",
        query_asset_family="handover_notes",
        hard_distractor_families=("string", "statistics"),
        entity_family="operations",
        plans=(
            Plan("cmp.v5", (note_a, note_b),
                 (S("n1", "string.normalize_whitespace", ("note_a",)),
                  S("n2", "string.normalize_whitespace", ("note_b",)),
                  S("n3", "string.count_length", ("@n1",)),
                  S("n4", "string.count_length", ("@n2",)),
                  S("n5", "rates.percent_change", ("@n3", "@n4"),
                    "how much longer the second write-up is")),
                 "n5", intent="length_gap"),
            Plan("cmp.v6", (note_a, note_b, gap_low, gap_high),
                 (S("n1", "string.normalize_whitespace", ("note_a",)),
                  S("n2", "string.normalize_whitespace", ("note_b",)),
                  S("n3", "string.count_length", ("@n1",)),
                  S("n4", "string.count_length", ("@n2",)),
                  S("n5", "rates.percent_change", ("@n3", "@n4")),
                  S("n6", "classification.three_bands",
                    ("@n5", "gap_low", "gap_high"))),
                 "n6", intent="length_gap_band"),
            Plan("cmp.v10", (note_a, note_b, places),
                 (S("n1", "string.normalize_whitespace", ("note_a",),
                    "first note, feeding two measurements"),
                  S("n2", "string.normalize_whitespace", ("note_b",),
                    "second note, feeding two measurements"),
                  S("n3", "string.count_length", ("@n1",)),
                  S("n4", "string.count_words", ("@n1",)),
                  S("n5", "string.count_length", ("@n2",)),
                  S("n6", "string.count_words", ("@n2",)),
                  S("n7", "rates.ratio_of", ("@n3", "@n4"),
                    "characters per word in the first note"),
                  S("n8", "rates.ratio_of", ("@n5", "@n6"),
                    "characters per word in the second note"),
                  S("n9", "rates.percent_change", ("@n7", "@n8")),
                  S("n10", "format.percent", ("@n9", "places"))),
                 "n10", intent="wordiness_shift"),
        )))

    # ── keyword screening ───────────────────────────────────────────────
    scr_note = R("note", "text_note", "the note that has to be screened")
    scr_word = R("keyword", "needle_text", "the term the screen looks for")
    scr_start = R("expected_start", "prefix_text",
                  "how a correctly filed note has to begin")
    scr_hits = R("hit_target", "threshold_count",
                 "how often the term has to appear")
    scr_words = R("word_target", "threshold_count",
                  "how many words the note has to reach")

    out.append(Blueprint(
        workflow_id="text.keyword_screen",
        domain="text",
        natural_user_goal=("check whether an incident note mentions what it is "
                           "supposed to mention before it is filed"),
        target_description="the screening verdict on the note",
        value_generator_id="text.incident_note",
        query_asset_family="incident_note",
        hard_distractor_families=("string", "boolean"),
        boolean_balancing_strategy="calibrate_keyword_and_length",
        entity_family="quality",
        plans=(
            Plan("scr.v2", (scr_note, scr_word),
                 (S("n1", "string.normalize_lower", ("note",)),
                  S("n2", "string.validate_contains", ("@n1", "keyword"))),
                 "n2", intent="mentions_term"),
            Plan("scr.v4", (scr_note, scr_word),
                 (S("n1", "string.normalize_whitespace", ("note",)),
                  S("n2", "string.normalize_lower", ("@n1",)),
                  S("n3", "string.validate_contains", ("@n2", "keyword")),
                  S("n4", "boolean.not", ("@n3",),
                    "the filing rule is that the term must be absent")),
                 "n4", intent="term_absent"),
            Plan("scr.v8", (scr_note, scr_word, scr_start, scr_hits, scr_words),
                 (S("n1", "string.normalize_whitespace", ("note",)),
                  S("n2", "string.normalize_lower", ("@n1",)),
                  S("n3", "string.count_substring", ("@n2", "keyword")),
                  S("n4", "string.count_words", ("@n1",),
                    "reuse of the cleaned note three calls later"),
                  S("n5", "comparison.at_least", ("@n3", "hit_target")),
                  S("n6", "comparison.at_least", ("@n4", "word_target")),
                  S("n7", "string.validate_prefix", ("@n2", "expected_start")),
                  S("n8", "decision.majority", ("@n5", "@n6", "@n7"))),
                 "n8", intent="screen_majority"),
        )))

    # ── label digest ────────────────────────────────────────────────────
    labels = R("labels", "text_list_labels", "the labels waiting to be printed")
    sep = R("separator", "separator", "the character the print file uses between labels")
    slot = R("slot", "index_position", "which label on the sheet we are looking at")
    width = R("width", "count_small", "how many characters fit on the small tag")
    dig_places = R("places", "places", "how many decimals the print report shows")

    out.append(Blueprint(
        workflow_id="text.label_digest",
        domain="text",
        natural_user_goal=("work out what the print run for a sheet of item "
                           "labels looks like"),
        target_description="the print line or the selected label",
        value_generator_id="text.label_sheet",
        query_asset_family="label_sheet",
        hard_distractor_families=("list", "string"),
        entity_family="fabrication",
        plans=(
            Plan("dig.v6", (labels, sep, dig_places),
                 (S("n1", "list.map_sort_text", ("labels",)),
                  S("n2", "list.combine_join_text", ("@n1", "separator"),
                    "the print line, measured two ways"),
                  S("n3", "string.count_substring", ("@n2", "separator")),
                  S("n4", "string.count_length", ("@n2",)),
                  S("n5", "rates.share_percent", ("@n3", "@n4")),
                  S("n6", "format.percent", ("@n5", "places"))),
                 "n6", intent="share_spent_on_separators"),
            Plan("dig.v5", (labels, sep, slot, width),
                 (S("n1", "list.map_sort_text", ("labels",)),
                  S("n2", "list.combine_join_text", ("@n1", "separator")),
                  S("n3", "string.split_take", ("@n2", "separator", "slot")),
                  S("n4", "string.normalize_title", ("@n3",)),
                  S("n5", "string.truncate", ("@n4", "width"))),
                 "n5", intent="tag_for_slot"),
            Plan("dig.v8", (labels, sep, slot),
                 (S("n1", "list.map_sort_text", ("labels",)),
                  S("n2", "list.combine_join_text", ("@n1", "separator"),
                    "the print line, measured again four calls later"),
                  S("n3", "string.split_take", ("@n2", "separator", "slot")),
                  S("n4", "string.normalize_slug", ("@n3",)),
                  S("n5", "string.count_length", ("@n2",)),
                  S("n6", "string.count_length", ("@n4",)),
                  S("n7", "rates.share_percent", ("@n6", "@n5")),
                  S("n8", "format.tag", ("@n4", "@n7"))),
                 "n8", intent="print_job_code"),
        )))

    return out
