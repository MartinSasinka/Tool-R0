"""Sequence workflows: reading audits, price shaping, batch reconciliation, digests.

Every plan here binds real ``list.*`` capabilities to a real sequence: the
filter threshold is computed from the series itself (so the filter can never
empty it), the two long reconciliation plans filter the *same* series two
different ways and then compare the two results, and the shaping family keeps
its answers as lists so the module is not another float-and-boolean producer.
The plans of one blueprint differ by how much of the series they have to hold
on to: the short ones consume it once, the long ones read it again four to
eight calls later.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── reading series audit ────────────────────────────────────────────
    readings = R("readings", "list_readings",
                 "the readings the logger recorded during the run")
    gap_floor = R("gap_floor", "threshold_value",
                  "the spread the flagged readings must reach")
    small_gap = R("small_gap", "cut_low",
                  "a concentration that still counts as even")
    large_gap = R("large_gap", "cut_high",
                  "the concentration that counts as heavily skewed")
    out.append(Blueprint(
        workflow_id="list.reading_series_audit",
        domain="list_processing",
        natural_user_goal=("understand how the readings of a run sit around "
                           "their own average"),
        target_description=("how many readings stand out, how far apart they "
                            "are, or how concentrated the run is"),
        value_generator_id="list.reading_series",
        query_asset_family="reading_series",
        hard_distractor_families=("list", "statistics"),
        boolean_balancing_strategy="threshold_band",
        entity_family="field_service",
        plans=(
            Plan("readings.v3", (readings,),
                 (S("n1", "statistics.mean", ("readings",),
                    "the average of the series, used as its own cut"),
                  S("n2", "list.filter", ("readings", "@n1"),
                    "the readings on one side of that average"),
                  S("n3", "list.reduce_count", ("@n2",))),
                 "n3", intent="off_average_count"),
            Plan("readings.v6", (readings, gap_floor),
                 (S("n1", "statistics.mean", ("readings",)),
                  S("n2", "list.filter", ("readings", "@n1"),
                    "the selected readings, reduced by two branches"),
                  S("n3", "list.reduce_max", ("@n2",)),
                  S("n4", "list.reduce_min", ("@n2",)),
                  S("n5", "arithmetic.subtract", ("@n3", "@n4"),
                    "spread inside the selection"),
                  S("n6", "comparison.at_least", ("@n5", "gap_floor"))),
                 "n6", intent="selection_spread_verdict"),
            Plan("readings.v9", (readings, small_gap, large_gap),
                 (S("n1", "statistics.mean", ("readings",)),
                  S("n2", "list.filter", ("readings", "@n1")),
                  S("n3", "list.reduce_sum", ("@n2",)),
                  S("n4", "list.reduce_count", ("@n2",)),
                  S("n5", "list.index_of_max", ("readings",),
                    "where the peak reading sits"),
                  S("n6", "list.index", ("readings", "@n5"),
                    "the peak reading itself"),
                  S("n7", "rates.share_percent", ("@n3", "@n6"),
                    "the selected total measured against the peak"),
                  S("n8", "statistics.mean_three", ("@n7", "@n4", "@n1"),
                    "share, selection size and average combined"),
                  S("n9", "classification.three_bands",
                    ("@n8", "small_gap", "large_gap"))),
                 "n9", intent="run_concentration_band"),
        )))

    # ── price series shaping ────────────────────────────────────────────
    line_values = R("line_values", "list_prices",
                    "the price of every line on the sheet")
    uplift = R("uplift_rate", "percent_growth",
               "the uplift the new price sheet applies to each line")
    extra = R("handling_fee", "money_fee",
              "the handling fee appended as a further line")
    keep_count = R("keep_count", "count_small",
                   "how many of the heaviest lines the summary keeps")
    out.append(Blueprint(
        workflow_id="list.price_series_shaping",
        domain="list_processing",
        natural_user_goal=("rebuild a price sheet after an uplift and keep the "
                           "result as a series"),
        target_description="the reshaped price series",
        value_generator_id="list.price_series",
        query_asset_family="price_sheet",
        hard_distractor_families=("list", "rates"),
        entity_family="finance",
        plans=(
            Plan("shaping.v2", (line_values, uplift, extra),
                 (S("n1", "list.map_percent", ("line_values", "uplift_rate"),
                    "the uplift owed by each line"),
                  S("n2", "list.combine_append", ("@n1", "handling_fee"))),
                 "n2", intent="uplift_series_with_fee"),
            Plan("shaping.v4", (line_values, uplift, keep_count),
                 (S("n1", "list.map_cumulative", ("line_values",),
                    "the sheet read as running totals"),
                  S("n2", "list.map_percent", ("line_values", "uplift_rate")),
                  S("n3", "list.combine_pairwise", ("@n1", "@n2"),
                    "running total and uplift added position by position"),
                  S("n4", "list.filter_top_k", ("@n3", "keep_count"))),
                 "n4", intent="heaviest_reshaped_lines"),
            Plan("shaping.v7", (line_values, uplift),
                 (S("n1", "list.map_cumulative", ("line_values",),
                    "running totals, offset again five calls later"),
                  S("n2", "list.map_percent", ("line_values", "uplift_rate"),
                    "uplift series, concatenated again five calls later"),
                  S("n3", "list.combine_pairwise", ("@n1", "@n2")),
                  S("n4", "list.reduce_max", ("@n3",)),
                  S("n5", "list.map_offset", ("@n1", "@n4")),
                  S("n6", "list.combine_concat", ("@n5", "@n2")),
                  S("n7", "list.map_sort_asc", ("@n6",))),
                 "n7", intent="merged_reshaped_sheet"),
        )))

    # ── batch reconciliation (one series, two different filters) ────────
    batch_counts = R("batch_counts", "list_quantities",
                     "the counted quantity of every batch")
    gap_limit = R("gap_limit", "threshold_value",
                  "the difference between the two selections that still passes")
    out.append(Blueprint(
        workflow_id="list.quantity_batch_reconciliation",
        domain="list_processing",
        natural_user_goal=("reconcile two readings of the same batch series, "
                           "one taken against the average and one against the "
                           "median"),
        target_description=("the merged selection or the verdict on how far "
                            "the two selections disagree"),
        value_generator_id="list.batch_counts",
        query_asset_family="batch_sheet",
        hard_distractor_families=("list", "statistics"),
        boolean_balancing_strategy="threshold_band",
        entity_family="quality",
        plans=(
            Plan("batches.v5", (batch_counts,),
                 (S("n1", "statistics.mean", ("batch_counts",)),
                  S("n2", "statistics.median", ("batch_counts",)),
                  S("n3", "list.filter", ("batch_counts", "@n1"),
                    "selection taken against the average"),
                  S("n4", "list.filter", ("batch_counts", "@n2"),
                    "selection taken against the median"),
                  S("n5", "list.combine_concat", ("@n3", "@n4"))),
                 "n5", intent="merged_selections"),
            Plan("batches.v8", (batch_counts, gap_limit),
                 (S("n1", "statistics.mean", ("batch_counts",)),
                  S("n2", "list.filter", ("batch_counts", "@n1")),
                  S("n3", "list.reduce_sum", ("@n2",)),
                  S("n4", "statistics.median", ("batch_counts",)),
                  S("n5", "list.filter", ("batch_counts", "@n4")),
                  S("n6", "list.reduce_sum", ("@n5",)),
                  S("n7", "arithmetic.abs_difference", ("@n3", "@n6"),
                    "how far the two selections disagree"),
                  S("n8", "comparison.at_least", ("@n7", "gap_limit"))),
                 "n8", intent="selection_disagreement_verdict"),
            Plan("batches.v10", (batch_counts,),
                 (S("n1", "statistics.mean", ("batch_counts",)),
                  S("n2", "list.filter", ("batch_counts", "@n1"),
                    "average-side selection, appended to again eight calls on"),
                  S("n3", "list.reduce_sum", ("@n2",)),
                  S("n4", "list.reduce_max", ("@n2",)),
                  S("n5", "arithmetic.subtract", ("@n3", "@n4"),
                    "the selection without its heaviest batch"),
                  S("n6", "statistics.median", ("batch_counts",)),
                  S("n7", "list.filter", ("batch_counts", "@n6")),
                  S("n8", "statistics.mean", ("@n7",)),
                  S("n9", "arithmetic.abs_difference", ("@n5", "@n8")),
                  S("n10", "list.combine_append", ("@n2", "@n9"),
                    "the disagreement recorded as a further entry")),
                 "n10", intent="reconciled_series"),
        )))

    # ── label / code catalogue digest (text sequences) ──────────────────
    labels = R("labels", "text_list_labels",
               "the labels the catalogue page lists")
    codes = R("codes", "text_list_codes",
              "the reference codes printed on the page")
    slot = R("slot", "index_position", "which entry of the page is asked about")
    glue = R("glue", "separator", "the separator the digest is printed with")
    stem_width = R("stem_width", "count_small",
                   "how many leading characters of a code form its family stem")
    out.append(Blueprint(
        workflow_id="list.label_catalogue_digest",
        domain="list_processing",
        natural_user_goal="produce the printed digest of a catalogue page",
        target_description="the digest line for the page",
        value_generator_id="list.catalogue_page",
        query_asset_family="catalogue_page",
        hard_distractor_families=("list", "string"),
        entity_family="retail",
        plans=(
            Plan("digest.v3", (labels, slot),
                 (S("n1", "list.map_sort_text", ("labels",)),
                  S("n2", "list.index_text", ("@n1", "slot")),
                  S("n3", "string.normalize_title", ("@n2",))),
                 "n3", intent="ranked_label"),
            Plan("digest.v5", (labels, glue),
                 (S("n1", "list.map_sort_text", ("labels",)),
                  S("n2", "list.combine_join_text", ("@n1", "glue")),
                  S("n3", "list.reduce_count_text", ("labels",),
                    "how many labels the page carries"),
                  S("n4", "format.tag", ("@n2", "@n3")),
                  S("n5", "string.normalize_upper", ("@n4",))),
                 "n5", intent="page_digest_line"),
            Plan("digest.v6", (codes, slot, stem_width),
                 (S("n1", "list.map_sort_text", ("codes",)),
                  S("n2", "list.index_text", ("@n1", "slot"),
                    "the code asked about, tagged again four calls later"),
                  S("n3", "string.truncate", ("@n2", "stem_width"),
                    "its family stem"),
                  S("n4", "list.filter_prefix", ("codes", "@n3"),
                    "every code of that family"),
                  S("n5", "list.reduce_count_text", ("@n4",)),
                  S("n6", "format.tag", ("@n2", "@n5"))),
                 "n6", intent="code_family_tag"),
        )))

    return out
