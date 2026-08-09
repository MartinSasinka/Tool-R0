"""Data-summary workflows: profiling a mapping, digesting records, headlines.

Summaries are where a structure has to be turned into something a person reads,
so these plans deliberately end in the answer types the numeric families never
produce: a trimmed mapping, a built record, a joined label, a rounded series.
The plans of a blueprint differ by how far the summary is taken -- one reduction
for the headline figure, a reduction plus a re-selection of the structure for
the digest, and a fan of independently thresholded reductions for the verdict.

Mappings are always trimmed at the mean of their own values so the filter can
never empty them.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── profile of an amount table ──────────────────────────────────────
    profile_roles = (
        R("site_amounts", "mapping_amounts", "amount booked per site"),
        R("places", "places", "decimals the summary should carry"),
    )
    out.append(Blueprint(
        workflow_id="summary.mapping_profile",
        domain="data_summary",
        natural_user_goal=("profile a table of per-site amounts: who leads it "
                           "and how far the leader sits above the rest"),
        target_description="the profile line or the trimmed table",
        value_generator_id="summary.amount_table",
        query_asset_family="amount_table",
        hard_distractor_families=("dictionary", "statistics"),
        entity_family="operations",
        plans=(
            Plan("profile.v3", profile_roles[:1],
                 (S("n1", "dictionary.aggregate_argmax", ("site_amounts",),
                    "site holding the largest amount"),
                  S("n2", "dictionary.aggregate_max", ("site_amounts",)),
                  S("n3", "format.tag", ("@n1", "@n2"))),
                 "n3", intent="leading_site_label"),
            Plan("profile.v5", profile_roles[:1],
                 (S("n1", "dictionary.values", ("site_amounts",)),
                  S("n2", "statistics.mean", ("@n1",),
                    "table average: the trim level and the replacement value"),
                  S("n3", "dictionary.aggregate_filter",
                    ("site_amounts", "@n2"),
                    "sites above the average"),
                  S("n4", "dictionary.aggregate_argmax", ("site_amounts",)),
                  S("n5", "dictionary.update", ("@n3", "@n4", "@n2"),
                    "leader restated at the table average")),
                 "n5", intent="normalised_leader_table"),
            Plan("profile.v7", profile_roles[:2],
                 (S("n1", "dictionary.values", ("site_amounts",),
                    "amounts as a series, read by two reductions"),
                  S("n2", "dictionary.aggregate_sum", ("site_amounts",)),
                  S("n3", "statistics.mean", ("@n1",)),
                  S("n4", "list.reduce_max", ("@n1",)),
                  S("n5", "arithmetic.subtract", ("@n4", "@n3"),
                    "how far the leader is above the average"),
                  S("n6", "rates.share_percent", ("@n5", "@n2")),
                  S("n7", "format.percent", ("@n6", "places"))),
                 "n7", intent="leader_headroom_label"),
        )))

    # ── digest of a row set ─────────────────────────────────────────────
    digest_roles = (
        R("rows", "record_list", "the rows to digest"),
        R("label_field", "text_field_name", "field naming each row"),
        R("separator", "separator", "separator between the listed names"),
        R("amount_field", "field_name", "measured field of each row"),
        R("level_limit", "threshold_value", "level the rows must reach"),
        R("spread_limit", "threshold_percent",
          "scatter the rows may show, in percent of their average"),
        R("peak_limit", "threshold_value", "value the heaviest row may reach"),
    )
    out.append(Blueprint(
        workflow_id="summary.record_digest",
        domain="data_summary",
        natural_user_goal=("digest a set of rows into one readable line and "
                           "check whether the set clears its limits"),
        target_description="the digest of the rows or the verdict on them",
        value_generator_id="summary.row_set",
        query_asset_family="row_set",
        hard_distractor_families=("record", "string"),
        boolean_balancing_strategy="threshold_band",
        entity_family="reporting",
        plans=(
            Plan("digest.v4", digest_roles[:3],
                 (S("n1", "record.project_text", ("rows", "label_field")),
                  S("n2", "list.map_sort_text", ("@n1",)),
                  S("n3", "list.combine_join_text", ("@n2", "separator")),
                  S("n4", "string.normalize_title", ("@n3",))),
                 "n4", intent="row_name_line"),
            Plan("digest.v7",
                 (digest_roles[0], digest_roles[1], digest_roles[3]),
                 (S("n1", "record.project", ("rows", "amount_field"),
                    "measured series, read by three steps"),
                  S("n2", "list.reduce_max", ("@n1",)),
                  S("n3", "list.index_of_value", ("@n1", "@n2"),
                    "where the heaviest row sits"),
                  S("n4", "record.project_text", ("rows", "label_field")),
                  S("n5", "list.index_text", ("@n4", "@n3"),
                    "name of that row"),
                  S("n6", "statistics.mean", ("@n1",)),
                  S("n7", "record.build", ("@n5", "@n6"),
                    "heaviest row's name against the average of the set")),
                 "n7", intent="digest_record"),
            Plan("digest.v9",
                 (digest_roles[0], digest_roles[3]) + digest_roles[4:7],
                 (S("n1", "record.project", ("rows", "amount_field")),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.stdev", ("@n1",)),
                  S("n4", "list.reduce_max", ("@n1",)),
                  S("n5", "rates.share_percent", ("@n3", "@n2"),
                    "scatter relative to the average"),
                  S("n6", "comparison.at_least", ("@n2", "level_limit")),
                  S("n7", "comparison.at_least", ("@n5", "spread_limit")),
                  S("n8", "comparison.at_least", ("@n4", "peak_limit")),
                  S("n9", "decision.all_of", ("@n6", "@n7", "@n8"),
                    "level, scatter and peak all clear")),
                 "n9", intent="row_set_acceptance"),
        )))

    # ── headline figure of a price series ───────────────────────────────
    headline_roles = (
        R("line_prices", "list_prices", "price of every line"),
        R("top_count", "count_small", "how many leading lines to report on"),
        R("currency", "currency_code", "currency the report is issued in"),
        R("concentration_cut", "threshold_ratio",
          "share of the series the leading lines may hold"),
    )
    out.append(Blueprint(
        workflow_id="summary.series_headline",
        domain="data_summary",
        natural_user_goal=("summarise a price series by its leading lines and "
                           "say how much of the series they carry"),
        target_description="the headline of the series or its concentration band",
        value_generator_id="summary.price_series",
        query_asset_family="price_series",
        hard_distractor_families=("list", "format"),
        entity_family="finance",
        plans=(
            Plan("headline.v2", headline_roles[:2],
                 (S("n1", "list.map_sort_asc", ("line_prices",)),
                  S("n2", "list.slice_last", ("@n1", "top_count"),
                    "the leading lines")),
                 "n2", intent="leading_lines"),
            Plan("headline.v4", headline_roles[:3],
                 (S("n1", "list.map_sort", ("line_prices",)),
                  S("n2", "list.slice_first", ("@n1", "top_count")),
                  S("n3", "list.reduce_sum", ("@n2",)),
                  S("n4", "format.currency", ("@n3", "currency"))),
                 "n4", intent="leading_lines_value"),
            Plan("headline.v6",
                 headline_roles[:2] + headline_roles[3:4],
                 (S("n1", "list.map_sort", ("line_prices",)),
                  S("n2", "list.slice_first", ("@n1", "top_count")),
                  S("n3", "list.reduce_sum", ("@n2",)),
                  S("n4", "list.reduce_sum", ("line_prices",)),
                  S("n5", "rates.ratio_of", ("@n3", "@n4"),
                    "share the leading lines carry"),
                  S("n6", "classification.ratio_band",
                    ("@n5", "concentration_cut"))),
                 "n6", intent="series_concentration_band"),
        )))

    # ── coverage of a stock table ───────────────────────────────────────
    coverage_roles = (
        R("stock_counts", "mapping_counts", "units held per product"),
        R("concentration_floor", "threshold_percent",
          "share of the stock the above-average products must carry"),
    )
    out.append(Blueprint(
        workflow_id="summary.coverage_report",
        domain="data_summary",
        natural_user_goal=("report how the held stock spreads over the products "
                           "of a catalogue"),
        target_description="the coverage figure, the restated table or the verdict",
        value_generator_id="summary.stock_table",
        query_asset_family="stock_table",
        hard_distractor_families=("dictionary", "comparison"),
        boolean_balancing_strategy="threshold_band",
        entity_family="inventory",
        plans=(
            Plan("coverage.v3", coverage_roles[:1],
                 (S("n1", "dictionary.values", ("stock_counts",),
                    "the held counts, reduced and then counted against"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "list.reduce_count_above", ("@n1", "@n2"))),
                 "n3", intent="above_average_product_count"),
            Plan("coverage.v4", coverage_roles[:1],
                 (S("n1", "dictionary.values", ("stock_counts",)),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "dictionary.aggregate_argmax", ("stock_counts",)),
                  S("n4", "dictionary.update", ("stock_counts", "@n3", "@n2"),
                    "busiest product restated at the catalogue average")),
                 "n4", intent="levelled_stock_table"),
            Plan("coverage.v7", coverage_roles[:2],
                 (S("n1", "dictionary.values", ("stock_counts",)),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "dictionary.aggregate_filter",
                    ("stock_counts", "@n2")),
                  S("n4", "dictionary.aggregate_sum", ("@n3",),
                    "stock held by the above-average products"),
                  S("n5", "dictionary.aggregate_sum", ("stock_counts",)),
                  S("n6", "rates.share_percent", ("@n4", "@n5")),
                  S("n7", "comparison.at_least",
                    ("@n6", "concentration_floor"))),
                 "n7", intent="stock_concentration_verdict"),
        )))

    return out
