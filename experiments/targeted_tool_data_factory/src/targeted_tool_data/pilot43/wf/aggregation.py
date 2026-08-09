"""Aggregation workflows: rolling two sources up and reconciling the results.

What separates this family from plain statistics is that the figures being
compared are produced by *different* aggregations -- a list reduction against a
mapping reduction, a projected field against the row count, a filtered
sub-group against the whole -- so the long plans are genuinely two independent
reduction trees that merge late instead of one chain.

The plans of a blueprint differ by how many sources they reconcile: two
reductions for the headline gap, a reduction plus its own share for the
verdict, and both sources reduced twice over for the full reconciliation.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── list total against mapping total ────────────────────────────────
    rollup_roles = (
        R("line_totals", "list_prices", "amount booked on every line"),
        R("site_amounts", "mapping_amounts", "amount the sites report"),
        R("gap_limit", "threshold_percent",
          "difference between the two sources that still passes, in percent"),
    )
    out.append(Blueprint(
        workflow_id="aggregation.two_source_rollup",
        domain="aggregation",
        natural_user_goal=("reconcile the totals the line ledger and the site "
                           "reports arrive at"),
        target_description="the reconciliation gap or the reconciliation verdict",
        value_generator_id="aggregation.reconciliation",
        query_asset_family="ledger_reconciliation",
        hard_distractor_families=("dictionary", "list"),
        boolean_balancing_strategy="threshold_band",
        entity_family="finance",
        plans=(
            Plan("rollup.v3", rollup_roles[:2],
                 (S("n1", "list.reduce_sum", ("line_totals",)),
                  S("n2", "dictionary.aggregate_sum", ("site_amounts",)),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2"),
                    "gap between the two independently built totals")),
                 "n3", intent="reconciliation_gap"),
            Plan("rollup.v5", rollup_roles[:3],
                 (S("n1", "list.reduce_sum", ("line_totals",)),
                  S("n2", "dictionary.aggregate_sum", ("site_amounts",),
                    "reported total: the counterpart and the base"),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2")),
                  S("n4", "rates.share_percent", ("@n3", "@n2")),
                  S("n5", "comparison.at_least", ("@n4", "gap_limit"))),
                 "n5", intent="reconciliation_verdict"),
            Plan("rollup.v10", rollup_roles[:2],
                 (S("n1", "list.reduce_sum", ("line_totals",)),
                  S("n2", "statistics.mean", ("line_totals",)),
                  S("n3", "dictionary.values", ("site_amounts",)),
                  S("n4", "statistics.mean", ("@n3",),
                    "average site report, base of the per-site gap"),
                  S("n5", "dictionary.aggregate_sum", ("site_amounts",),
                    "reported total, base of the overall gap"),
                  S("n6", "arithmetic.abs_difference", ("@n1", "@n5")),
                  S("n7", "arithmetic.abs_difference", ("@n2", "@n4")),
                  S("n8", "rates.share_percent", ("@n6", "@n5")),
                  S("n9", "rates.share_percent", ("@n7", "@n4")),
                  S("n10", "statistics.average_two", ("@n8", "@n9"),
                    "the gap seen at the total and at the single site, "
                    "averaged")),
                 "n10", intent="two_level_reconciliation"),
        )))

    # ── rolling a field up over a row set ───────────────────────────────
    group_roles = (
        R("rows", "record_list", "the rows to roll up"),
        R("label_field", "text_field_name", "field naming each row"),
        R("row_position", "index_position", "which listed row is asked about"),
        R("amount_field", "field_name", "field being rolled up"),
        R("share_cut", "threshold_ratio",
          "share of the rows that must clear their own average"),
    )
    out.append(Blueprint(
        workflow_id="aggregation.group_field_rollup",
        domain="aggregation",
        natural_user_goal=("roll one field of a row set up and relate a single "
                           "row to what the roll-up says"),
        target_description="a rolled-up figure, its band, or the row it belongs to",
        value_generator_id="aggregation.row_rollup",
        query_asset_family="row_set",
        hard_distractor_families=("record", "statistics"),
        entity_family="operations",
        plans=(
            Plan("group.v4", group_roles[:4],
                 (S("n1", "record.project_text", ("rows", "label_field")),
                  S("n2", "list.index_text", ("@n1", "row_position")),
                  S("n3", "record.lookup", ("rows", "label_field", "@n2"),
                    "the row carrying that name"),
                  S("n4", "record.select", ("@n3", "amount_field"))),
                 "n4", intent="single_row_value"),
            Plan("group.v6",
                 (group_roles[0], group_roles[3], group_roles[4]),
                 (S("n1", "record.project", ("rows", "amount_field")),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "record.aggregate_count",
                    ("rows", "amount_field", "@n2"),
                    "rows above their own average"),
                  S("n4", "record.aggregate_size", ("rows",)),
                  S("n5", "rates.ratio_of", ("@n3", "@n4")),
                  S("n6", "classification.ratio_band", ("@n5", "share_cut"))),
                 "n6", intent="above_average_share_band"),
            Plan("group.v8", group_roles[:4],
                 (S("n1", "record.project", ("rows", "amount_field"),
                    "rolled-up series, also the source of the sub-group"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "list.filter", ("@n1", "@n2")),
                  S("n4", "statistics.mean", ("@n3",),
                    "average of the sub-group"),
                  S("n5", "record.project_text", ("rows", "label_field")),
                  S("n6", "list.map_sort_text", ("@n5",)),
                  S("n7", "list.index_text", ("@n6", "row_position")),
                  S("n8", "record.build", ("@n7", "@n4"),
                    "the named row against the sub-group average")),
                 "n8", intent="sub_group_record"),
        )))

    # ── rolling a shift roster up ───────────────────────────────────────
    workload_roles = (
        R("shift_hours", "list_durations_h", "hours worked on every shift"),
        R("places", "places", "decimals the roster figure should carry"),
        R("peak_limit", "threshold_value",
          "how far the longest shift may exceed the average"),
        R("ceiling_limit", "threshold_value",
          "control ceiling the roster may reach"),
    )
    out.append(Blueprint(
        workflow_id="aggregation.workload_rollup",
        domain="aggregation",
        natural_user_goal=("roll a shift roster up and see whether one shift "
                           "carries too much of it"),
        target_description="the rolled-up roster figure or the overload verdict",
        value_generator_id="aggregation.roster",
        query_asset_family="shift_roster",
        hard_distractor_families=("list", "arithmetic"),
        boolean_balancing_strategy="threshold_band",
        entity_family="field service",
        plans=(
            Plan("workload.v2", workload_roles[:2],
                 (S("n1", "list.reduce_sum", ("shift_hours",)),
                  S("n2", "format.fixed", ("@n1", "places"))),
                 "n2", intent="rostered_total_label"),
            Plan("workload.v5", workload_roles[:1],
                 (S("n1", "list.reduce_sum", ("shift_hours",)),
                  S("n2", "statistics.mean", ("shift_hours",)),
                  S("n3", "list.reduce_max", ("shift_hours",)),
                  S("n4", "arithmetic.subtract", ("@n3", "@n2"),
                    "how far the longest shift is above the average"),
                  S("n5", "rates.share_percent", ("@n4", "@n1"),
                    "that excess against everything rostered")),
                 "n5", intent="peak_shift_share"),
            Plan("workload.v8",
                 workload_roles[:1] + workload_roles[2:4],
                 (S("n1", "statistics.mean", ("shift_hours",),
                    "roster average, used by both limits"),
                  S("n2", "statistics.stdev", ("shift_hours",)),
                  S("n3", "list.reduce_max", ("shift_hours",)),
                  S("n4", "arithmetic.add", ("@n1", "@n2"),
                    "control ceiling of the roster"),
                  S("n5", "arithmetic.subtract", ("@n3", "@n1"),
                    "excess carried by the longest shift"),
                  S("n6", "comparison.at_least", ("@n5", "peak_limit")),
                  S("n7", "comparison.at_least", ("@n4", "ceiling_limit")),
                  S("n8", "boolean.or", ("@n6", "@n7"),
                    "either reading of the roster raises the flag")),
                 "n8", intent="roster_overload_verdict"),
        )))

    # ── ordered quantities against held stock ───────────────────────────
    quantity_roles = (
        R("order_quantities", "list_quantities", "units ordered per line"),
        R("stock_counts", "mapping_counts", "units held per product"),
        R("line_label", "text_label", "product the order is booked against"),
    )
    out.append(Blueprint(
        workflow_id="aggregation.quantity_rollup",
        domain="aggregation",
        natural_user_goal=("put the quantities on an order next to the stock "
                           "the catalogue holds"),
        target_description="a rolled-up quantity figure, table or series",
        value_generator_id="aggregation.order_stock",
        query_asset_family="order_stock_pair",
        hard_distractor_families=("dictionary", "list"),
        entity_family="inventory",
        plans=(
            Plan("quantity.v3", quantity_roles[:2],
                 (S("n1", "dictionary.values", ("stock_counts",)),
                  S("n2", "list.combine_concat", ("@n1", "order_quantities")),
                  S("n3", "list.reduce_distinct", ("@n2",),
                    "how many different quantities occur across both sources")),
                 "n3", intent="distinct_quantity_count"),
            Plan("quantity.v5", quantity_roles[:3],
                 (S("n1", "dictionary.values", ("stock_counts",)),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "dictionary.aggregate_filter",
                    ("stock_counts", "@n2"),
                    "products held above the catalogue average"),
                  S("n4", "list.reduce_sum", ("order_quantities",)),
                  S("n5", "dictionary.update", ("@n3", "line_label", "@n4"),
                    "the ordered total booked into the trimmed table")),
                 "n5", intent="order_booked_table"),
            Plan("quantity.v6", quantity_roles[:2],
                 (S("n1", "list.reduce_sum", ("order_quantities",)),
                  S("n2", "dictionary.aggregate_sum", ("stock_counts",),
                    "stock held in total, base of both shares"),
                  S("n3", "rates.share_percent", ("@n1", "@n2")),
                  S("n4", "dictionary.aggregate_max", ("stock_counts",)),
                  S("n5", "rates.share_percent", ("@n4", "@n2")),
                  S("n6", "arithmetic.abs_difference", ("@n3", "@n5"),
                    "order share against the share the biggest product holds")),
                 "n6", intent="share_gap"),
            Plan("quantity.v7", quantity_roles[:2],
                 (S("n1", "dictionary.values", ("stock_counts",)),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "list.filter", ("@n1", "@n2"),
                    "the counts on one side of the average, reduced and "
                    "then rescaled"),
                  S("n4", "list.reduce_sum", ("@n3",)),
                  S("n5", "list.reduce_sum", ("order_quantities",)),
                  S("n6", "rates.share_percent", ("@n4", "@n5")),
                  S("n7", "list.map_percent", ("@n3", "@n6"),
                    "each of those counts at that coverage rate")),
                 "n7", intent="coverage_weighted_counts"),
        )))

    return out
