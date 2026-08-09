"""Statistics workflows: dispersion, central tendency, composition, outliers.

The family exists to bind ``statistics.*`` against real collections, so every
plan starts from a list (or from a list projected out of records) and reduces
it; the arithmetic that follows only relates two reductions to each other.
The plans of one blueprint differ by how much of the sample they look at: the
short ones reduce once, the long ones reduce, re-select a sub-sample from the
reduction and reduce again, which is where the nested aggregations and the
reused mean come from.

Every filter cuts at the sample's own mean rather than at a stated constant:
that keeps both admissible bindings of ``list.filter`` non-empty regardless of
the drawn values.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── dispersion of a measurement run ─────────────────────────────────
    dispersion_roles = (
        R("readings", "list_readings", "the readings taken during the run"),
        R("stability_limit", "threshold_value",
          "dispersion the run is allowed to reach"),
        R("places", "places", "decimals the report should carry"),
        R("spread_limit", "threshold_value", "dispersion the run must clear"),
        R("skew_limit", "threshold_percent",
          "how far the mean may sit from the median, in percent"),
    )
    out.append(Blueprint(
        workflow_id="statistics.reading_dispersion",
        domain="statistics",
        natural_user_goal=("judge how widely the readings of a measurement run "
                           "scatter around their own average"),
        target_description="the dispersion of the run or the stability verdict",
        value_generator_id="statistics.measurement_run",
        query_asset_family="measurement_run",
        hard_distractor_families=("list", "arithmetic"),
        boolean_balancing_strategy="threshold_band",
        entity_family="quality",
        plans=(
            Plan("dispersion.v2", dispersion_roles[:2],
                 (S("n1", "statistics.stdev", ("readings",),
                    "scatter of the run"),
                  S("n2", "comparison.at_least", ("@n1", "stability_limit"),
                    "stability verdict")),
                 "n2", intent="stability_verdict"),
            Plan("dispersion.v4",
                 (dispersion_roles[0], dispersion_roles[2]),
                 (S("n1", "statistics.mean", ("readings",)),
                  S("n2", "statistics.stdev", ("readings",)),
                  S("n3", "rates.share_percent", ("@n2", "@n1"),
                    "scatter relative to the level it scatters around"),
                  S("n4", "format.percent", ("@n3", "places"))),
                 "n4", intent="relative_dispersion_label"),
            Plan("dispersion.v6", dispersion_roles[:1],
                 (S("n1", "statistics.mean", ("readings",),
                    "level of the run, needed again by the blend"),
                  S("n2", "statistics.stdev", ("readings",)),
                  S("n3", "arithmetic.add", ("@n1", "@n2"),
                    "upper control limit"),
                  S("n4", "list.reduce_count_above", ("readings", "@n3"),
                    "readings beyond the control limit"),
                  S("n5", "list.reduce_count", ("readings",)),
                  S("n6", "statistics.weighted_average",
                    ("@n1", "@n5", "@n3", "@n4"),
                    "level and control limit blended by how many readings "
                    "each of them describes")),
                 "n6", intent="blended_reference_level"),
            Plan("dispersion.v8",
                 dispersion_roles[:1] + dispersion_roles[3:5],
                 (S("n1", "statistics.mean", ("readings",)),
                  S("n2", "statistics.median", ("readings",)),
                  S("n3", "statistics.stdev", ("readings",)),
                  S("n4", "arithmetic.abs_difference", ("@n1", "@n2"),
                    "distance between the two centres"),
                  S("n5", "rates.share_percent", ("@n4", "@n1"),
                    "that distance as a share of the level"),
                  S("n6", "comparison.at_least", ("@n3", "spread_limit")),
                  S("n7", "comparison.at_least", ("@n5", "skew_limit")),
                  S("n8", "boolean.and", ("@n6", "@n7"),
                    "wide and lopsided at the same time")),
                 "n8", intent="wide_and_skewed_verdict"),
        )))

    # ── mean against median ─────────────────────────────────────────────
    tendency_roles = (
        R("sample_values", "list_prices", "the values in the sample"),
        R("gap_low", "cut_low", "gap that still counts as a small one"),
        R("gap_high", "cut_high", "gap that counts as a large one"),
        R("gap_ratio_cut", "threshold_ratio",
          "share of the sample range the gap may take up"),
    )
    out.append(Blueprint(
        workflow_id="statistics.central_tendency_gap",
        domain="statistics",
        natural_user_goal=("find out how far the average of a sample sits from "
                           "its middle value"),
        target_description="the gap between the two centres, or its size band",
        value_generator_id="statistics.sample",
        query_asset_family="value_sample",
        hard_distractor_families=("statistics", "comparison"),
        entity_family="analytics",
        plans=(
            Plan("tendency.v3", tendency_roles[:1],
                 (S("n1", "statistics.mean", ("sample_values",)),
                  S("n2", "statistics.median", ("sample_values",)),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2"))),
                 "n3", intent="centre_gap"),
            Plan("tendency.v5", tendency_roles[:3],
                 (S("n1", "statistics.mean", ("sample_values",)),
                  S("n2", "statistics.median", ("sample_values",),
                    "middle value, also the base of the share"),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2")),
                  S("n4", "rates.share_percent", ("@n3", "@n2")),
                  S("n5", "classification.three_bands",
                    ("@n4", "gap_low", "gap_high"))),
                 "n5", intent="centre_gap_band"),
            Plan("tendency.v8",
                 tendency_roles[:1] + tendency_roles[3:4],
                 (S("n1", "statistics.mean", ("sample_values",)),
                  S("n2", "statistics.median", ("sample_values",)),
                  S("n3", "list.reduce_max", ("sample_values",)),
                  S("n4", "list.reduce_min", ("sample_values",)),
                  S("n5", "arithmetic.subtract", ("@n3", "@n4"),
                    "range the sample covers"),
                  S("n6", "arithmetic.abs_difference", ("@n1", "@n2")),
                  S("n7", "rates.ratio_of", ("@n6", "@n5"),
                    "how much of the range the centre gap takes up"),
                  S("n8", "classification.ratio_band",
                    ("@n7", "gap_ratio_cut"))),
                 "n8", intent="centre_gap_against_range"),
        )))

    # ── composition of a batch of sampled records ───────────────────────
    composition_roles = (
        R("samples", "record_list", "the sampled rows of the batch"),
        R("amount_field", "field_name", "field the analysis runs on"),
        R("units_field", "field_name", "second field to fold in"),
    )
    out.append(Blueprint(
        workflow_id="statistics.sample_composition",
        domain="statistics",
        natural_user_goal=("describe how the measured field of a sampled batch "
                           "is composed"),
        target_description="a composition figure of the sampled batch",
        value_generator_id="statistics.batch_sample",
        query_asset_family="sample_batch",
        hard_distractor_families=("record", "statistics"),
        entity_family="laboratory",
        plans=(
            Plan("composition.v4", composition_roles[:2],
                 (S("n1", "record.project", ("samples", "amount_field"),
                    "the field pulled out as a series"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.stdev", ("@n1",)),
                  S("n4", "rates.share_percent", ("@n3", "@n2"))),
                 "n4", intent="coefficient_of_variation"),
            Plan("composition.v6", composition_roles[:3],
                 (S("n1", "record.project", ("samples", "amount_field")),
                  S("n2", "record.project", ("samples", "units_field")),
                  S("n3", "list.combine_pairwise", ("@n1", "@n2"),
                    "the two fields added row by row"),
                  S("n4", "statistics.mean", ("@n3",)),
                  S("n5", "list.reduce_max", ("@n3",)),
                  S("n6", "rates.share_percent", ("@n4", "@n5"),
                    "average row against the heaviest row")),
                 "n6", intent="row_balance"),
            Plan("composition.v7", composition_roles[:2],
                 (S("n1", "record.project", ("samples", "amount_field"),
                    "series the whole plan reads from"),
                  S("n2", "statistics.mean", ("@n1",),
                    "batch average, also the cut for the sub-sample"),
                  S("n3", "list.filter", ("@n1", "@n2"),
                    "the rows on one side of the average"),
                  S("n4", "statistics.mean", ("@n3",)),
                  S("n5", "record.aggregate_size", ("samples",)),
                  S("n6", "list.reduce_count", ("@n3",)),
                  S("n7", "statistics.weighted_average",
                    ("@n4", "@n6", "@n2", "@n5"),
                    "sub-sample average and batch average, each weighted by "
                    "the rows behind it")),
                 "n7", intent="weighted_batch_level"),
        )))

    # ── screening a run for outliers ────────────────────────────────────
    outlier_roles = (
        R("readings", "list_readings", "the readings to screen"),
        R("excess_limit", "threshold_value",
          "how far the flagged group may exceed the run average"),
        R("places", "places", "decimals the returned series should carry"),
    )
    out.append(Blueprint(
        workflow_id="statistics.outlier_screen",
        domain="statistics",
        natural_user_goal=("screen a run for the readings that sit away from "
                           "the average and describe them"),
        target_description="the flagged readings or the verdict on them",
        value_generator_id="statistics.screen",
        query_asset_family="measurement_run",
        hard_distractor_families=("list", "statistics"),
        boolean_balancing_strategy="threshold_band",
        entity_family="quality",
        plans=(
            Plan("outlier.v3", outlier_roles[:1],
                 (S("n1", "statistics.mean", ("readings",)),
                  S("n2", "list.filter", ("readings", "@n1")),
                  S("n3", "list.reduce_count", ("@n2",))),
                 "n3", intent="flagged_reading_count"),
            Plan("outlier.v5", outlier_roles[:2],
                 (S("n1", "statistics.mean", ("readings",),
                    "run average, the cut and the baseline"),
                  S("n2", "list.filter", ("readings", "@n1")),
                  S("n3", "statistics.mean", ("@n2",)),
                  S("n4", "arithmetic.subtract", ("@n3", "@n1"),
                    "how far the flagged group sits from the run"),
                  S("n5", "comparison.at_least", ("@n4", "excess_limit"))),
                 "n5", intent="flagged_group_verdict"),
            Plan("outlier.v6",
                 (outlier_roles[0], outlier_roles[2]),
                 (S("n1", "statistics.mean", ("readings",)),
                  S("n2", "list.filter", ("readings", "@n1"),
                    "flagged group, reduced and then rescaled"),
                  S("n3", "statistics.mean", ("@n2",)),
                  S("n4", "rates.percent_change", ("@n1", "@n3"),
                    "drift of the flagged group against the run"),
                  S("n5", "list.map_percent", ("@n2", "@n4"),
                    "each flagged reading at that drift"),
                  S("n6", "list.map_round", ("@n5", "places"))),
                 "n6", intent="flagged_drift_series"),
        )))

    return out
