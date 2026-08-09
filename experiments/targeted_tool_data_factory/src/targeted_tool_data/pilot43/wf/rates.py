"""Rates and ratios: growth over periods, shares of a total, uplifts, blends.

Every blueprint here is about a *rate* rather than an amount, so the plans lean on
``rates.*`` throughout and keep the two rate types apart on purpose:
``rates.ratio_of`` produces a Ratio (0.32) and only a Ratio may reach a ratio
band, while ``rates.share_percent`` and ``rates.percent_change`` produce a
Percentage (32.0) and only a Percentage may drive a percentage uplift. Where a
plan needs to cross that line it does so with an explicit
``rates.ratio_to_percent`` call.

The plans of a blueprint differ in depth: the short ones report the rate itself,
the long ones derive two rates from one shared aggregate and end in a verdict or
a band whose comparison constant is calibrated after execution.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── growth over a horizon ───────────────────────────────────────────
    growth_roles = (
        R("base_volume", "quantity_units", "units shipped in the base period"),
        R("growth_rate", "percent_growth", "growth rate expected per period"),
        R("periods", "count_small", "periods the forecast runs for"),
        R("volume_target", "threshold_value",
          "volume the forecast has to reach"),
        R("share_limit", "threshold_percent",
          "share of the forecast that may come from growth"),
    )
    out.append(Blueprint(
        workflow_id="rates.volume_growth_forecast",
        domain="rates_and_ratios",
        natural_user_goal=("work out what compounding growth does to a shipping "
                           "volume and how much of the forecast is new"),
        target_description="the growth rate, the new share, or the forecast verdict",
        value_generator_id="rates.growth",
        query_asset_family="volume_forecast",
        hard_distractor_families=("arithmetic", "comparison"),
        boolean_balancing_strategy="threshold_band",
        entity_family="logistics",
        plans=(
            Plan("vgrowth.v2", growth_roles[:3],
                 (S("n1", "rates.compound_growth",
                    ("base_volume", "growth_rate", "periods")),
                  S("n2", "rates.percent_change", ("base_volume", "@n1"),
                    "total movement over the whole horizon")),
                 "n2", intent="horizon_percent_change"),
            Plan("vgrowth.v4", growth_roles[:3],
                 (S("n1", "rates.compound_growth",
                    ("base_volume", "growth_rate", "periods")),
                  S("n2", "arithmetic.subtract", ("@n1", "base_volume"),
                    "volume the growth adds"),
                  S("n3", "rates.ratio_of", ("@n2", "base_volume"),
                    "that addition as a ratio of the base"),
                  S("n4", "rates.ratio_to_percent", ("@n3",),
                    "the same figure restated as a percentage")),
                 "n4", intent="growth_ratio_as_percent"),
            Plan("vgrowth.v6", growth_roles,
                 (S("n1", "rates.compound_growth",
                    ("base_volume", "growth_rate", "periods"),
                    "forecast volume, read by three later calls"),
                  S("n2", "arithmetic.subtract", ("@n1", "base_volume")),
                  S("n3", "rates.share_percent", ("@n2", "@n1"),
                    "share of the forecast that is new volume"),
                  S("n4", "comparison.at_least", ("@n1", "volume_target")),
                  S("n5", "comparison.at_least", ("@n3", "share_limit")),
                  S("n6", "boolean.and", ("@n4", "@n5"),
                    "the forecast is big enough and growth-driven")),
                 "n6", intent="growth_quality_verdict"),
        )))

    # ── share of a network total ────────────────────────────────────────
    share_roles = (
        R("site_amounts", "mapping_amounts", "turnover booked at each site"),
        R("rebate_rate", "percent_margin", "rebate rate the top site is granted"),
        R("net_share_limit", "threshold_percent",
          "share of the network the top site may hold after its rebate"),
    )
    out.append(Blueprint(
        workflow_id="rates.site_share_audit",
        domain="rates_and_ratios",
        natural_user_goal=("find out how concentrated a network's turnover is on "
                           "its strongest site"),
        target_description="the leading site's share or the concentration verdict",
        value_generator_id="rates.shares",
        query_asset_family="site_turnover_table",
        hard_distractor_families=("dictionary", "arithmetic"),
        boolean_balancing_strategy="threshold_band",
        entity_family="retail",
        plans=(
            Plan("share.v3", share_roles[:1],
                 (S("n1", "dictionary.aggregate_max", ("site_amounts",)),
                  S("n2", "dictionary.aggregate_sum", ("site_amounts",)),
                  S("n3", "rates.share_percent", ("@n1", "@n2"),
                    "share the strongest site holds")),
                 "n3", intent="leader_share"),
            Plan("share.v5", share_roles[:1],
                 (S("n1", "dictionary.aggregate_sum", ("site_amounts",),
                    "network total, needed again by the last call"),
                  S("n2", "dictionary.aggregate_argmax", ("site_amounts",)),
                  S("n3", "dictionary.update_remove", ("site_amounts", "@n2"),
                    "the table with the strongest site taken out"),
                  S("n4", "dictionary.aggregate_sum", ("@n3",)),
                  S("n5", "rates.share_percent", ("@n4", "@n1"),
                    "share the remaining sites hold between them")),
                 "n5", intent="rest_of_network_share"),
            Plan("share.v8", share_roles,
                 (S("n1", "dictionary.aggregate_sum", ("site_amounts",)),
                  S("n2", "dictionary.aggregate_argmax", ("site_amounts",)),
                  S("n3", "dictionary.lookup", ("site_amounts", "@n2"),
                    "turnover of the strongest site"),
                  S("n4", "rates.percent_of", ("rebate_rate", "@n3"),
                    "rebate that site earns"),
                  S("n5", "arithmetic.subtract", ("@n3", "@n4")),
                  S("n6", "arithmetic.subtract", ("@n1", "@n4"),
                    "network total once the rebate is paid out"),
                  S("n7", "rates.share_percent", ("@n5", "@n6")),
                  S("n8", "comparison.at_least", ("@n7", "net_share_limit"),
                    "still too concentrated after the rebate?")),
                 "n8", intent="post_rebate_concentration_verdict"),
        )))

    # ── per-line uplift schedule ────────────────────────────────────────
    uplift_roles = (
        R("line_prices", "list_prices", "price of every line on the schedule"),
        R("uplift_rate", "percent_growth", "uplift applied to each line"),
        R("tax_rate", "percent_tax", "tax charged on the uplifted lines"),
        R("low_cut", "cut_low", "uplift share that counts as a light revision"),
        R("high_cut", "cut_high", "uplift share that counts as a heavy revision"),
        R("concentration_limit", "threshold_percent",
          "share of the new total one line may represent"),
        R("uplift_share_limit", "threshold_percent",
          "share of the new total that may be uplift"),
    )
    out.append(Blueprint(
        workflow_id="rates.line_uplift_schedule",
        domain="rates_and_ratios",
        natural_user_goal=("apply a percentage uplift to every line of a price "
                           "schedule and judge how heavy the revision is"),
        target_description="the uplifted lines, the uplift share, or its band",
        value_generator_id="rates.uplift",
        query_asset_family="price_schedule",
        hard_distractor_families=("list", "classification"),
        boolean_balancing_strategy="threshold_band",
        entity_family="pricing",
        plans=(
            Plan("lineuplift.v4", uplift_roles[:3],
                 (S("n1", "list.map_percent", ("line_prices", "uplift_rate"),
                    "uplift due on each line"),
                  S("n2", "list.combine_pairwise", ("line_prices", "@n1"),
                    "line values after the uplift, used by both later calls"),
                  S("n3", "list.map_percent", ("@n2", "tax_rate")),
                  S("n4", "list.combine_pairwise", ("@n2", "@n3"),
                    "the amount finally charged per line")),
                 "n4", intent="charged_lines"),
            Plan("lineuplift.v6", (uplift_roles[0], uplift_roles[1],
                               uplift_roles[3], uplift_roles[4]),
                 (S("n1", "list.map_percent", ("line_prices", "uplift_rate"),
                    "per-line uplift, aggregated three calls later"),
                  S("n2", "list.combine_pairwise", ("line_prices", "@n1")),
                  S("n3", "list.reduce_sum", ("@n2",)),
                  S("n4", "list.reduce_sum", ("@n1",)),
                  S("n5", "rates.share_percent", ("@n4", "@n3"),
                    "share of the new total that is pure uplift"),
                  S("n6", "classification.three_bands",
                    ("@n5", "low_cut", "high_cut"))),
                 "n6", intent="revision_weight_band"),
            Plan("lineuplift.v10", (uplift_roles[0], uplift_roles[1],
                                uplift_roles[5], uplift_roles[6]),
                 (S("n1", "list.map_percent", ("line_prices", "uplift_rate")),
                  S("n2", "list.combine_pairwise", ("line_prices", "@n1")),
                  S("n3", "list.reduce_sum", ("@n2",),
                    "uplifted total, the denominator of both shares"),
                  S("n4", "list.reduce_max", ("@n2",)),
                  S("n5", "list.reduce_sum", ("@n1",)),
                  S("n6", "rates.share_percent", ("@n4", "@n3")),
                  S("n7", "rates.share_percent", ("@n5", "@n3")),
                  S("n8", "comparison.at_least",
                    ("@n6", "concentration_limit")),
                  S("n9", "comparison.at_least",
                    ("@n7", "uplift_share_limit")),
                  S("n10", "boolean.and", ("@n8", "@n9"),
                    "the schedule is both concentrated and uplift-heavy")),
                 "n10", intent="schedule_risk_verdict"),
        )))

    # ── blend strength ──────────────────────────────────────────────────
    blend_roles = (
        R("concentrate_l", "volume_l", "litres of concentrate in the blend"),
        R("water_l", "volume_l", "litres of water in the blend"),
        R("target_ratio", "threshold_ratio",
          "concentrate ratio the specification asks for"),
        R("top_up_rate", "percent_growth", "share by which the batch is topped up"),
        R("dose_limit", "threshold_value",
          "litres of concentrate one dosing unit can add"),
    )
    out.append(Blueprint(
        workflow_id="rates.blend_strength_control",
        domain="rates_and_ratios",
        natural_user_goal=("check the strength of a blend and what a top-up does "
                           "to the concentrate it needs"),
        target_description="the strength band or the extra concentrate required",
        value_generator_id="rates.blend",
        query_asset_family="blend_batch",
        hard_distractor_families=("rates", "classification"),
        boolean_balancing_strategy="threshold_band",
        entity_family="quality",
        plans=(
            Plan("blend.v3", (blend_roles[0], blend_roles[1], blend_roles[2]),
                 (S("n1", "arithmetic.add", ("concentrate_l", "water_l")),
                  S("n2", "rates.ratio_of", ("concentrate_l", "@n1"),
                    "concentrate ratio of the batch"),
                  S("n3", "classification.ratio_band", ("@n2", "target_ratio"),
                    "ratio band against the specification")),
                 "n3", intent="strength_band"),
            Plan("blend.v5", (blend_roles[0], blend_roles[1], blend_roles[3]),
                 (S("n1", "arithmetic.add", ("concentrate_l", "water_l"),
                    "batch volume, read again once the top-up is applied"),
                  S("n2", "rates.ratio_of", ("concentrate_l", "@n1")),
                  S("n3", "rates.ratio_to_percent", ("@n2",),
                    "the strength as a percentage so it can drive a rate"),
                  S("n4", "rates.increase_by_percent", ("@n1", "top_up_rate")),
                  S("n5", "rates.percent_of", ("@n3", "@n4"),
                    "concentrate the topped-up batch needs to hold its strength")),
                 "n5", intent="concentrate_after_top_up"),
            Plan("blend.v7", (blend_roles[0], blend_roles[1], blend_roles[3],
                              blend_roles[4]),
                 (S("n1", "arithmetic.add", ("concentrate_l", "water_l")),
                  S("n2", "rates.ratio_of", ("concentrate_l", "@n1")),
                  S("n3", "rates.ratio_to_percent", ("@n2",)),
                  S("n4", "rates.increase_by_percent", ("@n1", "top_up_rate")),
                  S("n5", "rates.percent_of", ("@n3", "@n4")),
                  S("n6", "arithmetic.subtract", ("@n5", "concentrate_l"),
                    "extra concentrate the top-up calls for"),
                  S("n7", "comparison.at_least", ("@n6", "dose_limit"),
                    "more than one dosing unit can deliver?")),
                 "n7", intent="extra_dose_verdict"),
        )))

    return out
