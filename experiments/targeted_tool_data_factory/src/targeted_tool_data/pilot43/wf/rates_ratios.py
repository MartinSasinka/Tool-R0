"""Rate and ratio workflows: price uplifts, conversion funnels, growth forecasts.

The distinction the semantic types enforce is the point of this domain: a ratio
is never silently a percentage, so a funnel step that wants a band has to keep
its value as a ratio, while a step that wants a printable figure has to convert.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── price uplift review ─────────────────────────────────────────────
    base_price = R("base_price", "money_price", "price the factory charges")
    margin_rate = R("margin_rate", "percent_margin",
                    "margin the reseller adds on top")
    discount_rate = R("discount_rate", "percent_discount",
                      "discount promised to the account")
    tax_rate = R("tax_rate", "percent_tax", "tax rate that applies to the sale")
    volume = R("volume", "quantity_units", "units the account is buying")
    price_ceiling = R("price_ceiling", "threshold_money",
                      "amount the account has approved")
    share_limit = R("share_limit", "threshold_percent",
                    "share of the taxed price that may be net goods")
    currency = R("currency", "currency_code", "currency of the quote")
    places = R("places", "places", "decimals the quote is written to")

    out.append(Blueprint(
        workflow_id="rates_ratios.price_uplift_review",
        domain="rates_ratios",
        natural_user_goal=("see what a factory price becomes once margin, "
                           "discount and tax have all been applied"),
        target_description="the final price, its movement or the approval verdict",
        value_generator_id="rates_ratios.uplift",
        query_asset_family="price_quote",
        hard_distractor_families=("rates", "arithmetic"),
        boolean_balancing_strategy="calibrate_quote_ceiling",
        entity_family="wholesale",
        plans=(
            Plan("uplift.v2", (base_price, margin_rate, tax_rate),
                 (S("n1", "rates.increase_by_percent", ("base_price",
                                                        "margin_rate"),
                    "the reseller's price"),
                  S("n2", "rates.apply_tax", ("@n1", "tax_rate"),
                    "what the customer is billed per unit")),
                 "n2", intent="taxed_shelf_price"),
            Plan("uplift.v4", (base_price, margin_rate, tax_rate, volume,
                               price_ceiling),
                 (S("n1", "rates.increase_by_percent", ("base_price",
                                                        "margin_rate"),
                    "the reseller's price"),
                  S("n2", "rates.apply_tax", ("@n1", "tax_rate"),
                    "what the customer is billed per unit"),
                  S("n3", "arithmetic.multiply", ("@n2", "volume")),
                  S("n4", "comparison.at_least", ("@n3", "price_ceiling"))),
                 "n4", intent="order_approval_verdict"),
            Plan("uplift.v6", (base_price, margin_rate, discount_rate, tax_rate,
                               currency, places),
                 (S("n1", "rates.increase_by_percent", ("base_price",
                                                        "margin_rate")),
                  S("n2", "rates.decrease_by_percent", ("@n1",
                                                        "discount_rate")),
                  S("n3", "rates.apply_tax", ("@n2", "tax_rate")),
                  S("n4", "rates.percent_change", ("base_price", "@n3"),
                    "how far the price has moved from the factory gate"),
                  S("n5", "format.percent", ("@n4", "places")),
                  S("n6", "string.concat", ("currency", "@n5"),
                    "the note that goes on the quote")),
                 "n6", intent="price_movement_note"),
            Plan("uplift.v8", (base_price, margin_rate, discount_rate, tax_rate,
                               volume, price_ceiling, share_limit),
                 (S("n1", "rates.increase_by_percent", ("base_price",
                                                        "margin_rate")),
                  S("n2", "rates.decrease_by_percent", ("@n1", "discount_rate"),
                    "the discounted price, needed again for the tax share"),
                  S("n3", "rates.apply_tax", ("@n2", "tax_rate")),
                  S("n4", "arithmetic.multiply", ("@n3", "volume")),
                  S("n5", "rates.share_percent", ("@n2", "@n3")),
                  S("n6", "comparison.at_least", ("@n4", "price_ceiling")),
                  S("n7", "comparison.at_least", ("@n5", "share_limit")),
                  S("n8", "boolean.and", ("@n6", "@n7"))),
                 "n8", intent="quote_review_verdict"),
        )))

    # ── conversion funnel ───────────────────────────────────────────────
    visitors = R("visitors", "quantity_units", "people who reached the page")
    signups = R("signups", "quantity_units", "people who left their details")
    buyers = R("buyers", "quantity_units", "people who went on to buy")
    band_cut = R("band_cut", "threshold_ratio",
                 "conversion the team was aiming for")
    gap_floor = R("gap_floor", "threshold_value",
                  "difference between the two stages worth reporting")
    rate_floor = R("rate_floor", "threshold_percent",
                   "sign-up rate the campaign promised")
    close_floor = R("close_floor", "threshold_percent",
                    "closing rate the sales team promised")
    places2 = R("places", "places", "decimals the report is written to")

    out.append(Blueprint(
        workflow_id="rates_ratios.conversion_funnel",
        domain="rates_ratios",
        natural_user_goal=("understand how well visitors turn into sign-ups and "
                           "sign-ups into buyers"),
        target_description="the conversion band, the stage gap or the campaign verdict",
        value_generator_id="rates_ratios.funnel",
        query_asset_family="campaign_report",
        hard_distractor_families=("rates", "comparison"),
        boolean_balancing_strategy="calibrate_stage_rate_floors",
        entity_family="marketing",
        plans=(
            Plan("funnel.v2", (visitors, buyers, band_cut),
                 (S("n1", "rates.ratio_of", ("buyers", "visitors"),
                    "buyers per visitor"),
                  S("n2", "classification.ratio_band", ("@n1", "band_cut"))),
                 "n2", intent="conversion_band"),
            Plan("funnel.v4", (visitors, signups, buyers, gap_floor),
                 (S("n1", "rates.ratio_of", ("signups", "visitors"),
                    "first stage of the funnel"),
                  S("n2", "rates.ratio_of", ("buyers", "signups"),
                    "second stage, measured on its own"),
                  S("n3", "arithmetic.abs_difference", ("@n1", "@n2"),
                    "how differently the two stages behave"),
                  S("n4", "comparison.at_least", ("@n3", "gap_floor"))),
                 "n4", intent="stage_gap_verdict"),
            Plan("funnel.v6", (visitors, signups, buyers, places2),
                 (S("n1", "rates.ratio_of", ("signups", "visitors")),
                  S("n2", "rates.ratio_to_percent", ("@n1",)),
                  S("n3", "rates.ratio_of", ("buyers", "signups")),
                  S("n4", "rates.ratio_to_percent", ("@n3",)),
                  S("n5", "rates.percent_change", ("@n2", "@n4"),
                    "how much the funnel narrows between stages"),
                  S("n6", "format.percent", ("@n5", "places"))),
                 "n6", intent="funnel_narrowing_label"),
            Plan("funnel.v7", (visitors, signups, buyers, rate_floor,
                               close_floor),
                 (S("n1", "rates.ratio_of", ("signups", "visitors")),
                  S("n2", "rates.ratio_to_percent", ("@n1",)),
                  S("n3", "rates.ratio_of", ("buyers", "signups")),
                  S("n4", "rates.ratio_to_percent", ("@n3",)),
                  S("n5", "comparison.at_least", ("@n2", "rate_floor")),
                  S("n6", "comparison.at_least", ("@n4", "close_floor")),
                  S("n7", "boolean.xor", ("@n5", "@n6"),
                    "true when exactly one half of the funnel delivered")),
                 "n7", intent="single_stage_success"),
        )))

    # ── growth projection ───────────────────────────────────────────────
    baseline = R("baseline", "money_total", "revenue the year opened with")
    growth_rate = R("growth_rate", "percent_growth",
                    "growth the plan assumed each period")
    periods = R("periods", "count_small", "periods the plan covers")
    later_value = R("later_value", "money_total", "revenue actually recorded")
    target_value = R("target_value", "threshold_money",
                     "revenue the board signed off on")
    error_share_limit = R("error_share_limit", "threshold_percent",
                          "how far the forecast may miss before it is rewritten")
    growth_floor = R("growth_floor", "threshold_percent",
                     "growth the board treats as the minimum")
    band_low = R("band_low", "cut_low", "growth that counts as flat")
    band_high = R("band_high", "cut_high", "growth that counts as strong")

    out.append(Blueprint(
        workflow_id="rates_ratios.growth_projection",
        domain="rates_ratios",
        natural_user_goal=("compare what a growth plan predicted with what the "
                           "business actually did"),
        target_description="the growth band, the forecast error or the plan verdict",
        value_generator_id="rates_ratios.growth",
        query_asset_family="growth_plan",
        hard_distractor_families=("rates", "statistics"),
        boolean_balancing_strategy="calibrate_forecast_error_share",
        entity_family="finance",
        plans=(
            Plan("growth.v3", (baseline, growth_rate, periods, band_low,
                               band_high),
                 (S("n1", "rates.compound_growth", ("baseline", "growth_rate",
                                                    "periods")),
                  S("n2", "rates.percent_change", ("baseline", "@n1")),
                  S("n3", "classification.three_bands", ("@n2", "band_low",
                                                         "band_high"))),
                 "n3", intent="projected_growth_band"),
            Plan("growth.v6", (baseline, growth_rate, periods, later_value,
                               error_share_limit),
                 (S("n1", "rates.compound_growth", ("baseline", "growth_rate",
                                                    "periods")),
                  S("n2", "arithmetic.subtract", ("@n1", "baseline"),
                    "gain the plan predicted"),
                  S("n3", "arithmetic.subtract", ("later_value", "baseline"),
                    "gain the business really made"),
                  S("n4", "arithmetic.abs_difference", ("@n2", "@n3"),
                    "the forecast error"),
                  S("n5", "rates.share_percent", ("@n4", "@n2")),
                  S("n6", "comparison.at_least", ("@n5", "error_share_limit"))),
                 "n6", intent="forecast_error_verdict"),
            Plan("growth.v10", (baseline, growth_rate, periods, later_value,
                                target_value, error_share_limit, growth_floor),
                 (S("n1", "rates.compound_growth", ("baseline", "growth_rate",
                                                    "periods"),
                    "the projection, tested again right at the end"),
                  S("n2", "arithmetic.subtract", ("@n1", "baseline")),
                  S("n3", "arithmetic.subtract", ("later_value", "baseline")),
                  S("n4", "arithmetic.abs_difference", ("@n2", "@n3")),
                  S("n5", "rates.share_percent", ("@n4", "@n2")),
                  S("n6", "rates.percent_change", ("baseline", "later_value"),
                    "the growth that actually happened"),
                  S("n7", "comparison.at_least", ("@n5", "error_share_limit")),
                  S("n8", "comparison.at_least", ("@n6", "growth_floor")),
                  S("n9", "comparison.at_least", ("@n1", "target_value")),
                  S("n10", "decision.any_of", ("@n7", "@n8", "@n9"),
                    "any one of these is enough to keep the plan on the table")),
                 "n10", intent="growth_plan_review"),
        )))

    return out
