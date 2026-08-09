"""Commerce workflows: pricing, discounting, invoicing, supplier comparison.

Reference module for the blueprint DSL. Note how the long plans get their
structure from real reuse (a discounted subtotal is needed twice) rather than
from a pattern label, and how boolean sinks always end in a calibratable
predicate so the True/False share can be balanced.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── order settlement ────────────────────────────────────────────────
    settle_roles = (
        R("unit_price", "money_price", "price of a single unit"),
        R("quantity", "quantity_units", "number of units ordered"),
        R("shipping_fee", "money_fee", "flat shipping fee"),
        R("discount_rate", "percent_discount", "agreed discount rate"),
        R("tax_rate", "percent_tax", "applicable tax rate"),
        R("budget", "threshold_money", "budget the order must stay within"),
        R("share_limit", "threshold_percent",
          "maximum share of the total the goods may represent"),
    )
    out.append(Blueprint(
        workflow_id="commerce.order_settlement",
        domain="commerce",
        natural_user_goal=("work out what an order finally costs and whether it "
                           "stays inside the agreed budget"),
        target_description="the settled order total or the budget verdict",
        value_generator_id="commerce.order",
        query_asset_family="purchase_order",
        hard_distractor_families=("rates", "arithmetic"),
        boolean_balancing_strategy="calibrate_budget_threshold",
        entity_family="procurement",
        plans=(
            Plan("settle.v2", settle_roles[:2] + settle_roles[4:5],
                 (S("n1", "arithmetic.multiply", ("unit_price", "quantity"),
                    "line total before tax"),
                  S("n2", "rates.apply_tax", ("@n1", "tax_rate"),
                    "line total including tax")),
                 "n2", intent="amount_after_tax"),
            Plan("settle.v3", settle_roles[:2] + settle_roles[4:6],
                 (S("n1", "arithmetic.multiply", ("unit_price", "quantity")),
                  S("n2", "rates.apply_tax", ("@n1", "tax_rate")),
                  S("n3", "comparison.at_least", ("@n2", "budget"),
                    "budget verdict")),
                 "n3", intent="budget_verdict"),
            Plan("settle.v5", settle_roles[:6],
                 (S("n1", "arithmetic.multiply", ("unit_price", "quantity")),
                  S("n2", "rates.decrease_by_percent", ("@n1", "discount_rate")),
                  S("n3", "arithmetic.add", ("@n2", "shipping_fee")),
                  S("n4", "rates.apply_tax", ("@n3", "tax_rate")),
                  S("n5", "comparison.at_least", ("@n4", "budget"))),
                 "n5", intent="budget_verdict_full"),
            Plan("settle.v8", settle_roles,
                 (S("n1", "arithmetic.multiply", ("unit_price", "quantity")),
                  S("n2", "rates.decrease_by_percent", ("@n1", "discount_rate"),
                    "discounted goods value, needed twice"),
                  S("n3", "arithmetic.add", ("@n2", "shipping_fee")),
                  S("n4", "rates.apply_tax", ("@n3", "tax_rate")),
                  S("n5", "rates.share_percent", ("@n2", "@n4"),
                    "goods share of the taxed total"),
                  S("n6", "comparison.at_least", ("@n4", "budget")),
                  S("n7", "comparison.at_least", ("@n5", "share_limit")),
                  S("n8", "boolean.and", ("@n6", "@n7"),
                    "both conditions must hold")),
                 "n8", intent="two_condition_settlement"),
        )))

    # ── basket discount audit ───────────────────────────────────────────
    basket_roles = (
        R("line_prices", "list_prices", "price of every line in the basket"),
        R("discount_rate", "percent_discount", "discount applied to each line"),
        R("saving_target", "threshold_money", "saving the buyer wants to reach"),
        R("share_limit", "threshold_percent",
          "share of the saving the largest line may represent"),
    )
    out.append(Blueprint(
        workflow_id="commerce.basket_discount_audit",
        domain="commerce",
        natural_user_goal=("check how much a basket-wide discount saves and how "
                           "concentrated that saving is"),
        target_description="the saving spread or the saving verdict",
        value_generator_id="commerce.basket",
        query_asset_family="shopping_basket",
        hard_distractor_families=("list", "rates"),
        boolean_balancing_strategy="calibrate_saving_threshold",
        entity_family="retail",
        plans=(
            Plan("basket.v4", basket_roles[:2],
                 (S("n1", "list.map_percent", ("line_prices", "discount_rate"),
                    "per-line saving, consumed by two branches"),
                  S("n2", "list.reduce_sum", ("@n1",)),
                  S("n3", "list.reduce_max", ("@n1",)),
                  S("n4", "arithmetic.subtract", ("@n2", "@n3"),
                    "saving outside the largest line")),
                 "n4", intent="saving_spread"),
            Plan("basket.v6", (basket_roles[0], basket_roles[1],
                               basket_roles[3]),
                 (S("n1", "list.map_percent", ("line_prices", "discount_rate")),
                  S("n2", "list.reduce_sum", ("@n1",)),
                  S("n3", "list.reduce_max", ("@n1",)),
                  S("n4", "arithmetic.subtract", ("@n2", "@n3")),
                  S("n5", "rates.share_percent", ("@n4", "@n2"),
                    "share of the saving outside the largest line"),
                  S("n6", "comparison.at_least", ("@n5", "share_limit"))),
                 "n6", intent="saving_concentration_verdict"),
            Plan("basket.v8", basket_roles,
                 (S("n1", "list.map_percent", ("line_prices", "discount_rate")),
                  S("n2", "list.reduce_sum", ("@n1",)),
                  S("n3", "list.reduce_max", ("@n1",)),
                  S("n4", "arithmetic.subtract", ("@n2", "@n3")),
                  S("n5", "rates.share_percent", ("@n3", "@n2")),
                  S("n6", "comparison.at_least", ("@n4", "saving_target")),
                  S("n7", "comparison.at_least", ("@n5", "share_limit")),
                  S("n8", "boolean.or", ("@n6", "@n7"))),
                 "n8", intent="either_condition_saving"),
        )))

    # ── invoice line report (numbers -> text) ───────────────────────────
    invoice_roles = (
        R("item_name", "text_label", "name of the invoiced item"),
        R("unit_price", "money_price", "price per unit"),
        R("quantity", "quantity_units", "units invoiced"),
        R("currency", "currency_code", "currency the invoice is issued in"),
        R("tax_rate", "percent_tax", "tax rate on the line"),
    )
    out.append(Blueprint(
        workflow_id="commerce.invoice_line_report",
        domain="commerce",
        natural_user_goal="produce the printable invoice line for an item",
        target_description="the formatted invoice line",
        value_generator_id="commerce.invoice",
        query_asset_family="invoice_line",
        hard_distractor_families=("format", "string"),
        entity_family="finance",
        plans=(
            Plan("invoice.v3", invoice_roles[:4],
                 (S("n1", "arithmetic.multiply", ("unit_price", "quantity")),
                  S("n2", "format.currency", ("@n1", "currency")),
                  S("n3", "string.concat", ("item_name", "@n2"))),
                 "n3", intent="line_label"),
            Plan("invoice.v5", invoice_roles,
                 (S("n1", "arithmetic.multiply", ("unit_price", "quantity")),
                  S("n2", "rates.apply_tax", ("@n1", "tax_rate")),
                  S("n3", "format.currency", ("@n2", "currency")),
                  S("n4", "string.normalize_title", ("item_name",)),
                  S("n5", "string.concat", ("@n4", "@n3"))),
                 "n5", intent="line_label_taxed"),
        )))

    # ── supplier price comparison ──────────────────────────────────────
    supplier_roles = (
        R("price_a", "money_price", "unit price of the first supplier"),
        R("price_b", "money_price", "unit price of the second supplier"),
        R("quantity", "quantity_units", "units to be purchased"),
        R("discount_rate", "percent_discount",
          "discount the first supplier grants"),
        R("small_gap", "cut_low", "gap that still counts as a small difference"),
        R("large_gap", "cut_high", "gap that counts as a large difference"),
    )
    out.append(Blueprint(
        workflow_id="commerce.supplier_price_comparison",
        domain="commerce",
        natural_user_goal="decide how far apart two supplier quotes really are",
        target_description="the size band of the quote difference",
        value_generator_id="commerce.supplier",
        query_asset_family="supplier_quote",
        hard_distractor_families=("comparison", "rates"),
        entity_family="procurement",
        plans=(
            Plan("supplier.v4", supplier_roles[:4],
                 (S("n1", "arithmetic.multiply", ("price_a", "quantity")),
                  S("n2", "arithmetic.multiply", ("price_b", "quantity")),
                  S("n3", "rates.decrease_by_percent", ("@n1", "discount_rate")),
                  S("n4", "arithmetic.abs_difference", ("@n3", "@n2"))),
                 "n4", intent="quote_gap"),
            Plan("supplier.v6", supplier_roles,
                 (S("n1", "arithmetic.multiply", ("price_a", "quantity")),
                  S("n2", "arithmetic.multiply", ("price_b", "quantity")),
                  S("n3", "rates.decrease_by_percent", ("@n1", "discount_rate")),
                  S("n4", "arithmetic.abs_difference", ("@n3", "@n2")),
                  S("n5", "rates.share_percent", ("@n4", "@n2")),
                  S("n6", "classification.three_bands",
                    ("@n5", "small_gap", "large_gap"))),
                 "n6", intent="quote_gap_band"),
        )))

    return out
