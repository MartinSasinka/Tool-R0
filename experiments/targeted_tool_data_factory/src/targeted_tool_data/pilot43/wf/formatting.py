"""Formatting workflows: everything here ends in something a human reads.

The families differ in what has to be rendered and how much of the number
survives the rendering. An invoice line renders money and a share side by side;
a rounded amount is deliberately formatted, read back as a number and compared
with the amount it came from, which is the one place in the pilot where the
round trip number -> text -> number is the point; a code column renders a parsed
serial into a fixed-width entry; a summary line joins labels and totals into one
printed line. The padded plans always pad last so the printed width is the
answer's own width.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── invoice line render ─────────────────────────────────────────────
    item_label = R("item_label", "text_label", "the item the line is raised for")
    unit_price = R("unit_price", "money_price", "price of a single unit")
    quantity = R("quantity", "quantity_units", "units on the line")
    currency = R("currency", "currency_code", "currency the invoice is issued in")
    tax_rate = R("tax_rate", "percent_tax", "tax rate applied to the line")
    discount_rate = R("discount_rate", "percent_discount",
                      "discount agreed for the line")
    places = R("places", "places", "how many decimals the printed share shows")
    line_width = R("line_width", "count_items",
                   "the width the invoice column is printed at")
    fill_char = R("fill_char", "separator",
                  "the character the invoice pads a short line with")

    out.append(Blueprint(
        workflow_id="format.invoice_line_render",
        domain="formatting",
        natural_user_goal="render the printable line for an item on an invoice",
        target_description="the printed invoice line",
        value_generator_id="format.invoice_line",
        query_asset_family="invoice_print_line",
        hard_distractor_families=("format", "rates"),
        entity_family="finance",
        plans=(
            Plan("cur.v3", (item_label, unit_price, currency),
                 (S("n1", "string.normalize_title", ("item_label",)),
                  S("n2", "format.currency", ("unit_price", "currency")),
                  S("n3", "string.concat", ("@n1", "@n2"),
                    "the two independently rendered halves of the line")),
                 "n3", intent="unit_price_line"),
            Plan("cur.v5", (item_label, unit_price, quantity, currency, tax_rate),
                 (S("n1", "arithmetic.multiply", ("unit_price", "quantity"),
                    "goods value, needed again to isolate the tax"),
                  S("n2", "rates.apply_tax", ("@n1", "tax_rate")),
                  S("n3", "arithmetic.subtract", ("@n2", "@n1"),
                    "the tax the line actually carries"),
                  S("n4", "format.currency", ("@n3", "currency")),
                  S("n5", "string.concat", ("item_label", "@n4"))),
                 "n5", intent="tax_only_line"),
            Plan("cur.v8", (unit_price, quantity, currency, tax_rate,
                            discount_rate, places, line_width, fill_char),
                 (S("n1", "arithmetic.multiply", ("unit_price", "quantity")),
                  S("n2", "rates.decrease_by_percent", ("@n1", "discount_rate"),
                    "discounted goods value, read by two branches"),
                  S("n3", "rates.apply_tax", ("@n2", "tax_rate")),
                  S("n4", "rates.share_percent", ("@n2", "@n3"),
                    "goods share of what the customer pays"),
                  S("n5", "format.percent", ("@n4", "places")),
                  S("n6", "format.currency", ("@n3", "currency")),
                  S("n7", "string.concat", ("@n6", "@n5")),
                  S("n8", "format.pad", ("@n7", "line_width", "fill_char"))),
                 "n8", intent="full_invoice_line"),
        )))

    # ── rounded amount ──────────────────────────────────────────────────
    gross_total = R("gross_total", "money_total",
                    "the amount as the ledger holds it")
    round_places = R("places", "places", "how many decimals the report is kept to")
    unit_name = R("unit_name", "unit_word", "the unit the report labels amounts with")
    round_currency = R("currency", "currency_code",
                       "currency the report is issued in")
    round_tax = R("tax_rate", "percent_tax", "tax rate the report applies first")

    out.append(Blueprint(
        workflow_id="format.rounded_amount",
        domain="formatting",
        natural_user_goal=("see what rounding an amount for the report actually "
                           "costs before the report goes out"),
        target_description="the rounded label or the rounding error",
        value_generator_id="format.report_amount",
        query_asset_family="ledger_report_amount",
        hard_distractor_families=("format", "rounding"),
        entity_family="finance",
        plans=(
            Plan("round.v2", (gross_total, round_places, unit_name),
                 (S("n1", "format.fixed", ("gross_total", "places")),
                  S("n2", "string.concat", ("@n1", "unit_name"))),
                 "n2", intent="rounded_label"),
            Plan("round.v4", (gross_total, round_places),
                 (S("n1", "format.fixed", ("gross_total", "places")),
                  S("n2", "string.parse_number", ("@n1",),
                    "the printed amount read back as a number"),
                  S("n3", "arithmetic.subtract", ("gross_total", "@n2")),
                  S("n4", "rates.share_percent", ("@n3", "gross_total"))),
                 "n4", intent="rounding_error_share"),
            Plan("round.v6", (gross_total, round_places, round_currency, round_tax),
                 (S("n1", "rates.apply_tax", ("gross_total", "tax_rate"),
                    "the taxed amount, compared against its own printed form"),
                  S("n2", "format.fixed", ("@n1", "places")),
                  S("n3", "string.parse_number", ("@n2",)),
                  S("n4", "arithmetic.subtract", ("@n1", "@n3"),
                    "what rounding removed"),
                  S("n5", "format.currency", ("@n3", "currency")),
                  S("n6", "format.tag", ("@n5", "@n4"),
                    "the printed amount carrying its own rounding error")),
                 "n6", intent="rounded_amount_with_error"),
        )))

    # ── code column render ──────────────────────────────────────────────
    asset_code = R("asset_code", "identifier_code",
                   "the code stencilled on the asset")
    code_fill = R("fill_char", "separator",
                  "the character the column is padded with")
    code_width = R("column_width", "count_items",
                   "the width the register column is printed at")
    code_places = R("places", "places", "how many decimals the serial is printed to")
    code_price = R("unit_price", "money_price", "price booked against one serial")
    code_currency = R("currency", "currency_code", "currency the register uses")

    out.append(Blueprint(
        workflow_id="format.code_column_render",
        domain="formatting",
        natural_user_goal=("render an asset code into the fixed-width column the "
                           "register prints"),
        target_description="the rendered register column entry",
        value_generator_id="format.register_column",
        query_asset_family="register_column_entry",
        hard_distractor_families=("format", "string"),
        entity_family="facilities",
        plans=(
            Plan("pad.v3", (asset_code, code_width, code_fill),
                 (S("n1", "string.normalize_slug", ("asset_code",)),
                  S("n2", "string.normalize_upper", ("@n1",)),
                  S("n3", "format.pad", ("@n2", "column_width", "fill_char"))),
                 "n3", intent="padded_code"),
            Plan("pad.v6", (asset_code, code_places, code_width, code_fill),
                 (S("n1", "string.normalize_upper", ("asset_code",),
                    "the register form, needed again at the merge"),
                  S("n2", "string.extract_digits", ("asset_code",)),
                  S("n3", "string.parse_number", ("@n2",)),
                  S("n4", "format.fixed", ("@n3", "places")),
                  S("n5", "string.concat", ("@n1", "@n4"),
                    "the code and the serial it hides, four calls apart"),
                  S("n6", "format.pad", ("@n5", "column_width", "fill_char"))),
                 "n6", intent="padded_code_with_serial"),
            Plan("pad.v9", (asset_code, code_price, code_currency, code_fill),
                 (S("n1", "string.normalize_upper", ("asset_code",),
                    "the register form, read by three branches"),
                  S("n2", "string.extract_digits", ("@n1",)),
                  S("n3", "string.parse_number", ("@n2",)),
                  S("n4", "arithmetic.multiply", ("@n3", "unit_price")),
                  S("n5", "format.currency", ("@n4", "currency")),
                  S("n6", "string.count_length", ("@n1",),
                    "the column width the code itself dictates"),
                  S("n7", "format.pad", ("@n5", "@n6", "fill_char")),
                  S("n8", "string.concat", ("@n1", "@n7")),
                  S("n9", "format.tag", ("@n8", "@n6"))),
                 "n9", intent="register_row_for_code"),
        )))

    # ── summary line ────────────────────────────────────────────────────
    labels = R("labels", "text_list_labels", "the items the summary covers")
    joiner = R("joiner", "separator", "the character the summary lists items with")
    amounts = R("amounts", "list_prices", "what each item on the summary cost")
    sum_currency = R("currency", "currency_code", "currency the summary is written in")
    sum_places = R("places", "places", "how many decimals the summary shows")
    sum_tax = R("tax_rate", "percent_tax", "tax rate the summary adds at the end")

    out.append(Blueprint(
        workflow_id="format.summary_line",
        domain="formatting",
        natural_user_goal=("build the one-line summary that goes at the bottom of "
                           "a spend report"),
        target_description="the printed summary line",
        value_generator_id="format.spend_summary",
        query_asset_family="spend_report_footer",
        hard_distractor_families=("format", "list"),
        entity_family="finance",
        plans=(
            Plan("sum.v4", (labels, joiner, amounts, sum_currency),
                 (S("n1", "list.reduce_sum", ("amounts",)),
                  S("n2", "format.currency", ("@n1", "currency")),
                  S("n3", "list.combine_join_text", ("labels", "joiner")),
                  S("n4", "string.concat", ("@n3", "@n2"))),
                 "n4", intent="items_and_total"),
            Plan("sum.v7", (labels, joiner, amounts, sum_places),
                 (S("n1", "list.combine_join_text", ("labels", "joiner")),
                  S("n2", "string.normalize_title", ("@n1",),
                    "the item list, joined back on five calls later"),
                  S("n3", "list.reduce_sum", ("amounts",)),
                  S("n4", "list.reduce_max", ("amounts",)),
                  S("n5", "rates.share_percent", ("@n4", "@n3"),
                    "how concentrated the spend is on one item"),
                  S("n6", "format.percent", ("@n5", "places")),
                  S("n7", "string.concat", ("@n2", "@n6"))),
                 "n7", intent="items_and_concentration"),
            Plan("sum.v10", (labels, joiner, amounts, sum_currency, sum_places,
                             sum_tax),
                 (S("n1", "list.combine_join_text", ("labels", "joiner")),
                  S("n2", "string.normalize_title", ("@n1",),
                    "the item list, measured two ways"),
                  S("n3", "string.count_words", ("@n2",)),
                  S("n4", "string.count_length", ("@n2",)),
                  S("n5", "list.reduce_sum", ("amounts",)),
                  S("n6", "rates.apply_tax", ("@n5", "tax_rate")),
                  S("n7", "format.currency", ("@n6", "currency")),
                  S("n8", "rates.ratio_of", ("@n4", "@n3"),
                    "characters per word of the item list"),
                  S("n9", "format.fixed", ("@n8", "places")),
                  S("n10", "string.concat", ("@n7", "@n9"))),
                 "n10", intent="taxed_total_with_list_width"),
        )))

    # ── report line ─────────────────────────────────────────────────────
    # The families below render the same primitives against different assets:
    # a monthly report line, a share of a project budget, and an asset column.
    rep_label = R("item_label", "text_label", "what the line item is called")
    rep_price = R("unit_price", "money_price", "the price of one unit")
    rep_units = R("units", "quantity_units", "how many units the line covers")
    rep_currency = R("currency", "currency_code", "the currency the report is issued in")
    rep_width = R("column_width", "count_items", "how wide the report column is")
    rep_fill = R("fill_character", "separator",
                 "the character short entries are padded with")

    out.append(Blueprint(
        workflow_id="formatting.report_line",
        domain="formatting",
        natural_user_goal="write one line of the monthly report the way it is printed",
        target_description="the printed report line",
        value_generator_id="formatting.report",
        query_asset_family="monthly_report_line",
        hard_distractor_families=("format", "string"),
        entity_family="finance",
        plans=(
            Plan("rep.v2", (rep_price, rep_units, rep_currency),
                 (S("n1", "arithmetic.multiply", ("unit_price", "units")),
                  S("n2", "format.currency", ("@n1", "currency"))),
                 "n2", intent="line_amount"),
            Plan("rep.v4", (rep_label, rep_price, rep_units, rep_currency),
                 (S("n1", "arithmetic.multiply", ("unit_price", "units")),
                  S("n2", "format.currency", ("@n1", "currency")),
                  S("n3", "string.normalize_title", ("item_label",),
                    "the item name as the report writes it"),
                  S("n4", "string.concat", ("@n3", "@n2"))),
                 "n4", intent="report_line"),
            Plan("rep.v6", (rep_label, rep_price, rep_units, rep_currency,
                            rep_width, rep_fill),
                 (S("n1", "arithmetic.multiply", ("unit_price", "units")),
                  S("n2", "format.currency", ("@n1", "currency")),
                  S("n3", "string.normalize_title", ("item_label",)),
                  S("n4", "string.concat", ("@n3", "@n2")),
                  S("n5", "string.normalize_slug", ("@n4",)),
                  S("n6", "format.pad", ("@n5", "column_width", "fill_character"))),
                 "n6", intent="padded_report_key"),
        )))

    # ── share summary ───────────────────────────────────────────────────
    part = R("component_cost", "money_price", "what this component costs")
    whole = R("project_cost", "money_total", "what the whole project costs")
    share_places = R("places", "places", "how many decimals the summary shows")
    share_cut = R("target_share", "threshold_ratio",
                  "the proportion the component is supposed to stay under")

    out.append(Blueprint(
        workflow_id="formatting.share_summary",
        domain="formatting",
        natural_user_goal=("write up how much of the project budget one component "
                           "eats, the way the summary sheet shows it"),
        target_description="the rendered share and its verdict",
        value_generator_id="formatting.share",
        query_asset_family="cost_summary_cell",
        hard_distractor_families=("rates", "format"),
        entity_family="finance",
        plans=(
            Plan("pct.v2", (part, whole, share_places),
                 (S("n1", "rates.share_percent", ("component_cost", "project_cost")),
                  S("n2", "format.percent", ("@n1", "places"))),
                 "n2", intent="rendered_share"),
            Plan("pct.v4", (part, whole),
                 (S("n1", "rates.share_percent", ("component_cost", "project_cost")),
                  S("n2", "format.number_text", ("@n1",),
                    "the share as plain text, measured and reused"),
                  S("n3", "string.count_length", ("@n2",)),
                  S("n4", "format.tag", ("@n2", "@n3"))),
                 "n4", intent="share_cell"),
            Plan("pct.v6", (part, whole, share_places, share_cut),
                 (S("n1", "rates.share_percent", ("component_cost", "project_cost")),
                  S("n2", "format.percent", ("@n1", "places")),
                  S("n3", "rates.ratio_of", ("component_cost", "project_cost"),
                    "the same relation read as a proportion"),
                  S("n4", "classification.ratio_band", ("@n3", "target_share")),
                  S("n5", "string.concat", ("@n2", "@n4")),
                  S("n6", "string.normalize_upper", ("@n5",))),
                 "n6", intent="share_with_verdict"),
            Plan("pct.v8", (part, whole, share_places, share_cut),
                 (S("n1", "rates.share_percent", ("component_cost", "project_cost"),
                    "the share, rendered again five calls later"),
                  S("n2", "format.fixed", ("@n1", "places")),
                  S("n3", "string.parse_number", ("@n2",),
                    "read the rounded share back as a number"),
                  S("n4", "rates.ratio_of", ("component_cost", "project_cost")),
                  S("n5", "classification.ratio_band", ("@n4", "target_share")),
                  S("n6", "format.percent", ("@n1", "places")),
                  S("n7", "string.concat", ("@n6", "@n5")),
                  S("n8", "format.tag", ("@n7", "@n3"))),
                 "n8", intent="summary_sheet_cell"),
        )))

    # ── stock label ─────────────────────────────────────────────────────
    lbl_code = R("asset_code", "identifier_code", "the code printed on the asset")
    lbl_units = R("units", "quantity_units", "how many units the label covers")
    lbl_unit = R("unit", "unit_word", "the unit the label is written in")
    lbl_fill = R("fill_character", "separator",
                 "the character the label is padded with")
    lbl_places = R("places", "places", "how many decimals the label shows")

    out.append(Blueprint(
        workflow_id="formatting.stock_label",
        domain="formatting",
        natural_user_goal=("print the shelf label that carries the asset code and "
                           "what is stored under it"),
        target_description="the printed shelf label",
        value_generator_id="formatting.shelf_label",
        query_asset_family="shelf_label",
        hard_distractor_families=("format", "string"),
        entity_family="warehouse",
        plans=(
            Plan("lbl.v3", (lbl_code, lbl_units, lbl_unit),
                 (S("n1", "string.normalize_upper", ("asset_code",)),
                  S("n2", "format.with_unit", ("units", "unit")),
                  S("n3", "string.concat", ("@n1", "@n2"))),
                 "n3", intent="shelf_label"),
            Plan("lbl.v6", (lbl_code, lbl_fill, lbl_places),
                 (S("n1", "string.extract_digits", ("asset_code",)),
                  S("n2", "string.parse_number", ("@n1",)),
                  S("n3", "format.fixed", ("@n2", "places")),
                  S("n4", "string.count_length", ("asset_code",),
                    "the label is as wide as the code itself"),
                  S("n5", "format.pad", ("@n3", "@n4", "fill_character")),
                  S("n6", "string.concat", ("asset_code", "@n5"))),
                 "n6", intent="code_with_aligned_serial"),
            Plan("lbl.v7", (lbl_code, lbl_fill),
                 (S("n1", "string.normalize_upper", ("asset_code",),
                    "the printed code, measured and reused at the end"),
                  S("n2", "string.extract_digits", ("asset_code",)),
                  S("n3", "string.parse_number", ("@n2",)),
                  S("n4", "format.number_text", ("@n3",)),
                  S("n5", "string.count_length", ("@n1",)),
                  S("n6", "format.pad", ("@n4", "@n5", "fill_character")),
                  S("n7", "string.concat", ("@n1", "@n6"))),
                 "n7", intent="aligned_label_entry"),
        )))

    return out
