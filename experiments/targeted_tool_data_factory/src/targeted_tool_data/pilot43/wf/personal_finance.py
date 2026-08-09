"""Personal-finance workflows: savings projections, budget shares, affordability.

Growth is always applied with :capability:`rates.compound_growth` so the
projected balance really is a compounded figure, and the interest earned is
recovered by differencing the projection against what was paid in -- which is
where the joins in the longer plans come from.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── savings growth ──────────────────────────────────────────────────
    opening_balance = R("opening_balance", "money_total",
                        "money already sitting in the account")
    annual_rate = R("annual_rate", "percent_growth",
                    "interest the account pays each year")
    years = R("years", "count_small", "years the money is left untouched")
    monthly_deposit = R("monthly_deposit", "money_fee",
                        "amount paid in every month")
    months = R("months", "count_items", "months of deposits already made")
    savings_target = R("savings_target", "threshold_money",
                       "amount the saver wants to end up with")
    growth_target = R("growth_target", "threshold_percent",
                      "growth the saver is hoping for")

    out.append(Blueprint(
        workflow_id="personal_finance.savings_growth",
        domain="personal_finance",
        natural_user_goal=("see what a savings pot is likely to be worth later "
                           "and whether that reaches what the saver wants"),
        target_description="the projected balance, its growth or the target verdict",
        value_generator_id="personal_finance.savings",
        query_asset_family="savings_statement",
        hard_distractor_families=("rates", "arithmetic"),
        boolean_balancing_strategy="calibrate_savings_target",
        entity_family="household",
        plans=(
            Plan("savings.v2", (opening_balance, annual_rate, years,
                                savings_target),
                 (S("n1", "rates.compound_growth", ("opening_balance",
                                                    "annual_rate", "years"),
                    "balance after compounding"),
                  S("n2", "comparison.at_least", ("@n1", "savings_target"))),
                 "n2", intent="savings_target_verdict"),
            Plan("savings.v4", (opening_balance, annual_rate, years,
                                monthly_deposit, months),
                 (S("n1", "arithmetic.multiply", ("monthly_deposit", "months"),
                    "everything paid in by hand"),
                  S("n2", "arithmetic.add", ("opening_balance", "@n1")),
                  S("n3", "rates.compound_growth", ("@n2", "annual_rate",
                                                    "years")),
                  S("n4", "rates.percent_change", ("opening_balance", "@n3"),
                    "how far the pot has moved from where it started")),
                 "n4", intent="pot_movement_percent"),
            Plan("savings.v6", (opening_balance, annual_rate, years,
                                monthly_deposit, months, growth_target),
                 (S("n1", "arithmetic.multiply", ("monthly_deposit", "months")),
                  S("n2", "arithmetic.add", ("opening_balance", "@n1"),
                    "total paid in, needed again to isolate the interest"),
                  S("n3", "rates.compound_growth", ("@n2", "annual_rate",
                                                    "years")),
                  S("n4", "arithmetic.subtract", ("@n3", "@n2"),
                    "interest the account added"),
                  S("n5", "rates.share_percent", ("@n4", "@n3")),
                  S("n6", "comparison.at_least", ("@n5", "growth_target"))),
                 "n6", intent="interest_share_verdict"),
            Plan("savings.v8", (opening_balance, annual_rate, years,
                                monthly_deposit, months, savings_target,
                                growth_target),
                 (S("n1", "arithmetic.multiply", ("monthly_deposit", "months")),
                  S("n2", "arithmetic.add", ("opening_balance", "@n1")),
                  S("n3", "rates.compound_growth", ("@n2", "annual_rate",
                                                    "years")),
                  S("n4", "arithmetic.subtract", ("@n3", "@n2")),
                  S("n5", "rates.share_percent", ("@n4", "@n2")),
                  S("n6", "comparison.at_least", ("@n3", "savings_target")),
                  S("n7", "comparison.at_least", ("@n5", "growth_target")),
                  S("n8", "boolean.and", ("@n6", "@n7"),
                    "the pot has to be big enough and to have earned enough")),
                 "n8", intent="savings_double_verdict"),
        )))

    # ── household budget split ──────────────────────────────────────────
    category_amounts = R("category_amounts", "list_prices",
                         "amount spent in each budget category")
    income = R("income", "money_total", "money coming into the household")
    uplift_rate = R("uplift_rate", "percent_growth",
                    "price rise expected in every category")
    essential_share = R("essential_share", "threshold_percent",
                        "share of income the household wants to stay under")
    income_share_limit = R("income_share_limit", "threshold_percent",
                           "share of income the household refuses to go past")
    cut_low = R("cut_low", "cut_low", "share that still counts as comfortable")
    cut_high = R("cut_high", "cut_high", "share that counts as stretched")
    places = R("places", "places", "decimals the summary is written to")
    household_label = R("household_label", "text_label",
                        "name the household files its budget under")

    out.append(Blueprint(
        workflow_id="personal_finance.budget_split",
        domain="personal_finance",
        natural_user_goal=("understand how much of a month's income the "
                           "household's spending swallows and how lopsided it is"),
        target_description="the spending share, its band or the affordability verdict",
        value_generator_id="personal_finance.budget",
        query_asset_family="budget_sheet",
        hard_distractor_families=("list", "rates"),
        boolean_balancing_strategy="calibrate_income_share",
        entity_family="household",
        plans=(
            Plan("budget.v3", (category_amounts, income, essential_share),
                 (S("n1", "list.reduce_sum", ("category_amounts",),
                    "everything spent this month"),
                  S("n2", "rates.share_percent", ("@n1", "income")),
                  S("n3", "comparison.at_least", ("@n2", "essential_share"))),
                 "n3", intent="income_share_verdict"),
            Plan("budget.v5", (category_amounts, income, cut_low, cut_high),
                 (S("n1", "list.reduce_sum", ("category_amounts",)),
                  S("n2", "list.reduce_max", ("category_amounts",),
                    "the single largest category"),
                  S("n3", "arithmetic.subtract", ("@n1", "@n2"),
                    "spend outside the largest category"),
                  S("n4", "rates.share_percent", ("@n3", "income")),
                  S("n5", "classification.three_bands", ("@n4", "cut_low",
                                                         "cut_high"))),
                 "n5", intent="discretionary_band"),
            Plan("budget.v6", (category_amounts, income, uplift_rate, places,
                               household_label),
                 (S("n1", "list.map_percent", ("category_amounts",
                                               "uplift_rate"),
                    "the rise each category will see"),
                  S("n2", "list.combine_pairwise", ("category_amounts", "@n1"),
                    "next month's category amounts"),
                  S("n3", "list.reduce_sum", ("@n2",)),
                  S("n4", "rates.share_percent", ("@n3", "income")),
                  S("n5", "format.percent", ("@n4", "places")),
                  S("n6", "string.concat", ("household_label", "@n5"),
                    "the line for the budget file")),
                 "n6", intent="projected_share_label"),
            Plan("budget.v9", (category_amounts, income, uplift_rate,
                               essential_share, income_share_limit),
                 (S("n1", "list.map_percent", ("category_amounts",
                                               "uplift_rate")),
                  S("n2", "list.combine_pairwise", ("category_amounts", "@n1"),
                    "next month's amounts, read three ways"),
                  S("n3", "list.reduce_sum", ("@n2",),
                    "next month's total, used twice"),
                  S("n4", "list.reduce_max", ("@n2",)),
                  S("n5", "rates.share_percent", ("@n4", "@n3"),
                    "how concentrated the spending is"),
                  S("n6", "rates.share_percent", ("@n3", "income"),
                    "what the whole basket costs relative to pay"),
                  S("n7", "comparison.at_least", ("@n5", "essential_share")),
                  S("n8", "comparison.at_least", ("@n6", "income_share_limit")),
                  S("n9", "boolean.or", ("@n7", "@n8"),
                    "either concentration or the total can trip the review")),
                 "n9", intent="budget_pressure_verdict"),
        )))

    # ── loan affordability ──────────────────────────────────────────────
    loan_amount = R("loan_amount", "money_total", "amount being borrowed")
    loan_rate = R("loan_rate", "percent_growth", "interest charged each year")
    term_years = R("term_years", "count_small", "years the loan runs for")
    term_months = R("term_months", "quantity_units",
                    "monthly payments the loan is spread over")
    monthly_income = R("monthly_income", "money_total", "take-home pay each month")
    existing_costs = R("existing_costs", "money_fee",
                       "other fixed costs already paid every month")
    affordable_share = R("affordable_share", "threshold_percent",
                         "share of pay the lender is willing to see committed")
    payment_cap = R("payment_cap", "threshold_money",
                    "monthly outgoing the borrower can actually manage")
    interest_share_limit = R("interest_share_limit", "threshold_percent",
                             "share of the repayment the borrower accepts as interest")
    band_low = R("band_low", "cut_low", "share that still looks affordable")
    band_high = R("band_high", "cut_high", "share that looks unaffordable")

    out.append(Blueprint(
        workflow_id="personal_finance.loan_affordability",
        domain="personal_finance",
        natural_user_goal=("judge whether the monthly cost of a loan is "
                           "manageable next to what someone actually earns"),
        target_description="the monthly commitment, its band or the lending verdict",
        value_generator_id="personal_finance.loan",
        query_asset_family="loan_offer",
        hard_distractor_families=("rates", "comparison"),
        boolean_balancing_strategy="calibrate_affordability_share",
        entity_family="household",
        plans=(
            Plan("loan.v4", (loan_amount, loan_rate, term_years, term_months,
                             monthly_income, affordable_share),
                 (S("n1", "rates.compound_growth", ("loan_amount", "loan_rate",
                                                    "term_years"),
                    "total amount repayable"),
                  S("n2", "arithmetic.divide", ("@n1", "term_months")),
                  S("n3", "rates.share_percent", ("@n2", "monthly_income")),
                  S("n4", "comparison.at_least", ("@n3", "affordable_share"))),
                 "n4", intent="affordability_verdict"),
            Plan("loan.v5", (loan_amount, loan_rate, term_years, term_months,
                             existing_costs, monthly_income, band_low, band_high),
                 (S("n1", "rates.compound_growth", ("loan_amount", "loan_rate",
                                                    "term_years")),
                  S("n2", "arithmetic.divide", ("@n1", "term_months")),
                  S("n3", "arithmetic.add", ("@n2", "existing_costs"),
                    "everything committed each month"),
                  S("n4", "rates.share_percent", ("@n3", "monthly_income")),
                  S("n5", "classification.three_bands", ("@n4", "band_low",
                                                         "band_high"))),
                 "n5", intent="commitment_band"),
            Plan("loan.v7", (loan_amount, loan_rate, term_years, term_months,
                             existing_costs, monthly_income, affordable_share,
                             payment_cap),
                 (S("n1", "rates.compound_growth", ("loan_amount", "loan_rate",
                                                    "term_years")),
                  S("n2", "arithmetic.divide", ("@n1", "term_months")),
                  S("n3", "arithmetic.add", ("@n2", "existing_costs"),
                    "monthly commitment, tested twice"),
                  S("n4", "rates.share_percent", ("@n3", "monthly_income")),
                  S("n5", "comparison.at_least", ("@n4", "affordable_share")),
                  S("n6", "comparison.greater", ("@n3", "payment_cap")),
                  S("n7", "boolean.xor", ("@n5", "@n6"),
                    "the lender's rule and the borrower's own cap can disagree")),
                 "n7", intent="lender_borrower_disagreement"),
            Plan("loan.v10", (loan_amount, loan_rate, term_years, term_months,
                              existing_costs, monthly_income, affordable_share,
                              payment_cap, interest_share_limit),
                 (S("n1", "rates.compound_growth", ("loan_amount", "loan_rate",
                                                    "term_years"),
                    "total repayable, read again much later"),
                  S("n2", "arithmetic.subtract", ("@n1", "loan_amount"),
                    "interest the loan costs"),
                  S("n3", "arithmetic.divide", ("@n1", "term_months")),
                  S("n4", "arithmetic.add", ("@n3", "existing_costs")),
                  S("n5", "rates.share_percent", ("@n4", "monthly_income")),
                  S("n6", "rates.share_percent", ("@n2", "@n1"),
                    "share of the repayment that is pure interest"),
                  S("n7", "comparison.at_least", ("@n5", "affordable_share")),
                  S("n8", "comparison.at_least", ("@n6",
                                                  "interest_share_limit")),
                  S("n9", "comparison.greater", ("@n4", "payment_cap")),
                  S("n10", "decision.majority", ("@n7", "@n8", "@n9"),
                    "most of the lending rules have to bite")),
                 "n10", intent="lending_majority_verdict"),
        )))

    return out
