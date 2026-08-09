"""Personal finance: monthly budgets, credit instalments, savings goals, subscriptions.

Every role here is a fact a household actually states -- take-home pay, a
category-by-category spending plan, the fee on a loan, what is already in the
savings pot, the charges on a subscription statement -- so the money, percentage
and count hints carry the realistic ranges and nothing is a bare generic value.

The plans of a blueprint differ in how far the household has got with the
question. The short ones answer the arithmetic of the account (what is left, how
long the goal takes); the long ones derive two independent tests from the same
aggregate -- affordability and concentration -- and end in a verdict whose
threshold is calibrated after execution so the yes/no split stays balanced.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── monthly budget headroom ─────────────────────────────────────────
    budget_roles = (
        R("take_home", "money_total", "pay that reaches the current account"),
        R("essential_spend", "mapping_amounts",
          "amount the plan commits to each essential category"),
        R("savings_rate", "percent_share",
          "share of the pay the savings rule sets aside"),
        R("buffer_target", "threshold_money",
          "buffer the household wants left at the end of the month"),
        R("headroom_target", "threshold_money",
          "money that has to be free once the essentials are paid"),
    )
    out.append(Blueprint(
        workflow_id="finance.budget_headroom",
        domain="personal_finance",
        natural_user_goal=("see what a monthly budget leaves over once the "
                           "essentials and the savings rule are honoured"),
        target_description=("the discretionary share, the heavy categories, or "
                            "the headroom verdict"),
        value_generator_id="finance.budget",
        query_asset_family="monthly_budget",
        hard_distractor_families=("dictionary", "rates"),
        boolean_balancing_strategy="threshold_band",
        entity_family="household",
        plans=(
            Plan("headroom.v3", budget_roles[1:2],
                 (S("n1", "dictionary.values", ("essential_spend",)),
                  S("n2", "statistics.mean", ("@n1",),
                    "average commitment per category"),
                  S("n3", "dictionary.aggregate_filter",
                    ("essential_spend", "@n2"),
                    "the categories running above that average")),
                 "n3", intent="heavy_categories"),
            Plan("headroom.v5", budget_roles[:3],
                 (S("n1", "dictionary.aggregate_sum", ("essential_spend",)),
                  S("n2", "arithmetic.subtract", ("take_home", "@n1"),
                    "money left once the essentials are paid"),
                  S("n3", "rates.percent_of", ("savings_rate", "take_home"),
                    "what the savings rule takes"),
                  S("n4", "arithmetic.subtract", ("@n2", "@n3")),
                  S("n5", "rates.share_percent", ("@n4", "take_home"),
                    "share of the pay that stays discretionary")),
                 "n5", intent="discretionary_share"),
            Plan("headroom.v7", budget_roles,
                 (S("n1", "dictionary.aggregate_sum", ("essential_spend",)),
                  S("n2", "arithmetic.subtract", ("take_home", "@n1"),
                    "money after the essentials, tested again four calls later"),
                  S("n3", "rates.percent_of", ("savings_rate", "take_home")),
                  S("n4", "arithmetic.subtract", ("@n2", "@n3")),
                  S("n5", "comparison.at_least", ("@n4", "buffer_target")),
                  S("n6", "comparison.at_least", ("@n2", "headroom_target")),
                  S("n7", "boolean.and", ("@n5", "@n6"),
                    "buffer and headroom both hold")),
                 "n7", intent="headroom_verdict"),
        )))

    # ── loan instalment plan ────────────────────────────────────────────
    loan_roles = (
        R("loan_amount", "money_total", "sum the household wants to borrow"),
        R("annual_rate", "percent_growth", "interest rate quoted for the year"),
        R("arrangement_fee", "money_fee", "one-off arrangement fee"),
        R("term_months", "count_items", "months the loan runs for"),
        R("currency", "currency_code", "currency the loan is written in"),
        R("take_home", "money_total", "pay that reaches the current account"),
        R("disposable_rate", "percent_share",
          "share of the pay the lender treats as disposable"),
        R("instalment_cap", "threshold_money",
          "instalment the household refuses to go above"),
        R("cost_share_limit", "threshold_percent",
          "cost of credit the household tolerates as a share of the sum borrowed"),
        R("affordability_floor", "threshold_money",
          "disposable income the lender wants to see"),
    )
    out.append(Blueprint(
        workflow_id="finance.loan_instalment_plan",
        domain="personal_finance",
        natural_user_goal=("work out the monthly instalment on a loan and whether "
                           "it is affordable"),
        target_description="the instalment, its cost content, or the lending verdict",
        value_generator_id="finance.loan",
        query_asset_family="loan_illustration",
        hard_distractor_families=("rates", "format"),
        boolean_balancing_strategy="threshold_band",
        entity_family="household",
        plans=(
            Plan("instalment.v4", loan_roles[:5],
                 (S("n1", "rates.increase_by_percent",
                    ("loan_amount", "annual_rate")),
                  S("n2", "arithmetic.add", ("@n1", "arrangement_fee")),
                  S("n3", "arithmetic.divide", ("@n2", "term_months")),
                  S("n4", "format.currency", ("@n3", "currency"),
                    "instalment written the way the offer letter states it")),
                 "n4", intent="instalment_label"),
            Plan("instalment.v6", loan_roles[:4],
                 (S("n1", "rates.increase_by_percent",
                    ("loan_amount", "annual_rate")),
                  S("n2", "arithmetic.add", ("@n1", "arrangement_fee"),
                    "total repayable, split into cost and instalment later"),
                  S("n3", "arithmetic.subtract", ("@n2", "loan_amount"),
                    "cost of the credit"),
                  S("n4", "arithmetic.divide", ("@n3", "term_months")),
                  S("n5", "arithmetic.divide", ("@n2", "term_months")),
                  S("n6", "rates.share_percent", ("@n4", "@n5"),
                    "share of each instalment that is pure cost")),
                 "n6", intent="cost_content_of_instalment"),
            Plan("instalment.v10", loan_roles[:4] + loan_roles[5:],
                 (S("n1", "rates.increase_by_percent",
                    ("loan_amount", "annual_rate")),
                  S("n2", "arithmetic.add", ("@n1", "arrangement_fee")),
                  S("n3", "arithmetic.divide", ("@n2", "term_months"),
                    "the monthly instalment"),
                  S("n4", "arithmetic.subtract", ("@n2", "loan_amount")),
                  S("n5", "rates.share_percent", ("@n4", "loan_amount")),
                  S("n6", "rates.percent_of", ("disposable_rate", "take_home"),
                    "disposable income the lender works with"),
                  S("n7", "comparison.at_least", ("@n3", "instalment_cap")),
                  S("n8", "comparison.at_least", ("@n5", "cost_share_limit")),
                  S("n9", "comparison.at_least", ("@n6", "affordability_floor")),
                  S("n10", "decision.majority", ("@n7", "@n8", "@n9"),
                    "most of the lending tests point the same way")),
                 "n10", intent="lending_decision"),
        )))

    # ── savings goal ────────────────────────────────────────────────────
    goal_roles = (
        R("goal_amount", "money_budget", "sum the household is saving towards"),
        R("saved_so_far", "money_total", "amount already in the savings pot"),
        R("monthly_saving", "money_fee", "amount paid into the pot each month"),
        R("interest_rate", "percent_growth", "interest the pot earns per year"),
        R("interest_years", "count_small",
          "years of interest the bank has already confirmed"),
        R("months_limit", "threshold_value",
          "longest the household is prepared to keep saving"),
        R("gap_share_limit", "threshold_percent",
          "share of the original gap that may still be open"),
    )
    out.append(Blueprint(
        workflow_id="finance.savings_goal_tracker",
        domain="personal_finance",
        natural_user_goal=("find out how long a savings goal still takes once the "
                           "interest on the pot is counted"),
        target_description="the months still needed or the at-risk verdict",
        value_generator_id="finance.savings",
        query_asset_family="savings_plan",
        hard_distractor_families=("arithmetic", "rates"),
        boolean_balancing_strategy="threshold_band",
        entity_family="household",
        plans=(
            Plan("goal.v2", goal_roles[:3],
                 (S("n1", "arithmetic.subtract",
                    ("goal_amount", "saved_so_far")),
                  S("n2", "arithmetic.divide", ("@n1", "monthly_saving"),
                    "months of contributions the gap needs")),
                 "n2", intent="months_at_current_rate"),
            Plan("goal.v5", goal_roles[:5],
                 (S("n1", "arithmetic.subtract",
                    ("goal_amount", "saved_so_far"),
                    "gap if the pot earned nothing"),
                  S("n2", "rates.compound_growth",
                    ("saved_so_far", "interest_rate", "interest_years")),
                  S("n3", "arithmetic.subtract", ("goal_amount", "@n2"),
                    "gap once the interest is counted"),
                  S("n4", "comparison.min", ("@n1", "@n3"),
                    "the gap that actually has to be saved"),
                  S("n5", "arithmetic.divide", ("@n4", "monthly_saving"))),
                 "n5", intent="months_after_interest"),
            Plan("goal.v9", goal_roles,
                 (S("n1", "arithmetic.subtract",
                    ("goal_amount", "saved_so_far"),
                    "original gap, used again as a yardstick five calls later"),
                  S("n2", "rates.compound_growth",
                    ("saved_so_far", "interest_rate", "interest_years")),
                  S("n3", "arithmetic.subtract", ("goal_amount", "@n2")),
                  S("n4", "comparison.min", ("@n1", "@n3")),
                  S("n5", "arithmetic.divide", ("@n4", "monthly_saving")),
                  S("n6", "rates.share_percent", ("@n4", "@n1"),
                    "share of the original gap the interest leaves open"),
                  S("n7", "comparison.at_least", ("@n5", "months_limit")),
                  S("n8", "comparison.at_least", ("@n6", "gap_share_limit")),
                  S("n9", "boolean.or", ("@n7", "@n8"),
                    "the goal is at risk on either count")),
                 "n9", intent="goal_at_risk_verdict"),
        )))

    # ── subscription statement ──────────────────────────────────────────
    sub_roles = (
        R("charges", "record_list", "one row per subscription on the statement"),
        R("amount_field", "field_name", "column holding the amount to audit"),
        R("tax_rate", "percent_tax", "tax added to each charge"),
        R("annual_rate", "percent_growth", "yearly rise the providers announced"),
        R("rise_periods", "count_small", "years the announced rises run for"),
    )
    out.append(Blueprint(
        workflow_id="finance.subscription_statement_audit",
        domain="personal_finance",
        natural_user_goal=("audit a subscription statement and see what the "
                           "announced price rises will add to it"),
        target_description=("the number of expensive subscriptions or what the "
                            "rises add to the bill"),
        value_generator_id="finance.subscriptions",
        query_asset_family="subscription_statement",
        hard_distractor_families=("record", "rates"),
        entity_family="household",
        plans=(
            Plan("sub.v3", (sub_roles[0], sub_roles[1], sub_roles[3]),
                 (S("n1", "record.aggregate_mean", ("charges", "amount_field")),
                  S("n2", "rates.increase_by_percent", ("@n1", "annual_rate"),
                    "what the average charge becomes after the rise"),
                  S("n3", "record.aggregate_count",
                    ("charges", "amount_field", "@n2"),
                    "subscriptions already dearer than that")),
                 "n3", intent="already_expensive_count"),
            Plan("sub.v6", sub_roles,
                 (S("n1", "record.aggregate_sum", ("charges", "amount_field"),
                    "the whole monthly bill, the yardstick of the last call"),
                  S("n2", "record.aggregate_max", ("charges", "amount_field")),
                  S("n3", "arithmetic.subtract", ("@n1", "@n2"),
                    "what the smaller subscriptions cost together"),
                  S("n4", "rates.apply_tax", ("@n3", "tax_rate")),
                  S("n5", "rates.compound_growth",
                    ("@n4", "annual_rate", "rise_periods")),
                  S("n6", "rates.share_percent", ("@n5", "@n1"),
                    "share of today's bill the smaller ones will reach")),
                 "n6", intent="small_subscription_drift"),
            Plan("sub.v9", sub_roles,
                 (S("n1", "record.aggregate_sum", ("charges", "amount_field")),
                  S("n2", "record.aggregate_mean", ("charges", "amount_field"),
                    "average charge, taxed and counted against separately"),
                  S("n3", "record.aggregate_count",
                    ("charges", "amount_field", "@n2")),
                  S("n4", "rates.apply_tax", ("@n2", "tax_rate")),
                  S("n5", "arithmetic.multiply", ("@n4", "@n3"),
                    "taxed cost of the above-average subscriptions"),
                  S("n6", "rates.compound_growth",
                    ("@n5", "annual_rate", "rise_periods")),
                  S("n7", "arithmetic.subtract", ("@n6", "@n5"),
                    "what the rises add to that block"),
                  S("n8", "rates.apply_tax", ("@n1", "tax_rate"),
                    "today's whole bill including tax"),
                  S("n9", "rates.share_percent", ("@n7", "@n8"))),
                 "n9", intent="rise_share_of_bill"),
        )))

    return out
