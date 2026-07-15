"""synthetic_tools — versioned executable synthetic tool registry (v5).

One authoritative registry containing BOTH the tool schema exposed to the model
AND the deterministic executable implementation used during training. The
generator (lib/synthetic_gen_v5.py) and the trainer executor
(nestful_mtgrpo_minimal/executor.py, mode="synthetic") load this same module;
run manifests record REGISTRY_VERSION and registry_hash().

Design rules:
  * every tool is a safe, reviewed pure-Python function — NO LLM-generated code
    is ever executed;
  * deterministic: same arguments -> same observation, always;
  * semantic types on every parameter and output prevent nonsensical
    compositions (money never flows into a kilograms parameter);
  * heterogeneous output keys (output_0 / result / value) and output types
    (number / integer / boolean / string / array / object) mirror NESTFUL's
    mixed-API style WITHOUT copying any NESTFUL tool;
  * genuinely distinct tools: different formulas, arities, parameter names and
    semantics — families like unit conversions differ in factor, units and
    semantic direction, and the validation report flags exact behavioural
    duplicates.

CONTAMINATION: no NESTFUL questions, gold traces, tool schemas or
implementations are copied. Only aggregate style (naming, $varN.key$ reference
convention, offered-tool counts) informed the design.

Public API:
    REGISTRY_VERSION            semver string
    TOOLS                       {name: spec dict}
    ALL_TOOL_NAMES, FAMILIES, DOMAINS
    tool_schema(name)           NESTFUL-style schema row for a task's `tools`
    registry_hash()             deterministic sha256 over schemas + semantics
    sample_args(name, rng)      literal argument sample for the generator
    semantics_compatible(producer_out_sem, consumer_param_sem)
    validate_registry()         determinism / typing / duplicate report
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

REGISTRY_VERSION = "5.0.0"

TOOLS: Dict[str, Dict[str, Any]] = {}

# Semantic vocabulary. "generic_number" params accept ANY numeric producer;
# a param with a specific semantic accepts ONLY a producer with the SAME
# semantic. Non-numeric semantics: text, flag, list_number, object.
_NON_NUMERIC_SEMS = {"text", "flag", "list_number", "object"}


def _r2(x: float) -> float:
    return round(float(x), 2)


def _p(typ: str, desc: str, semantic: str = "generic_number",
       required: bool = True, **constraints: Any) -> Dict[str, Any]:
    d = {"type": typ, "desc": desc, "semantic": semantic, "required": required}
    d.update(constraints)
    return d


def _add(name: str, domain: str, family: str, description: str,
         params: Dict[str, Dict[str, Any]], out_key: str, out_type: str,
         out_semantic: str, fn: Callable[..., Any],
         sample: Callable[[random.Random], Dict[str, Any]],
         phrase: Callable[[Dict[str, Any]], str],
         chain_in: Optional[str] = None,
         out_fields: Optional[Dict[str, Tuple[str, str]]] = None) -> None:
    if name in TOOLS:
        raise ValueError(f"duplicate tool name: {name}")
    if chain_in is not None and chain_in not in params:
        raise ValueError(f"{name}: chain_in '{chain_in}' not in params")
    TOOLS[name] = {
        "name": name, "domain": domain, "family": family,
        "description": description, "params": params,
        "out_key": out_key, "out_type": out_type, "out_semantic": out_semantic,
        "fn": fn, "sample": sample, "phrase": phrase,
        "chain_in": chain_in, "out_fields": out_fields,
    }


def _mk_fn1(param: str, f: Callable[[Any], Any]) -> Callable[..., Any]:
    def fn(**kw: Any) -> Any:
        return f(kw[param])
    return fn


def _mk_fn2(p1: str, p2: str, f: Callable[[Any, Any], Any]) -> Callable[..., Any]:
    def fn(**kw: Any) -> Any:
        return f(kw[p1], kw[p2])
    return fn


def _mk_fn3(p1: str, p2: str, p3: str,
            f: Callable[[Any, Any, Any], Any]) -> Callable[..., Any]:
    def fn(**kw: Any) -> Any:
        return f(kw[p1], kw[p2], kw[p3])
    return fn


def _rng_num(lo: float, hi: float, choices=None):
    def sample(rng: random.Random) -> Any:
        if choices is not None:
            return rng.choice(choices)
        if isinstance(lo, int) and isinstance(hi, int):
            return rng.randrange(lo, hi)
        return _r2(rng.uniform(lo, hi))
    return sample


# ─────────────────────────────────────────────────────────────────────────────
#  Family: unit conversions (unary, factor-based; distinct units + semantics)
# ─────────────────────────────────────────────────────────────────────────────
def _add_unary(name: str, domain: str, family: str, desc: str, param: str,
               psem: str, out_key: str, out_sem: str,
               f: Callable[[float], Any], lo, hi, phrase_fmt: str,
               choices=None, out_type: str = "number",
               ptype: str = "number", constraints: Optional[dict] = None) -> None:
    sampler = _rng_num(lo, hi, choices)
    _add(name, domain, family, desc,
         {param: _p(ptype, desc, psem, **(constraints or {}))},
         out_key, out_type, out_sem, _mk_fn1(param, f),
         lambda rng, s=sampler, p=param: {p: s(rng)},
         lambda a, fmt=phrase_fmt, p=param: fmt.format(v=a[p]),
         chain_in=param)


_CONVERSIONS = [
    # name, unit_in, unit_out, param, in_sem, out_sem, factor, out_key, lo, hi
    ("kilometers_to_miles", "kilometers", "miles", "kilometers", "length_km", "length_mi", 0.621371, "value", 5, 900),
    ("miles_to_kilometers", "miles", "kilometers", "miles", "length_mi", "length_km", 1.609344, "value", 5, 600),
    ("meters_to_feet", "meters", "feet", "meters", "length_m", "length_ft", 3.28084, "result", 1, 400),
    ("feet_to_meters", "feet", "meters", "feet", "length_ft", "length_m", 0.3048, "result", 3, 900),
    ("centimeters_to_inches", "centimeters", "inches", "centimeters", "length_cm", "length_in", 0.393701, "output_0", 2, 500),
    ("inches_to_centimeters", "inches", "centimeters", "inches", "length_in", "length_cm", 2.54, "output_0", 1, 200),
    ("kilograms_to_pounds", "kilograms", "pounds", "kilograms", "mass_kg", "mass_lb", 2.20462, "value", 1, 250),
    ("pounds_to_kilograms", "pounds", "kilograms", "pounds", "mass_lb", "mass_kg", 0.453592, "value", 2, 500),
    ("grams_to_ounces", "grams", "ounces", "grams", "mass_g", "mass_oz", 0.035274, "output_0", 10, 2000),
    ("ounces_to_grams", "ounces", "grams", "ounces", "mass_oz", "mass_g", 28.3495, "output_0", 1, 80),
    ("liters_to_gallons", "liters", "US gallons", "liters", "volume_l", "volume_gal", 0.264172, "result", 2, 400),
    ("gallons_to_liters", "US gallons", "liters", "gallons", "volume_gal", "volume_l", 3.78541, "result", 1, 120),
    ("milliliters_to_fluid_ounces", "milliliters", "fluid ounces", "milliliters", "volume_ml", "volume_floz", 0.033814, "value", 50, 2000),
    ("kmh_to_mph", "kilometers per hour", "miles per hour", "speed_kmh", "speed_kmh", "speed_mph", 0.621371, "value", 20, 300),
    ("mph_to_kmh", "miles per hour", "kilometers per hour", "speed_mph", "speed_mph", "speed_kmh", 1.609344, "value", 15, 200),
    ("hours_to_minutes", "hours", "minutes", "hours", "duration_h", "duration_min", 60.0, "output_0", 1, 48),
    ("minutes_to_seconds", "minutes", "seconds", "minutes", "duration_min", "duration_s", 60.0, "output_0", 1, 600),
    ("days_to_hours", "days", "hours", "days", "duration_day", "duration_h", 24.0, "result", 1, 60),
    ("weeks_to_days", "weeks", "days", "weeks", "duration_week", "duration_day", 7.0, "result", 1, 52),
    ("megabytes_to_gigabytes", "megabytes", "gigabytes", "megabytes", "data_mb", "data_gb", 1.0 / 1024.0, "value", 512, 90000),
    ("kilowatt_hours_to_megajoules", "kilowatt-hours", "megajoules", "kilowatt_hours", "energy_kwh", "energy_mj", 3.6, "output_0", 1, 900),
    ("acres_to_square_meters", "acres", "square meters", "acres", "area_acre", "area_m2", 4046.86, "result", 1, 50),
]
for _n, _ui, _uo, _pm, _si, _so, _f, _ok, _lo, _hi in _CONVERSIONS:
    _add_unary(_n, "conversion", "unit_conversion",
               f"Converts a value from {_ui} to {_uo}.",
               _pm, _si, _ok, _so,
               (lambda v, f=_f: _r2(float(v) * f)), _lo, _hi,
               "{v} " + _ui + " converted to " + _uo)

_TEMPS = [
    ("celsius_to_fahrenheit", "Celsius", "Fahrenheit", "celsius", "temp_c", "temp_f",
     lambda v: _r2(float(v) * 9 / 5 + 32), -20, 45),
    ("fahrenheit_to_celsius", "Fahrenheit", "Celsius", "fahrenheit", "temp_f", "temp_c",
     lambda v: _r2((float(v) - 32) * 5 / 9), 0, 110),
    ("celsius_to_kelvin", "Celsius", "Kelvin", "celsius", "temp_c", "temp_k",
     lambda v: _r2(float(v) + 273.15), -30, 60),
    ("kelvin_to_celsius", "Kelvin", "Celsius", "kelvin", "temp_k", "temp_c",
     lambda v: _r2(float(v) - 273.15), 250, 380),
]
for _n, _ui, _uo, _pm, _si, _so, _fn, _lo, _hi in _TEMPS:
    _add_unary(_n, "conversion", "temperature_conversion",
               f"Converts a temperature from degrees {_ui} to {_uo}.",
               _pm, _si, "value", _so, _fn, _lo, _hi,
               "{v} degrees " + _ui + " converted to " + _uo)


# ─────────────────────────────────────────────────────────────────────────────
#  Family: finance
# ─────────────────────────────────────────────────────────────────────────────
def _range_sampler(spec):
    """spec: (lo, hi) numeric range, or ([choices],) / [choices] choice list."""
    if isinstance(spec, tuple) and len(spec) == 2 \
            and all(isinstance(x, (int, float)) for x in spec):
        return _rng_num(spec[0], spec[1])
    if isinstance(spec, tuple) and len(spec) == 1 and isinstance(spec[0], list):
        return _rng_num(0, 0, spec[0])
    if isinstance(spec, list):
        return _rng_num(0, 0, spec)
    raise ValueError(f"bad sampler spec: {spec!r}")


def _add_binary(name: str, domain: str, family: str, desc: str,
                p1: str, s1: str, r1, p2: str, s2: str, r2,
                out_key: str, out_sem: str, f, phrase_fmt: str,
                chain_in: Optional[str] = None, out_type: str = "number",
                t1: str = "number", t2: str = "number",
                c1: Optional[dict] = None, c2: Optional[dict] = None) -> None:
    samp1 = _range_sampler(r1)
    samp2 = _range_sampler(r2)
    _add(name, domain, family, desc,
         {p1: _p(t1, f"{p1.replace('_', ' ')}.", s1, **(c1 or {})),
          p2: _p(t2, f"{p2.replace('_', ' ')}.", s2, **(c2 or {}))},
         out_key, out_type, out_sem, _mk_fn2(p1, p2, f),
         lambda rng, a=samp1, b=samp2, x=p1, y=p2: {x: a(rng), y: b(rng)},
         lambda a, fmt=phrase_fmt: fmt.format(**a),
         chain_in=chain_in if chain_in is not None else p1)


_add(  # ternary example with explicit spec
    "calculate_simple_interest", "finance", "interest",
    "Calculates simple interest earned on a principal at an annual rate over a number of years.",
    {"principal_amount": _p("number", "Initial amount invested.", "money"),
     "annual_rate_percent": _p("number", "Yearly interest rate in percent.", "percent"),
     "years": _p("number", "Investment duration in years.", "duration_year")},
    "output_0", "number", "money",
    _mk_fn3("principal_amount", "annual_rate_percent", "years",
            lambda p, r, y: _r2(float(p) * float(r) / 100.0 * float(y))),
    lambda rng: {"principal_amount": rng.randrange(500, 20000, 100),
                 "annual_rate_percent": rng.choice([2, 2.5, 3, 4, 5, 6, 7.5]),
                 "years": rng.randrange(1, 12)},
    lambda a: (f"the simple interest on {a['principal_amount']} at "
               f"{a['annual_rate_percent']}% per year for {a['years']} years"),
    chain_in="principal_amount")

_add(
    "compound_savings_amount", "finance", "interest",
    "Computes the final amount of a savings account with annual compounding.",
    {"principal_amount": _p("number", "Initial deposit.", "money"),
     "annual_rate_percent": _p("number", "Annual interest rate in percent.", "percent"),
     "years": _p("integer", "Number of full years.", "duration_year", min=0, max=50)},
    "output_0", "number", "money",
    _mk_fn3("principal_amount", "annual_rate_percent", "years",
            lambda p, r, y: _r2(float(p) * (1 + float(r) / 100.0) ** int(y))),
    lambda rng: {"principal_amount": rng.randrange(1000, 15000, 500),
                 "annual_rate_percent": rng.choice([2, 3, 4, 5]),
                 "years": rng.randrange(1, 10)},
    lambda a: (f"the final balance of {a['principal_amount']} compounded at "
               f"{a['annual_rate_percent']}% for {a['years']} years"),
    chain_in="principal_amount")

_add_binary("apply_discount", "finance", "pricing",
            "Applies a percentage discount to a price and returns the discounted price.",
            "price", "money", (20, 900), "discount_percent", "percent",
            ([5, 10, 15, 20, 25, 30, 40],),
            "output_0", "money",
            lambda price, discount_percent: _r2(float(price) * (1 - float(discount_percent) / 100.0)),
            "the price after a {discount_percent}% discount on {price}")

_add_binary("add_sales_tax", "finance", "pricing",
            "Adds sales tax to a net price and returns the gross price.",
            "net_price", "money", (10, 600), "tax_rate_percent", "percent",
            ([5, 7, 8, 10, 19, 21],),
            "output_0", "money",
            lambda net_price, tax_rate_percent: _r2(float(net_price) * (1 + float(tax_rate_percent) / 100.0)),
            "the total after adding {tax_rate_percent}% sales tax to {net_price}")

_add_binary("remove_sales_tax", "finance", "pricing",
            "Removes sales tax from a gross price and returns the net price.",
            "gross_price", "money", (12, 700), "tax_rate_percent", "percent",
            ([5, 7, 8, 10, 19, 21],),
            "output_0", "money",
            lambda gross_price, tax_rate_percent: _r2(float(gross_price) / (1 + float(tax_rate_percent) / 100.0)),
            "the net price when {gross_price} already includes {tax_rate_percent}% tax")

_add_binary("calculate_tip_amount", "finance", "pricing",
            "Calculates the tip for a bill given a tip percentage.",
            "bill_amount", "money", (15, 400), "tip_percent", "percent",
            ([10, 12, 15, 18, 20],),
            "output_0", "money",
            lambda bill_amount, tip_percent: _r2(float(bill_amount) * float(tip_percent) / 100.0),
            "a {tip_percent}% tip on a bill of {bill_amount}")

_add_binary("split_bill_evenly", "finance", "pricing",
            "Splits a bill between people; each share is rounded UP to the next cent so the bill is always covered.",
            "total_amount", "money", (40, 700), "num_people", "count", (2, 9),
            "output_0", "money",
            lambda total_amount, num_people: math.ceil(float(total_amount) / int(num_people) * 100) / 100.0,
            "each person's share when {total_amount} is split between {num_people} people",
            t2="integer", c2={"min": 1})

_add_binary("monthly_installment", "finance", "loans",
            "Computes the monthly installment for a zero-interest loan, rounded up to a whole currency unit.",
            "loan_amount", "money", (1000, 30000), "num_months", "count",
            ([6, 12, 24, 36, 48],),
            "output_0", "money",
            lambda loan_amount, num_months: float(math.ceil(float(loan_amount) / int(num_months))),
            "the monthly installment for a {loan_amount} loan over {num_months} months",
            t2="integer", c2={"min": 1})

_add_binary("loan_total_repayment", "finance", "loans",
            "Computes the total amount repaid given a monthly payment and the number of months.",
            "monthly_payment", "money", (80, 1500), "num_months", "count",
            ([6, 12, 24, 36, 48],),
            "output_0", "money",
            lambda monthly_payment, num_months: _r2(float(monthly_payment) * int(num_months)),
            "the total repaid at {monthly_payment} per month for {num_months} months",
            t2="integer", c2={"min": 1})

_add_binary("convert_currency", "finance", "pricing",
            "Converts an amount of money using a fixed exchange rate.",
            "amount", "money", (50, 3000), "exchange_rate", "ratio",
            ([0.85, 0.92, 1.08, 1.25, 4.5, 23.5],),
            "output_0", "money",
            lambda amount, exchange_rate: _r2(float(amount) * float(exchange_rate)),
            "{amount} converted at an exchange rate of {exchange_rate}")

_add_binary("profit_margin_percent", "finance", "business",
            "Computes the profit margin in percent from revenue and cost.",
            "revenue", "money", (200, 9000), "cost", "money", (50, 4000),
            "result", "percent",
            lambda revenue, cost: _r2((float(revenue) - float(cost)) / float(revenue) * 100.0),
            "the profit margin when revenue is {revenue} and cost is {cost}")

_add_binary("price_from_target_margin", "finance", "business",
            "Computes the selling price that achieves a target profit margin on a cost price.",
            "cost_price", "money", (10, 800), "target_margin_percent", "percent",
            ([20, 25, 30, 40, 50],),
            "result", "money",
            lambda cost_price, target_margin_percent: _r2(float(cost_price) / (1 - float(target_margin_percent) / 100.0)),
            "the selling price for a {target_margin_percent}% margin on a cost of {cost_price}")

_add(
    "break_even_units", "finance", "business",
    "Computes how many units must be sold to cover fixed costs (rounded up).",
    {"fixed_costs": _p("number", "Total fixed costs.", "money"),
     "unit_price": _p("number", "Selling price per unit.", "money"),
     "unit_cost": _p("number", "Variable cost per unit.", "money")},
    "result", "integer", "count",
    _mk_fn3("fixed_costs", "unit_price", "unit_cost",
            lambda f, p, c: int(math.ceil(float(f) / (float(p) - float(c))))),
    lambda rng: {"fixed_costs": rng.randrange(1000, 20000, 500),
                 "unit_price": rng.randrange(20, 90),
                 "unit_cost": rng.randrange(2, 18)},
    lambda a: (f"the break-even units with fixed costs {a['fixed_costs']}, "
               f"unit price {a['unit_price']} and unit cost {a['unit_cost']}"),
    chain_in="fixed_costs")

_add(
    "linear_depreciation_value", "finance", "business",
    "Computes the remaining value of an asset after linear yearly depreciation.",
    {"initial_value": _p("number", "Purchase value of the asset.", "money"),
     "annual_depreciation": _p("number", "Value lost each year.", "money"),
     "years": _p("integer", "Years of use.", "duration_year", min=0)},
    "result", "number", "money",
    _mk_fn3("initial_value", "annual_depreciation", "years",
            lambda v, d, y: _r2(float(v) - float(d) * int(y))),
    lambda rng: {"initial_value": rng.randrange(5000, 40000, 1000),
                 "annual_depreciation": rng.randrange(300, 2000, 100),
                 "years": rng.randrange(1, 8)},
    lambda a: (f"the value of an asset bought for {a['initial_value']} after "
               f"{a['years']} years at {a['annual_depreciation']} depreciation per year"),
    chain_in="initial_value")

_add_binary("return_on_investment_percent", "finance", "business",
            "Computes the return on investment in percent from net gain and cost.",
            "net_gain", "money", (100, 5000), "investment_cost", "money", (500, 9000),
            "result", "percent",
            lambda net_gain, investment_cost: _r2(float(net_gain) / float(investment_cost) * 100.0),
            "the ROI when the gain is {net_gain} on an investment of {investment_cost}")

_add_binary("hourly_to_annual_salary", "finance", "payroll",
            "Converts an hourly wage to an annual salary (52 paid weeks).",
            "hourly_rate", "money", (12, 90), "hours_per_week", "duration_h",
            ([20, 30, 37.5, 40],),
            "output_0", "money",
            lambda hourly_rate, hours_per_week: _r2(float(hourly_rate) * float(hours_per_week) * 52),
            "the annual salary at {hourly_rate} per hour for {hours_per_week} hours a week")

_add(
    "overtime_pay", "finance", "payroll",
    "Computes overtime pay given an hourly rate, overtime hours and a pay multiplier.",
    {"hourly_rate": _p("number", "Base hourly rate.", "money"),
     "overtime_hours": _p("number", "Overtime hours worked.", "duration_h"),
     "multiplier": _p("number", "Overtime pay multiplier.", "ratio", required=False)},
    "output_0", "number", "money",
    lambda hourly_rate, overtime_hours, multiplier=1.5:
        _r2(float(hourly_rate) * float(overtime_hours) * float(multiplier)),
    lambda rng: {"hourly_rate": rng.randrange(12, 60),
                 "overtime_hours": rng.randrange(2, 30)},
    lambda a: (f"the overtime pay for {a['overtime_hours']} extra hours at "
               f"{a['hourly_rate']} per hour"),
    chain_in="hourly_rate")

_add_binary("unit_price", "finance", "pricing",
            "Computes the price per unit from a total price and a quantity.",
            "total_price", "money", (20, 900), "quantity", "count", (2, 60),
            "output_0", "money",
            lambda total_price, quantity: _r2(float(total_price) / int(quantity)),
            "the unit price when {quantity} items cost {total_price} in total",
            t2="integer", c2={"min": 1})

_add(
    "bulk_order_total", "finance", "pricing",
    "Computes the total for a bulk order after a bulk discount.",
    {"unit_price": _p("number", "Price of one item.", "money"),
     "quantity": _p("integer", "Number of items ordered.", "count", min=1),
     "bulk_discount_percent": _p("number", "Discount applied to the whole order.", "percent")},
    "output_0", "number", "money",
    _mk_fn3("unit_price", "quantity", "bulk_discount_percent",
            lambda u, q, d: _r2(float(u) * int(q) * (1 - float(d) / 100.0))),
    lambda rng: {"unit_price": rng.choice([1.5, 2.5, 4, 7.99, 12, 25]),
                 "quantity": rng.randrange(10, 200, 5),
                 "bulk_discount_percent": rng.choice([5, 8, 10, 12, 15])},
    lambda a: (f"the total for {a['quantity']} units at {a['unit_price']} each "
               f"with a {a['bulk_discount_percent']}% bulk discount"),
    chain_in="unit_price")

_add(
    "late_payment_total", "finance", "loans",
    "Computes the total owed after a monthly late fee percentage is applied for several months.",
    {"amount_due": _p("number", "Original amount due.", "money"),
     "monthly_fee_percent": _p("number", "Late fee percent added each month.", "percent"),
     "months_late": _p("integer", "Months overdue.", "count", min=0)},
    "output_0", "number", "money",
    _mk_fn3("amount_due", "monthly_fee_percent", "months_late",
            lambda a, f, m: _r2(float(a) * (1 + float(f) / 100.0) ** int(m))),
    lambda rng: {"amount_due": rng.randrange(100, 3000, 50),
                 "monthly_fee_percent": rng.choice([1, 1.5, 2, 3]),
                 "months_late": rng.randrange(1, 7)},
    lambda a: (f"the total owed on {a['amount_due']} after {a['months_late']} months "
               f"of {a['monthly_fee_percent']}% monthly late fees"),
    chain_in="amount_due")


# ─────────────────────────────────────────────────────────────────────────────
#  Family: geometry
# ─────────────────────────────────────────────────────────────────────────────
_add_binary("rectangle_area", "geometry", "area",
            "Computes the area of a rectangle from its side lengths.",
            "length", "length_m", (2, 60), "width", "length_m", (2, 40),
            "result", "area_m2",
            lambda length, width: _r2(float(length) * float(width)),
            "the area of a rectangle {length} by {width}")

_add_binary("rectangle_perimeter", "geometry", "perimeter",
            "Computes the perimeter of a rectangle from its side lengths.",
            "length", "length_m", (2, 60), "width", "length_m", (2, 40),
            "result", "length_m",
            lambda length, width: _r2(2 * (float(length) + float(width))),
            "the perimeter of a rectangle {length} by {width}")

_add_unary("square_area", "geometry", "area",
           "Computes the area of a square from its side length.",
           "side_length", "length_m", "result", "area_m2",
           lambda v: _r2(float(v) ** 2), 2, 45,
           "the area of a square with side {v}")

_add_unary("circle_area", "geometry", "area",
           "Computes the area of a circle from its radius.",
           "radius", "length_m", "result", "area_m2",
           lambda v: _r2(math.pi * float(v) ** 2), 1, 25,
           "the area of a circle with radius {v}")

_add_unary("circle_circumference", "geometry", "perimeter",
           "Computes the circumference of a circle from its radius.",
           "radius", "length_m", "result", "length_m",
           lambda v: _r2(2 * math.pi * float(v)), 1, 30,
           "the circumference of a circle with radius {v}")

_add_binary("triangle_area", "geometry", "area",
            "Computes the area of a triangle from base and height.",
            "base", "length_m", (2, 50), "height", "length_m", (2, 40),
            "result", "area_m2",
            lambda base, height: _r2(float(base) * float(height) / 2.0),
            "the area of a triangle with base {base} and height {height}")

_add(
    "trapezoid_area", "geometry", "area",
    "Computes the area of a trapezoid from its two parallel sides and height.",
    {"base_a": _p("number", "First parallel side.", "length_m"),
     "base_b": _p("number", "Second parallel side.", "length_m"),
     "height": _p("number", "Height between the parallel sides.", "length_m")},
    "result", "number", "area_m2",
    _mk_fn3("base_a", "base_b", "height",
            lambda a, b, h: _r2((float(a) + float(b)) / 2.0 * float(h))),
    lambda rng: {"base_a": rng.randrange(3, 40), "base_b": rng.randrange(3, 40),
                 "height": rng.randrange(2, 25)},
    lambda a: (f"the area of a trapezoid with sides {a['base_a']} and "
               f"{a['base_b']} and height {a['height']}"),
    chain_in="base_a")

_add_binary("ellipse_area", "geometry", "area",
            "Computes the area of an ellipse from its semi-major and semi-minor axes.",
            "semi_major_axis", "length_m", (3, 40), "semi_minor_axis", "length_m", (2, 25),
            "result", "area_m2",
            lambda semi_major_axis, semi_minor_axis: _r2(math.pi * float(semi_major_axis) * float(semi_minor_axis)),
            "the area of an ellipse with axes {semi_major_axis} and {semi_minor_axis}")

_add_unary("cube_volume", "geometry", "volume",
           "Computes the volume of a cube from its edge length.",
           "edge_length", "length_m", "result", "volume_m3",
           lambda v: _r2(float(v) ** 3), 1, 20,
           "the volume of a cube with edge {v}")

_add(
    "box_volume", "geometry", "volume",
    "Computes the volume of a rectangular box.",
    {"length": _p("number", "Box length.", "length_m"),
     "width": _p("number", "Box width.", "length_m"),
     "height": _p("number", "Box height.", "length_m")},
    "result", "number", "volume_m3",
    _mk_fn3("length", "width", "height",
            lambda l, w, h: _r2(float(l) * float(w) * float(h))),
    lambda rng: {"length": rng.randrange(2, 30), "width": rng.randrange(2, 20),
                 "height": rng.randrange(1, 15)},
    lambda a: f"the volume of a box {a['length']} x {a['width']} x {a['height']}",
    chain_in="length")

_add_binary("cylinder_volume", "geometry", "volume",
            "Computes the volume of a cylinder from radius and height.",
            "radius", "length_m", (1, 15), "height", "length_m", (2, 30),
            "result", "volume_m3",
            lambda radius, height: _r2(math.pi * float(radius) ** 2 * float(height)),
            "the volume of a cylinder with radius {radius} and height {height}")

_add_unary("sphere_volume", "geometry", "volume",
           "Computes the volume of a sphere from its radius.",
           "radius", "length_m", "result", "volume_m3",
           lambda v: _r2(4.0 / 3.0 * math.pi * float(v) ** 3), 1, 12,
           "the volume of a sphere with radius {v}")

_add_binary("right_triangle_hypotenuse", "geometry", "pythagoras",
            "Computes the hypotenuse of a right triangle from its two legs.",
            "leg_a", "length_m", (3, 40), "leg_b", "length_m", (3, 40),
            "result", "length_m",
            lambda leg_a, leg_b: _r2(math.hypot(float(leg_a), float(leg_b))),
            "the hypotenuse of a right triangle with legs {leg_a} and {leg_b}")

_add_binary("rectangle_aspect_ratio", "geometry", "scaling",
            "Computes the aspect ratio of a rectangle (length divided by width).",
            "length", "length_m", (10, 90), "width", "length_m", (3, 40),
            "result", "ratio",
            lambda length, width: _r2(float(length) / float(width)),
            "the aspect ratio of a rectangle {length} by {width}")

_add_binary("scale_dimension", "geometry", "scaling",
            "Scales a dimension by a multiplicative factor.",
            "dimension", "length_m", (2, 80), "scale_factor", "ratio",
            ([0.5, 1.5, 2, 2.5, 3],),
            "result", "length_m",
            lambda dimension, scale_factor: _r2(float(dimension) * float(scale_factor)),
            "{dimension} scaled by a factor of {scale_factor}")

_add_binary("regular_polygon_perimeter", "geometry", "perimeter",
            "Computes the perimeter of a regular polygon from side length and number of sides.",
            "side_length", "length_m", (2, 30), "num_sides", "count", (3, 12),
            "result", "length_m",
            lambda side_length, num_sides: _r2(float(side_length) * int(num_sides)),
            "the perimeter of a regular {num_sides}-gon with side {side_length}",
            t2="integer", c2={"min": 3})

_add(
    "box_surface_area", "geometry", "area",
    "Computes the total surface area of a rectangular box.",
    {"length": _p("number", "Box length.", "length_m"),
     "width": _p("number", "Box width.", "length_m"),
     "height": _p("number", "Box height.", "length_m")},
    "result", "number", "area_m2",
    _mk_fn3("length", "width", "height",
            lambda l, w, h: _r2(2 * (float(l) * float(w) + float(l) * float(h)
                                     + float(w) * float(h)))),
    lambda rng: {"length": rng.randrange(2, 25), "width": rng.randrange(2, 18),
                 "height": rng.randrange(1, 12)},
    lambda a: f"the surface area of a box {a['length']} x {a['width']} x {a['height']}",
    chain_in="length")

_add_binary("cone_volume", "geometry", "volume",
            "Computes the volume of a cone from base radius and height.",
            "radius", "length_m", (1, 12), "height", "length_m", (2, 25),
            "result", "volume_m3",
            lambda radius, height: _r2(math.pi * float(radius) ** 2 * float(height) / 3.0),
            "the volume of a cone with radius {radius} and height {height}")


# ─────────────────────────────────────────────────────────────────────────────
#  Family: scalar math
# ─────────────────────────────────────────────────────────────────────────────
_add_binary("add_numbers", "math", "arithmetic",
            "Adds two numbers.",
            "first_number", "generic_number", (2, 900), "second_number", "generic_number", (2, 900),
            "output_0", "generic_number",
            lambda first_number, second_number: _r2(float(first_number) + float(second_number)),
            "the sum of {first_number} and {second_number}")

_add_binary("subtract_numbers", "math", "arithmetic",
            "Subtracts the second number from the first.",
            "minuend", "generic_number", (100, 900), "subtrahend", "generic_number", (2, 90),
            "output_0", "generic_number",
            lambda minuend, subtrahend: _r2(float(minuend) - float(subtrahend)),
            "the difference between {minuend} and {subtrahend}")

_add_binary("multiply_numbers", "math", "arithmetic",
            "Multiplies two numbers.",
            "first_factor", "generic_number", (2, 60), "second_factor", "generic_number", (2, 40),
            "output_0", "generic_number",
            lambda first_factor, second_factor: _r2(float(first_factor) * float(second_factor)),
            "the product of {first_factor} and {second_factor}")

_add_binary("divide_numbers", "math", "arithmetic",
            "Divides the first number by the second.",
            "dividend", "generic_number", (50, 900), "divisor", "generic_number", (2, 40),
            "output_0", "generic_number",
            lambda dividend, divisor: _r2(float(dividend) / float(divisor)),
            "{dividend} divided by {divisor}")

_add_binary("power_of", "math", "arithmetic",
            "Raises a base to an integer exponent.",
            "base_value", "generic_number", (2, 12), "exponent", "count", (2, 4),
            "output_0", "generic_number",
            lambda base_value, exponent: _r2(float(base_value) ** int(exponent)),
            "{base_value} raised to the power of {exponent}",
            t2="integer", c2={"min": 0, "max": 6})

_add_unary("square_root_of", "math", "arithmetic",
           "Computes the square root of a non-negative number.",
           "value", "generic_number", "output_0", "generic_number",
           lambda v: _r2(math.sqrt(float(v))), 4, 900,
           "the square root of {v}", constraints={"min": 0})

_add_binary("absolute_difference", "math", "arithmetic",
            "Computes the absolute difference between two numbers.",
            "first_number", "generic_number", (5, 500), "second_number", "generic_number", (5, 500),
            "output_0", "generic_number",
            lambda first_number, second_number: _r2(abs(float(first_number) - float(second_number))),
            "the absolute difference between {first_number} and {second_number}")

_add(
    "round_to_decimals", "math", "rounding",
    "Rounds a number to a given number of decimal places.",
    {"value": _p("number", "The number to round.", "generic_number"),
     "decimals": _p("integer", "Decimal places to keep.", "count", required=False,
                    min=0, max=6)},
    "output_0", "number", "generic_number",
    lambda value, decimals=2: round(float(value), int(decimals)),
    lambda rng: {"value": _r2(rng.uniform(1, 900)), "decimals": rng.randrange(0, 3)},
    lambda a: f"{a['value']} rounded to {a.get('decimals', 2)} decimal places",
    chain_in="value")

_add_binary("floor_divide", "math", "arithmetic",
            "Computes integer division of the first number by the second (rounded down).",
            "dividend", "generic_number", (50, 900), "divisor", "count", (2, 20),
            "output_0", "generic_number",
            lambda dividend, divisor: float(int(float(dividend)) // int(divisor)),
            "how many whole times {divisor} fits into {dividend}",
            t2="integer", c2={"min": 1})

_add_binary("remainder_of", "math", "arithmetic",
            "Computes the remainder when the first integer is divided by the second.",
            "dividend", "count", (20, 900), "divisor", "count", (2, 30),
            "output_0", "count",
            lambda dividend, divisor: int(int(float(dividend)) % int(divisor)),
            "the remainder of {dividend} divided by {divisor}",
            t1="integer", t2="integer", c2={"min": 1}, out_type="integer")

_add_binary("percent_change", "math", "percentages",
            "Computes the percentage change from an old value to a new value.",
            "old_value", "generic_number", (50, 500), "new_value", "generic_number", (50, 900),
            "result", "percent",
            lambda old_value, new_value: _r2((float(new_value) - float(old_value)) / float(old_value) * 100.0),
            "the percentage change from {old_value} to {new_value}")

_add_binary("percentage_of", "math", "percentages",
            "Computes what percentage the part is of the whole.",
            "part", "generic_number", (5, 90), "whole", "generic_number", (100, 500),
            "output_0", "percent",
            lambda part, whole: _r2(float(part) / float(whole) * 100.0),
            "what percentage {part} is of {whole}")

_add_binary("increase_by_percent", "math", "percentages",
            "Increases a value by a given percentage.",
            "value", "generic_number", (20, 800), "percent", "percent",
            ([5, 10, 15, 20, 25, 50],),
            "output_0", "generic_number",
            lambda value, percent: _r2(float(value) * (1 + float(percent) / 100.0)),
            "{value} increased by {percent}%")

_add_binary("decrease_by_percent", "math", "percentages",
            "Decreases a value by a given percentage.",
            "value", "generic_number", (20, 800), "percent", "percent",
            ([5, 10, 15, 20, 25, 50],),
            "output_0", "generic_number",
            lambda value, percent: _r2(float(value) * (1 - float(percent) / 100.0)),
            "{value} decreased by {percent}%")

_add_binary("ratio_of", "math", "percentages",
            "Computes the ratio of two values (first divided by second).",
            "numerator", "generic_number", (50, 900), "denominator", "generic_number", (2, 40),
            "output_0", "ratio",
            lambda numerator, denominator: _r2(float(numerator) / float(denominator)),
            "the ratio of {numerator} to {denominator}")

_add_binary("average_of_two", "math", "arithmetic",
            "Computes the average of two numbers.",
            "first_number", "generic_number", (10, 900), "second_number", "generic_number", (10, 900),
            "output_0", "generic_number",
            lambda first_number, second_number: _r2((float(first_number) + float(second_number)) / 2.0),
            "the average of {first_number} and {second_number}")

_add(
    "clamp_value", "math", "rounding",
    "Clamps a value into the inclusive range [lower_bound, upper_bound].",
    {"value": _p("number", "The value to clamp.", "generic_number"),
     "lower_bound": _p("number", "Minimum allowed value.", "generic_number"),
     "upper_bound": _p("number", "Maximum allowed value.", "generic_number")},
    "output_0", "number", "generic_number",
    _mk_fn3("value", "lower_bound", "upper_bound",
            lambda v, lo, hi: _r2(min(max(float(v), float(lo)), float(hi)))),
    lambda rng: {"value": rng.randrange(0, 300), "lower_bound": rng.randrange(20, 80),
                 "upper_bound": rng.randrange(120, 250)},
    lambda a: (f"{a['value']} clamped between {a['lower_bound']} and "
               f"{a['upper_bound']}"),
    chain_in="value")

_add_unary("reciprocal_of", "math", "arithmetic",
           "Computes the reciprocal (1 divided by the value).",
           "value", "generic_number", "output_0", "ratio",
           lambda v: round(1.0 / float(v), 6), 2, 50,
           "the reciprocal of {v}")


# ─────────────────────────────────────────────────────────────────────────────
#  Family: list statistics
# ─────────────────────────────────────────────────────────────────────────────
def _num_list(rng: random.Random, lo=1, hi=100, n_lo=4, n_hi=9) -> List[float]:
    return [rng.randrange(lo, hi) for _ in range(rng.randrange(n_lo, n_hi))]


def _add_list(name: str, family: str, desc: str, f: Callable[[List[float]], Any],
              phrase_fmt: str, out_key: str = "output_0",
              out_type: str = "number", out_sem: str = "generic_number") -> None:
    _add(name, "statistics", family, desc,
         {"values": _p("array", "The list of numeric values.", "list_number",
                       min_len=1)},
         out_key, out_type, out_sem,
         _mk_fn1("values", f),
         lambda rng: {"values": _num_list(rng)},
         lambda a, fmt=phrase_fmt: fmt.format(v=a["values"]),
         chain_in=None)


_add_list("mean_of_values", "aggregation",
          "Computes the arithmetic mean of a list of numbers.",
          lambda vs: _r2(sum(float(x) for x in vs) / len(vs)),
          "the mean of {v}")
_add_list("median_of_values", "aggregation",
          "Computes the median of a list of numbers.",
          lambda vs: _r2(sorted(float(x) for x in vs)[len(vs) // 2]
                         if len(vs) % 2 == 1 else
                         (sorted(float(x) for x in vs)[len(vs) // 2 - 1]
                          + sorted(float(x) for x in vs)[len(vs) // 2]) / 2.0),
          "the median of {v}")
_add_list("sum_of_values", "aggregation",
          "Computes the sum of a list of numbers.",
          lambda vs: _r2(sum(float(x) for x in vs)), "the sum of {v}")
_add_list("min_of_values", "aggregation",
          "Returns the smallest number in a list.",
          lambda vs: _r2(min(float(x) for x in vs)), "the smallest of {v}")
_add_list("max_of_values", "aggregation",
          "Returns the largest number in a list.",
          lambda vs: _r2(max(float(x) for x in vs)), "the largest of {v}")
_add_list("range_of_values", "aggregation",
          "Computes the range (max minus min) of a list of numbers.",
          lambda vs: _r2(max(float(x) for x in vs) - min(float(x) for x in vs)),
          "the range of {v}")
_add_list("product_of_values", "aggregation",
          "Computes the product of a list of numbers.",
          lambda vs: _r2(math.prod(float(x) for x in vs)),
          "the product of {v}")
_add_list("count_of_values", "aggregation",
          "Counts how many items are in a list.",
          lambda vs: int(len(vs)), "the number of items in {v}",
          out_type="integer", out_sem="count")
_add_list("population_std_dev", "dispersion",
          "Computes the population standard deviation of a list of numbers.",
          lambda vs: _r2(math.sqrt(sum((float(x) - sum(map(float, vs)) / len(vs)) ** 2
                                       for x in vs) / len(vs))),
          "the population standard deviation of {v}")
_add_list("population_variance", "dispersion",
          "Computes the population variance of a list of numbers.",
          lambda vs: _r2(sum((float(x) - sum(map(float, vs)) / len(vs)) ** 2
                             for x in vs) / len(vs)),
          "the population variance of {v}")
_add_list("first_value", "selection",
          "Returns the first item of a list of numbers.",
          lambda vs: _r2(float(vs[0])), "the first item of {v}")
_add_list("last_value", "selection",
          "Returns the last item of a list of numbers.",
          lambda vs: _r2(float(vs[-1])), "the last item of {v}")
_add_list("sorted_ascending", "selection",
          "Returns the list sorted in ascending order.",
          lambda vs: sorted(_r2(float(x)) for x in vs),
          "the values {v} sorted ascending",
          out_type="array", out_sem="list_number")

_add(
    "count_above_threshold", "statistics", "filtering",
    "Counts how many values in a list are strictly greater than a threshold.",
    {"values": _p("array", "The list of numeric values.", "list_number", min_len=1),
     "threshold": _p("number", "The comparison threshold.", "generic_number")},
    "output_0", "integer", "count",
    _mk_fn2("values", "threshold",
            lambda vs, t: int(sum(1 for x in vs if float(x) > float(t)))),
    lambda rng: {"values": _num_list(rng), "threshold": rng.randrange(20, 80)},
    lambda a: f"how many of {a['values']} exceed {a['threshold']}",
    chain_in="threshold")

_add(
    "count_below_threshold", "statistics", "filtering",
    "Counts how many values in a list are strictly less than a threshold.",
    {"values": _p("array", "The list of numeric values.", "list_number", min_len=1),
     "threshold": _p("number", "The comparison threshold.", "generic_number")},
    "output_0", "integer", "count",
    _mk_fn2("values", "threshold",
            lambda vs, t: int(sum(1 for x in vs if float(x) < float(t)))),
    lambda rng: {"values": _num_list(rng), "threshold": rng.randrange(20, 80)},
    lambda a: f"how many of {a['values']} are below {a['threshold']}",
    chain_in="threshold")


# ─────────────────────────────────────────────────────────────────────────────
#  Family: text
# ─────────────────────────────────────────────────────────────────────────────
_WORDS = ["ledger", "harbor", "signal", "meadow", "copper", "lantern", "orchid",
          "summit", "quartz", "violet", "anchor", "breeze", "cinder", "drift"]


def _text_sample(rng: random.Random, n_lo=2, n_hi=5) -> str:
    return " ".join(rng.sample(_WORDS, rng.randrange(n_lo, n_hi)))


def _add_text1(name: str, family: str, desc: str, f: Callable[[str], Any],
               phrase_fmt: str, out_type: str = "string",
               out_sem: str = "text", out_key: str = "output_0",
               chainable: bool = True) -> None:
    _add(name, "text", family, desc,
         {"text": _p("string", "The input text.", "text")},
         out_key, out_type, out_sem, _mk_fn1("text", lambda v: f(str(v))),
         lambda rng: {"text": _text_sample(rng)},
         lambda a, fmt=phrase_fmt: fmt.format(v=a["text"]),
         chain_in="text" if chainable else None)


_add_text1("to_uppercase", "casing", "Converts a text to upper case.",
           lambda s: s.upper(), "'{v}' in upper case")
_add_text1("to_lowercase", "casing", "Converts a text to lower case.",
           lambda s: s.lower(), "'{v}' in lower case")
_add_text1("to_title_case", "casing", "Converts a text to title case.",
           lambda s: s.title(), "'{v}' in title case")
_add_text1("reverse_text", "transform", "Reverses the characters of a text.",
           lambda s: s[::-1], "'{v}' reversed")
_add_text1("character_count", "measurement", "Counts the characters in a text.",
           lambda s: int(len(s)), "the number of characters in '{v}'",
           out_type="integer", out_sem="count")
_add_text1("word_count", "measurement", "Counts the words in a text.",
           lambda s: int(len(s.split())), "the number of words in '{v}'",
           out_type="integer", out_sem="count")
_add_text1("slugify_text", "transform",
           "Converts a text to a lowercase slug with dashes instead of spaces.",
           lambda s: "-".join(s.lower().split()), "'{v}' as a URL slug")
_add_text1("extract_initials", "transform",
           "Extracts the upper-case initials of each word in a text.",
           lambda s: "".join(w[0].upper() for w in s.split() if w),
           "the initials of '{v}'")
_add_text1("wrap_in_brackets", "formatting",
           "Wraps a text in square brackets.",
           lambda s: f"[{s}]", "'{v}' wrapped in brackets")

_add(
    "concat_texts", "text", "transform",
    "Concatenates two texts with an optional separator.",
    {"first_text": _p("string", "The first text.", "text"),
     "second_text": _p("string", "The second text.", "text"),
     "separator": _p("string", "Separator inserted between them.", "text",
                     required=False)},
    "output_0", "string", "text",
    lambda first_text, second_text, separator=" ":
        f"{first_text}{separator}{second_text}",
    lambda rng: {"first_text": rng.choice(_WORDS), "second_text": rng.choice(_WORDS)},
    lambda a: f"'{a['first_text']}' joined with '{a['second_text']}'",
    chain_in="first_text")

_add_binary("repeat_word", "text", "transform",
            "Repeats a word a given number of times separated by hyphens.",
            "word", "text", (0, 0), "times", "count", (2, 5),
            "output_0", "text",
            lambda word, times: "-".join([str(word)] * int(times)),
            "the word '{word}' repeated {times} times",
            chain_in="word", out_type="string", t1="string", t2="integer",
            c2={"min": 1, "max": 20})
TOOLS["repeat_word"]["sample"] = lambda rng: {"word": rng.choice(_WORDS),
                                              "times": rng.randrange(2, 5)}

_add_binary("first_characters", "text", "slicing",
            "Returns the first N characters of a text.",
            "text", "text", (0, 0), "count", "count", (2, 6),
            "output_0", "text",
            lambda text, count: str(text)[: int(count)],
            "the first {count} characters of '{text}'",
            chain_in="text", out_type="string", t1="string", t2="integer",
            c2={"min": 1})
TOOLS["first_characters"]["sample"] = lambda rng: {"text": _text_sample(rng),
                                                   "count": rng.randrange(2, 6)}

_add_binary("last_characters", "text", "slicing",
            "Returns the last N characters of a text.",
            "text", "text", (0, 0), "count", "count", (2, 6),
            "output_0", "text",
            lambda text, count: str(text)[-int(count):],
            "the last {count} characters of '{text}'",
            chain_in="text", out_type="string", t1="string", t2="integer",
            c2={"min": 1})
TOOLS["last_characters"]["sample"] = lambda rng: {"text": _text_sample(rng),
                                                  "count": rng.randrange(2, 6)}

_add_binary("pad_number_with_zeros", "text", "formatting",
            "Formats an integer left-padded with zeros to a fixed width.",
            "number", "count", (1, 9999), "width", "count", ([4, 5, 6],),
            "output_0", "text",
            lambda number, width: str(int(float(number))).zfill(int(width)),
            "{number} padded with zeros to width {width}",
            out_type="string", t1="integer", t2="integer",
            c2={"min": 1, "max": 12})

_add(
    "mask_account_number", "text", "formatting",
    "Masks all but the last visible digits of an account number with asterisks.",
    {"account_number": _p("string", "The account number to mask.", "text"),
     "visible_digits": _p("integer", "How many trailing digits stay visible.",
                          "count", required=False, min=1, max=8)},
    "output_0", "string", "text",
    lambda account_number, visible_digits=4:
        "*" * max(0, len(str(account_number)) - int(visible_digits))
        + str(account_number)[-int(visible_digits):],
    lambda rng: {"account_number": str(rng.randrange(10 ** 9, 10 ** 10))},
    lambda a: f"the account number '{a['account_number']}' masked",
    chain_in="account_number")

_add(
    "format_as_currency", "text", "formatting",
    "Formats a numeric amount as a currency string with two decimals.",
    {"amount": _p("number", "The numeric amount.", "money"),
     "currency_symbol": _p("string", "Currency symbol prefix.", "text",
                           required=False)},
    "output_0", "string", "text",
    lambda amount, currency_symbol="$": f"{currency_symbol}{float(amount):.2f}",
    lambda rng: {"amount": _r2(rng.uniform(5, 900)),
                 "currency_symbol": rng.choice(["$", "€", "£"])},
    lambda a: "that amount formatted as a currency string",
    chain_in="amount")

_add(
    "format_as_percent", "text", "formatting",
    "Formats a number as a percent string with a fixed number of decimals.",
    {"value": _p("number", "The numeric value (already in percent units).", "percent"),
     "decimals": _p("integer", "Decimal places.", "count", required=False,
                    min=0, max=4)},
    "output_0", "string", "text",
    lambda value, decimals=1: f"{float(value):.{int(decimals)}f}%",
    lambda rng: {"value": _r2(rng.uniform(1, 99)), "decimals": rng.randrange(0, 3)},
    lambda a: "that value formatted as a percent string",
    chain_in="value")


# ─────────────────────────────────────────────────────────────────────────────
#  Family: boolean / comparisons
# ─────────────────────────────────────────────────────────────────────────────
_add_binary("is_above_threshold", "comparison", "threshold",
            "Checks whether a value is strictly greater than a threshold.",
            "value", "generic_number", (10, 500), "threshold", "generic_number", (10, 500),
            "result", "flag",
            lambda value, threshold: bool(float(value) > float(threshold)),
            "whether it exceeds {threshold}", out_type="boolean")

_add_binary("is_below_threshold", "comparison", "threshold",
            "Checks whether a value is strictly less than a threshold.",
            "value", "generic_number", (10, 500), "threshold", "generic_number", (10, 500),
            "result", "flag",
            lambda value, threshold: bool(float(value) < float(threshold)),
            "whether it is below {threshold}", out_type="boolean")

_add(
    "is_within_range", "comparison", "threshold",
    "Checks whether a value lies inside the inclusive range [lower_bound, upper_bound].",
    {"value": _p("number", "The value to check.", "generic_number"),
     "lower_bound": _p("number", "Lower bound.", "generic_number"),
     "upper_bound": _p("number", "Upper bound.", "generic_number")},
    "result", "boolean", "flag",
    _mk_fn3("value", "lower_bound", "upper_bound",
            lambda v, lo, hi: bool(float(lo) <= float(v) <= float(hi))),
    lambda rng: {"value": rng.randrange(0, 300), "lower_bound": rng.randrange(10, 90),
                 "upper_bound": rng.randrange(120, 280)},
    lambda a: f"whether it lies between {a['lower_bound']} and {a['upper_bound']}",
    chain_in="value")

_add(
    "values_equal", "comparison", "equality",
    "Checks whether two numbers are equal within an optional tolerance.",
    {"first_value": _p("number", "First number.", "generic_number"),
     "second_value": _p("number", "Second number.", "generic_number"),
     "tolerance": _p("number", "Allowed absolute difference.", "generic_number",
                     required=False, min=0)},
    "result", "boolean", "flag",
    lambda first_value, second_value, tolerance=0.01:
        bool(abs(float(first_value) - float(second_value)) <= float(tolerance)),
    lambda rng: {"first_value": rng.randrange(10, 200),
                 "second_value": rng.randrange(10, 200)},
    lambda a: f"whether it equals {a['second_value']}",
    chain_in="first_value")

_add_unary("is_even_number", "comparison", "parity",
           "Checks whether an integer is even.",
           "number", "count", "result", "flag",
           lambda v: bool(int(float(v)) % 2 == 0), 1, 900,
           "whether {v} is even", out_type="boolean", ptype="integer")

_add_unary("is_odd_number", "comparison", "parity",
           "Checks whether an integer is odd.",
           "number", "count", "result", "flag",
           lambda v: bool(int(float(v)) % 2 == 1), 1, 900,
           "whether {v} is odd", out_type="boolean", ptype="integer")

_add_binary("is_multiple_of", "comparison", "parity",
            "Checks whether the first integer is an exact multiple of the second.",
            "number", "count", (10, 900), "divisor", "count", (2, 12),
            "result", "flag",
            lambda number, divisor: bool(int(float(number)) % int(divisor) == 0),
            "whether {number} is a multiple of {divisor}",
            out_type="boolean", t1="integer", t2="integer", c2={"min": 1})

_add_unary("is_positive", "comparison", "sign",
           "Checks whether a number is strictly positive.",
           "value", "generic_number", "result", "flag",
           lambda v: bool(float(v) > 0), -200, 400,
           "whether {v} is positive", out_type="boolean")

_add_binary("exceeds_budget", "comparison", "budget",
            "Checks whether spending exceeds a budget.",
            "amount_spent", "money", (50, 900), "budget", "money", (100, 800),
            "result", "flag",
            lambda amount_spent, budget: bool(float(amount_spent) > float(budget)),
            "whether that spending exceeds the budget of {budget}",
            out_type="boolean")

_add_binary("meets_minimum_quantity", "comparison", "budget",
            "Checks whether a quantity meets a required minimum.",
            "quantity", "count", (1, 200), "minimum_required", "count", (10, 150),
            "result", "flag",
            lambda quantity, minimum_required: bool(int(float(quantity)) >= int(float(minimum_required))),
            "whether that quantity meets the minimum of {minimum_required}",
            out_type="boolean", t1="integer", t2="integer")

_add_binary("text_longer_than", "comparison", "text_check",
            "Checks whether a text has more characters than a given length.",
            "text", "text", (0, 0), "min_length", "count", (3, 20),
            "result", "flag",
            lambda text, min_length: bool(len(str(text)) > int(min_length)),
            "whether '{text}' is longer than {min_length} characters",
            chain_in="text", out_type="boolean", t1="string", t2="integer",
            c2={"min": 0})
TOOLS["text_longer_than"]["sample"] = lambda rng: {"text": _text_sample(rng),
                                                   "min_length": rng.randrange(3, 20)}

_add_binary("starts_with_letter", "comparison", "text_check",
            "Checks whether a text starts with a given letter (case-insensitive).",
            "text", "text", (0, 0), "letter", "text", (0, 0),
            "result", "flag",
            lambda text, letter: bool(str(text).lower().startswith(str(letter).lower())),
            "whether '{text}' starts with '{letter}'",
            chain_in="text", out_type="boolean", t1="string", t2="string")
TOOLS["starts_with_letter"]["sample"] = lambda rng: {
    "text": rng.choice(_WORDS), "letter": rng.choice("lhsmcoq")}


# ─────────────────────────────────────────────────────────────────────────────
#  Family: logistics
# ─────────────────────────────────────────────────────────────────────────────
_add_binary("boxes_needed", "logistics", "packing",
            "Computes how many boxes are needed for a number of units (rounded up).",
            "total_units", "count", (20, 900), "box_capacity", "count",
            ([6, 8, 10, 12, 24],),
            "output_0", "count",
            lambda total_units, box_capacity: int(math.ceil(int(float(total_units)) / int(box_capacity))),
            "the number of boxes of {box_capacity} needed for {total_units} units",
            t1="integer", t2="integer", c2={"min": 1}, out_type="integer")

_add(
    "pallets_needed", "logistics", "packing",
    "Computes how many pallets are needed for a number of boxes given the pallet layout (rounded up).",
    {"num_boxes": _p("integer", "Boxes to load.", "count", min=1),
     "boxes_per_layer": _p("integer", "Boxes per pallet layer.", "count", min=1),
     "layers_per_pallet": _p("integer", "Layers stacked per pallet.", "count", min=1)},
    "output_0", "integer", "count",
    _mk_fn3("num_boxes", "boxes_per_layer", "layers_per_pallet",
            lambda n, bl, lp: int(math.ceil(int(float(n)) / (int(bl) * int(lp))))),
    lambda rng: {"num_boxes": rng.randrange(20, 500),
                 "boxes_per_layer": rng.choice([4, 6, 8]),
                 "layers_per_pallet": rng.choice([2, 3, 4])},
    lambda a: (f"the pallets needed for {a['num_boxes']} boxes with "
               f"{a['boxes_per_layer']} per layer and {a['layers_per_pallet']} layers"),
    chain_in="num_boxes")

_add_binary("fuel_needed_liters", "logistics", "transport",
            "Computes fuel needed for a trip from distance and consumption per 100 km.",
            "distance_km", "length_km", (50, 1200), "consumption_per_100km", "ratio",
            ([4.5, 5.5, 6, 7, 8, 9.5],),
            "value", "volume_l",
            lambda distance_km, consumption_per_100km: _r2(float(distance_km) * float(consumption_per_100km) / 100.0),
            "the fuel needed for {distance_km} km at {consumption_per_100km} l/100km")

_add_binary("travel_time_hours", "logistics", "transport",
            "Computes travel time in hours from distance and constant speed.",
            "distance_km", "length_km", (40, 1500), "speed_kmh", "speed_kmh",
            ([50, 60, 80, 90, 100, 120],),
            "value", "duration_h",
            lambda distance_km, speed_kmh: _r2(float(distance_km) / float(speed_kmh)),
            "the travel time for {distance_km} km at {speed_kmh} km/h")

_add_binary("average_speed", "logistics", "transport",
            "Computes average speed from distance and travel time.",
            "distance_km", "length_km", (30, 1200), "time_hours", "duration_h",
            ([0.5, 1, 1.5, 2, 3, 4, 5, 8],),
            "value", "speed_kmh",
            lambda distance_km, time_hours: _r2(float(distance_km) / float(time_hours)),
            "the average speed for {distance_km} km in {time_hours} hours")

_add_binary("distance_travelled_km", "logistics", "transport",
            "Computes the distance travelled from speed and time.",
            "speed_kmh", "speed_kmh", ([40, 60, 80, 100, 120],), "time_hours", "duration_h",
            ([0.5, 1, 1.5, 2, 3, 5],),
            "value", "length_km",
            lambda speed_kmh, time_hours: _r2(float(speed_kmh) * float(time_hours)),
            "the distance covered at {speed_kmh} km/h in {time_hours} hours")

_add(
    "shipping_cost", "logistics", "transport",
    "Computes shipping cost from weight, a per-kilogram rate and an optional base fee.",
    {"weight_kg": _p("number", "Shipment weight in kilograms.", "mass_kg"),
     "rate_per_kg": _p("number", "Cost per kilogram.", "money"),
     "base_fee": _p("number", "Flat base fee.", "money", required=False, min=0)},
    "output_0", "number", "money",
    lambda weight_kg, rate_per_kg, base_fee=5.0:
        _r2(float(weight_kg) * float(rate_per_kg) + float(base_fee)),
    lambda rng: {"weight_kg": rng.randrange(2, 80),
                 "rate_per_kg": rng.choice([1.2, 2.5, 3, 4.5])},
    lambda a: (f"the shipping cost for {a['weight_kg']} kg at "
               f"{a['rate_per_kg']} per kg"),
    chain_in="weight_kg")

_add_binary("total_cargo_weight", "logistics", "packing",
            "Computes total cargo weight from the unit weight and the item count.",
            "unit_weight_kg", "mass_kg", ([0.5, 1.2, 2, 3.5, 5, 8],), "item_count", "count", (5, 300),
            "output_0", "mass_kg",
            lambda unit_weight_kg, item_count: _r2(float(unit_weight_kg) * int(float(item_count))),
            "the total weight of {item_count} items at {unit_weight_kg} kg each",
            t2="integer", c2={"min": 1})

_add_binary("delivery_days", "logistics", "transport",
            "Computes delivery days for a route (rounded up) given daily driving distance.",
            "distance_km", "length_km", (200, 4000), "km_per_day", "length_km",
            ([300, 400, 500, 600],),
            "output_0", "count",
            lambda distance_km, km_per_day: int(math.ceil(float(distance_km) / float(km_per_day))),
            "the delivery days for {distance_km} km at {km_per_day} km per day",
            out_type="integer")

_add_binary("route_fuel_cost", "logistics", "transport",
            "Computes the fuel cost of a route from liters needed and price per liter.",
            "fuel_liters", "volume_l", (10, 400), "price_per_liter", "money",
            ([1.4, 1.6, 1.8, 2.1],),
            "output_0", "money",
            lambda fuel_liters, price_per_liter: _r2(float(fuel_liters) * float(price_per_liter)),
            "the fuel cost for {fuel_liters} liters at {price_per_liter} per liter")

_add_binary("remaining_stock", "logistics", "inventory",
            "Computes remaining stock after a number of units are sold.",
            "initial_stock", "count", (100, 900), "units_sold", "count", (10, 90),
            "output_0", "count",
            lambda initial_stock, units_sold: int(float(initial_stock)) - int(float(units_sold)),
            "the remaining stock after selling {units_sold} of {initial_stock} units",
            t1="integer", t2="integer", out_type="integer")

_add_binary("warehouse_utilization_percent", "logistics", "inventory",
            "Computes warehouse utilization in percent from used and total slots.",
            "used_slots", "count", (10, 900), "total_slots", "count", (1000, 2000),
            "result", "percent",
            lambda used_slots, total_slots: _r2(int(float(used_slots)) / int(float(total_slots)) * 100.0),
            "the utilization when {used_slots} of {total_slots} slots are used",
            t1="integer", t2="integer", c2={"min": 1})


# ─────────────────────────────────────────────────────────────────────────────
#  Family: health / science
# ─────────────────────────────────────────────────────────────────────────────
_add_binary("body_mass_index", "health", "body",
            "Computes the body mass index from weight and height.",
            "weight_kg", "mass_kg", (45, 120), "height_m", "length_m",
            ([1.55, 1.62, 1.7, 1.78, 1.85, 1.92],),
            "value", "ratio",
            lambda weight_kg, height_m: _r2(float(weight_kg) / float(height_m) ** 2),
            "the BMI for {weight_kg} kg and {height_m} m")

_add(
    "calories_burned", "health", "body",
    "Estimates calories burned from a MET value, body weight and duration.",
    {"met_value": _p("number", "Metabolic equivalent of the activity.", "ratio"),
     "weight_kg": _p("number", "Body weight in kilograms.", "mass_kg"),
     "hours": _p("number", "Activity duration in hours.", "duration_h")},
    "value", "number", "energy_kcal",
    _mk_fn3("met_value", "weight_kg", "hours",
            lambda m, w, h: _r2(float(m) * float(w) * float(h))),
    lambda rng: {"met_value": rng.choice([3, 5, 6, 8, 10]),
                 "weight_kg": rng.randrange(50, 110),
                 "hours": rng.choice([0.5, 1, 1.5, 2])},
    lambda a: (f"the calories burned at MET {a['met_value']} for a "
               f"{a['weight_kg']} kg person over {a['hours']} hours"),
    chain_in="weight_kg")

_add_unary("daily_water_intake_liters", "health", "body",
           "Estimates recommended daily water intake in liters from body weight.",
           "weight_kg", "mass_kg", "value", "volume_l",
           lambda v: _r2(float(v) * 0.033), 40, 120,
           "the recommended daily water intake for {v} kg")

_add_binary("medication_dose_mg", "health", "body",
            "Computes a medication dose from body weight and mg-per-kg dosing.",
            "weight_kg", "mass_kg", (10, 100), "mg_per_kg", "ratio",
            ([2, 5, 7.5, 10],),
            "value", "mass_mg",
            lambda weight_kg, mg_per_kg: _r2(float(weight_kg) * float(mg_per_kg)),
            "the dose for {weight_kg} kg at {mg_per_kg} mg per kg")

_add_unary("maximum_heart_rate", "health", "body",
           "Estimates maximum heart rate from age (220 minus age).",
           "age_years", "duration_year", "value", "count",
           lambda v: int(220 - int(float(v))), 18, 80,
           "the maximum heart rate for age {v}", out_type="integer",
           ptype="integer", constraints={"min": 1, "max": 120})

_add_binary("density_of_object", "science", "physics",
            "Computes density from mass and volume.",
            "mass_g", "mass_g", (50, 2000), "volume_cm3", "volume_cm3", (10, 500),
            "result", "ratio",
            lambda mass_g, volume_cm3: _r2(float(mass_g) / float(volume_cm3)),
            "the density of an object with mass {mass_g} g and volume {volume_cm3} cm³")

_add_binary("force_newtons", "science", "physics",
            "Computes force from mass and acceleration (F = m * a).",
            "mass_kg", "mass_kg", (1, 200), "acceleration", "ratio",
            ([1.5, 2, 3, 5, 9.81],),
            "result", "force_n",
            lambda mass_kg, acceleration: _r2(float(mass_kg) * float(acceleration)),
            "the force on {mass_kg} kg accelerating at {acceleration} m/s²")

_add_binary("kinetic_energy_joules", "science", "physics",
            "Computes kinetic energy from mass and velocity (0.5 * m * v²).",
            "mass_kg", "mass_kg", (1, 100), "velocity_ms", "speed_ms", (2, 40),
            "result", "energy_j",
            lambda mass_kg, velocity_ms: _r2(0.5 * float(mass_kg) * float(velocity_ms) ** 2),
            "the kinetic energy of {mass_kg} kg moving at {velocity_ms} m/s")

_add(
    "potential_energy_joules", "science", "physics",
    "Computes gravitational potential energy (m * g * h).",
    {"mass_kg": _p("number", "Mass in kilograms.", "mass_kg"),
     "height_m": _p("number", "Height in meters.", "length_m"),
     "gravity": _p("number", "Gravitational acceleration.", "ratio",
                   required=False, min=0)},
    "result", "number", "energy_j",
    lambda mass_kg, height_m, gravity=9.81:
        _r2(float(mass_kg) * float(height_m) * float(gravity)),
    lambda rng: {"mass_kg": rng.randrange(1, 80), "height_m": rng.randrange(2, 60)},
    lambda a: f"the potential energy of {a['mass_kg']} kg at {a['height_m']} m",
    chain_in="mass_kg")

_add_binary("ohms_law_voltage", "science", "electricity",
            "Computes voltage from current and resistance (V = I * R).",
            "current_amps", "current_a", ([0.5, 1, 1.5, 2, 3, 5],), "resistance_ohms", "resistance_ohm", (5, 200),
            "result", "voltage_v",
            lambda current_amps, resistance_ohms: _r2(float(current_amps) * float(resistance_ohms)),
            "the voltage across {resistance_ohms} ohms at {current_amps} amps")

_add_binary("electrical_power_watts", "science", "electricity",
            "Computes electrical power from voltage and current (P = V * I).",
            "voltage_v", "voltage_v", ([12, 24, 110, 230],), "current_amps", "current_a",
            ([0.5, 1, 2, 3, 5, 8],),
            "result", "power_w",
            lambda voltage_v, current_amps: _r2(float(voltage_v) * float(current_amps)),
            "the power drawn at {voltage_v} V and {current_amps} A")

_add_binary("pressure_pascals", "science", "physics",
            "Computes pressure from force and area (P = F / A).",
            "force_n", "force_n", (10, 900), "area_m2", "area_m2",
            ([0.5, 1, 2, 4],),
            "result", "pressure_pa",
            lambda force_n, area_m2: _r2(float(force_n) / float(area_m2)),
            "the pressure of {force_n} N over {area_m2} m²")

_add_binary("speed_of_object_ms", "science", "physics",
            "Computes speed in meters per second from distance and time.",
            "distance_m", "length_m", (10, 2000), "time_s", "duration_s", (2, 300),
            "result", "speed_ms",
            lambda distance_m, time_s: _r2(float(distance_m) / float(time_s)),
            "the speed of an object covering {distance_m} m in {time_s} s")


# ─────────────────────────────────────────────────────────────────────────────
#  Family: scheduling (deterministic day/time arithmetic)
# ─────────────────────────────────────────────────────────────────────────────
_add_binary("days_between_day_numbers", "scheduling", "days",
            "Computes the number of days between two day numbers of the year.",
            "start_day", "day_number", (1, 150), "end_day", "day_number", (151, 365),
            "output_0", "count",
            lambda start_day, end_day: int(float(end_day)) - int(float(start_day)),
            "the days between day {start_day} and day {end_day} of the year",
            t1="integer", t2="integer", out_type="integer")

_add_binary("add_days_to_day_number", "scheduling", "days",
            "Adds a number of days to a day number of the year (wraps at 365).",
            "day_number", "day_number", (1, 300), "days_to_add", "count", (1, 90),
            "output_0", "day_number",
            lambda day_number, days_to_add: (int(float(day_number)) + int(float(days_to_add)) - 1) % 365 + 1,
            "the day of the year {days_to_add} days after day {day_number}",
            t1="integer", t2="integer", out_type="integer")

_add_unary("minutes_to_clock_text", "scheduling", "clock",
           "Formats a number of minutes since midnight as an HH:MM string.",
           "total_minutes", "duration_min", "output_0", "text",
           lambda v: f"{int(float(v)) // 60:02d}:{int(float(v)) % 60:02d}",
           0, 1439, "{v} minutes since midnight as a clock time",
           out_type="string", ptype="integer",
           constraints={"min": 0, "max": 100000})

_add_binary("total_minutes", "scheduling", "clock",
            "Converts hours and minutes into total minutes.",
            "hours", "duration_h", (0, 23), "minutes", "duration_min", (0, 59),
            "output_0", "duration_min",
            lambda hours, minutes: int(float(hours)) * 60 + int(float(minutes)),
            "the total minutes in {hours} hours and {minutes} minutes",
            t1="integer", t2="integer", out_type="integer")

_add_unary("seconds_to_full_minutes", "scheduling", "clock",
           "Converts seconds into full minutes (rounded down).",
           "seconds", "duration_s", "output_0", "duration_min",
           lambda v: int(float(v)) // 60, 60, 90000,
           "the full minutes in {v} seconds", out_type="integer",
           ptype="integer", constraints={"min": 0})


# ─────────────────────────────────────────────────────────────────────────────
#  Family: object outputs (nested fields; consumers reference $varN.field$)
# ─────────────────────────────────────────────────────────────────────────────
_add(
    "rectangle_metrics", "geometry", "object_output",
    "Computes both the area and the perimeter of a rectangle.",
    {"length": _p("number", "Rectangle length.", "length_m"),
     "width": _p("number", "Rectangle width.", "length_m")},
    "result", "object", "object",
    _mk_fn2("length", "width",
            lambda l, w: {"area": _r2(float(l) * float(w)),
                          "perimeter": _r2(2 * (float(l) + float(w)))}),
    lambda rng: {"length": rng.randrange(3, 50), "width": rng.randrange(2, 35)},
    lambda a: f"the area and perimeter of a rectangle {a['length']} by {a['width']}",
    chain_in="length",
    out_fields={"area": ("number", "area_m2"), "perimeter": ("number", "length_m")})

_add(
    "list_statistics", "statistics", "object_output",
    "Computes the minimum, maximum and mean of a list of numbers.",
    {"values": _p("array", "The list of numeric values.", "list_number", min_len=1)},
    "result", "object", "object",
    _mk_fn1("values", lambda vs: {"minimum": _r2(min(float(x) for x in vs)),
                                  "maximum": _r2(max(float(x) for x in vs)),
                                  "mean": _r2(sum(float(x) for x in vs) / len(vs))}),
    lambda rng: {"values": _num_list(rng)},
    lambda a: f"the min, max and mean of {a['values']}",
    chain_in=None,
    out_fields={"minimum": ("number", "generic_number"),
                "maximum": ("number", "generic_number"),
                "mean": ("number", "generic_number")})

_add(
    "price_with_tax_breakdown", "finance", "object_output",
    "Computes the tax amount and gross price for a net price and tax rate.",
    {"net_price": _p("number", "Price before tax.", "money"),
     "tax_rate_percent": _p("number", "Tax rate in percent.", "percent")},
    "result", "object", "object",
    _mk_fn2("net_price", "tax_rate_percent",
            lambda n, r: {"net_price": _r2(float(n)),
                          "tax_amount": _r2(float(n) * float(r) / 100.0),
                          "gross_price": _r2(float(n) * (1 + float(r) / 100.0))}),
    lambda rng: {"net_price": rng.randrange(20, 800),
                 "tax_rate_percent": rng.choice([7, 10, 19, 21])},
    lambda a: (f"the tax breakdown of {a['net_price']} at "
               f"{a['tax_rate_percent']}% tax"),
    chain_in="net_price",
    out_fields={"net_price": ("number", "money"),
                "tax_amount": ("number", "money"),
                "gross_price": ("number", "money")})

_add(
    "trip_plan", "logistics", "object_output",
    "Computes the travel time and fuel needed for a road trip.",
    {"distance_km": _p("number", "Trip distance in kilometers.", "length_km"),
     "speed_kmh": _p("number", "Average speed in km/h.", "speed_kmh"),
     "consumption_per_100km": _p("number", "Fuel use per 100 km.", "ratio")},
    "result", "object", "object",
    _mk_fn3("distance_km", "speed_kmh", "consumption_per_100km",
            lambda d, s, c: {"travel_hours": _r2(float(d) / float(s)),
                             "fuel_liters": _r2(float(d) * float(c) / 100.0)}),
    lambda rng: {"distance_km": rng.randrange(80, 1200),
                 "speed_kmh": rng.choice([60, 80, 90, 100, 110]),
                 "consumption_per_100km": rng.choice([5, 6, 7, 8])},
    lambda a: f"the trip plan for {a['distance_km']} km at {a['speed_kmh']} km/h",
    chain_in="distance_km",
    out_fields={"travel_hours": ("number", "duration_h"),
                "fuel_liters": ("number", "volume_l")})

_add(
    "loan_breakdown", "finance", "object_output",
    "Computes the monthly payment and total repaid for a zero-interest loan.",
    {"loan_amount": _p("number", "Amount borrowed.", "money"),
     "num_months": _p("integer", "Number of monthly payments.", "count", min=1)},
    "result", "object", "object",
    _mk_fn2("loan_amount", "num_months",
            lambda a_, m: {"monthly_payment": _r2(float(a_) / int(m)),
                           "total_repaid": _r2(float(a_))}),
    lambda rng: {"loan_amount": rng.randrange(2000, 25000, 500),
                 "num_months": rng.choice([12, 24, 36, 48])},
    lambda a: f"the loan breakdown for {a['loan_amount']} over {a['num_months']} months",
    chain_in="loan_amount",
    out_fields={"monthly_payment": ("number", "money"),
                "total_repaid": ("number", "money")})

_add(
    "temperature_report", "conversion", "object_output",
    "Reports a Celsius temperature together with its Fahrenheit equivalent.",
    {"celsius": _p("number", "Temperature in degrees Celsius.", "temp_c")},
    "result", "object", "object",
    _mk_fn1("celsius", lambda c: {"celsius": _r2(float(c)),
                                  "fahrenheit": _r2(float(c) * 9 / 5 + 32)}),
    lambda rng: {"celsius": rng.randrange(-15, 42)},
    lambda a: f"the temperature report for {a['celsius']} °C",
    chain_in="celsius",
    out_fields={"celsius": ("number", "temp_c"), "fahrenheit": ("number", "temp_f")})

_add(
    "text_profile", "text", "object_output",
    "Profiles a text: character count and word count.",
    {"text": _p("string", "The text to profile.", "text")},
    "result", "object", "object",
    _mk_fn1("text", lambda s: {"characters": int(len(str(s))),
                               "words": int(len(str(s).split()))}),
    lambda rng: {"text": _text_sample(rng)},
    lambda a: f"the character and word counts of '{a['text']}'",
    chain_in="text",
    out_fields={"characters": ("integer", "count"), "words": ("integer", "count")})


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────
ALL_TOOL_NAMES = sorted(TOOLS.keys())
DOMAINS = sorted({t["domain"] for t in TOOLS.values()})
FAMILIES = sorted({t["family"] for t in TOOLS.values()})


def tool_schema(name: str) -> Dict[str, Any]:
    """NESTFUL-style schema row for a task's `tools` list."""
    t = TOOLS[name]
    props = {}
    required = []
    for p, meta in t["params"].items():
        props[p] = {"type": meta["type"], "description": meta["desc"]}
        if meta.get("required", True):
            required.append(p)
    if t["out_type"] == "object" and t["out_fields"]:
        out_params = {f: {"type": ft, "description": f"The {f.replace('_', ' ')}."}
                      for f, (ft, _sem) in t["out_fields"].items()}
    else:
        out_params = {t["out_key"]: {"type": t["out_type"],
                                     "description": t["description"]}}
    return {
        "name": t["name"],
        "description": t["description"],
        "parameters": {"type": "object", "properties": props, "required": required},
        "output_parameters": out_params,
    }


def sample_args(name: str, rng: random.Random) -> Dict[str, Any]:
    return TOOLS[name]["sample"](rng)


def registry_hash() -> str:
    """Deterministic hash over version + schemas + semantic metadata.

    Implementation bytecode is deliberately NOT hashed (not stable across
    Python versions); any behavioural change must bump REGISTRY_VERSION.
    """
    payload = {"version": REGISTRY_VERSION, "tools": {}}
    for name in ALL_TOOL_NAMES:
        t = TOOLS[name]
        payload["tools"][name] = {
            "domain": t["domain"], "family": t["family"],
            "description": t["description"],
            "params": {p: {k: v for k, v in meta.items() if k != "desc"}
                       for p, meta in t["params"].items()},
            "out_key": t["out_key"], "out_type": t["out_type"],
            "out_semantic": t["out_semantic"], "chain_in": t["chain_in"],
            "out_fields": t["out_fields"],
        }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def semantics_compatible(producer_out_sem: str, consumer_param_sem: str) -> bool:
    """True when a producer output may flow into a consumer parameter.

    Rules (strict to prevent nonsensical compositions):
      * consumer 'generic_number' accepts ANY numeric producer;
      * a consumer with a specific numeric semantic accepts ONLY the same
        specific semantic (money never flows into kilograms);
      * text accepts text; nothing else crosses type classes.
    """
    if consumer_param_sem == "generic_number":
        return producer_out_sem not in _NON_NUMERIC_SEMS
    return producer_out_sem == consumer_param_sem


def numeric_output(name: str) -> bool:
    return TOOLS[name]["out_type"] in ("number", "integer")


# ─────────────────────────────────────────────────────────────────────────────
#  Validation (used by validate_registry.sh and unit tests)
# ─────────────────────────────────────────────────────────────────────────────
# Shared numeric probe values used for behavioural-duplicate detection.
# (5, 12) has first < second so subtraction, absolute difference and
# floor-at-zero variants produce different outputs.
_DUP_PROBES_1 = [7, 33.5, 120]
_DUP_PROBES_2 = [(7, 3), (33.5, 12), (120, 48), (5, 12)]


def _behaviour_signature(t: Dict[str, Any]) -> Optional[str]:
    """Signature from outputs on SHARED probes (None when not probe-compatible).

    Only pure-numeric tools are probed this way; tools taking strings/arrays
    are distinguished by their sampled-args checks instead.
    """
    ptypes = [m["type"] for m in t["params"].values()]
    pnames = list(t["params"].keys())
    required = [p for p, m in t["params"].items() if m.get("required", True)]
    if any(pt not in ("number", "integer") for pt in ptypes):
        return None
    try:
        if len(required) == 1 and len(pnames) == 1:
            outs = [t["fn"](**{pnames[0]: v}) for v in _DUP_PROBES_1]
        elif len(required) == 2 and len(pnames) == 2:
            outs = [t["fn"](**{pnames[0]: a, pnames[1]: b})
                    for a, b in _DUP_PROBES_2]
        else:
            return None
    except Exception:  # noqa: BLE001 — probe outside a tool's domain: skip
        return None
    return f"arity{len(pnames)}::" + json.dumps([str(o) for o in outs])


def _semantic_signature(t: Dict[str, Any]) -> str:
    """Sorted param semantics + output semantic. Two tools with the same
    behaviour AND the same semantic signature are renamed clones (fatal);
    same behaviour with DIFFERENT semantics (rectangle_area vs multiply) is a
    legitimate domain-specific wrapper that changes tool selection."""
    psems = sorted(m["semantic"] for m in t["params"].values())
    return ",".join(psems) + "->" + t["out_semantic"]


def validate_registry(seed: int = 20260715, probes: int = 3) -> Dict[str, Any]:
    """Deterministic self-check of every tool. Returns a report dict.

    Checks per tool: sampled args execute; execution is deterministic (two
    identical runs); output type matches the declaration; object outputs match
    out_fields. Behavioural duplicates are detected on SHARED probe inputs so
    renamed-identical functions cannot masquerade as diversity; allowlisted
    physical-factor twins are reported separately.
    """
    errors: List[str] = []
    rng = random.Random(seed)

    for name in ALL_TOOL_NAMES:
        t = TOOLS[name]
        for _ in range(probes):
            args = t["sample"](rng)
            try:
                o1 = t["fn"](**args)
                o2 = t["fn"](**args)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: execution failed on sampled args {args}: {exc!r}")
                break
            if o1 != o2:
                errors.append(f"{name}: non-deterministic output for {args}")
                break
            ok, why = _check_out_type(t, o1)
            if not ok:
                errors.append(f"{name}: {why} (args={args}, out={o1!r})")
                break

    behaviour_sigs: Dict[str, List[str]] = {}
    for name in ALL_TOOL_NAMES:
        sig = _behaviour_signature(TOOLS[name])
        if sig is not None:
            behaviour_sigs.setdefault(sig, []).append(name)
    duplicates, behaviour_twins = [], []
    for names in behaviour_sigs.values():
        if len(names) <= 1:
            continue
        # Renamed clones: same behaviour AND same semantic signature = fatal.
        by_sem: Dict[str, List[str]] = {}
        for n in names:
            by_sem.setdefault(_semantic_signature(TOOLS[n]), []).append(n)
        for sem_group in by_sem.values():
            if len(sem_group) > 1:
                duplicates.append(sorted(sem_group))
        if len(by_sem) > 1:
            behaviour_twins.append(sorted(names))

    fam_counts: Dict[str, int] = {}
    dom_counts: Dict[str, int] = {}
    arity_counts: Dict[int, int] = {}
    out_type_counts: Dict[str, int] = {}
    for t in TOOLS.values():
        fam_counts[t["family"]] = fam_counts.get(t["family"], 0) + 1
        dom_counts[t["domain"]] = dom_counts.get(t["domain"], 0) + 1
        arity_counts[len(t["params"])] = arity_counts.get(len(t["params"]), 0) + 1
        out_type_counts[t["out_type"]] = out_type_counts.get(t["out_type"], 0) + 1

    return {
        "registry_version": REGISTRY_VERSION,
        "registry_hash": registry_hash(),
        "num_tools": len(TOOLS),
        "num_domains": len(DOMAINS),
        "num_families": len(FAMILIES),
        "family_counts": dict(sorted(fam_counts.items())),
        "domain_counts": dict(sorted(dom_counts.items())),
        "arity_counts": {str(k): v for k, v in sorted(arity_counts.items())},
        "out_type_counts": dict(sorted(out_type_counts.items())),
        "behaviour_duplicates": duplicates,
        "behaviour_twins_distinct_semantics": behaviour_twins,
        "errors": errors,
        "ok": not errors and not duplicates,
    }


def _check_out_type(t: Dict[str, Any], out: Any) -> Tuple[bool, str]:
    ot = t["out_type"]
    if ot == "number":
        if isinstance(out, bool) or not isinstance(out, (int, float)):
            return False, f"declared number, got {type(out).__name__}"
    elif ot == "integer":
        if isinstance(out, bool) or not isinstance(out, int):
            return False, f"declared integer, got {type(out).__name__}"
    elif ot == "boolean":
        if not isinstance(out, bool):
            return False, f"declared boolean, got {type(out).__name__}"
    elif ot == "string":
        if not isinstance(out, str):
            return False, f"declared string, got {type(out).__name__}"
    elif ot == "array":
        if not isinstance(out, list):
            return False, f"declared array, got {type(out).__name__}"
    elif ot == "object":
        if not isinstance(out, dict):
            return False, f"declared object, got {type(out).__name__}"
        fields = t.get("out_fields") or {}
        if set(out.keys()) != set(fields.keys()):
            return False, f"object fields {sorted(out)} != declared {sorted(fields)}"
    return True, ""


if __name__ == "__main__":
    report = validate_registry()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["ok"] else 1)
