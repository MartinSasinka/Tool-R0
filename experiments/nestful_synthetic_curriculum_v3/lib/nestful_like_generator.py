"""curriculum_v4_nestful_like — NESTFUL-style synthetic task generator (Phase 1j).

Design goals (RESEARCH_FIX_PLAN E5): distributionally closer to NESTFUL than
curriculum v3.1 on the dimensions that matter for transfer —
  * realistic snake_case tool names and DESCRIPTIVE parameter names
    (principal_amount, annual_rate_percent, ...), heterogeneous output-key
    styles (output_0 / result / value) like NESTFUL's mixed API corpus;
  * many offered tools per task (NESTFUL offers ~10-25, v3.1 offered ~5-10);
  * call counts 2-6 (NESTFUL median 3; v3.1 was 1-4 with a 1-call stage);
  * NESTFUL-style variable references: label "$varN", reference
    "$varN.<output_key>$";
  * motifs: long chains, argument binding (literal+ref mixes), reference
    reuse, distractor-heavy tool menus, continuation-pressure phrasing.

CONTAMINATION RULES (hard):
  * NO NESTFUL questions, gold traces, tool schemas or answers are copied —
    the tool library below is written from scratch and every question is
    template-generated from sampled numbers;
  * only AGGREGATE NESTFUL statistics (call-count / offered-tool-count
    targets, naming style) informed the design, recorded in `PROVENANCE`;
  * the build script verifies zero overlap with NESTFUL by question hash,
    trace hash and sample_id.

All tools are deterministic pure-python; gold traces are re-executed by
`replay_task` and must reproduce `gold_answer` exactly (gold replay = 1.0).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

GENERATOR_VERSION = "v4.0"

PROVENANCE = {
    "generator": "nestful_like_generator.py",
    "generator_version": GENERATOR_VERSION,
    "nestful_sources_used": [
        "AGGREGATE ONLY: call-count distribution, offered-tools-per-task range, "
        "tool/parameter naming style, $varN.output_key$ reference convention "
        "(observed on experiments/nestful_mtgrpo_minimal/data/splits/*.jsonl)."
    ],
    "nestful_content_copied": "NONE (no questions, no gold traces, no tool schemas)",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Tool library — realistic domains, heterogeneous API styles
# ─────────────────────────────────────────────────────────────────────────────
# ToolSpec fields:
#   name, domain, description, params: {pname: (json_type, description)},
#   out_key (heterogeneous: output_0 / result / value), out_type,
#   fn: python implementation, sample: rng -> {pname: literal},
#   phrase: (args) -> natural-language fragment describing this step,
#   chain_in: name of the numeric param that can receive a previous output
#             (None = tool cannot be chained INTO),
#   chain_out_numeric: whether output can feed a numeric param downstream.

def _r2(x: float) -> float:
    return round(float(x), 2)


TOOLS: Dict[str, Dict[str, Any]] = {}


def _add_tool(name: str, domain: str, description: str,
              params: Dict[str, Tuple[str, str]], out_key: str, out_type: str,
              fn: Callable[..., Any], sample: Callable[[random.Random], Dict[str, Any]],
              phrase: Callable[[Dict[str, Any]], str],
              chain_in: Optional[str], chain_out_numeric: bool) -> None:
    TOOLS[name] = {
        "name": name, "domain": domain, "description": description,
        "params": params, "out_key": out_key, "out_type": out_type,
        "fn": fn, "sample": sample, "phrase": phrase,
        "chain_in": chain_in, "chain_out_numeric": chain_out_numeric,
    }


# --- finance (out_key: output_0) --------------------------------------------
_add_tool(
    "calculate_simple_interest", "finance",
    "Calculates simple interest earned on a principal at an annual rate over a number of years.",
    {"principal_amount": ("number", "The initial amount of money invested."),
     "annual_rate_percent": ("number", "The yearly interest rate in percent."),
     "years": ("number", "The investment duration in years.")},
    "output_0", "number",
    lambda principal_amount, annual_rate_percent, years:
        _r2(principal_amount * annual_rate_percent / 100.0 * years),
    lambda rng: {"principal_amount": rng.randrange(500, 20000, 100),
                 "annual_rate_percent": rng.choice([2, 2.5, 3, 4, 5, 6, 7.5]),
                 "years": rng.randrange(1, 12)},
    lambda a: (f"the simple interest on {a['principal_amount']} at "
               f"{a['annual_rate_percent']}% per year for {a['years']} years"),
    "principal_amount", True)

_add_tool(
    "apply_discount", "finance",
    "Applies a percentage discount to a price and returns the discounted price.",
    {"price": ("number", "The original price."),
     "discount_percent": ("number", "The discount in percent.")},
    "output_0", "number",
    lambda price, discount_percent: _r2(price * (1 - discount_percent / 100.0)),
    lambda rng: {"price": rng.randrange(20, 900),
                 "discount_percent": rng.choice([5, 10, 15, 20, 25, 30, 40])},
    lambda a: f"the price after a {a['discount_percent']}% discount on {a['price']}",
    "price", True)

_add_tool(
    "add_sales_tax", "finance",
    "Adds sales tax to a net price and returns the gross price.",
    {"net_price": ("number", "The price before tax."),
     "tax_rate_percent": ("number", "The sales tax rate in percent.")},
    "output_0", "number",
    lambda net_price, tax_rate_percent: _r2(net_price * (1 + tax_rate_percent / 100.0)),
    lambda rng: {"net_price": rng.randrange(10, 600),
                 "tax_rate_percent": rng.choice([5, 7, 8, 10, 19, 21])},
    lambda a: f"the total after adding {a['tax_rate_percent']}% sales tax to {a['net_price']}",
    "net_price", True)

_add_tool(
    "split_bill_evenly", "finance",
    "Splits a total bill evenly between a number of people and returns the per-person share.",
    {"total_amount": ("number", "The total bill amount."),
     "num_people": ("number", "How many people share the bill.")},
    "output_0", "number",
    lambda total_amount, num_people: _r2(total_amount / num_people),
    lambda rng: {"total_amount": rng.randrange(40, 700),
                 "num_people": rng.randrange(2, 9)},
    lambda a: f"each person's share when {a['total_amount']} is split between {a['num_people']} people",
    "total_amount", True)

_add_tool(
    "calculate_tip_amount", "finance",
    "Calculates the tip for a bill given a tip percentage.",
    {"bill_amount": ("number", "The bill total."),
     "tip_percent": ("number", "The tip percentage.")},
    "output_0", "number",
    lambda bill_amount, tip_percent: _r2(bill_amount * tip_percent / 100.0),
    lambda rng: {"bill_amount": rng.randrange(15, 400),
                 "tip_percent": rng.choice([10, 12, 15, 18, 20])},
    lambda a: f"a {a['tip_percent']}% tip on a bill of {a['bill_amount']}",
    "bill_amount", True)

_add_tool(
    "convert_currency", "finance",
    "Converts an amount of money using a fixed exchange rate.",
    {"amount": ("number", "The amount in the source currency."),
     "exchange_rate": ("number", "Units of target currency per source unit.")},
    "output_0", "number",
    lambda amount, exchange_rate: _r2(amount * exchange_rate),
    lambda rng: {"amount": rng.randrange(50, 3000),
                 "exchange_rate": rng.choice([0.85, 0.92, 1.08, 1.25, 4.5, 23.5])},
    lambda a: f"{a['amount']} converted at an exchange rate of {a['exchange_rate']}",
    "amount", True)

_add_tool(
    "monthly_installment", "finance",
    "Computes the flat monthly installment for a loan repaid over a number of months (no interest).",
    {"loan_amount": ("number", "Total amount borrowed."),
     "num_months": ("number", "Number of monthly payments.")},
    "output_0", "number",
    lambda loan_amount, num_months: _r2(loan_amount / num_months),
    lambda rng: {"loan_amount": rng.randrange(1000, 30000, 500),
                 "num_months": rng.choice([6, 12, 24, 36, 48])},
    lambda a: f"the monthly installment for a {a['loan_amount']} loan over {a['num_months']} months",
    "loan_amount", True)

# --- geometry (out_key: result) ----------------------------------------------
_add_tool(
    "rectangle_area", "geometry",
    "Computes the area of a rectangle from its side lengths.",
    {"length": ("number", "The rectangle length."),
     "width": ("number", "The rectangle width.")},
    "result", "number",
    lambda length, width: _r2(length * width),
    lambda rng: {"length": rng.randrange(2, 60), "width": rng.randrange(2, 40)},
    lambda a: f"the area of a rectangle {a['length']} by {a['width']}",
    "length", True)

_add_tool(
    "rectangle_perimeter", "geometry",
    "Computes the perimeter of a rectangle from its side lengths.",
    {"length": ("number", "The rectangle length."),
     "width": ("number", "The rectangle width.")},
    "result", "number",
    lambda length, width: _r2(2 * (length + width)),
    lambda rng: {"length": rng.randrange(2, 60), "width": rng.randrange(2, 40)},
    lambda a: f"the perimeter of a rectangle {a['length']} by {a['width']}",
    "length", True)

_add_tool(
    "circle_area", "geometry",
    "Computes the area of a circle from its radius (pi = 3.14159).",
    {"radius": ("number", "The circle radius.")},
    "result", "number",
    lambda radius: _r2(3.14159 * radius * radius),
    lambda rng: {"radius": rng.randrange(1, 25)},
    lambda a: f"the area of a circle with radius {a['radius']}",
    "radius", True)

_add_tool(
    "triangle_area", "geometry",
    "Computes the area of a triangle from base and height.",
    {"base": ("number", "The triangle base."),
     "height": ("number", "The triangle height.")},
    "result", "number",
    lambda base, height: _r2(base * height / 2.0),
    lambda rng: {"base": rng.randrange(2, 50), "height": rng.randrange(2, 40)},
    lambda a: f"the area of a triangle with base {a['base']} and height {a['height']}",
    "base", True)

_add_tool(
    "scale_dimension", "geometry",
    "Scales a dimension by a multiplicative factor.",
    {"dimension": ("number", "The dimension to scale."),
     "scale_factor": ("number", "The multiplicative scale factor.")},
    "result", "number",
    lambda dimension, scale_factor: _r2(dimension * scale_factor),
    lambda rng: {"dimension": rng.randrange(2, 80),
                 "scale_factor": rng.choice([0.5, 1.5, 2, 2.5, 3])},
    lambda a: f"{a['dimension']} scaled by a factor of {a['scale_factor']}",
    "dimension", True)

# --- physics / units (out_key: value) ----------------------------------------
_add_tool(
    "celsius_to_fahrenheit", "units",
    "Converts a temperature from Celsius to Fahrenheit.",
    {"celsius": ("number", "Temperature in degrees Celsius.")},
    "value", "number",
    lambda celsius: _r2(celsius * 9 / 5 + 32),
    lambda rng: {"celsius": rng.randrange(-20, 45)},
    lambda a: f"{a['celsius']} degrees Celsius converted to Fahrenheit",
    "celsius", True)

_add_tool(
    "kilometers_to_miles", "units",
    "Converts a distance from kilometers to miles.",
    {"kilometers": ("number", "Distance in kilometers.")},
    "value", "number",
    lambda kilometers: _r2(kilometers * 0.621371),
    lambda rng: {"kilometers": rng.randrange(5, 900)},
    lambda a: f"{a['kilometers']} kilometers converted to miles",
    "kilometers", True)

_add_tool(
    "kilograms_to_pounds", "units",
    "Converts a mass from kilograms to pounds.",
    {"kilograms": ("number", "Mass in kilograms.")},
    "value", "number",
    lambda kilograms: _r2(kilograms * 2.20462),
    lambda rng: {"kilograms": rng.randrange(1, 250)},
    lambda a: f"{a['kilograms']} kilograms converted to pounds",
    "kilograms", True)

_add_tool(
    "average_speed", "units",
    "Computes average speed from distance and travel time.",
    {"distance_km": ("number", "Distance travelled in kilometers."),
     "time_hours": ("number", "Travel time in hours.")},
    "value", "number",
    lambda distance_km, time_hours: _r2(distance_km / time_hours),
    lambda rng: {"distance_km": rng.randrange(30, 1200),
                 "time_hours": rng.choice([0.5, 1, 1.5, 2, 3, 4, 5, 8])},
    lambda a: f"the average speed for {a['distance_km']} km in {a['time_hours']} hours",
    "distance_km", True)

_add_tool(
    "travel_time_hours", "units",
    "Computes travel time in hours from distance and constant speed.",
    {"distance_km": ("number", "Distance to travel in kilometers."),
     "speed_kmh": ("number", "Constant speed in km/h.")},
    "value", "number",
    lambda distance_km, speed_kmh: _r2(distance_km / speed_kmh),
    lambda rng: {"distance_km": rng.randrange(40, 1500),
                 "speed_kmh": rng.choice([50, 60, 80, 90, 100, 120])},
    lambda a: f"the travel time for {a['distance_km']} km at {a['speed_kmh']} km/h",
    "distance_km", True)

_add_tool(
    "hours_to_minutes", "units",
    "Converts a duration from hours to minutes.",
    {"hours": ("number", "Duration in hours.")},
    "value", "number",
    lambda hours: _r2(hours * 60),
    lambda rng: {"hours": rng.choice([0.5, 1, 1.5, 2, 3, 4, 6, 8])},
    lambda a: f"{a['hours']} hours converted to minutes",
    "hours", True)

_add_tool(
    "fuel_needed_liters", "units",
    "Computes fuel needed for a trip from distance and consumption per 100 km.",
    {"distance_km": ("number", "Trip distance in kilometers."),
     "consumption_per_100km": ("number", "Fuel consumption in liters per 100 km.")},
    "value", "number",
    lambda distance_km, consumption_per_100km: _r2(distance_km * consumption_per_100km / 100.0),
    lambda rng: {"distance_km": rng.randrange(50, 1200),
                 "consumption_per_100km": rng.choice([4.5, 5.5, 6, 7, 8, 9.5])},
    lambda a: (f"the fuel needed for {a['distance_km']} km at "
               f"{a['consumption_per_100km']} liters per 100 km"),
    "distance_km", True)

# --- statistics (out_key: output_0; array-typed params) -----------------------
def _num_list(rng: random.Random, lo=1, hi=100, n_lo=4, n_hi=8) -> List[float]:
    return [rng.randrange(lo, hi) for _ in range(rng.randrange(n_lo, n_hi))]


_add_tool(
    "mean_of_values", "statistics",
    "Computes the arithmetic mean of a list of numbers.",
    {"values": ("array", "The list of numeric values.")},
    "output_0", "number",
    lambda values: _r2(sum(values) / len(values)),
    lambda rng: {"values": _num_list(rng)},
    lambda a: f"the mean of {a['values']}",
    None, True)

_add_tool(
    "sum_of_values", "statistics",
    "Computes the sum of a list of numbers.",
    {"values": ("array", "The list of numeric values.")},
    "output_0", "number",
    lambda values: _r2(sum(values)),
    lambda rng: {"values": _num_list(rng)},
    lambda a: f"the sum of {a['values']}",
    None, True)

_add_tool(
    "max_of_values", "statistics",
    "Returns the largest number in a list.",
    {"values": ("array", "The list of numeric values.")},
    "output_0", "number",
    lambda values: _r2(max(values)),
    lambda rng: {"values": _num_list(rng)},
    lambda a: f"the largest of {a['values']}",
    None, True)

_add_tool(
    "value_range", "statistics",
    "Computes the range (max minus min) of a list of numbers.",
    {"values": ("array", "The list of numeric values.")},
    "output_0", "number",
    lambda values: _r2(max(values) - min(values)),
    lambda rng: {"values": _num_list(rng)},
    lambda a: f"the range of {a['values']}",
    None, True)

_add_tool(
    "percentage_of", "statistics",
    "Computes what percentage the part is of the whole.",
    {"part": ("number", "The part value."),
     "whole": ("number", "The whole value.")},
    "output_0", "number",
    lambda part, whole: _r2(part / whole * 100.0),
    lambda rng: {"part": rng.randrange(5, 90), "whole": rng.randrange(100, 500)},
    lambda a: f"what percentage {a['part']} is of {a['whole']}",
    "part", True)

_add_tool(
    "increase_by_percent", "statistics",
    "Increases a value by a given percentage.",
    {"value": ("number", "The starting value."),
     "percent": ("number", "The percentage increase.")},
    "output_0", "number",
    lambda value, percent: _r2(value * (1 + percent / 100.0)),
    lambda rng: {"value": rng.randrange(20, 800),
                 "percent": rng.choice([5, 10, 15, 20, 25, 50])},
    lambda a: f"{a['value']} increased by {a['percent']}%",
    "value", True)

_add_tool(
    "difference_of", "statistics",
    "Computes the difference between two values (first minus second).",
    {"minuend": ("number", "The value to subtract from."),
     "subtrahend": ("number", "The value to subtract.")},
    "output_0", "number",
    lambda minuend, subtrahend: _r2(minuend - subtrahend),
    lambda rng: {"minuend": rng.randrange(100, 900), "subtrahend": rng.randrange(5, 90)},
    lambda a: f"the difference between {a['minuend']} and {a['subtrahend']}",
    "minuend", True)

_add_tool(
    "ratio_of", "statistics",
    "Computes the ratio of two values (first divided by second).",
    {"numerator": ("number", "The numerator."),
     "denominator": ("number", "The denominator.")},
    "output_0", "number",
    lambda numerator, denominator: _r2(numerator / denominator),
    lambda rng: {"numerator": rng.randrange(50, 900),
                 "denominator": rng.randrange(2, 40)},
    lambda a: f"the ratio of {a['numerator']} to {a['denominator']}",
    "numerator", True)

# --- shopping / inventory ------------------------------------------------------
_add_tool(
    "total_price", "shopping",
    "Computes the total price for a quantity of items at a unit price.",
    {"unit_price": ("number", "Price of one item."),
     "quantity": ("number", "Number of items.")},
    "output_0", "number",
    lambda unit_price, quantity: _r2(unit_price * quantity),
    lambda rng: {"unit_price": rng.choice([1.5, 2.5, 4, 7.99, 12, 25, 49.9]),
                 "quantity": rng.randrange(2, 40)},
    lambda a: f"the total price of {a['quantity']} items at {a['unit_price']} each",
    "unit_price", True)

_add_tool(
    "remaining_stock", "shopping",
    "Computes remaining stock after a number of units are sold.",
    {"initial_stock": ("number", "Units in stock initially."),
     "units_sold": ("number", "Units sold.")},
    "output_0", "number",
    lambda initial_stock, units_sold: _r2(initial_stock - units_sold),
    lambda rng: {"initial_stock": rng.randrange(100, 900),
                 "units_sold": rng.randrange(10, 90)},
    lambda a: f"the remaining stock after selling {a['units_sold']} of {a['initial_stock']} units",
    "initial_stock", True)

_add_tool(
    "is_above_threshold", "shopping",
    "Checks whether a value is strictly greater than a threshold.",
    {"value": ("number", "The value to check."),
     "threshold": ("number", "The threshold to compare against.")},
    "result", "boolean",
    lambda value, threshold: bool(value > threshold),
    lambda rng: {"value": rng.randrange(10, 500), "threshold": rng.randrange(10, 500)},
    lambda a: f"whether it exceeds {a['threshold']}",
    "value", False)

_add_tool(
    "units_per_box", "shopping",
    "Computes how many full boxes are needed for a number of units (integer division, rounding up).",
    {"total_units": ("number", "Total units to pack."),
     "box_capacity": ("number", "Units per box.")},
    "output_0", "number",
    lambda total_units, box_capacity: float(-(-int(total_units) // int(box_capacity))),
    lambda rng: {"total_units": rng.randrange(20, 900),
                 "box_capacity": rng.choice([6, 8, 10, 12, 24])},
    lambda a: f"the number of boxes of {a['box_capacity']} needed for {a['total_units']} units",
    "total_units", True)

# --- text (out_key: output_0; string outputs) ---------------------------------
_WORDS = ["ledger", "harbor", "signal", "meadow", "copper", "lantern", "orchid",
          "summit", "quartz", "violet", "anchor", "breeze"]

_add_tool(
    "format_as_currency", "text",
    "Formats a numeric amount as a currency string with two decimals and a symbol prefix.",
    {"amount": ("number", "The numeric amount."),
     "currency_symbol": ("string", "The currency symbol to prefix.")},
    "output_0", "string",
    lambda amount, currency_symbol: f"{currency_symbol}{float(amount):.2f}",
    lambda rng: {"amount": _r2(rng.uniform(5, 900)),
                 "currency_symbol": rng.choice(["$", "€", "£"])},
    lambda a: f"formatted as a currency string with the symbol {a['currency_symbol']}",
    "amount", False)

_add_tool(
    "repeat_word", "text",
    "Repeats a word a given number of times separated by hyphens.",
    {"word": ("string", "The word to repeat."),
     "times": ("number", "How many times to repeat it.")},
    "output_0", "string",
    lambda word, times: "-".join([str(word)] * int(times)),
    lambda rng: {"word": rng.choice(_WORDS), "times": rng.randrange(2, 5)},
    lambda a: f"the word '{a['word']}' repeated {a['times']} times separated by hyphens",
    None, False)  # never chain INTO 'times' (an upstream value could be huge)

_add_tool(
    "character_count", "text",
    "Counts the number of characters in a text string.",
    {"text": ("string", "The text to measure.")},
    "output_0", "number",
    lambda text: float(len(str(text))),
    lambda rng: {"text": " ".join(rng.sample(_WORDS, rng.randrange(2, 5)))},
    lambda a: f"the number of characters in '{a['text']}'",
    "text", True)


ALL_TOOL_NAMES = sorted(TOOLS.keys())
DOMAINS = sorted({t["domain"] for t in TOOLS.values()})


# ─────────────────────────────────────────────────────────────────────────────
#  Schema / execution helpers
# ─────────────────────────────────────────────────────────────────────────────

def tool_schema(name: str) -> Dict[str, Any]:
    """OpenAI-ish schema row for the task's `tools` list (NESTFUL-style)."""
    t = TOOLS[name]
    props = {p: {"type": typ, "description": desc}
             for p, (typ, desc) in t["params"].items()}
    return {
        "name": t["name"],
        "description": t["description"],
        "parameters": {"type": "object", "properties": props,
                       "required": list(props.keys())},
        "output_parameters": {t["out_key"]: {"type": t["out_type"],
                                             "description": t["description"]}},
    }


def _ref(label: str, out_key: str) -> str:
    return f"${label.lstrip('$')}.{out_key}$"


_REF_RE = re.compile(r"^\$([A-Za-z_]\w*)(?:\.\w+)?\$$")


def execute_call(name: str, args: Dict[str, Any], scope: Dict[str, Any]) -> Any:
    """Execute one gold call, resolving $varN.key$ references from `scope`."""
    t = TOOLS[name]
    resolved = {}
    for k, v in args.items():
        m = _REF_RE.match(v.strip()) if isinstance(v, str) else None
        if m:
            var = m.group(1)
            if var not in scope:
                raise KeyError(f"unresolved reference {v}")
            resolved[k] = scope[var]
        else:
            resolved[k] = v
    return t["fn"](**resolved)


def replay_task(row: Dict[str, Any]) -> Tuple[bool, Any]:
    """Re-execute gold_calls; True iff the final observation equals gold_answer."""
    scope: Dict[str, Any] = {}
    obs = None
    try:
        for call in row["gold_calls"]:
            obs = execute_call(call["name"], call["arguments"], scope)
            scope[call["label"].lstrip("$")] = obs
    except Exception as exc:  # noqa: BLE001
        return False, f"replay_error: {type(exc).__name__}: {exc}"
    return obs == row["gold_answer"], obs


# ─────────────────────────────────────────────────────────────────────────────
#  Chain construction
# ─────────────────────────────────────────────────────────────────────────────

_CHAINABLE = [n for n, t in TOOLS.items() if t["chain_in"] and t["chain_out_numeric"]]
_CHAIN_STARTERS = [n for n, t in TOOLS.items() if t["chain_out_numeric"]]
_CHAIN_ENDERS = [n for n, t in TOOLS.items() if t["chain_in"]]


def _build_chain(rng: random.Random, n_calls: int, motif: str
                 ) -> Tuple[List[Dict[str, Any]], List[Any], List[str]]:
    """Build an executable gold chain. Returns (gold_calls, observations, phrases)."""
    calls: List[Dict[str, Any]] = []
    observations: List[Any] = []
    phrases: List[str] = []
    scope: Dict[str, Any] = {}

    reuse_slot: Optional[Tuple[str, str]] = None  # (label, out_key) reused later

    for i in range(n_calls):
        label = f"$var{i + 1}"
        if i == 0:
            name = rng.choice(_CHAIN_STARTERS if n_calls > 1 else list(TOOLS))
            t = TOOLS[name]
            args = t["sample"](rng)
            phrases.append(f"compute {t['phrase'](args)}")
        else:
            last = i == n_calls - 1
            pool = _CHAIN_ENDERS if last else _CHAINABLE
            name = rng.choice(pool)
            t = TOOLS[name]
            args = t["sample"](rng)
            prev_label = f"var{i}"
            prev_tool = TOOLS[calls[-1]["name"]]
            args[t["chain_in"]] = _ref(prev_label, prev_tool["out_key"])
            # question text stays natural language: refs are described, never
            # shown as $varN$ syntax (that would leak the trace format)
            display = dict(args)
            display[t["chain_in"]] = "that result"
            # reference-reuse motif: one extra arg (if any numeric arg exists)
            # rebinds an EARLIER variable instead of a literal
            if motif == "reference_reuse" and reuse_slot and i >= 2:
                for p, (typ, _d) in t["params"].items():
                    if p != t["chain_in"] and typ == "number":
                        args[p] = _ref(*reuse_slot)
                        display[p] = "the first step's value"
                        break
                reuse_slot = None
            phrases.append(f"use that result to get {t['phrase'](display)}")
        calls.append({"name": name, "arguments": args, "label": label})
        obs = execute_call(name, args, scope)
        scope[f"var{i + 1}"] = obs
        observations.append(obs)
        if motif == "reference_reuse" and i == 0:
            reuse_slot = (f"var{i + 1}", TOOLS[name]["out_key"])
    return calls, observations, phrases


def _question_from_phrases(rng: random.Random, phrases: List[str], n_calls: int) -> str:
    """Natural-language question with continuation pressure (multi-step phrasing)."""
    style = rng.choice(["enumerated", "flowing", "imperative"])
    if style == "enumerated":
        steps = [f"{i + 1}) {p}" for i, p in enumerate(phrases)]
        q = ("Solve the following in order, using each intermediate result for the "
             "next step: " + "; ".join(steps)
             + f". Report the final value after all {n_calls} steps.")
    elif style == "flowing":
        connectors = ["Then", "Next", "After that", "Finally"]
        parts = [phrases[0].capitalize()]
        for i, p in enumerate(phrases[1:]):
            c = connectors[min(i, len(connectors) - 1)] if i < len(phrases) - 2 \
                else "Finally"
            parts.append(f"{c}, {p}")
        q = ". ".join(parts) + ". What is the final result?"
    else:
        q = ("First " + phrases[0] + ", " + ", then ".join(phrases[1:])
             + ". Give the value produced by the last step.")
    return q


def _offered_tools(rng: random.Random, used: List[str], n_offered: int) -> List[str]:
    """Used tools + same-domain and off-domain distractors, shuffled."""
    used_set = set(used)
    domains_used = {TOOLS[n]["domain"] for n in used}
    same_domain = [n for n in ALL_TOOL_NAMES
                   if n not in used_set and TOOLS[n]["domain"] in domains_used]
    other = [n for n in ALL_TOOL_NAMES
             if n not in used_set and TOOLS[n]["domain"] not in domains_used]
    rng.shuffle(same_domain)
    rng.shuffle(other)
    need = max(0, n_offered - len(used))
    n_same = min(len(same_domain), max(1, need // 2))
    distractors = same_domain[:n_same] + other[:need - n_same]
    offered = list(used) + distractors
    rng.shuffle(offered)
    return offered


def _answer_type(val: Any) -> str:
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, (int, float)):
        return "scalar"
    if isinstance(val, str):
        return "string"
    if isinstance(val, list):
        return "list"
    return "object"


def question_hash(q: str) -> str:
    return hashlib.sha256(" ".join(str(q).lower().split()).encode()).hexdigest()


def trace_hash(gold_calls: List[Dict[str, Any]]) -> str:
    canon = json.dumps([[c["name"], c["arguments"]] for c in gold_calls],
                       sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
#  Task generation
# ─────────────────────────────────────────────────────────────────────────────

STAGES = {
    "v4_stage1_2call": {"n_calls": (2, 2)},
    "v4_stage2_3call": {"n_calls": (3, 3)},
    "v4_stage3_4call": {"n_calls": (4, 4)},
    "v4_stage4_5to6call": {"n_calls": (5, 6)},
}

MOTIFS = ("long_chain", "argument_binding", "reference_reuse", "distractor_heavy")


def generate_task(rng: random.Random, stage: str, motif: str, seed: int,
                  idx: int) -> Dict[str, Any]:
    lo, hi = STAGES[stage]["n_calls"]
    n_calls = rng.randrange(lo, hi + 1)
    calls, observations, phrases = _build_chain(rng, n_calls, motif)
    question = _question_from_phrases(rng, phrases, n_calls)
    gold_answer = observations[-1]
    used = [c["name"] for c in calls]
    # NESTFUL-like offered-tool counts; distractor_heavy pushes the upper range
    n_offered = rng.randrange(16, 26) if motif == "distractor_heavy" \
        else rng.randrange(10, 20)
    offered = _offered_tools(rng, used, n_offered)
    sid_body = f"{stage}_{motif}_{seed}_{idx:05d}"
    sid = f"v4_{sid_body}_{question_hash(question)[:8]}"
    return {
        "sample_id": sid,
        "question": question,
        "tools": [tool_schema(n) for n in offered],
        "gold_calls": calls,
        "observations": observations,
        "gold_answer": gold_answer,
        "num_calls": n_calls,
        "stage": stage,
        "motif_type": motif,
        "answer_type": _answer_type(gold_answer),
        "terminal_stage": True,
        "source": "curriculum_v4_nestful_like",
        "generation_seed": seed,
        "provenance": PROVENANCE,
    }


def generate_stage(stage: str, n_examples: int, seed: int,
                   forbidden_question_hashes: Optional[set] = None,
                   forbidden_trace_hashes: Optional[set] = None,
                   ) -> List[Dict[str, Any]]:
    """Deterministic per-stage generation with in-corpus and external dedup."""
    rng = random.Random(f"{GENERATOR_VERSION}|{stage}|{seed}")
    rows: List[Dict[str, Any]] = []
    seen_q: set = set(forbidden_question_hashes or set())
    seen_t: set = set(forbidden_trace_hashes or set())
    attempts = 0
    max_attempts = n_examples * 60
    while len(rows) < n_examples and attempts < max_attempts:
        attempts += 1
        motif = MOTIFS[len(rows) % len(MOTIFS)]
        row = generate_task(rng, stage, motif, seed, len(rows))
        qh, th = question_hash(row["question"]), trace_hash(row["gold_calls"])
        if qh in seen_q or th in seen_t:
            continue
        ok, _obs = replay_task(row)
        if not ok:
            raise RuntimeError(f"gold replay failed during generation: {row['sample_id']}")
        seen_q.add(qh)
        seen_t.add(th)
        rows.append(row)
    if len(rows) < n_examples:
        raise RuntimeError(
            f"{stage}: exhausted {max_attempts} attempts at {len(rows)}/{n_examples} "
            "unique examples — enlarge the tool library or value ranges.")
    return rows
