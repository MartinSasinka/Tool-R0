"""Semantic primitive registry.

A primitive = typed deterministic operation with:
  - semantic id + category,
  - typed params (with semantics for composition),
  - unit metadata for semantic-plausibility control (pilot2),
  - deterministic fn,
  - direct-value samplers,
  - surface variants per track (A = NESTFUL-like morphology, G = independent
    vocabulary). Surfaces vary names, param names and descriptions; semantics
    never change (capability 2, 8 — schema reading vs name matching).

Surface names are GLOBALLY UNIQUE and each name maps to exactly one parameter
signature (enforced at import). This is required by the trainer's synthetic
executor, which resolves a call by tool name against a single global registry
(pilot1 violated it: "add" existed with two different parameter sets).

REGISTRY_HASH covers ids, types, units and surface signatures; any change is a
new dataset version.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..util import sha256_obj

NUM = "number"
INT = "integer"
STR = "string"
ARR = "array"
BOOL = "boolean"

# ── units (semantic plausibility, pilot2) ────────────────────────────────
U_ABSTRACT = "abstract"       # plain dimensionless quantity
U_COUNT = "count"
U_RATIO = "ratio"
U_PERCENT = "percent"
U_DUR_S = "duration_s"
U_DUR_MIN = "duration_min"
U_DUR_H = "duration_h"
U_LEN_KM = "length_km"
U_LEN_M = "length_m"
U_TEMP_C = "temperature_c"
U_TEMP_F = "temperature_f"
U_TEXT = "text"
U_NUMTEXT = "numeric_text"
U_LIST = "list_number"
U_FLAG = "flag"

PHYSICAL_UNITS = {U_DUR_S, U_DUR_MIN, U_DUR_H, U_LEN_KM, U_LEN_M,
                  U_TEMP_C, U_TEMP_F}
NEUTRAL_UNITS = {U_ABSTRACT, U_COUNT, U_RATIO}

# param unit expectations
E_ANY = "any"                 # any numeric quantity is acceptable
E_NEUTRAL = "neutral"         # must be dimensionless (percentage, exponent…)
PRESERVE = "preserve"         # output unit == unit of the first numeric input


def _r6(x: float) -> float:
    return round(float(x) + 0.0, 6)


@dataclass
class Surface:
    surface_id: str
    name: str
    param_names: List[str]
    description: str
    output_field: str = "output_0"
    param_descs: Optional[List[str]] = None


@dataclass
class Primitive:
    sid: str
    category: str
    params: List[Tuple[str, str, str]]       # (canonical_name, type, semantic)
    out_type: str
    out_semantic: str
    fn: Callable[..., Any]
    sampler: Callable[[random.Random], List[Any]]
    surfaces_a: List[Surface] = field(default_factory=list)
    surfaces_g: List[Surface] = field(default_factory=list)
    phrase: str = ""                         # query phrase template
    phrase_ref: str = ""                     # variant used when input is a ref
    param_units: List[str] = field(default_factory=list)   # per param
    unit_out: str = U_ABSTRACT               # specific unit or PRESERVE
    answer_kind: str = "float"               # float|int|bool|string|list|numeric_string

    def surfaces(self, track: str) -> List[Surface]:
        return self.surfaces_a if track == "A" else self.surfaces_g

    def unit_of_output(self, in_units: List[str]) -> str:
        if self.unit_out == PRESERVE:
            for u in in_units:
                if u not in (U_TEXT, U_NUMTEXT, U_LIST, U_FLAG):
                    return u
            return U_ABSTRACT
        return self.unit_out


_REG: Dict[str, Primitive] = {}


def _int(lo: int, hi: int):
    return lambda rng: rng.randint(lo, hi)


def _flt(lo: float, hi: float, nd: int = 2):
    return lambda rng: round(rng.uniform(lo, hi), nd)


def _sam(*gens):
    return lambda rng: [g(rng) for g in gens]


def _lst(n_lo=3, n_hi=6, lo=1, hi=60):
    return lambda rng: [rng.randint(lo, hi) for _ in range(rng.randint(n_lo, n_hi))]


def _add_prim(p: Primitive) -> None:
    assert p.sid not in _REG, p.sid
    if not p.param_units:
        p.param_units = [E_ANY] * len(p.params)
    assert len(p.param_units) == len(p.params), p.sid
    _REG[p.sid] = p


def _S(sfid, name, pnames, desc, out="output_0"):
    return Surface(surface_id=sfid, name=name, param_names=list(pnames),
                   description=desc, output_field=out)


# ── arithmetic (binary number,number -> number) ──────────────────────────
def _bin(sid, fn, phrase, a_surfs, g_surfs, sam=None, sem=("operand", "operand"),
         punits=(E_ANY, E_ANY), unit_out=PRESERVE, phrase_ref=""):
    _add_prim(Primitive(
        sid=sid, category="arithmetic",
        params=[("a", NUM, sem[0]), ("b", NUM, sem[1])],
        out_type=NUM, out_semantic="quantity",
        fn=lambda a, b: _r6(fn(float(a), float(b))),
        sampler=sam or _sam(_int(3, 900), _int(2, 90)),
        surfaces_a=a_surfs, surfaces_g=g_surfs, phrase=phrase,
        phrase_ref=phrase_ref or phrase,
        param_units=list(punits), unit_out=unit_out))


_bin("add", lambda a, b: a + b, "add {a} and {b}",
     [_S("a1", "add", ["arg_0", "arg_1"], "Adds two numbers and returns the result."),
      _S("a2", "sum_two_numbers", ["first_number", "second_number"],
         "Returns the sum of the two inputs.")],
     [_S("g1", "combine_amounts", ["amount_a", "amount_b"],
         "Combines two quantities into their total."),
      _S("g2", "total_of_pair", ["left", "right"],
         "Returns the total of the given pair of values.")])

_bin("subtract", lambda a, b: a - b, "subtract {b} from {a}",
     [_S("a1", "subtract", ["arg_0", "arg_1"], "Subtracts the second number from the first."),
      _S("a2", "difference_of_numbers", ["minuend", "subtrahend"],
         "Returns minuend minus subtrahend.")],
     [_S("g1", "reduce_amount", ["base_value", "reduction"],
         "Reduces a base value by the given amount."),
      _S("g2", "remaining_after", ["from_value", "take_away"],
         "Computes how much remains after removal.")])

_bin("multiply", lambda a, b: a * b, "multiply {a} by {b}",
     [_S("a1", "multiply", ["arg_0", "arg_1"], "Multiplies two numbers."),
      _S("a2", "product_of_numbers", ["first_factor", "second_factor"],
         "Returns the product of the factors.")],
     [_S("g1", "scale_quantity", ["quantity", "factor"], "Scales a quantity by a factor."),
      _S("g2", "pairwise_product", ["value_one", "value_two"],
         "Returns the pairwise product.")],
     sam=_sam(_int(3, 400), _int(2, 40)), punits=(E_ANY, E_NEUTRAL))

_bin("divide", lambda a, b: a / b, "divide {a} by {b}",
     [_S("a1", "divide", ["arg_0", "arg_1"], "Divides the first number by the second."),
      _S("a2", "quotient_of", ["numerator", "denominator"],
         "Returns numerator / denominator.")],
     [_S("g1", "split_evenly", ["total", "parts"],
         "Splits a total into equal parts and returns one part."),
      _S("g2", "per_unit_value", ["whole", "units"], "Computes the value per unit.")],
     sam=_sam(_int(53, 8971), _int(3, 47)), punits=(E_ANY, E_NEUTRAL))

_bin("power", lambda a, b: a ** b, "raise {a} to the power of {b}",
     [_S("a1", "power", ["arg_0", "arg_1"], "Raises the base to the given exponent."),
      _S("a2", "exponent_of", ["base", "exponent"], "Computes base ** exponent.")],
     [_S("g1", "exponentiate", ["base_value", "power_value"],
         "Applies exponentiation to the base.")],
     sam=_sam(_int(2, 12), _int(2, 3)), punits=(E_NEUTRAL, E_NEUTRAL),
     unit_out=U_ABSTRACT)

_bin("floor_divide", lambda a, b: math.floor(a / b),
     "compute how many whole times {b} fits into {a}",
     [_S("a1", "floor_divide", ["arg_0", "arg_1"], "Integer division rounded down."),
      _S("a2", "whole_quotient", ["dividend", "divisor"],
         "Returns the whole-number quotient.")],
     [_S("g1", "full_groups_of", ["total_items", "group_size"],
         "Counts complete groups of the given size.")],
     sam=_sam(_int(47, 4993), _int(3, 59)), punits=(E_ANY, E_NEUTRAL),
     unit_out=U_COUNT)

_bin("modulo", lambda a, b: a % b, "find the remainder of {a} divided by {b}",
     [_S("a1", "reminder", ["arg_0", "arg_1"], "Returns the remainder after division."),
      _S("a2", "modulo_of", ["dividend", "divisor"], "Computes dividend mod divisor.")],
     [_S("g1", "leftover_after_grouping", ["total_items", "group_size"],
         "Returns items left over after grouping.")],
     sam=_sam(_int(43, 2999), _int(3, 53)), punits=(E_ANY, E_NEUTRAL))

_bin("percent_of", lambda a, b: a * b / 100.0, "compute {a} percent of {b}",
     [_S("a1", "percent_of", ["percent", "whole"],
         "Computes the given percent of a value."),
      _S("a2", "percentage_value", ["rate_percent", "base"],
         "Returns rate_percent% of base.")],
     [_S("g1", "portion_by_rate", ["rate", "reference_value"],
         "Applies a percentage rate to a reference value.")],
     sam=_sam(_int(4, 95), _int(43, 3989)), sem=("percentage", "base"),
     punits=(E_NEUTRAL, E_ANY), unit_out=PRESERVE)

_bin("ratio_of", lambda a, b: a / b, "compute the ratio of {a} to {b}",
     [_S("a1", "ratio_of", ["numerator", "denominator"],
         "Computes the ratio of two quantities.")],
     [_S("g1", "proportion_between", ["part_value", "whole_value"],
         "Returns the proportion between two values.")],
     sam=_sam(_int(31, 907), _int(7, 89)), unit_out=U_RATIO)

_bin("abs_difference", lambda a, b: abs(a - b),
     "find the absolute difference between {a} and {b}",
     [_S("a1", "absolute_difference", ["arg_0", "arg_1"], "Returns |a - b|.")],
     [_S("g1", "gap_between", ["first_value", "second_value"],
         "Measures the gap between two values.")])

_bin("average_two", lambda a, b: (a + b) / 2.0, "average {a} and {b}",
     [_S("a1", "average_of_two", ["arg_0", "arg_1"], "Averages two numbers.")],
     [_S("g1", "midpoint_value", ["value_one", "value_two"],
         "Returns the midpoint of two values.")])

_bin("max_two", lambda a, b: max(a, b), "take the larger of {a} and {b}",
     [_S("a1", "maximum_of_two", ["arg_0", "arg_1"], "Returns the larger number.")],
     [_S("g1", "pick_higher", ["candidate_a", "candidate_b"],
         "Picks the higher of two candidates.")])

_bin("min_two", lambda a, b: min(a, b), "take the smaller of {a} and {b}",
     [_S("a1", "minimum_of_two", ["arg_0", "arg_1"], "Returns the smaller number.")],
     [_S("g1", "pick_lower", ["candidate_a", "candidate_b"],
         "Picks the lower of two candidates.")])

_bin("increase_by_percent", lambda a, b: a * (1 + b / 100.0),
     "increase {a} by {b} percent",
     [_S("a1", "increase_by_percent", ["base", "percent"],
         "Increases a value by the given percent.")],
     [_S("g1", "grow_by_rate", ["starting_value", "growth_rate"],
         "Grows a value by a percentage rate.")],
     sam=_sam(_int(43, 1997), _int(3, 60)), sem=("base", "percentage"),
     punits=(E_ANY, E_NEUTRAL))

_bin("decrease_by_percent", lambda a, b: a * (1 - b / 100.0),
     "decrease {a} by {b} percent",
     [_S("a1", "decrease_by_percent", ["base", "percent"],
         "Decreases a value by the given percent.")],
     [_S("g1", "shrink_by_rate", ["starting_value", "discount_rate"],
         "Reduces a value by a percentage rate.")],
     sam=_sam(_int(43, 1997), _int(3, 60)), sem=("base", "percentage"),
     punits=(E_ANY, E_NEUTRAL))


# ── unary (number -> number) ──────────────────────────────────────────────
def _un(sid, fn, phrase, a_surfs, g_surfs, sam=None, cat="unary",
        punits=(E_ANY,), unit_out=PRESERVE, phrase_ref="", out_type=NUM,
        answer_kind="float"):
    _add_prim(Primitive(
        sid=sid, category=cat, params=[("a", NUM, "operand")],
        out_type=out_type, out_semantic="quantity",
        fn=(lambda a: _r6(fn(float(a)))) if out_type == NUM
        else (lambda a: fn(float(a))),
        sampler=sam or _sam(_int(4, 900)),
        surfaces_a=a_surfs, surfaces_g=g_surfs, phrase=phrase,
        phrase_ref=phrase_ref or phrase,
        param_units=list(punits), unit_out=unit_out, answer_kind=answer_kind))


_un("sqrt", math.sqrt, "take the square root of {a}",
    [_S("a1", "sqrt", ["arg_0"], "Returns the square root.")],
    [_S("g1", "root_extract", ["input_value"], "Extracts the square root of the input.")],
    sam=_sam(lambda rng: rng.choice([169, 196, 225, 289, 324, 361, 441, 484,
                                     529, 576, 676, 729, 841, 961, 1156])),
    punits=(E_NEUTRAL,), unit_out=U_ABSTRACT)

_un("negate", lambda a: -a, "negate {a}",
    [_S("a1", "negate", ["arg_0"], "Returns the negation of the number.")],
    [_S("g1", "flip_sign", ["value"], "Flips the sign of the value.")])

_un("inverse", lambda a: 1.0 / a, "take the reciprocal of {a}",
    [_S("a1", "inverse", ["arg_0"], "Returns 1 divided by the number.")],
    [_S("g1", "reciprocal_value", ["value"], "Computes the reciprocal of the value.")],
    sam=_sam(_int(3, 41)), punits=(E_NEUTRAL,), unit_out=U_RATIO)

_un("floor_value", math.floor, "round {a} down to a whole number",
    [_S("a1", "floor", ["arg_0"], "Rounds down to the nearest integer.")],
    [_S("g1", "round_down_whole", ["value"], "Drops the fractional part, rounding down.")],
    sam=_sam(_flt(2.3, 900.7)))

_un("ceil_value", math.ceil, "round {a} up to a whole number",
    [_S("a1", "ceiling", ["arg_0"], "Rounds up to the nearest integer.")],
    [_S("g1", "round_up_whole", ["value"], "Rounds up to the next whole number.")],
    sam=_sam(_flt(2.3, 900.7)))

_un("square", lambda a: a * a, "square {a}",
    [_S("a1", "square_value", ["arg_0"], "Multiplies the number by itself.")],
    [_S("g1", "self_product", ["value"], "Returns the value multiplied by itself.")],
    sam=_sam(_int(3, 60)), punits=(E_NEUTRAL,), unit_out=U_ABSTRACT)

# int-typed unary sinks (genuine integer answers)
_un("round_to_int", lambda a: int(round(a)), "round {a} to the nearest whole number",
    [_S("a1", "nearest_integer", ["arg_0"], "Rounds a number to the nearest integer.")],
    [_S("g1", "whole_number_of", ["value"], "Returns the nearest whole number.")],
    sam=_sam(_flt(3.3, 899.7)), out_type=INT, answer_kind="int")

_un("digit_sum", lambda a: sum(int(c) for c in str(abs(int(a)))),
    "sum the digits of {a}",
    [_S("a1", "digit_sum", ["arg_0"], "Sums the decimal digits of a whole number.")],
    [_S("g1", "digits_added_up", ["whole_value"],
        "Adds together the digits of the given whole number.")],
    sam=_sam(_int(1043, 98999)), punits=(E_NEUTRAL,), unit_out=U_COUNT,
    out_type=INT, answer_kind="int")


# round_to_places (number, integer) -> number
_add_prim(Primitive(
    sid="round_places", category="unary",
    params=[("a", NUM, "operand"), ("places", INT, "precision")],
    out_type=NUM, out_semantic="quantity",
    fn=lambda a, places: _r6(round(float(a), int(places))),
    sampler=_sam(_flt(3.1234, 900.9876, 4), _int(1, 3)),
    surfaces_a=[_S("a1", "round_to_decimals", ["value", "places"],
                   "Rounds a number to the given number of decimal places.")],
    surfaces_g=[_S("g1", "trim_precision", ["raw_value", "digits"],
                   "Trims a number to the requested precision.")],
    phrase="round {a} to {places} decimal places",
    param_units=[E_ANY, E_NEUTRAL], unit_out=PRESERVE))

# clamp (number, number, number) -> number
_add_prim(Primitive(
    sid="clamp", category="selection",
    params=[("a", NUM, "operand"), ("lo", NUM, "lower_bound"),
            ("hi", NUM, "upper_bound")],
    out_type=NUM, out_semantic="quantity",
    fn=lambda a, lo, hi: _r6(min(max(float(a), float(lo)), float(hi))),
    sampler=_sam(_int(1, 900), _int(11, 79), _int(121, 797)),
    surfaces_a=[_S("a1", "clamp_value", ["value", "minimum", "maximum"],
                   "Clamps a value into the inclusive range.")],
    surfaces_g=[_S("g1", "bound_within", ["input_value", "low_limit", "high_limit"],
                   "Bounds the input within the limits.")],
    phrase="clamp {a} between {lo} and {hi}",
    param_units=[E_ANY, E_ANY, E_ANY], unit_out=PRESERVE))

# round_direction with enum mode
_add_prim(Primitive(
    sid="round_direction", category="selection",
    params=[("a", NUM, "operand"), ("mode", "enum:up,down,nearest", "rounding_mode")],
    out_type=NUM, out_semantic="quantity",
    fn=lambda a, mode: _r6({"up": math.ceil, "down": math.floor,
                            "nearest": lambda x: round(x)}[str(mode)](float(a))),
    sampler=_sam(_flt(2.2, 800.8), lambda rng: rng.choice(["up", "down", "nearest"])),
    surfaces_a=[_S("a1", "round_with_mode", ["value", "mode"],
                   "Rounds a value using the given mode: up, down, or nearest.")],
    surfaces_g=[_S("g1", "adjust_to_whole", ["value", "direction"],
                   "Adjusts to a whole number in the given direction.")],
    phrase="round {a} using mode {mode}",
    param_units=[E_ANY, E_NEUTRAL], unit_out=PRESERVE))


# ── three-way scalar aggregators ──────────────────────────────────────────
# NESTFUL never puts a reference inside an array argument (0/200 dev rows) and
# the trainer's executor cannot resolve one, so branch aggregation over three
# independent branches uses three SCALAR reference slots instead.
def _tri(sid, fn, phrase, a_surf, g_surf):
    _add_prim(Primitive(
        sid=sid, category="aggregate",
        params=[("a", NUM, "operand"), ("b", NUM, "operand"),
                ("c", NUM, "operand")],
        out_type=NUM, out_semantic="quantity",
        fn=lambda a, b, c: _r6(fn(float(a), float(b), float(c))),
        sampler=_sam(_int(11, 899), _int(7, 743), _int(13, 617)),
        surfaces_a=[a_surf], surfaces_g=[g_surf], phrase=phrase,
        param_units=[E_NEUTRAL, E_NEUTRAL, E_NEUTRAL], unit_out=U_ABSTRACT))


_tri("sum_three", lambda a, b, c: a + b + c,
     "add up {a}, {b} and {c}",
     _S("a1", "sum_of_three", ["arg_0", "arg_1", "arg_2"],
        "Adds three numbers and returns the total."),
     _S("g1", "total_of_three_parts", ["part_one", "part_two", "part_three"],
        "Combines three parts into one total."))

_tri("mean_three", lambda a, b, c: (a + b + c) / 3.0,
     "average {a}, {b} and {c}",
     _S("a1", "mean_of_three", ["arg_0", "arg_1", "arg_2"],
        "Returns the arithmetic mean of three numbers."),
     _S("g1", "balance_of_three", ["reading_one", "reading_two", "reading_three"],
        "Returns the average of three readings."))

_tri("range_three", lambda a, b, c: max(a, b, c) - min(a, b, c),
     "find the spread between the largest and smallest of {a}, {b} and {c}",
     _S("a1", "range_of_three", ["arg_0", "arg_1", "arg_2"],
        "Returns the difference between the largest and smallest of three numbers."),
     _S("g1", "dispersion_of_three", ["value_one", "value_two", "value_three"],
        "Measures the dispersion across three values."))


# ── boolean predicates (genuine bool answers) ─────────────────────────────
_add_prim(Primitive(
    sid="is_greater", category="predicate",
    params=[("a", NUM, "operand"), ("b", NUM, "threshold")],
    out_type=BOOL, out_semantic="flag",
    fn=lambda a, b: bool(float(a) > float(b)),
    sampler=_sam(_int(23, 887), _int(17, 653)),
    surfaces_a=[_S("a1", "is_greater", ["arg_0", "arg_1"],
                   "Returns true when the first number is greater than the second.")],
    surfaces_g=[_S("g1", "exceeds_value", ["measured_value", "limit"],
                   "Checks whether the measured value exceeds the limit.")],
    phrase="check whether {a} is greater than {b}",
    param_units=[E_ANY, E_ANY], unit_out=U_FLAG, answer_kind="bool"))

_add_prim(Primitive(
    sid="is_within_range", category="predicate",
    params=[("a", NUM, "operand"), ("lo", NUM, "lower_bound"),
            ("hi", NUM, "upper_bound")],
    out_type=BOOL, out_semantic="flag",
    fn=lambda a, lo, hi: bool(float(lo) <= float(a) <= float(hi)),
    sampler=_sam(_int(23, 887), _int(13, 97), _int(211, 863)),
    surfaces_a=[_S("a1", "is_within_range", ["value", "minimum", "maximum"],
                   "Returns true when the value lies inside the inclusive range.")],
    surfaces_g=[_S("g1", "inside_limits", ["reading", "low_limit", "high_limit"],
                   "Checks whether a reading lies inside the given limits.")],
    phrase="check whether {a} lies between {lo} and {hi}",
    param_units=[E_ANY, E_ANY, E_ANY], unit_out=U_FLAG, answer_kind="bool"))

_add_prim(Primitive(
    sid="is_divisible_by", category="predicate",
    params=[("a", NUM, "operand"), ("k", INT, "divisor")],
    out_type=BOOL, out_semantic="flag",
    fn=lambda a, k: bool(abs(float(a) - round(float(a))) < 1e-9
                         and int(round(float(a))) % int(k) == 0),
    sampler=_sam(_int(102, 9997), _int(3, 17)),
    surfaces_a=[_S("a1", "is_divisible_by", ["arg_0", "arg_1"],
                   "Returns true when the first number divides evenly by the second.")],
    surfaces_g=[_S("g1", "divides_evenly", ["total_amount", "group_size"],
                   "Checks whether the amount splits into equal groups.")],
    phrase="check whether {a} divides evenly by {k}",
    param_units=[E_ANY, E_NEUTRAL], unit_out=U_FLAG, answer_kind="bool"))


# ── string ops ────────────────────────────────────────────────────────────
_add_prim(Primitive(
    sid="number_to_string", category="string",
    params=[("a", NUM, "operand")],
    out_type=STR, out_semantic="numeric_text",
    fn=lambda a: (str(int(a)) if float(a) == int(float(a)) else str(_r6(a))),
    sampler=_sam(_int(13, 8999)),
    surfaces_a=[_S("a1", "number_to_text", ["value"],
                   "Converts a number to its string representation.")],
    surfaces_g=[_S("g1", "stringify_amount", ["amount"], "Renders an amount as text.")],
    phrase="convert {a} to text",
    param_units=[E_ANY], unit_out=U_NUMTEXT, answer_kind="numeric_string"))

_add_prim(Primitive(
    sid="parse_number", category="string",
    params=[("text", STR, "numeric_text")],
    out_type=NUM, out_semantic="quantity",
    fn=lambda text: _r6(float(str(text).strip())),
    sampler=_sam(lambda rng: str(rng.randint(137, 8971))),
    surfaces_a=[_S("a1", "parse_numeric_text", ["text"],
                   "Parses a numeric string into a number.")],
    surfaces_g=[_S("g1", "read_amount_from_text", ["raw_text"],
                   "Reads a numeric amount out of a text field.")],
    phrase="parse the numeric text {text}",
    param_units=[U_NUMTEXT], unit_out=U_ABSTRACT))

_add_prim(Primitive(
    sid="format_fixed", category="string",
    params=[("a", NUM, "operand"), ("places", INT, "precision")],
    out_type=STR, out_semantic="numeric_text",
    fn=lambda a, places: f"{float(a):.{int(places)}f}",
    sampler=_sam(_flt(3.111, 900.999, 3), _int(1, 2)),
    surfaces_a=[_S("a1", "format_number_fixed", ["value", "places"],
                   "Formats a number with a fixed number of decimals as a string.")],
    surfaces_g=[_S("g1", "render_with_decimals", ["value", "decimal_count"],
                   "Renders the value as text with fixed decimals.")],
    phrase="format {a} with {places} decimal places as text",
    param_units=[E_ANY, E_NEUTRAL], unit_out=U_NUMTEXT,
    answer_kind="numeric_string"))

_add_prim(Primitive(
    sid="format_with_unit", category="string",
    params=[("a", NUM, "operand"), ("unit", STR, "unit_text")],
    out_type=STR, out_semantic="text",
    fn=lambda a, unit: f"{(int(a) if float(a) == int(float(a)) else _r6(a))} {unit}",
    sampler=_sam(_int(17, 941), lambda rng: rng.choice(
        ["kg", "units", "items", "points", "boxes", "litres"])),
    surfaces_a=[_S("a1", "label_with_unit", ["value", "unit"],
                   "Appends a unit label to a numeric value and returns text.")],
    surfaces_g=[_S("g1", "annotate_measurement", ["measurement", "unit_name"],
                   "Formats a measurement together with its unit name.")],
    phrase="label {a} with the unit {unit}",
    param_units=[E_ANY, U_TEXT], unit_out=U_TEXT, answer_kind="string"))

_add_prim(Primitive(
    sid="tag_value", category="string",
    params=[("prefix", STR, "text"), ("a", NUM, "operand")],
    out_type=STR, out_semantic="text",
    fn=lambda prefix, a: f"{prefix}-{int(a) if float(a) == int(float(a)) else _r6(a)}",
    sampler=_sam(lambda rng: rng.choice(["batch", "order", "run", "lot", "ticket"]),
                 _int(17, 941)),
    surfaces_a=[_S("a1", "build_tag", ["prefix", "value"],
                   "Builds an identifier by joining a prefix and a number with a dash.")],
    surfaces_g=[_S("g1", "compose_reference_code", ["code_prefix", "code_number"],
                   "Composes a reference code from a prefix and a number.")],
    phrase="build an identifier from the prefix {prefix} and {a}",
    param_units=[U_TEXT, E_ANY], unit_out=U_TEXT, answer_kind="string"))

_add_prim(Primitive(
    sid="text_length", category="string",
    params=[("text", STR, "text")],
    out_type=INT, out_semantic="count",
    fn=lambda text: len(str(text)),
    sampler=_sam(lambda rng: str(rng.randint(1013, 99999997))),
    surfaces_a=[_S("a1", "text_length", ["text"],
                   "Returns the number of characters in the text.")],
    surfaces_g=[_S("g1", "character_count", ["input_text"],
                   "Counts characters in the input text.")],
    phrase="count the characters of {text}",
    param_units=[U_TEXT], unit_out=U_COUNT, answer_kind="int"))

_add_prim(Primitive(
    sid="concat_texts", category="string",
    params=[("first", STR, "text"), ("second", STR, "text")],
    out_type=STR, out_semantic="text",
    fn=lambda first, second: str(first) + str(second),
    sampler=_sam(lambda rng: rng.choice(["batch-", "run-", "lot-", "id-"]),
                 lambda rng: str(rng.randint(13, 97))),
    surfaces_a=[_S("a1", "concat_strings", ["first", "second"],
                   "Concatenates two strings in order.")],
    surfaces_g=[_S("g1", "join_text_pieces", ["piece_one", "piece_two"],
                   "Joins two text pieces together.")],
    phrase="join the texts {first} and {second}",
    param_units=[U_TEXT, U_TEXT], unit_out=U_TEXT, answer_kind="string"))

_add_prim(Primitive(
    sid="join_values", category="string",
    params=[("values", ARR, "number_list"), ("separator", STR, "separator")],
    out_type=STR, out_semantic="text",
    fn=lambda values, separator: str(separator).join(
        str(int(v)) if float(v) == int(float(v)) else str(_r6(v)) for v in values),
    sampler=_sam(_lst(3, 5, 2, 89), lambda rng: rng.choice([", ", " | ", "-", "; "])),
    surfaces_a=[_S("a1", "join_values_with", ["values", "separator"],
                   "Joins the numbers into one string using the separator.")],
    surfaces_g=[_S("g1", "serialize_series", ["number_list", "delimiter"],
                   "Serializes the series into a single delimited string.")],
    phrase="join {values} into one string separated by {separator}",
    param_units=[U_LIST, U_TEXT], unit_out=U_TEXT, answer_kind="string"))


# ── list ops (array -> number/integer) ────────────────────────────────────
def _larr(sid, fn, phrase, a_surfs, g_surfs, out=NUM, out_sem="quantity",
          unit_out=U_ABSTRACT, answer_kind="float"):
    _add_prim(Primitive(
        sid=sid, category="list",
        params=[("values", ARR, "number_list")],
        out_type=out, out_semantic=out_sem,
        fn=(lambda values: _r6(fn([float(v) for v in values]))) if out == NUM
        else (lambda values: fn([float(v) for v in values])),
        sampler=_sam(_lst()),
        surfaces_a=a_surfs, surfaces_g=g_surfs, phrase=phrase,
        param_units=[U_LIST], unit_out=unit_out, answer_kind=answer_kind))


_larr("sum_values", sum, "sum the list {values}",
      [_S("a1", "sum_of_values", ["values"], "Sums a list of numbers.")],
      [_S("g1", "aggregate_total", ["number_list"], "Aggregates a list into its total.")])

_larr("mean_values", lambda v: sum(v) / len(v), "average the list {values}",
      [_S("a1", "mean_of_values", ["values"],
          "Computes the arithmetic mean of a list.")],
      [_S("g1", "central_value", ["number_list"], "Returns the average of the list.")])

_larr("max_values", max, "find the maximum of {values}",
      [_S("a1", "max_of_values", ["values"], "Returns the largest element of a list.")],
      [_S("g1", "peak_of_series", ["number_list"], "Finds the peak value in the series.")])

_larr("min_values", min, "find the minimum of {values}",
      [_S("a1", "min_of_values", ["values"], "Returns the smallest element of a list.")],
      [_S("g1", "valley_of_series", ["number_list"],
          "Finds the lowest value in the series.")])

_larr("count_values", lambda v: len(v), "count the items of {values}",
      [_S("a1", "count_of_values", ["values"], "Counts the elements of a list.")],
      [_S("g1", "series_length", ["number_list"],
          "Returns how many items the series has.")],
      out=INT, out_sem="count", unit_out=U_COUNT, answer_kind="int")

_larr("range_spread", lambda v: max(v) - min(v),
      "find the spread (max minus min) of {values}",
      [_S("a1", "range_of_values", ["values"], "Returns max minus min of a list.")],
      [_S("g1", "series_spread", ["number_list"], "Measures the spread of the series.")])

_larr("index_of_max", lambda v: int(v.index(max(v)) + 1),
      "find the 1-based position of the largest item in {values}",
      [_S("a1", "position_of_maximum", ["values"],
          "Returns the 1-based position of the largest element.")],
      [_S("g1", "rank_of_peak", ["number_list"],
          "Returns which position in the series holds the peak.")],
      out=INT, out_sem="count", unit_out=U_COUNT, answer_kind="int")


# ── list-producing ops (genuine list answers) ─────────────────────────────
_add_prim(Primitive(
    sid="sort_values_desc", category="list",
    params=[("values", ARR, "number_list")],
    out_type=ARR, out_semantic="number_list",
    fn=lambda values: sorted((_r6(v) for v in values), reverse=True),
    sampler=_sam(_lst(4, 6, 3, 97)),
    surfaces_a=[_S("a1", "sort_values_descending", ["values"],
                   "Sorts the numbers from largest to smallest.")],
    surfaces_g=[_S("g1", "order_series_high_to_low", ["number_list"],
                   "Orders the series from the highest value down.")],
    phrase="sort {values} from largest to smallest",
    param_units=[U_LIST], unit_out=U_LIST, answer_kind="list"))

_add_prim(Primitive(
    sid="scale_list", category="list",
    params=[("values", ARR, "number_list"), ("factor", NUM, "factor")],
    out_type=ARR, out_semantic="number_list",
    fn=lambda values, factor: [_r6(float(v) * float(factor)) for v in values],
    sampler=_sam(_lst(3, 5, 2, 47), _int(3, 19)),
    surfaces_a=[_S("a1", "scale_values", ["values", "factor"],
                   "Multiplies every element of the list by the factor.")],
    surfaces_g=[_S("g1", "rescale_series", ["number_list", "multiplier"],
                   "Rescales every entry of the series by the multiplier.")],
    phrase="scale every item of {values} by {factor}",
    param_units=[U_LIST, E_NEUTRAL], unit_out=U_LIST, answer_kind="list"))

_add_prim(Primitive(
    sid="filter_above", category="list",
    params=[("values", ARR, "number_list"), ("threshold", NUM, "threshold")],
    out_type=ARR, out_semantic="number_list",
    fn=lambda values, threshold: [_r6(v) for v in values
                                  if float(v) > float(threshold)],
    sampler=_sam(_lst(5, 8, 5, 95), _int(11, 43)),
    surfaces_a=[_S("a1", "filter_values_above", ["values", "threshold"],
                   "Keeps only the numbers strictly greater than the threshold.")],
    surfaces_g=[_S("g1", "entries_over_limit", ["number_list", "limit"],
                   "Returns the series entries that exceed the limit.")],
    phrase="keep the items of {values} above {threshold}",
    param_units=[U_LIST, E_ANY], unit_out=U_LIST, answer_kind="list"))

_add_prim(Primitive(
    sid="top_k_values", category="list",
    params=[("values", ARR, "number_list"), ("k", INT, "count")],
    out_type=ARR, out_semantic="number_list",
    fn=lambda values, k: sorted((_r6(v) for v in values), reverse=True)[:int(k)],
    sampler=_sam(_lst(5, 8, 3, 93), _int(2, 3)),
    surfaces_a=[_S("a1", "take_top_values", ["values", "k"],
                   "Returns the k largest numbers in descending order.")],
    surfaces_g=[_S("g1", "leading_entries", ["number_list", "how_many"],
                   "Returns the leading entries of the series.")],
    phrase="take the {k} largest items of {values}",
    phrase_ref="take the top items of {values}, where the number of items is {k}",
    param_units=[U_LIST, E_NEUTRAL], unit_out=U_LIST, answer_kind="list"))

_add_prim(Primitive(
    sid="append_value", category="list",
    params=[("values", ARR, "number_list"), ("value", NUM, "operand")],
    out_type=ARR, out_semantic="number_list",
    fn=lambda values, value: [_r6(v) for v in values] + [_r6(value)],
    sampler=_sam(_lst(3, 5, 2, 79), _int(11, 397)),
    surfaces_a=[_S("a1", "append_to_values", ["values", "value"],
                   "Appends the value to the end of the list and returns the list.")],
    surfaces_g=[_S("g1", "extend_series_with", ["number_list", "new_entry"],
                   "Extends the series with one further entry.")],
    phrase="append {value} to the list {values}",
    param_units=[U_LIST, E_ANY], unit_out=U_LIST, answer_kind="list"))

_add_prim(Primitive(
    sid="cumulative_sums", category="list",
    params=[("values", ARR, "number_list")],
    out_type=ARR, out_semantic="number_list",
    fn=lambda values: [_r6(sum(float(x) for x in values[:i + 1]))
                       for i in range(len(values))],
    sampler=_sam(_lst(3, 5, 2, 61)),
    surfaces_a=[_S("a1", "running_totals", ["values"],
                   "Returns the running totals of the list.")],
    surfaces_g=[_S("g1", "accumulate_series", ["number_list"],
                   "Accumulates the series into running totals.")],
    phrase="compute the running totals of {values}",
    param_units=[U_LIST], unit_out=U_LIST, answer_kind="list"))


# ── deterministic conversions (typed units) ───────────────────────────────
_un("seconds_to_minutes", lambda a: math.floor(a / 60.0),
    "find how many whole minutes fit into {a} seconds",
    [_S("a1", "seconds_to_full_minutes", ["seconds"],
        "Converts seconds into full minutes (rounded down).")],
    [_S("g1", "whole_minutes_from_seconds", ["second_count"],
        "Counts complete minutes contained in the seconds.")],
    sam=_sam(_int(157, 89993)), cat="conversion",
    punits=(U_DUR_S,), unit_out=U_DUR_MIN,
    phrase_ref="convert {a} from seconds into whole minutes")

_un("hours_to_minutes", lambda a: a * 60.0, "convert {a} hours to minutes",
    [_S("a1", "hours_to_minutes", ["hours"], "Converts hours to minutes.")],
    [_S("g1", "minutes_from_hours", ["hour_count"], "Expresses hours as minutes.")],
    sam=_sam(_int(2, 79)), cat="conversion",
    punits=(U_DUR_H,), unit_out=U_DUR_MIN,
    phrase_ref="convert {a} from hours into minutes")

_un("minutes_to_seconds", lambda a: a * 60.0, "convert {a} minutes to seconds",
    [_S("a1", "minutes_to_seconds", ["minutes"], "Converts minutes to seconds.")],
    [_S("g1", "seconds_from_minutes", ["minute_count"], "Expresses minutes as seconds.")],
    sam=_sam(_int(3, 97)), cat="conversion",
    punits=(U_DUR_MIN,), unit_out=U_DUR_S,
    phrase_ref="convert {a} from minutes into seconds")

_un("km_to_meters", lambda a: a * 1000.0, "convert {a} kilometers to meters",
    [_S("a1", "km_to_meters", ["kilometers"], "Converts kilometers to meters.")],
    [_S("g1", "meters_from_km", ["distance_km"], "Expresses a distance in meters.")],
    sam=_sam(_flt(0.7, 89.3, 1)), cat="conversion",
    punits=(U_LEN_KM,), unit_out=U_LEN_M,
    phrase_ref="convert {a} from kilometers into meters")

_un("meters_to_km", lambda a: a / 1000.0, "convert {a} meters to kilometers",
    [_S("a1", "meters_to_km", ["meters"], "Converts meters to kilometers.")],
    [_S("g1", "km_from_meters", ["distance_m"], "Expresses a distance in kilometers.")],
    sam=_sam(_int(1300, 89000)), cat="conversion",
    punits=(U_LEN_M,), unit_out=U_LEN_KM,
    phrase_ref="convert {a} from meters into kilometers")

_un("celsius_to_fahrenheit", lambda a: a * 9.0 / 5.0 + 32.0,
    "convert {a} degrees Celsius to Fahrenheit",
    [_S("a1", "celsius_to_fahrenheit", ["celsius"], "Converts Celsius to Fahrenheit.")],
    [_S("g1", "fahrenheit_from_celsius", ["temp_c"],
        "Converts a temperature to Fahrenheit.")],
    sam=_sam(_int(-18, 89)), cat="conversion",
    punits=(U_TEMP_C,), unit_out=U_TEMP_F,
    phrase_ref="convert {a} from degrees Celsius into Fahrenheit")

_un("fahrenheit_to_celsius", lambda a: (a - 32.0) * 5.0 / 9.0,
    "convert {a} degrees Fahrenheit to Celsius",
    [_S("a1", "fahrenheit_to_celsius", ["fahrenheit"],
        "Converts Fahrenheit to Celsius.")],
    [_S("g1", "celsius_from_fahrenheit", ["temp_f"],
        "Converts a temperature to Celsius.")],
    sam=_sam(_int(33, 197)), cat="conversion",
    punits=(U_TEMP_F,), unit_out=U_TEMP_C,
    phrase_ref="convert {a} from degrees Fahrenheit into Celsius")


# ── public API ────────────────────────────────────────────────────────────
def unit_of_constant_expect(expect: str) -> str:
    """A literal argument carries whatever unit its parameter expects."""
    return expect if expect not in (E_ANY, E_NEUTRAL) else U_ABSTRACT


def all_primitives() -> Dict[str, Primitive]:
    return dict(_REG)


def get(sid: str) -> Primitive:
    return _REG[sid]


def numeric_producers() -> List[str]:
    return [sid for sid, p in _REG.items() if p.out_type in (NUM, INT)]


def by_answer_kind(kind: str) -> List[str]:
    return sorted(sid for sid, p in _REG.items() if p.answer_kind == kind)


def registry_hash() -> str:
    payload = {}
    for sid, p in sorted(_REG.items()):
        payload[sid] = {
            "params": [(n, t, s) for n, t, s in p.params],
            "out": p.out_type,
            "units": p.param_units + [p.unit_out],
            "surfaces_a": [(s.name, tuple(s.param_names)) for s in p.surfaces_a],
            "surfaces_g": [(s.name, tuple(s.param_names)) for s in p.surfaces_g],
        }
    return sha256_obj(payload)


def all_surfaces() -> List[Tuple[str, str, Surface]]:
    """(sid, track, surface) for every registered surface."""
    out = []
    for sid, p in sorted(_REG.items()):
        for surf in p.surfaces_a:
            out.append((sid, "A", surf))
        for surf in p.surfaces_g:
            out.append((sid, "G", surf))
    return out


def validate_surface_uniqueness() -> List[str]:
    """Every surface tool name must map to exactly one (sid, param signature).

    Required by the trainer's synthetic executor, which looks tools up by name
    in a single global registry.
    """
    errs: List[str] = []
    seen: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
    for sid, _track, surf in all_surfaces():
        sig = tuple(surf.param_names)
        prev = seen.get(surf.name)
        if prev is None:
            seen[surf.name] = (sid, sig)
        elif prev != (sid, sig):
            errs.append(f"tool name {surf.name!r} maps to {prev} and {(sid, sig)}")
    return errs


def _load_extensions() -> None:
    """Pilot4 capability-gap primitives (boolean, geometry, bitwise, ...)."""
    import sys

    from . import extensions

    extensions.register(sys.modules[__name__])


_load_extensions()

_UNIQUENESS_ERRORS = validate_surface_uniqueness()
if _UNIQUENESS_ERRORS:                                   # pragma: no cover
    raise RuntimeError("registry surface name collision: "
                       + "; ".join(_UNIQUENESS_ERRORS))


# param type acceptance for chaining: producer out -> consumer param
def type_accepts(param_type: str, value_type: str) -> bool:
    if param_type.startswith("enum:"):
        return False
    if param_type == NUM:
        return value_type in (NUM, INT)
    if param_type == INT:
        return value_type == INT
    return param_type == value_type
