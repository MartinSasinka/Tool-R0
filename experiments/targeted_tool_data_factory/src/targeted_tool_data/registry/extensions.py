"""Pilot4 primitive extensions: capability families the pilot3 registry lacked.

The pilot3 registry covered arithmetic, list reduction, string formatting and
unit conversion well, but had no boolean composition, geometry, date/time,
path/url, bitwise, keyed-lookup or deterministic-classification capability.
Long-horizon programs therefore had very few type transitions to work with.

Every primitive here is general, deterministic, typed and testable, and is
written from the capability spec rather than copied from any benchmark's tool
implementations. Registered through ``register(module)`` so the parent registry
stays the single source of truth for uniqueness and hashing.
"""
from __future__ import annotations

import math
from typing import Any


def register(m: Any) -> None:  # noqa: PLR0915 - one flat registration table
    P, S, add = m.Primitive, m._S, m._add_prim
    NUM, INT, STR, ARR, BOOL = m.NUM, m.INT, m.STR, m.ARR, m.BOOL
    r6, sam, i_, f_ = m._r6, m._sam, m._int, m._flt
    ANY, NEUTRAL, PRESERVE = m.E_ANY, m.E_NEUTRAL, m.PRESERVE
    U_ABS, U_COUNT, U_FLAG, U_TEXT = m.U_ABSTRACT, m.U_COUNT, m.U_FLAG, m.U_TEXT
    U_LIST, U_RATIO = m.U_LIST, m.U_RATIO
    U_DUR_H, U_DUR_MIN, U_LEN_M = m.U_DUR_H, m.U_DUR_MIN, m.U_LEN_M

    def prim(sid, category, params, out_type, fn, sampler, a, g, phrase,
             units=None, unit_out=U_ABS, answer_kind="float", phrase_ref=""):
        add(P(sid=sid, category=category, params=params, out_type=out_type,
              out_semantic="quantity", fn=fn, sampler=sampler,
              surfaces_a=a, surfaces_g=g, phrase=phrase,
              phrase_ref=phrase_ref or phrase,
              param_units=list(units) if units else [ANY] * len(params),
              unit_out=unit_out, answer_kind=answer_kind))

    # ── boolean.logic ─────────────────────────────────────────────────────
    # Boolean inputs let predicate outputs feed a real join instead of dying
    # at a leaf, which is what made pilot3 fan-in almost purely numeric.
    _b = lambda rng: rng.random() < 0.5  # noqa: E731

    prim("logical_and", "predicate",
         [("a", BOOL, "flag"), ("b", BOOL, "flag")], BOOL,
         lambda a, b: bool(a) and bool(b), sam(_b, _b),
         [S("a1", "logical_and", ["arg_0", "arg_1"],
            "Returns true only when both flags are true.")],
         [S("g1", "both_conditions_hold", ["condition_one", "condition_two"],
            "Reports whether both conditions hold at the same time.")],
         "check whether both {a} and {b} hold",
         units=[U_FLAG, U_FLAG], unit_out=U_FLAG, answer_kind="bool")

    prim("logical_or", "predicate",
         [("a", BOOL, "flag"), ("b", BOOL, "flag")], BOOL,
         lambda a, b: bool(a) or bool(b), sam(_b, _b),
         [S("a1", "logical_or", ["arg_0", "arg_1"],
            "Returns true when at least one flag is true.")],
         [S("g1", "either_condition_holds", ["first_condition", "second_condition"],
            "Reports whether at least one condition holds.")],
         "check whether at least one of {a} and {b} holds",
         units=[U_FLAG, U_FLAG], unit_out=U_FLAG, answer_kind="bool")

    prim("logical_not", "predicate", [("a", BOOL, "flag")], BOOL,
         lambda a: not bool(a), sam(_b),
         [S("a1", "logical_not", ["arg_0"], "Inverts a boolean flag.")],
         [S("g1", "condition_fails", ["condition"],
            "Reports whether the condition does not hold.")],
         "invert the flag {a}",
         units=[U_FLAG], unit_out=U_FLAG, answer_kind="bool")

    prim("is_non_negative", "predicate", [("a", NUM, "operand")], BOOL,
         lambda a: float(a) >= 0.0, sam(i_(-60, 90)),
         [S("a1", "is_non_negative", ["arg_0"],
            "Checks whether the number is zero or positive.")],
         [S("g1", "value_not_below_zero", ["value"],
            "Checks that a value has not dropped below zero.")],
         "check whether {a} is zero or positive",
         units=[ANY], unit_out=U_FLAG, answer_kind="bool")

    # ── classification.deterministic ──────────────────────────────────────
    prim("classify_threshold", "classification",
         [("a", NUM, "operand"), ("threshold", NUM, "threshold")], STR,
         lambda a, threshold: "above" if float(a) > float(threshold) else "at_or_below",
         sam(i_(10, 400), i_(20, 300)),
         [S("a1", "classify_against_threshold", ["arg_0", "arg_1"],
            "Labels a value as above or at_or_below a threshold.")],
         [S("g1", "threshold_band_label", ["measurement", "limit"],
            "Returns the band label of a measurement against a limit.")],
         "label {a} as above or at_or_below {threshold}",
         units=[ANY, ANY], unit_out=U_TEXT, answer_kind="string")

    prim("grade_band", "classification", [("a", NUM, "operand")], STR,
         lambda a: ("high" if float(a) >= 80 else
                    "medium" if float(a) >= 50 else "low"),
         sam(i_(5, 99)),
         [S("a1", "grade_band", ["arg_0"],
            "Maps a score to the band low, medium or high.")],
         [S("g1", "performance_tier", ["score"],
            "Maps a score onto a performance tier.")],
         "map {a} onto a low/medium/high band",
         units=[ANY], unit_out=U_TEXT, answer_kind="string")

    # ── dictionary.lookup / dictionary.update ─────────────────────────────
    # Keyed lookup over a controlled key domain: dictionary semantics without
    # introducing an object-typed parameter the trainer executor cannot read.
    _FACTORS = {"km": 1000.0, "m": 1.0, "cm": 0.01, "mm": 0.001}
    prim("lookup_unit_factor", "conversion",
         [("unit_code", "enum:km,m,cm,mm", "unit_key")], NUM,
         lambda unit_code: r6(_FACTORS[str(unit_code)]),
         sam(lambda rng: rng.choice(["km", "m", "cm", "mm"])),
         [S("a1", "lookup_unit_factor", ["arg_0"],
            "Looks up the metre factor of a length unit code.")],
         [S("g1", "metre_factor_for_unit", ["unit_key"],
            "Returns how many metres one unit of the given code represents.")],
         "look up the metre factor for the unit {unit_code}",
         units=[NEUTRAL], unit_out=U_ABS)

    _ADJUST = {"none": 1.0, "half": 0.5, "double": 2.0, "quarter": 0.25}
    prim("apply_rate_override", "conversion",
         [("a", NUM, "operand"), ("override_code", "enum:none,half,double,quarter",
                                  "override_key")], NUM,
         lambda a, override_code: r6(float(a) * _ADJUST[str(override_code)]),
         sam(i_(20, 900), lambda rng: rng.choice(["none", "half", "double", "quarter"])),
         [S("a1", "apply_rate_override", ["arg_0", "arg_1"],
            "Applies a named override factor to a value.")],
         [S("g1", "adjusted_by_policy", ["base_value", "policy_key"],
            "Adjusts a base value according to a named policy.")],
         "apply the {override_code} override to {a}",
         units=[ANY, NEUTRAL], unit_out=PRESERVE)

    # ── geometry ──────────────────────────────────────────────────────────
    prim("rectangle_area", "geometry",
         [("width", NUM, "length"), ("height", NUM, "length")], NUM,
         lambda width, height: r6(float(width) * float(height)),
         sam(i_(2, 90), i_(2, 90)),
         [S("a1", "rectangle_area", ["arg_0", "arg_1"],
            "Computes the area of a rectangle from its two sides.")],
         [S("g1", "surface_of_rectangle", ["side_a", "side_b"],
            "Returns the surface covered by a rectangle.")],
         "compute the area of a rectangle {width} by {height}",
         units=[ANY, ANY], unit_out=U_ABS)

    prim("rectangle_perimeter", "geometry",
         [("width", NUM, "length"), ("height", NUM, "length")], NUM,
         lambda width, height: r6(2.0 * (float(width) + float(height))),
         sam(i_(2, 90), i_(2, 90)),
         [S("a1", "rectangle_perimeter", ["arg_0", "arg_1"],
            "Computes the perimeter of a rectangle.")],
         [S("g1", "outline_length_of_rectangle", ["first_side", "second_side"],
            "Returns the outline length around a rectangle.")],
         "compute the perimeter of a rectangle {width} by {height}",
         units=[ANY, ANY], unit_out=PRESERVE)

    prim("circle_area", "geometry", [("radius", NUM, "length")], NUM,
         lambda radius: r6(math.pi * float(radius) ** 2),
         sam(f_(1.5, 40.0, 2)),
         [S("a1", "circle_area", ["arg_0"],
            "Computes the area of a circle from its radius.")],
         [S("g1", "disc_surface", ["radius_value"],
            "Returns the surface of a disc with the given radius.")],
         "compute the area of a circle with radius {radius}",
         units=[ANY], unit_out=U_ABS)

    prim("hypotenuse", "geometry",
         [("a", NUM, "length"), ("b", NUM, "length")], NUM,
         lambda a, b: r6(math.hypot(float(a), float(b))),
         sam(i_(3, 80), i_(3, 80)),
         [S("a1", "hypotenuse_length", ["arg_0", "arg_1"],
            "Computes the hypotenuse of a right triangle from its legs.")],
         [S("g1", "diagonal_from_sides", ["leg_one", "leg_two"],
            "Returns the diagonal spanned by two perpendicular sides.")],
         "compute the hypotenuse for legs {a} and {b}",
         units=[ANY, ANY], unit_out=PRESERVE)

    # ── date_time ─────────────────────────────────────────────────────────
    prim("days_to_hours", "conversion", [("days", NUM, "duration")], NUM,
         lambda days: r6(float(days) * 24.0), sam(i_(1, 45)),
         [S("a1", "days_to_hours", ["arg_0"], "Converts days into hours.")],
         [S("g1", "hours_in_days", ["day_count"],
            "Returns how many hours the given number of days spans.")],
         "convert {days} days to hours",
         units=[ANY], unit_out=U_DUR_H,
         phrase_ref="convert {days} from days into hours")

    prim("weeks_to_days", "conversion", [("weeks", NUM, "duration")], NUM,
         lambda weeks: r6(float(weeks) * 7.0), sam(i_(1, 40)),
         [S("a1", "weeks_to_days", ["arg_0"], "Converts weeks into days.")],
         [S("g1", "days_in_weeks", ["week_count"],
            "Returns how many days the given number of weeks spans.")],
         "convert {weeks} weeks to days",
         units=[ANY], unit_out=U_ABS,
         phrase_ref="convert {weeks} from weeks into days")

    prim("minutes_since_midnight", "conversion",
         [("hours", INT, "duration"), ("minutes", INT, "duration")], INT,
         lambda hours, minutes: int(hours) * 60 + int(minutes),
         sam(i_(0, 23), i_(0, 59)),
         [S("a1", "minutes_since_midnight", ["arg_0", "arg_1"],
            "Converts a clock time into minutes elapsed since midnight.")],
         [S("g1", "clock_to_minute_offset", ["hour_part", "minute_part"],
            "Turns an hour/minute pair into a minute offset in the day.")],
         "convert the time {hours}:{minutes} into minutes since midnight",
         units=[ANY, ANY], unit_out=U_DUR_MIN, answer_kind="int")

    # ── path_url ──────────────────────────────────────────────────────────
    _SEG = lambda rng: rng.choice(["data", "reports", "archive", "exports", "logs"])  # noqa: E731
    prim("join_path_segments", "string",
         [("first", STR, "path_segment"), ("second", STR, "path_segment")], STR,
         lambda first, second: str(first).rstrip("/") + "/" + str(second).lstrip("/"),
         sam(_SEG, lambda rng: rng.choice(["q1", "q2", "raw", "final", "v2"])),
         [S("a1", "join_path_segments", ["first", "second"],
            "Joins two path segments with a single separator.")],
         [S("g1", "build_folder_path", ["parent_segment", "child_segment"],
            "Builds a folder path out of a parent and a child segment.")],
         "join the path segments {first} and {second}",
         units=[U_TEXT, U_TEXT], unit_out=U_TEXT, answer_kind="string")

    prim("file_extension", "string", [("filename", STR, "file_name")], STR,
         lambda filename: (str(filename).rsplit(".", 1)[-1]
                           if "." in str(filename) else ""),
         sam(lambda rng: rng.choice(["report.csv", "notes.txt", "table.json",
                                     "chart.png", "dump.tsv"])),
         [S("a1", "file_extension", ["filename"],
            "Extracts the extension of a file name.")],
         [S("g1", "suffix_of_file", ["file_label"],
            "Returns the suffix following the last dot of a file label.")],
         "take the extension of the file name {filename}",
         units=[U_TEXT], unit_out=U_TEXT, answer_kind="string")

    prim("domain_of_url", "string", [("url", STR, "url")], STR,
         lambda url: str(url).split("://")[-1].split("/")[0],
         sam(lambda rng: rng.choice([
             "https://example.org/a/b", "http://data.internal/x",
             "https://files.local/reports/2024", "https://node4.cluster/status"])),
         [S("a1", "domain_of_url", ["url"], "Extracts the host part of a URL.")],
         [S("g1", "host_from_address", ["address"],
            "Returns the host portion of a web address.")],
         "take the host of the address {url}",
         units=[U_TEXT], unit_out=U_TEXT, answer_kind="string")

    # ── bitwise ───────────────────────────────────────────────────────────
    for sid, op, sym, desc_a, desc_g, ph in [
        ("bitwise_and", lambda a, b: int(a) & int(b), "and",
         "Computes the bitwise AND of two integers.",
         "Keeps only the bits set in both inputs.", "compute the bitwise and of {a} and {b}"),
        ("bitwise_or", lambda a, b: int(a) | int(b), "or",
         "Computes the bitwise OR of two integers.",
         "Keeps the bits set in either input.", "compute the bitwise or of {a} and {b}"),
        ("bitwise_xor", lambda a, b: int(a) ^ int(b), "xor",
         "Computes the bitwise XOR of two integers.",
         "Keeps the bits set in exactly one input.", "compute the bitwise xor of {a} and {b}"),
    ]:
        prim(sid, "bitwise", [("a", INT, "bits"), ("b", INT, "bits")], INT,
             op, sam(i_(3, 255), i_(3, 255)),
             [S("a1", sid, ["arg_0", "arg_1"], desc_a)],
             [S("g1", f"mask_{sym}_values", ["mask_one", "mask_two"], desc_g)],
             ph, units=[NEUTRAL, NEUTRAL], unit_out=U_ABS, answer_kind="int")

    prim("left_shift", "bitwise",
         [("a", INT, "bits"), ("places", INT, "shift")], INT,
         lambda a, places: int(a) << int(places), sam(i_(1, 200), i_(1, 6)),
         [S("a1", "left_shift", ["arg_0", "arg_1"],
            "Shifts an integer left by the given number of bits.")],
         [S("g1", "shift_bits_up", ["bit_value", "shift_by"],
            "Moves the bits of a value up by a number of positions.")],
         "shift {a} left by {places} bits",
         units=[NEUTRAL, NEUTRAL], unit_out=U_ABS, answer_kind="int")

    # ── sequence.index / sequence.map / sequence.combine / statistics ─────
    prim("value_at_position", "list",
         [("values", ARR, "values"), ("position", INT, "index")], NUM,
         lambda values, position: r6(float(
             list(values)[(int(position) - 1) % max(len(list(values)), 1)])),
         sam(m._lst(3, 6, 2, 80), i_(1, 3)),
         [S("a1", "value_at_position", ["values", "position"],
            "Returns the item at the given 1-based position of the list.")],
         [S("g1", "entry_at_rank", ["entries", "rank"],
            "Returns the entry sitting at the requested rank.")],
         "take the item at position {position} of {values}",
         units=[U_LIST, NEUTRAL], unit_out=U_ABS)

    prim("offset_list", "list",
         [("values", ARR, "values"), ("delta", NUM, "offset")], ARR,
         lambda values, delta: [r6(float(v) + float(delta)) for v in values],
         sam(m._lst(3, 6, 2, 60), i_(2, 40)),
         [S("a1", "offset_list", ["values", "delta"],
            "Adds the same offset to every item of the list.")],
         [S("g1", "shift_series_by", ["series", "shift_amount"],
            "Shifts every element of a series by a fixed amount.")],
         "add {delta} to every item of {values}",
         units=[U_LIST, ANY], unit_out=U_LIST, answer_kind="list")

    prim("concat_lists", "list",
         [("first", ARR, "values"), ("second", ARR, "values")], ARR,
         lambda first, second: [r6(float(v)) for v in list(first) + list(second)],
         sam(m._lst(2, 4, 2, 50), m._lst(2, 4, 2, 50)),
         [S("a1", "concat_lists", ["first", "second"],
            "Concatenates two numeric lists in order.")],
         [S("g1", "merge_series", ["series_one", "series_two"],
            "Merges two numeric series one after the other.")],
         "concatenate the lists {first} and {second}",
         units=[U_LIST, U_LIST], unit_out=U_LIST, answer_kind="list")

    def _median(values):
        xs = sorted(float(v) for v in values)
        n = len(xs)
        if not n:
            raise ValueError("empty list")
        mid = n // 2
        return r6(xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0)

    prim("median_values", "list", [("values", ARR, "values")], NUM,
         _median, sam(m._lst(3, 7, 2, 90)),
         [S("a1", "median_values", ["values"],
            "Returns the median of a numeric list.")],
         [S("g1", "middle_of_series", ["series"],
            "Returns the middle value of a numeric series.")],
         "find the median of {values}",
         units=[U_LIST], unit_out=U_ABS)

    prim("product_three", "aggregate",
         [("a", NUM, "operand"), ("b", NUM, "operand"), ("c", NUM, "operand")], NUM,
         lambda a, b, c: r6(float(a) * float(b) * float(c)),
         sam(i_(2, 30), i_(2, 30), i_(2, 30)),
         [S("a1", "product_three", ["arg_0", "arg_1", "arg_2"],
            "Multiplies three numbers together.")],
         [S("g1", "combined_product", ["factor_one", "factor_two", "factor_three"],
            "Returns the product of three factors.")],
         "multiply {a}, {b} and {c} together",
         units=[ANY, ANY, ANY], unit_out=U_ABS)

    prim("text_upper", "string", [("text", STR, "text")], STR,
         lambda text: str(text).upper(),
         sam(lambda rng: rng.choice(["batch alpha", "run beta", "lot gamma",
                                     "unit delta"])),
         [S("a1", "text_upper", ["text"], "Uppercases the given text.")],
         [S("g1", "capitalize_label", ["label"],
            "Returns the label in upper case letters.")],
         "uppercase the text {text}",
         units=[U_TEXT], unit_out=U_TEXT, answer_kind="string")

    prim("ratio_to_percent", "arithmetic", [("a", NUM, "ratio")], NUM,
         lambda a: r6(float(a) * 100.0), sam(f_(0.02, 0.98, 3)),
         [S("a1", "ratio_to_percent", ["arg_0"],
            "Converts a ratio into a percentage.")],
         [S("g1", "percent_from_share", ["share"],
            "Expresses a share as a percentage.")],
         "express {a} as a percentage",
         units=[NEUTRAL], unit_out=m.U_PERCENT)
