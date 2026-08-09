"""Pilot4.3 op registry v3 = adapted base primitives + real generic/coding ops.

Two things went wrong in Pilot4.2 and both are fixed here.

1. The capability map covered ~30 mostly-arithmetic capabilities, so a workflow
   called ``file_path_audit`` still bound ``add -> increase_by_percent -> compare``.
   Pilot4.3 adds ~80 genuinely non-arithmetic ops (list, dictionary, record,
   string, path, url, date/duration, boolean decision, formatting) in
   :mod:`.ops_new`, and every op declares the capability it *is*, not the domain
   it is used in.
2. The registry was mutated globally, so "coverage" was registry size. Here the
   registry is a plain dict built on demand; coverage is only ever counted from
   the ops that appear in exported gold calls.

The base registry is adapted rather than re-declared: its unit metadata
(``param_units`` / ``unit_out``) is a precise source for semantic types, so
``increase_by_percent`` keeps "second argument is a percentage" and
``km_to_meters`` keeps "input is kilometres" without a hand-written table.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Sequence, Tuple

from .. import registry as reg
from ..repro import sha256_obj
from . import semtypes as st

PRESERVE = "@preserve"

#: capability families counted as generic/coding (not arithmetic-with-a-label)
CODING_FAMILIES = frozenset({
    "list", "dictionary", "record", "string", "path", "url", "date",
    "duration", "boolean", "decision", "validation", "bitwise", "format",
    "classification", "lookup",
})


@dataclass(frozen=True)
class Param:
    name: str
    sem: str
    role: str = "value"

    @property
    def runtime(self) -> str:
        return st.runtime_of(self.sem)


@dataclass(frozen=True)
class Surface:
    track: str
    name: str
    param_names: Tuple[str, ...]
    description: str
    output_field: str = "output_0"


@dataclass(frozen=True)
class Op:
    pid: str
    capability: str
    params: Tuple[Param, ...]
    out_sem: str
    fn: Callable[..., Any]
    surfaces: Tuple[Surface, ...]
    source: str = "base"
    notes: str = ""

    @property
    def family(self) -> str:
        return self.capability.split(".")[0]

    @property
    def coding_like(self) -> bool:
        return self.family in CODING_FAMILIES

    @property
    def arity(self) -> int:
        return len(self.params)

    def surface(self, track: str) -> Surface:
        for s in self.surfaces:
            if s.track == track:
                return s
        # a silent fallback is how the surface holdout leaked: a task rendered on
        # G_GENERAL_2 would carry A_NATIVE tool names and still claim the track
        raise KeyError(f"{self.pid} has no {track} surface")

    def tracks(self) -> Tuple[str, ...]:
        return tuple(s.track for s in self.surfaces)

    def resolve_out_sem(self, in_sems: Sequence[str]) -> str:
        """Output semantic type, resolving the ``@preserve`` sentinel.

        Rate-like inputs are skipped: ``percent_of(20, 500 EUR)`` yields money,
        not a percentage, and that distinction is what keeps "844 % of a price"
        out of the dataset.
        """
        if self.out_sem != PRESERVE:
            return self.out_sem
        rate_like = {st.PERCENTAGE, st.RATIO, st.INDEX}
        for sem in in_sems:
            if sem in st.NUMERIC and sem != st.GENERIC and sem not in rate_like:
                return sem
        return st.GENERIC


# ── base-registry adaptation ─────────────────────────────────────────────
_UNIT_TO_SEM = {
    reg.U_ABSTRACT: st.GENERIC,
    reg.U_COUNT: st.COUNT,
    reg.U_RATIO: st.RATIO,
    reg.U_PERCENT: st.PERCENTAGE,
    reg.U_DUR_S: st.DUR_S,
    reg.U_DUR_MIN: st.DUR_MIN,
    reg.U_DUR_H: st.DUR_H,
    reg.U_LEN_KM: st.LEN_KM,
    reg.U_LEN_M: st.LEN_M,
    reg.U_TEMP_C: st.TEMP_C,
    reg.U_TEMP_F: st.TEMP_F,
    reg.U_TEXT: st.TEXT,
    reg.U_NUMTEXT: st.NUMERIC_TEXT,
    reg.U_LIST: st.NUMBER_LIST,
    reg.U_FLAG: st.FLAG,
}

#: primitive id -> capability. Written out in full so the taxonomy is auditable
#: and so that renaming a workflow can never change a capability count.
BASE_CAPABILITY: Dict[str, str] = {
    # arithmetic
    "add": "arithmetic.add", "subtract": "arithmetic.subtract",
    "multiply": "arithmetic.multiply", "divide": "arithmetic.divide",
    "power": "arithmetic.power", "floor_divide": "arithmetic.floor_divide",
    "modulo": "arithmetic.modulo", "abs_difference": "arithmetic.abs_difference",
    "negate": "arithmetic.negate", "inverse": "arithmetic.inverse",
    "sqrt": "arithmetic.sqrt", "square": "arithmetic.square",
    "digit_sum": "arithmetic.digit_sum",
    "sum_three": "arithmetic.sum_three", "product_three": "arithmetic.product_three",
    # rates and ratios
    "percent_of": "rates.percent_of", "ratio_of": "rates.ratio_of",
    "increase_by_percent": "rates.increase_by_percent",
    "decrease_by_percent": "rates.decrease_by_percent",
    "ratio_to_percent": "rates.ratio_to_percent",
    # comparison / selection
    "is_greater": "comparison.greater", "max_two": "comparison.max",
    "min_two": "comparison.min",
    # rounding
    "round_to_int": "rounding.to_int", "round_places": "rounding.places",
    "floor_value": "rounding.floor", "ceil_value": "rounding.ceil",
    "round_direction": "rounding.direction",
    # statistics
    "average_two": "statistics.average_two", "mean_three": "statistics.mean_three",
    "mean_values": "statistics.mean", "median_values": "statistics.median",
    "range_spread": "statistics.range", "range_three": "statistics.range_three",
    # list / sequence
    "sum_values": "list.reduce_sum", "count_values": "list.reduce_count",
    "max_values": "list.reduce_max", "min_values": "list.reduce_min",
    "filter_above": "list.filter", "top_k_values": "list.filter_top_k",
    "index_of_max": "list.index_of_max", "value_at_position": "list.index",
    "cumulative_sums": "list.map_cumulative", "offset_list": "list.map_offset",
    "scale_list": "list.map_scale", "sort_values_desc": "list.map_sort",
    "append_value": "list.combine_append", "concat_lists": "list.combine_concat",
    "join_values": "list.combine_join",
    # dictionary
    "lookup_unit_factor": "dictionary.lookup",
    "apply_rate_override": "dictionary.update",
    # string
    "text_length": "string.count_length", "text_upper": "string.normalize_upper",
    "concat_texts": "string.concat", "parse_number": "string.parse_number",
    # formatting
    "format_fixed": "format.fixed", "number_to_string": "format.number_text",
    "format_with_unit": "format.with_unit", "tag_value": "format.tag",
    # path / url
    "join_path_segments": "path.join", "file_extension": "path.extension",
    "domain_of_url": "url.domain",
    # date / duration
    "days_to_hours": "duration.convert_days_hours",
    "hours_to_minutes": "duration.convert_hours_minutes",
    "minutes_to_seconds": "duration.convert_minutes_seconds",
    "seconds_to_minutes": "duration.convert_seconds_minutes",
    "weeks_to_days": "duration.convert_weeks_days",
    "minutes_since_midnight": "date.time_of_day",
    # unit conversion
    "km_to_meters": "unit_conversion.length_km_m",
    "meters_to_km": "unit_conversion.length_m_km",
    "celsius_to_fahrenheit": "unit_conversion.temperature_c_f",
    "fahrenheit_to_celsius": "unit_conversion.temperature_f_c",
    # geometry
    "rectangle_area": "geometry.rectangle_area",
    "rectangle_perimeter": "geometry.rectangle_perimeter",
    "circle_area": "geometry.circle_area", "hypotenuse": "geometry.hypotenuse",
    # boolean / decision
    "logical_and": "boolean.and", "logical_or": "boolean.or",
    "logical_not": "boolean.not", "is_divisible_by": "boolean.divisible",
    # validation / classification
    "clamp": "validation.clamp", "is_non_negative": "validation.non_negative",
    "is_within_range": "validation.in_range",
    "classify_threshold": "classification.threshold",
    "grade_band": "classification.band",
    # bitwise
    "bitwise_and": "bitwise.and", "bitwise_or": "bitwise.or",
    "bitwise_xor": "bitwise.xor", "left_shift": "bitwise.shift",
}

#: semantic-type overrides where the base unit metadata is too coarse
_BASE_SEM_OVERRIDE: Dict[str, Dict[str, str]] = {
    "join_path_segments": {"__out__": st.PATH},
    "file_extension": {"__out__": st.TEXT},
    "domain_of_url": {"__out__": st.TEXT},
    "parse_number": {"__out__": st.GENERIC},
    "classify_threshold": {"__out__": st.CATEGORY},
    "grade_band": {"__out__": st.CATEGORY},
    "join_values": {"__out__": st.TEXT},
    "index_of_max": {"__out__": st.INDEX},
    "count_values": {"__out__": st.COUNT},
    "text_length": {"__out__": st.COUNT},
    "digit_sum": {"__out__": st.COUNT},
    "ratio_to_percent": {"__out__": st.PERCENTAGE},
    "ratio_of": {"__out__": st.RATIO},
}

_TRACK_ORDER = ("A_NATIVE", "G_GENERAL_1", "G_GENERAL_2")

#: Second generic surface for the base primitives that ship with fewer than two
#: generic names. Without these, ``Op.surface("G_GENERAL_2")`` had nothing to
#: return for 81 of 199 ops, the surface holdout silently trained and tested on
#: the same tool names, and the "unseen surface" split measured nothing. Each
#: entry is ``(tool name, parameter names, description)`` and is deliberately
#: worded away from the operation's own vocabulary.
_BASE_G2: Dict[str, Tuple[str, Tuple[str, ...], str]] = {
    "abs_difference": ("distance_between_values", ("value_left", "value_right"),
                       "Returns how far apart two values lie."),
    "append_value": ("series_with_extra_entry",
                     ("existing_series", "extra_entry"),
                     "Returns the series with one more entry at the end."),
    "apply_rate_override": ("value_under_named_rule",
                            ("input_amount", "rule_name"),
                            "Applies the named rule to the input amount."),
    "average_two": ("mean_of_pair", ("left_reading", "right_reading"),
                    "Averages a pair of readings."),
    "bitwise_and": ("common_flag_bits", ("flags_left", "flags_right"),
                    "Returns the flag bits present in both inputs."),
    "bitwise_or": ("union_of_flag_bits", ("flags_first", "flags_second"),
                   "Returns the flag bits present in at least one input."),
    "bitwise_xor": ("differing_flag_bits", ("flags_a", "flags_b"),
                    "Returns the flag bits that differ between the inputs."),
    "ceil_value": ("next_whole_up", ("raw_amount",),
                   "Returns the next whole number at or above the amount."),
    "celsius_to_fahrenheit": ("temperature_in_fahrenheit", ("reading_celsius",),
                              "Restates a Celsius reading in Fahrenheit."),
    "circle_area": ("round_face_area", ("circle_radius",),
                    "Returns the area of a round face of the given radius."),
    "clamp": ("kept_inside_bounds",
              ("raw_reading", "lower_bound", "upper_bound"),
              "Returns the reading pulled inside the bounds."),
    "classify_threshold": ("limit_comparison_label",
                           ("observed_value", "threshold_value"),
                           "Labels an observed value against a threshold."),
    "concat_lists": ("appended_series", ("head_series", "tail_series"),
                     "Returns the first series followed by the second."),
    "concat_texts": ("combined_text", ("left_text", "right_text"),
                     "Returns the two text fragments as one string."),
    "count_values": ("entry_count", ("counted_series",),
                     "Counts the entries in the series."),
    "cumulative_sums": ("running_total_series", ("accumulated_series",),
                        "Returns the series of running totals."),
    "days_to_hours": ("hour_span_of_days", ("days",),
                      "Restates a number of days as hours."),
    "decrease_by_percent": ("value_after_reduction",
                            ("original_value", "reduction_rate"),
                            "Returns the value after the percentage reduction."),
    "digit_sum": ("sum_of_digits", ("whole_number",),
                  "Adds up the digits of a whole number."),
    "domain_of_url": ("site_host", ("web_address",),
                      "Returns the site host of a web address."),
    "fahrenheit_to_celsius": ("temperature_in_celsius", ("reading_fahrenheit",),
                              "Restates a Fahrenheit reading in Celsius."),
    "file_extension": ("file_type_suffix", ("file_name",),
                       "Returns the type suffix of a file name."),
    "filter_above": ("entries_exceeding", ("checked_series", "cut_off"),
                     "Keeps the entries above the cut-off."),
    "floor_value": ("next_whole_down", ("truncated_amount",),
                    "Returns the whole number at or below the amount."),
    "format_fixed": ("decimal_rendering", ("rendered_value", "places"),
                     "Renders a value with the requested number of decimals."),
    "format_with_unit": ("measurement_with_unit_label",
                         ("quantity", "unit_label"),
                         "Writes a quantity together with its unit label."),
    "grade_band": ("score_band", ("achieved_score",),
                   "Returns the band a score falls into."),
    "hours_to_minutes": ("minute_span_of_hours", ("hours",),
                         "Restates a number of hours as minutes."),
    "hypotenuse": ("longest_side_of_right_triangle",
                   ("short_side", "other_side"),
                   "Returns the longest side of a right triangle."),
    "increase_by_percent": ("value_after_uplift",
                            ("original_amount", "uplift_rate"),
                            "Returns the amount after the percentage uplift."),
    "index_of_max": ("position_of_largest_entry", ("searched_series",),
                     "Returns the position of the largest entry."),
    "inverse": ("one_divided_by", ("divisor_value",),
                "Returns one divided by the value."),
    "is_divisible_by": ("splits_without_remainder",
                        ("total_units", "group_units"),
                        "Reports whether the total splits without remainder."),
    "is_greater": ("above_threshold", ("candidate_value", "threshold"),
                   "Reports whether the candidate is above the threshold."),
    "is_non_negative": ("not_negative", ("checked_value",),
                        "Reports whether the value is zero or more."),
    "is_within_range": ("between_bounds",
                        ("checked_reading", "lower_edge", "upper_edge"),
                        "Reports whether the reading lies between the bounds."),
    "join_path_segments": ("nested_location", ("outer_folder", "inner_name"),
                           "Returns the location of a name inside a folder."),
    "join_values": ("delimited_listing", ("listed_series", "separator"),
                    "Writes the series as one separated listing."),
    "km_to_meters": ("metre_distance", ("kilometres",),
                     "Restates a distance in metres."),
    "left_shift": ("bits_moved_up", ("flag_value", "positions"),
                   "Moves the bits of a value up by the given positions."),
    "logical_and": ("all_conditions_true", ("check_one", "check_two"),
                    "Reports whether both checks are true."),
    "logical_not": ("condition_negated", ("negated_check",),
                    "Reports the opposite of the check."),
    "logical_or": ("any_condition_true", ("check_a", "check_b"),
                   "Reports whether at least one check is true."),
    "lookup_unit_factor": ("metres_per_unit", ("unit_code",),
                           "Returns how many metres one unit of the code covers."),
    "max_two": ("higher_of_pair", ("option_one", "option_two"),
                "Returns the higher of two options."),
    "max_values": ("largest_entry_of_series", ("scanned_series",),
                   "Returns the largest entry of the series."),
    "mean_three": ("average_of_three_readings",
                   ("first_reading", "second_reading", "third_reading"),
                   "Averages three readings."),
    "mean_values": ("series_average", ("averaged_series",),
                    "Returns the average of the series."),
    "median_values": ("middle_entry_of_series", ("ordered_series",),
                      "Returns the middle entry of the ordered series."),
    "meters_to_km": ("kilometre_distance", ("metres",),
                     "Restates a distance in kilometres."),
    "min_two": ("lower_of_pair", ("choice_one", "choice_two"),
                "Returns the lower of two choices."),
    "min_values": ("smallest_entry_of_series", ("swept_series",),
                   "Returns the smallest entry of the series."),
    "minutes_since_midnight": ("minutes_into_the_day",
                               ("hours_part", "minutes_part"),
                               "Returns how far into the day a clock time is."),
    "minutes_to_seconds": ("second_span_of_minutes", ("minutes",),
                           "Restates a number of minutes as seconds."),
    "negate": ("opposite_value", ("flipped_amount",),
               "Returns the value with the opposite sign."),
    "number_to_string": ("amount_as_text", ("numeric_amount",),
                         "Writes an amount as text."),
    "offset_list": ("series_shifted_by", ("shifted_series", "offset_amount"),
                    "Adds a fixed offset to every entry."),
    "parse_number": ("amount_from_text", ("text_field",),
                     "Extracts the amount contained in a text field."),
    "product_three": ("three_way_product",
                      ("first_factor", "second_factor", "third_factor"),
                      "Multiplies three factors together."),
    "range_spread": ("series_range", ("measured_series",),
                     "Returns the range covered by the series."),
    "range_three": ("spread_of_three", ("reading_a", "reading_b", "reading_c"),
                    "Returns the spread across three readings."),
    "ratio_of": ("share_of_whole", ("part_amount", "whole_amount"),
                 "Returns the part as a share of the whole."),
    "ratio_to_percent": ("share_as_percentage", ("share_value",),
                         "Restates a share as a percentage."),
    "rectangle_area": ("area_from_sides", ("length_side", "width_side"),
                       "Returns the area enclosed by two sides."),
    "rectangle_perimeter": ("border_length_from_sides",
                            ("long_side", "short_side"),
                            "Returns the border length around two sides."),
    "round_direction": ("whole_number_in_direction",
                        ("rounded_amount", "rounding_mode"),
                        "Rounds to a whole number in the given mode."),
    "round_places": ("value_at_precision", ("precise_amount", "decimal_places"),
                     "Returns the value at the requested precision."),
    "round_to_int": ("closest_whole_number", ("approximated_amount",),
                     "Returns the closest whole number."),
    "scale_list": ("series_multiplied_by", ("scaled_series", "factor"),
                   "Multiplies every entry by the factor."),
    "seconds_to_minutes": ("full_minutes_in_seconds", ("seconds",),
                           "Counts the full minutes inside a span of seconds."),
    "sort_values_desc": ("series_ranked_downwards", ("ranked_series",),
                         "Returns the series ranked from largest to smallest."),
    "sqrt": ("square_root_of", ("radicand",),
             "Returns the square root of the value."),
    "square": ("value_squared", ("base_value",),
               "Returns the value raised to the second power."),
    "sum_three": ("total_of_three", ("first_part", "second_part", "third_part"),
                  "Adds three parts into one total."),
    "sum_values": ("series_total", ("totalled_series",),
                   "Returns the total of the series."),
    "tag_value": ("reference_label", ("label_prefix", "label_number"),
                  "Builds a reference label from a prefix and a number."),
    "text_length": ("length_of_text", ("measured_text",),
                    "Returns how many characters the text has."),
    "text_upper": ("upper_case_text", ("raw_label",),
                   "Returns the text in upper case."),
    "top_k_values": ("first_n_entries", ("trimmed_series", "entry_count"),
                     "Returns the leading n entries of the series."),
    "value_at_position": ("entry_by_index", ("series_entries", "index_position"),
                          "Returns the entry at the given index."),
    "weeks_to_days": ("day_span_of_weeks", ("weeks",),
                      "Restates a number of weeks as days."),
}


def _sem_for_param(sid: str, index: int, prim: reg.Primitive) -> str:
    name, rtype, semantic = prim.params[index]
    override = _BASE_SEM_OVERRIDE.get(sid, {}).get(name)
    if override:
        return override
    expect = prim.param_units[index] if index < len(prim.param_units) else reg.E_ANY
    if rtype == reg.BOOL:
        return st.FLAG
    if rtype == reg.ARR:
        return st.NUMBER_LIST
    if rtype == reg.STR:
        if sid == "join_path_segments":
            return st.PATH
        if sid in ("file_extension",):
            return st.PATH
        if sid == "domain_of_url":
            return st.URL
        if sid == "parse_number":
            return st.NUMERIC_TEXT
        if semantic in ("unit", "unit_name"):
            return st.UNIT_NAME
        return st.TEXT
    if semantic == "percentage":
        return st.PERCENTAGE
    if expect in _UNIT_TO_SEM and expect not in (reg.E_ANY, reg.E_NEUTRAL):
        return _UNIT_TO_SEM[expect]
    if expect == reg.E_NEUTRAL:
        # dimensionless by contract: a factor, exponent, divisor or rate
        return st.GENERIC if semantic in ("operand", "value") else st.GENERIC
    if rtype == reg.INT:
        return st.COUNT if semantic in ("count", "k", "position") else st.GENERIC
    return st.GENERIC


def _out_sem_for(sid: str, prim: reg.Primitive) -> str:
    override = _BASE_SEM_OVERRIDE.get(sid, {}).get("__out__")
    if override:
        return override
    if prim.out_type == reg.BOOL:
        return st.FLAG
    if prim.out_type == reg.ARR:
        return st.NUMBER_LIST
    if prim.out_type == reg.STR:
        return st.NUMERIC_TEXT if prim.answer_kind == "numeric_string" else st.TEXT
    if prim.unit_out == reg.PRESERVE:
        return PRESERVE
    return _UNIT_TO_SEM.get(prim.unit_out, st.GENERIC)


def _adapt_base(sid: str, prim: reg.Primitive) -> Op:
    params = tuple(
        Param(name=p[0], sem=_sem_for_param(sid, i, prim), role=p[2])
        for i, p in enumerate(prim.params))
    surfaces: List[Surface] = []
    pool = [("A_NATIVE", prim.surfaces_a[0] if prim.surfaces_a else None)]
    g = list(prim.surfaces_g)
    a_extra = list(prim.surfaces_a[1:])
    pool.append(("G_GENERAL_1", g[0] if g else (a_extra[0] if a_extra else None)))
    g2 = g[1] if len(g) > 1 else (a_extra[0] if a_extra else None)
    pool.append(("G_GENERAL_2", g2))
    for track, surf in pool:
        if surf is None:
            continue
        surfaces.append(Surface(track=track, name=surf.name,
                                param_names=tuple(surf.param_names),
                                description=surf.description,
                                output_field=surf.output_field))
    tracks = {s.track for s in surfaces}
    if "G_GENERAL_2" not in tracks:
        name, pnames, desc = _BASE_G2[sid]
        output = surfaces[0].output_field if surfaces else "output_0"
        surfaces.append(Surface(track="G_GENERAL_2", name=name,
                                param_names=pnames, description=desc,
                                output_field=output))
    return Op(pid=sid, capability=BASE_CAPABILITY[sid], params=params,
              out_sem=_out_sem_for(sid, prim), fn=prim.fn,
              surfaces=tuple(surfaces), source="base")


# ── registry assembly ────────────────────────────────────────────────────
_CACHE: Dict[str, Op] | None = None

#: Integers wider than this are rejected on the way in and on the way out of every
#: op. Nothing in this dataset legitimately exceeds MAX_ABS = 1e12, but the shortcut
#: search composes ops freely, and ``left_shift(1184, 4503)`` produces a number with
#: thousands of digits: harmless to the executor, which rejects it, yet every
#: subsequent float() and str() on it costs milliseconds. One task in the smoke run
#: spent nine minutes in exactly that way.
MAX_INT_BITS = 64


def _bounded(fn):
    """Wrap an op so oversized integers cannot enter or leave it."""
    def inner(**kwargs):
        for value in kwargs.values():
            if type(value) is int and value.bit_length() > MAX_INT_BITS:
                raise ValueError("oversized integer input")
        out = fn(**kwargs)
        if type(out) is int and out.bit_length() > MAX_INT_BITS:
            raise ValueError("oversized integer result")
        return out
    return inner


def build_ops(*, reload: bool = False) -> Dict[str, Op]:
    """The Pilot4.3 op registry. Base primitives + Pilot4.3 generic/coding ops."""
    global _CACHE
    if _CACHE is not None and not reload:
        return _CACHE
    from .ops_new import new_ops

    ops: Dict[str, Op] = {}
    for sid, prim in sorted(reg.all_primitives().items()):
        if sid not in BASE_CAPABILITY:
            continue                     # never silently absorb an unmapped op
        ops[sid] = _adapt_base(sid, prim)
    for op in new_ops():
        if op.pid in ops:
            raise ValueError(f"duplicate op id {op.pid}")
        ops[op.pid] = op
    for pid, op in ops.items():
        ops[pid] = replace(op, fn=_bounded(op.fn))
    errs = validate_ops(ops)
    if errs:
        raise ValueError("op registry invalid: " + "; ".join(errs[:12]))
    _CACHE = ops
    return ops


def validate_ops(ops: Dict[str, Op]) -> List[str]:
    """Fail-closed registry validation (surface uniqueness, types, capabilities)."""
    errs: List[str] = []
    seen: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
    for pid, op in sorted(ops.items()):
        if not op.capability or "." not in op.capability:
            errs.append(f"{pid}: capability must be family.name")
        if not op.params:
            errs.append(f"{pid}: no parameters")
        for p in op.params:
            if p.sem not in st.ALL:
                errs.append(f"{pid}: unknown param semantic {p.sem}")
        if op.out_sem != PRESERVE and op.out_sem not in st.ALL:
            errs.append(f"{pid}: unknown output semantic {op.out_sem}")
        if not op.surfaces:
            errs.append(f"{pid}: no surfaces")
        for s in op.surfaces:
            if len(s.param_names) != len(op.params):
                errs.append(f"{pid}/{s.name}: surface arity mismatch")
            key = seen.get(s.name)
            sig = tuple(s.param_names)
            if key is None:
                seen[s.name] = (pid, sig)
            elif key != (pid, sig):
                errs.append(f"tool name {s.name!r} maps to {key} and {(pid, sig)}")
    return errs


def ops_by_capability(ops: Dict[str, Op] | None = None) -> Dict[str, List[str]]:
    ops = ops or build_ops()
    out: Dict[str, List[str]] = {}
    for pid, op in sorted(ops.items()):
        out.setdefault(op.capability, []).append(pid)
    return out


def ops_by_family(ops: Dict[str, Op] | None = None) -> Dict[str, List[str]]:
    ops = ops or build_ops()
    out: Dict[str, List[str]] = {}
    for pid, op in sorted(ops.items()):
        out.setdefault(op.family, []).append(pid)
    return out


def get(pid: str) -> Op:
    return build_ops()[pid]


def registry_hash(ops: Dict[str, Op] | None = None) -> str:
    ops = ops or build_ops()
    payload = {
        pid: {
            "capability": op.capability,
            "params": [(p.name, p.sem, p.role) for p in op.params],
            "out": op.out_sem,
            "surfaces": [(s.track, s.name, list(s.param_names)) for s in op.surfaces],
            "source": op.source,
        }
        for pid, op in sorted(ops.items())
    }
    return sha256_obj(payload)


def export_registry() -> Dict[str, Any]:
    ops = build_ops()
    rows = {
        pid: {
            "primitive_id": pid,
            "capability": op.capability,
            "capability_family": op.family,
            "coding_like": op.coding_like,
            "arity": op.arity,
            "semantic_input_types": [p.sem for p in op.params],
            "runtime_input_types": [p.runtime for p in op.params],
            "semantic_output_type": op.out_sem,
            "runtime_output_type": (
                "number" if op.out_sem == PRESERVE else st.runtime_of(op.out_sem)),
            "answer_type": (
                "float" if op.out_sem == PRESERVE else st.answer_type_of(op.out_sem)),
            "source": op.source,
            "surfaces": {s.track: {"name": s.name,
                                   "param_names": list(s.param_names),
                                   "description": s.description}
                         for s in op.surfaces},
        }
        for pid, op in sorted(ops.items())
    }
    families = ops_by_family(ops)
    return {
        "schema_version": "ttdf.pilot43.primitive_registry.v3",
        "n_primitives": len(rows),
        "n_capabilities": len(ops_by_capability(ops)),
        "n_families": len(families),
        "n_coding_primitives": sum(1 for op in ops.values() if op.coding_like),
        "coding_families": sorted(f for f in families if f in CODING_FAMILIES),
        "primitives": rows,
        "capability_map": ops_by_capability(ops),
        "registry_hash": registry_hash(ops),
    }
