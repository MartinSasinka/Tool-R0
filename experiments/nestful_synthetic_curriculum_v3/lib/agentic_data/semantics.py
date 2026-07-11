"""Semantic-type compatibility for tool argument/output bindings.

JSON-type compatibility (number == number) is not enough: a Fahrenheit
temperature is JSON-number-compatible with a monetary `net_price`, but the
composition is semantically nonsensical — no meaningful real-world
transformation turns a temperature into a price. This module assigns every
tool parameter and output a SEMANTIC type from a small closed vocabulary and
rejects cross-family references unless one side is GENERIC — a deliberately
unit-agnostic math slot (`part`, `whole`, `value`, `numerator`, ...) which is
exactly the mechanism NESTFUL-style problems use to explicitly reinterpret a
quantity ("using that area as a PART of a whole").

Vocabulary: temperature_celsius, temperature_fahrenheit, money, percentage,
distance, duration, area, mass, count, speed, fuel_volume, boolean, text,
generic_scalar.

Three real (accepted) pilot rows motivated this gate — all execute cleanly
but compose unrelated real-world quantities:
  agentic_v4_stage2_000003: Fahrenheit temperature -> add_sales_tax.net_price
    (temperature -> money).
  agentic_v4_stage2_000006: Fahrenheit temperature ->
    calculate_simple_interest.annual_rate_percent (temperature -> percentage).
  agentic_v4_stage2_000009: fuel liters ->
    calculate_simple_interest.principal_amount (fuel_volume -> money).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

GENERIC = "generic_scalar"

# Families that may consume EACH OTHER directly (besides GENERIC on either
# side). Every family is at least self-compatible.
_FAMILY_COMPATIBLE: Dict[str, set] = {
    "temperature_celsius": {"temperature_celsius", "temperature_fahrenheit"},
    "temperature_fahrenheit": {"temperature_celsius", "temperature_fahrenheit"},
    "money": {"money"},
    "percentage": {"percentage"},
    "distance": {"distance"},
    "duration": {"duration"},
    "area": {"area"},
    "mass": {"mass"},
    "count": {"count"},
    "speed": {"speed"},
    "fuel_volume": {"fuel_volume"},
    "boolean": {"boolean"},
    "text": {"text"},
}

# tool_name -> {"params": {pname: semtype}, "output": semtype}. Params not
# listed default to GENERIC (safe: GENERIC never blocks a binding).
SEMANTIC_TYPES: Dict[str, Dict[str, Any]] = {
    # finance
    "calculate_simple_interest": {
        "params": {"principal_amount": "money", "annual_rate_percent": "percentage",
                   "years": "duration"},
        "output": "money"},
    "apply_discount": {
        "params": {"price": "money", "discount_percent": "percentage"},
        "output": "money"},
    "add_sales_tax": {
        "params": {"net_price": "money", "tax_rate_percent": "percentage"},
        "output": "money"},
    "split_bill_evenly": {
        "params": {"total_amount": "money", "num_people": "count"},
        "output": "money"},
    "calculate_tip_amount": {
        "params": {"bill_amount": "money", "tip_percent": "percentage"},
        "output": "money"},
    "convert_currency": {
        "params": {"amount": "money", "exchange_rate": GENERIC},
        "output": "money"},
    "monthly_installment": {
        "params": {"loan_amount": "money", "num_months": "count"},
        "output": "money"},
    # geometry
    "rectangle_area": {
        "params": {"length": "distance", "width": "distance"}, "output": "area"},
    "rectangle_perimeter": {
        "params": {"length": "distance", "width": "distance"}, "output": "distance"},
    "circle_area": {"params": {"radius": "distance"}, "output": "area"},
    "triangle_area": {
        "params": {"base": "distance", "height": "distance"}, "output": "area"},
    "scale_dimension": {
        "params": {"dimension": GENERIC, "scale_factor": GENERIC}, "output": GENERIC},
    # units / physics
    "celsius_to_fahrenheit": {
        "params": {"celsius": "temperature_celsius"}, "output": "temperature_fahrenheit"},
    "kilometers_to_miles": {
        "params": {"kilometers": "distance"}, "output": "distance"},
    "kilograms_to_pounds": {"params": {"kilograms": "mass"}, "output": "mass"},
    "average_speed": {
        "params": {"distance_km": "distance", "time_hours": "duration"},
        "output": "speed"},
    "travel_time_hours": {
        "params": {"distance_km": "distance", "speed_kmh": "speed"},
        "output": "duration"},
    "hours_to_minutes": {"params": {"hours": "duration"}, "output": "duration"},
    "fuel_needed_liters": {
        "params": {"distance_km": "distance", "consumption_per_100km": "fuel_volume"},
        "output": "fuel_volume"},
    # statistics — deliberately unit-agnostic (GENERIC): these are the
    # legitimate "reinterpret a quantity" slots NESTFUL-style tasks rely on
    "mean_of_values": {"params": {"values": GENERIC}, "output": GENERIC},
    "sum_of_values": {"params": {"values": GENERIC}, "output": GENERIC},
    "max_of_values": {"params": {"values": GENERIC}, "output": GENERIC},
    "value_range": {"params": {"values": GENERIC}, "output": GENERIC},
    "percentage_of": {
        "params": {"part": GENERIC, "whole": GENERIC}, "output": "percentage"},
    "increase_by_percent": {
        "params": {"value": GENERIC, "percent": "percentage"}, "output": GENERIC},
    "difference_of": {
        "params": {"minuend": GENERIC, "subtrahend": GENERIC}, "output": GENERIC},
    "ratio_of": {
        "params": {"numerator": GENERIC, "denominator": GENERIC}, "output": GENERIC},
    # shopping / inventory
    "total_price": {
        "params": {"unit_price": "money", "quantity": "count"}, "output": "money"},
    "remaining_stock": {
        "params": {"initial_stock": "count", "units_sold": "count"}, "output": "count"},
    "is_above_threshold": {
        "params": {"value": GENERIC, "threshold": GENERIC}, "output": "boolean"},
    "units_per_box": {
        "params": {"total_units": "count", "box_capacity": "count"}, "output": "count"},
    # text
    "format_as_currency": {
        "params": {"amount": "money", "currency_symbol": "text"}, "output": "text"},
    "repeat_word": {"params": {"word": "text", "times": "count"}, "output": "text"},
    "character_count": {"params": {"text": "text"}, "output": "count"},
}


def param_type(tool_name: str, param: str) -> str:
    return SEMANTIC_TYPES.get(tool_name, {}).get("params", {}).get(param, GENERIC)


def output_type(tool_name: str) -> str:
    return SEMANTIC_TYPES.get(tool_name, {}).get("output", GENERIC)


def compatible(producer_type: str, consumer_type: str) -> bool:
    """A binding is allowed when either side is GENERIC (unit-agnostic), or
    both sides belong to the same semantic family."""
    if producer_type == GENERIC or consumer_type == GENERIC:
        return True
    return consumer_type in _FAMILY_COMPATIBLE.get(producer_type, {producer_type})


_REF_RE = re.compile(r"^\$([A-Za-z_]\w*)\.(\w+)\$$")


def semantic_errors(gold_calls: List[Dict[str, Any]],
                    tools: Dict[str, Dict[str, Any]]) -> List[str]:
    """Reject cross-family bindings not covered by a GENERIC slot on either
    side (e.g. temperature -> net_price, fuel volume -> principal_amount)."""
    errs: List[str] = []
    label_to_tool: Dict[str, str] = {}
    for i, call in enumerate(gold_calls):
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        args = call.get("arguments")
        if isinstance(args, dict):
            for k, v in args.items():
                if not isinstance(v, str):
                    continue
                m = _REF_RE.match(v.strip())
                if not m:
                    continue
                ref_label = "$" + m.group(1)
                producer = label_to_tool.get(ref_label)
                if not producer:
                    continue   # unknown/forward ref — reported by trace_validation
                p_type = output_type(producer)
                c_type = param_type(name, k) if name else GENERIC
                if not compatible(p_type, c_type):
                    errs.append(
                        f"call {i + 1} arg {k!r} ({c_type}) is fed by "
                        f"{ref_label} = {producer}() output ({p_type}) — "
                        "semantically incompatible: different real-world "
                        "quantity families and neither side is a generic/"
                        "unit-agnostic slot")
        own_label = "$" + str(call.get("label") or "").lstrip("$")
        if name:
            label_to_tool[own_label] = name
    return errs
