"""Semantic-type compatibility for tool argument/output bindings (v5 registry).

JSON-type compatibility (number == number) is not enough: a Fahrenheit
temperature is JSON-number-compatible with a monetary `net_price`, but the
composition is semantically nonsensical. Every v5 tool parameter/output
already carries a fine-grained semantic tag from ``lib/synthetic_tools.py``
(e.g. ``length_km``, ``length_mi``, ``temp_c``, ``temp_f``) — one tag PER
UNIT, not per real-world quantity. The trainer's deterministic generator
(``synthetic_gen_v5.py``) deliberately keeps bindings STRICT at that
unit-exact granularity (``semantics_compatible()``: a consumer only accepts
the identical tag, unless it is the unit-agnostic ``generic_number`` slot).

The agentic challenger/solver loop is more permissive by design (matches the
old v4 behavior): it groups unit tags into real-world-quantity FAMILIES
(``length_km`` and ``length_mi`` are both "length") and accepts ANY binding
within the same family, plus the ``generic_number`` slot on EITHER side
(symmetric — a generic result may flow into a specific slot and vice versa,
exactly like NESTFUL problems that explicitly reinterpret a quantity as a
"part of a whole"). It still rejects cross-family bindings (temperature ->
money, fuel volume -> interest principal) which is the actual goal of this
gate.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .exec_bridge import TOOLS

GENERIC = "generic_number"

# unit-tag -> real-world-quantity family. Tags not listed here are their own
# singleton family (self-compatible only): count, money, percent, ratio,
# text, flag, list_number, object, day_number, current_a, data_gb, data_mb,
# force_n, power_w, pressure_pa, resistance_ohm, voltage_v, generic_number.
_UNIT_FAMILY: Dict[str, str] = {
    "length_cm": "length", "length_ft": "length", "length_in": "length",
    "length_km": "length", "length_m": "length", "length_mi": "length",
    "mass_g": "mass", "mass_kg": "mass", "mass_lb": "mass",
    "mass_mg": "mass", "mass_oz": "mass",
    "duration_day": "duration", "duration_h": "duration",
    "duration_min": "duration", "duration_s": "duration",
    "duration_week": "duration", "duration_year": "duration",
    "temp_c": "temperature", "temp_f": "temperature", "temp_k": "temperature",
    "volume_cm3": "volume", "volume_floz": "volume", "volume_gal": "volume",
    "volume_l": "volume", "volume_m3": "volume", "volume_ml": "volume",
    "speed_kmh": "speed", "speed_mph": "speed", "speed_ms": "speed",
    "area_acre": "area", "area_m2": "area",
    "energy_j": "energy", "energy_kcal": "energy", "energy_kwh": "energy",
    "energy_mj": "energy",
}


def _family(sem: str) -> str:
    return _UNIT_FAMILY.get(sem, sem)


def param_type(tool_name: str, param: str) -> str:
    """Semantic tag of a tool parameter (``generic_number`` if unknown)."""
    t = TOOLS.get(tool_name)
    if not t:
        return GENERIC
    meta = t["params"].get(param)
    return meta["semantic"] if meta else GENERIC


def output_type(tool_name: str, field: str = None) -> str:  # noqa: RUF013
    """Semantic tag of a tool's output, or of one ``out_fields`` field for
    object-typed outputs (``$varN.field$`` references)."""
    t = TOOLS.get(tool_name)
    if not t:
        return GENERIC
    if field is not None and t.get("out_fields") and field in t["out_fields"]:
        return t["out_fields"][field][1]
    return t["out_semantic"]


def compatible(producer_type: str, consumer_type: str) -> bool:
    """A binding is allowed when either side is GENERIC (unit-agnostic), or
    both sides belong to the same real-world-quantity family."""
    if producer_type == GENERIC or consumer_type == GENERIC:
        return True
    return _family(producer_type) == _family(consumer_type)


_REF_RE = re.compile(r"^\$([A-Za-z_]\w*)\.(\w+)\$$")


def semantic_errors(gold_calls: List[Dict[str, Any]],
                    tools: Dict[str, Dict[str, Any]] = None  # noqa: RUF013
                    ) -> List[str]:
    """Reject cross-family bindings not covered by a GENERIC slot on either
    side (e.g. temperature -> net_price, fuel volume -> principal_amount).

    ``tools`` is accepted for backward-compatible call sites but ignored —
    this module always resolves tool metadata from the canonical v5
    registry (``exec_bridge.TOOLS``) so generator/trainer stay in sync.
    """
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
                ref_field = m.group(2)
                producer = label_to_tool.get(ref_label)
                if not producer:
                    continue   # unknown/forward ref — reported by trace_validation
                p_type = output_type(producer, ref_field)
                c_type = param_type(name, k) if name else GENERIC
                if not compatible(p_type, c_type):
                    errs.append(
                        f"call {i + 1} arg {k!r} ({c_type}) is fed by "
                        f"{ref_label}.{ref_field} = {producer}() output "
                        f"({p_type}) — semantically incompatible: different "
                        "real-world quantity families and neither side is "
                        "a generic/unit-agnostic slot")
        own_label = "$" + str(call.get("label") or "").lstrip("$")
        if name:
            label_to_tool[own_label] = name
    return errs
