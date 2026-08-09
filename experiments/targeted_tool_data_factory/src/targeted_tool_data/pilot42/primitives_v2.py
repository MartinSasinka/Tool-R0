"""Fail-closed semantic overlay for the existing primitive registry."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from .. import registry as reg
from ..capability import family_of
from ..repro import sha256_obj, write_json
from ..pilot41.primitive_semantics import semantics_for

SCHEMA_VERSION = "ttdf.pilot42.primitive_registry.v2"

CAPABILITY_TO_PRIMITIVES: Dict[str, List[str]] = {
    "arithmetic.add": ["add"], "arithmetic.subtract": ["subtract"],
    "arithmetic.multiply": ["multiply"], "arithmetic.divide": ["divide"],
    "arithmetic.binary": ["add", "subtract", "multiply", "divide"],
    "arithmetic.percentage_of": ["percent_of"],
    "arithmetic.increase_by_percent": ["increase_by_percent"],
    "arithmetic.decrease_by_percent": ["decrease_by_percent"],
    "comparison": ["is_greater"],
    "comparison.greater_than": ["is_greater"],
    "rounding": ["round_to_int", "round_places", "floor_value", "ceil_value"],
    "statistics": ["average_two"],
    "statistics.average_two": ["average_two"],
    "statistics.mean_three": ["mean_three"],
    "sequence.filter": ["filter_above", "top_k_values"],
    "sequence.reduce": ["sum_values", "count_values", "max_values", "min_values"],
    "conversion.numeric": ["seconds_to_minutes", "hours_to_minutes",
                           "minutes_to_seconds", "km_to_meters", "meters_to_km",
                           "celsius_to_fahrenheit", "fahrenheit_to_celsius"],
}


@dataclass(frozen=True)
class PrimitiveSpecV2:
    primitive_id: str
    capability_family: str
    exact_capabilities: List[str]
    runtime_input_types: List[str]
    semantic_input_types: List[str]
    runtime_output_type: str
    semantic_output_type: str
    unit_behavior: str
    deterministic: bool = True


def build_primitive_registry() -> Dict[str, PrimitiveSpecV2]:
    inverse: Dict[str, List[str]] = {}
    for cap, sids in CAPABILITY_TO_PRIMITIVES.items():
        for sid in sids:
            inverse.setdefault(sid, []).append(cap)
    out: Dict[str, PrimitiveSpecV2] = {}
    for sid, primitive in sorted(reg.all_primitives().items()):
        sem = semantics_for(sid)
        out[sid] = PrimitiveSpecV2(
            primitive_id=sid, capability_family=family_of(sid),
            exact_capabilities=sorted(set(inverse.get(sid, []) + [family_of(sid)])),
            runtime_input_types=[p[1] for p in primitive.params],
            semantic_input_types=[str(v) for v in sem.semantic_input_types],
            runtime_output_type=primitive.out_type,
            semantic_output_type=str(sem.semantic_output_type),
            unit_behavior=sem.unit_behavior or "preserve")
    return out


def validate_primitive_registry(
        specs: Dict[str, PrimitiveSpecV2] | None = None,
        required_capabilities: List[str] | None = None) -> List[str]:
    specs = specs or build_primitive_registry()
    errors: List[str] = []
    for sid, spec in specs.items():
        if not spec.semantic_input_types:
            errors.append(f"{sid}: missing semantic inputs")
        if not spec.semantic_output_type:
            errors.append(f"{sid}: missing semantic output")
        if not spec.unit_behavior:
            errors.append(f"{sid}: missing unit behavior")
        if spec.capability_family == "unknown":
            errors.append(f"{sid}: missing capability taxonomy")
    for cap in required_capabilities or []:
        choices = [sid for sid in CAPABILITY_TO_PRIMITIVES.get(cap, []) if sid in specs]
        if not choices:
            errors.append(f"capability {cap}: no concrete primitive")
    return errors


def bind_capability(capability: str, *, seed: int = 0) -> PrimitiveSpecV2:
    specs = build_primitive_registry()
    choices = [sid for sid in CAPABILITY_TO_PRIMITIVES.get(capability, []) if sid in specs]
    if not choices:
        raise ValueError(f"no primitive binding for capability {capability!r}")
    return specs[sorted(choices)[seed % len(choices)]]


def export_registry(path: Path) -> Dict[str, Any]:
    specs = build_primitive_registry()
    errors = validate_primitive_registry(specs)
    if errors:
        raise ValueError("primitive registry invalid: " + "; ".join(errors))
    rows = {sid: asdict(spec) for sid, spec in specs.items()}
    payload = {"schema_version": SCHEMA_VERSION, "primitives": rows,
               "capability_map": CAPABILITY_TO_PRIMITIVES,
               "registry_hash": sha256_obj(rows)}
    write_json(path, payload)
    return payload
