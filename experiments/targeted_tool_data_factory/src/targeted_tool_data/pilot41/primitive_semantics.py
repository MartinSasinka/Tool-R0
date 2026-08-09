"""Overlay semantic-type metadata on existing Pilot4 primitives.

Does not mutate the frozen registry hash path used by Pilot4; Pilot4.1 looks
semantics up here and falls back to conservative GenericScalar / FreeText.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import registry as reg
from ..capability import family_of
from .semantic_types import SemanticType

SCHEMA_VERSION = "ttdf.primitive_semantics.v1"

# Maps out_semantic / param semantic strings from the registry onto types.
_OUT_MAP = {
    "quantity": "GenericScalar",
    "flag": "Boolean",
    "count": "Count",
    "text": "FreeText",
    "numeric_text": "FreeText",
    "number_list": "List[GenericScalar]",
}
_PARAM_MAP = {
    "operand": "GenericScalar",
    "percentage": "Percentage",
    "base": "GenericScalar",
    "rate": "Ratio",
    "values": "List[GenericScalar]",
    "text": "FreeText",
    "string": "FreeText",
    "filename": "Filename",
    "extension": "FileExtension",
    "url": "URL",
    "domain": "Domain",
    "bits": "GenericScalar",
    "index": "Index",
    "position": "Index",
    "angle": "Angle",
    "radius": "Length",
    "side": "Length",
    "temperature": "TemperatureCelsius",
    "celsius": "TemperatureCelsius",
    "fahrenheit": "TemperatureFahrenheit",
    "seconds": "DurationSeconds",
    "minutes": "DurationMinutes",
    "hours": "DurationHours",
    "days": "DurationDays",
    "money": "Money",
    "price": "Money",
    "amount": "Money",
    "flag": "Boolean",
    "predicate": "Boolean",
}

# Explicit overrides for well-known conversion / domain primitives
_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "add": {"semantic_inputs": ["GenericScalar", "GenericScalar"],
            "semantic_output": "GenericScalar", "unit_behavior": "preserve",
            "allowed_workflow_families": ["*"],
            "forbidden_successor_families": []},
    "subtract": {"semantic_inputs": ["GenericScalar", "GenericScalar"],
                 "semantic_output": "GenericScalar", "unit_behavior": "preserve",
                 "allowed_workflow_families": ["*"]},
    "multiply": {"semantic_inputs": ["GenericScalar", "GenericScalar"],
                 "semantic_output": "GenericScalar", "unit_behavior": "preserve",
                 "allowed_workflow_families": ["*"]},
    "divide": {"semantic_inputs": ["GenericScalar", "GenericScalar"],
               "semantic_output": "Ratio", "unit_behavior": "ignore",
               "allowed_workflow_families": ["*"]},
    "percent_of": {"semantic_inputs": ["Percentage", "GenericScalar"],
                   "semantic_output": "GenericScalar", "unit_behavior": "preserve",
                   "allowed_workflow_families": ["commerce", "personal_finance",
                                                 "rates_and_ratios", "*"]},
    "increase_by_percent": {
        "semantic_inputs": ["GenericScalar", "Percentage"],
        "semantic_output": "GenericScalar", "unit_behavior": "preserve",
        "output_role": "adjusted_value",
        "allowed_workflow_families": ["commerce", "personal_finance", "*"]},
    "decrease_by_percent": {
        "semantic_inputs": ["GenericScalar", "Percentage"],
        "semantic_output": "GenericScalar", "unit_behavior": "preserve",
        "output_role": "discounted_value",
        "allowed_workflow_families": ["commerce", "personal_finance", "*"]},
}


@dataclass
class PrimitiveSemantics:
    primitive_id: str
    capability_family: str
    runtime_input_types: List[str]
    semantic_input_types: List[SemanticType]
    runtime_output_type: str
    semantic_output_type: SemanticType
    unit_behavior: str = "preserve"
    allowed_input_roles: List[str] = field(default_factory=list)
    output_role: str = ""
    allowed_workflow_families: List[str] = field(default_factory=lambda: ["*"])
    forbidden_successor_families: List[str] = field(default_factory=list)
    semantic_neighbors: List[str] = field(default_factory=list)
    confusable_non_equivalents: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "primitive_id": self.primitive_id,
            "capability_family": self.capability_family,
            "runtime_input_types": list(self.runtime_input_types),
            "semantic_input_types": [str(s) for s in self.semantic_input_types],
            "runtime_output_type": self.runtime_output_type,
            "semantic_output_type": str(self.semantic_output_type),
            "unit_behavior": self.unit_behavior,
            "allowed_input_roles": list(self.allowed_input_roles),
            "output_role": self.output_role,
            "allowed_workflow_families": list(self.allowed_workflow_families),
            "forbidden_successor_families": list(self.forbidden_successor_families),
            "semantic_neighbors": list(self.semantic_neighbors),
            "confusable_non_equivalents": list(self.confusable_non_equivalents),
        }


_CACHE: Dict[str, PrimitiveSemantics] = {}


def _infer_param_type(semantic: str, runtime: str) -> SemanticType:
    key = (semantic or "").lower()
    if key in _PARAM_MAP:
        return SemanticType.parse(_PARAM_MAP[key])
    if runtime == "boolean":
        return SemanticType.parse("Boolean")
    if runtime == "string":
        return SemanticType.parse("FreeText")
    if runtime == "array":
        return SemanticType.parse("List[GenericScalar]")
    if runtime == "integer":
        return SemanticType.parse("Count")
    return SemanticType.parse("GenericScalar")


def semantics_for(sid: str) -> PrimitiveSemantics:
    if sid in _CACHE:
        return _CACHE[sid]
    prim = reg.get(sid)
    ov = _OVERRIDES.get(sid, {})
    runtime_ins = [t for _n, t, _s in prim.params]
    if "semantic_inputs" in ov:
        sem_ins = [SemanticType.parse(x) for x in ov["semantic_inputs"]]
    else:
        sem_ins = [_infer_param_type(s, t) for _n, t, s in prim.params]
    if "semantic_output" in ov:
        sem_out = SemanticType.parse(ov["semantic_output"])
    else:
        sem_out = SemanticType.parse(
            _OUT_MAP.get(prim.out_semantic, "GenericScalar"))
    # Unit-based refinement
    uout = (prim.unit_out or "").lower()
    if "duration" in uout or uout in ("s", "sec", "seconds"):
        sem_out = SemanticType.parse("DurationSeconds")
    elif uout in ("min", "minutes"):
        sem_out = SemanticType.parse("DurationMinutes")
    elif uout in ("h", "hr", "hours"):
        sem_out = SemanticType.parse("DurationHours")
    elif uout in ("d", "day", "days"):
        sem_out = SemanticType.parse("DurationDays")
    elif "celsius" in uout:
        sem_out = SemanticType.parse("TemperatureCelsius")
    elif "fahrenheit" in uout:
        sem_out = SemanticType.parse("TemperatureFahrenheit")

    ps = PrimitiveSemantics(
        primitive_id=sid,
        capability_family=family_of(sid),
        runtime_input_types=runtime_ins,
        semantic_input_types=sem_ins,
        runtime_output_type=prim.out_type,
        semantic_output_type=sem_out,
        unit_behavior=ov.get("unit_behavior", "preserve"),
        allowed_input_roles=list(ov.get("allowed_input_roles") or []),
        output_role=str(ov.get("output_role") or prim.out_semantic or ""),
        allowed_workflow_families=list(
            ov.get("allowed_workflow_families") or ["*"]),
        forbidden_successor_families=list(
            ov.get("forbidden_successor_families") or []),
        semantic_neighbors=list(ov.get("semantic_neighbors") or []),
        confusable_non_equivalents=list(
            ov.get("confusable_non_equivalents") or []),
    )
    _CACHE[sid] = ps
    return ps


def all_semantics() -> Dict[str, PrimitiveSemantics]:
    return {sid: semantics_for(sid) for sid in reg.all_primitives()}
