"""Semantic type system for Pilot4.1.

Runtime types remain the executor wire types (number/integer/string/...).
Semantic types carry meaning (Money, DurationDays, ...) and block random
typed-but-nonsensical DAG edges.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "ttdf.semantic_type.v1"

# Base semantic types (parametrized forms use SemanticType.parse)
BASE_TYPES = [
    "Money", "Percentage", "Ratio", "Probability", "Count", "GenericScalar",
    "DurationSeconds", "DurationMinutes", "DurationHours", "DurationDays",
    "TemperatureCelsius", "TemperatureFahrenheit",
    "Length", "Area", "Volume", "Mass", "Speed", "Angle",
    "Index", "Boolean", "Category", "Label", "FreeText", "Token",
    "Filename", "FileExtension", "URL", "Domain", "Date", "DateTime",
]

# Families that may freely convert among themselves via arithmetic
NUMERIC_FAMILY = {
    "Money", "Percentage", "Ratio", "Probability", "Count", "GenericScalar",
    "Length", "Area", "Volume", "Mass", "Speed", "Angle", "Index",
}
DURATION_FAMILY = {
    "DurationSeconds", "DurationMinutes", "DurationHours", "DurationDays",
}
TEMP_FAMILY = {"TemperatureCelsius", "TemperatureFahrenheit"}
TEXT_FAMILY = {"FreeText", "Label", "Token", "Category", "Filename",
               "FileExtension", "URL", "Domain"}


@dataclass(frozen=True)
class SemanticType:
    name: str
    params: Tuple[Tuple[str, str], ...] = ()

    def __str__(self) -> str:
        if not self.params:
            return self.name
        inner = ",".join(f"{k}={v}" for k, v in self.params)
        return f"{self.name}[{inner}]"

    @staticmethod
    def parse(spec: str) -> "SemanticType":
        spec = (spec or "GenericScalar").strip()
        m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[([^\]]*)\])?", spec)
        if not m:
            return SemanticType("GenericScalar")
        name = m.group(1)
        params: List[Tuple[str, str]] = []
        if m.group(2):
            for part in m.group(2).split(","):
                part = part.strip()
                if not part:
                    continue
                if "=" in part:
                    k, v = part.split("=", 1)
                    params.append((k.strip(), v.strip()))
                else:
                    params.append(("arg0", part))
        if name == "List" and params:
            return SemanticType("List", tuple(params))
        if name == "Dictionary":
            return SemanticType("Dictionary", tuple(params))
        return SemanticType(name, tuple(params))

    @property
    def is_list(self) -> bool:
        return self.name == "List"

    @property
    def element(self) -> Optional["SemanticType"]:
        if self.is_list and self.params:
            return SemanticType.parse(self.params[0][1] if self.params[0][0]
                                      else self.params[0][1])
        return None


@dataclass
class TypedValue:
    runtime_type: str
    semantic_type: SemanticType
    unit: str = ""
    role: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "runtime_type": self.runtime_type,
            "semantic_type": str(self.semantic_type),
            "unit": self.unit,
            "role": self.role,
            "schema_version": SCHEMA_VERSION,
        }


def family_of(st: SemanticType) -> str:
    if st.name in DURATION_FAMILY or st.name == "Duration":
        return "duration"
    if st.name in TEMP_FAMILY or st.name == "Temperature":
        return "temperature"
    if st.name in TEXT_FAMILY:
        return "text"
    if st.name == "Boolean":
        return "boolean"
    if st.name == "Money":
        return "money"
    if st.name in ("Percentage", "Ratio", "Probability"):
        return "rate"
    if st.name == "Count":
        return "count"
    if st.is_list:
        return "list"
    if st.name in NUMERIC_FAMILY:
        return "numeric"
    return st.name.lower()


def semantic_compatible(src: SemanticType, dst: SemanticType, *,
                        allow_generic: bool = False) -> Tuple[bool, str]:
    if src.name == dst.name:
        return True, "same_semantic_type"
    if dst.name == "GenericScalar" and src.name in NUMERIC_FAMILY:
        return True, "numeric_into_generic"
    if src.name == "GenericScalar":
        if allow_generic and dst.name in NUMERIC_FAMILY:
            return True, "generic_into_numeric_allowed"
        if dst.name == "GenericScalar":
            return True, "generic_to_generic"
        return False, f"forbidden_generic_to_{dst.name}"
    if family_of(src) == family_of(dst) and family_of(src) != src.name.lower():
        # same family but different concrete type needs conversion unless both numeric
        if family_of(src) == "numeric":
            return True, "numeric_family"
        if family_of(src) in ("duration", "temperature"):
            return False, f"needs_conversion_{src.name}_to_{dst.name}"
    if src.name == "Count" and dst.name in NUMERIC_FAMILY:
        return True, "count_as_numeric"
    if src.name in ("Percentage", "Ratio") and dst.name in ("Percentage", "Ratio",
                                                             "GenericScalar"):
        return True, "rate_family"
    if src.is_list and dst.is_list:
        se, de = src.element, dst.element
        if se and de:
            ok, reason = semantic_compatible(se, de, allow_generic=allow_generic)
            return ok, f"list_elem:{reason}"
        return True, "list_untyped"
    return False, f"incompatible_{src.name}_to_{dst.name}"


def unit_compatible(src_unit: str, dst_unit: str, *,
                    unit_behavior: str = "preserve") -> Tuple[bool, str]:
    su, du = (src_unit or "").strip(), (dst_unit or "").strip()
    if not du or du in ("*", "any", "neutral", ""):
        return True, "dst_unconstrained"
    if not su or su in ("*", "any", "neutral", "abstract"):
        return True, "src_unconstrained"
    if su == du:
        return True, "same_unit"
    if unit_behavior in ("convert", "rescale", "ignore"):
        return True, f"behavior_{unit_behavior}"
    # duration unit synonyms
    dur = {"s": "seconds", "sec": "seconds", "seconds": "seconds",
           "min": "minutes", "minutes": "minutes",
           "h": "hours", "hr": "hours", "hours": "hours",
           "d": "days", "day": "days", "days": "days"}
    if dur.get(su.lower(), su) == dur.get(du.lower(), du):
        return True, "duration_synonym"
    return False, f"unit_mismatch_{su}_vs_{du}"
