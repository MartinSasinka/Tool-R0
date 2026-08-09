"""Strict semantic types used by workflow plans and instances."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple

SCHEMA_VERSION = "ttdf.pilot42.semantic_type.v1"
BASE_TYPES = {
    "GenericScalar", "Money", "Percentage", "Ratio", "Probability", "Count",
    "DurationSeconds", "DurationMinutes", "DurationHours", "DurationDays",
    "TemperatureCelsius", "TemperatureFahrenheit", "Length", "Area", "Volume",
    "Mass", "Speed", "Angle", "Index", "Boolean", "Category", "Label",
    "FreeText", "Filename", "FileExtension", "URL", "Domain", "Date", "DateTime",
}
SENSITIVE = {"Money", "DurationSeconds", "DurationMinutes", "DurationHours",
             "DurationDays", "TemperatureCelsius", "TemperatureFahrenheit"}
NUMERIC = {"GenericScalar", "Money", "Percentage", "Ratio", "Probability",
           "Count", "Length", "Area", "Volume", "Mass", "Speed", "Angle", "Index"}


@dataclass(frozen=True)
class SemanticType:
    name: str
    element_type: Optional["SemanticType"] = None

    @classmethod
    def parse(cls, value: str | "SemanticType") -> "SemanticType":
        if isinstance(value, cls):
            return value
        text = str(value or "GenericScalar").strip()
        m = re.fullmatch(r"List\[(.+)\]", text)
        if m:
            return cls("List", cls.parse(m.group(1)))
        if text not in BASE_TYPES:
            raise ValueError(f"unknown semantic type: {text}")
        return cls(text)

    def __str__(self) -> str:
        return f"List[{self.element_type}]" if self.name == "List" else self.name


@dataclass(frozen=True)
class TypedValue:
    value: Any
    semantic_type: SemanticType
    role: str
    unit: str = ""
    entity: str = ""

    def as_dict(self) -> dict:
        return {"value": self.value, "semantic_type": str(self.semantic_type),
                "role": self.role, "unit": self.unit, "entity": self.entity,
                "schema_version": SCHEMA_VERSION}


def semantic_compatible(src: SemanticType | str, dst: SemanticType | str,
                        *, explicit_conversion: bool = False) -> Tuple[bool, str]:
    src, dst = SemanticType.parse(src), SemanticType.parse(dst)
    if src == dst:
        return True, "exact"
    if src.name == "List" or dst.name == "List":
        if src.name == dst.name == "List" and src.element_type and dst.element_type:
            ok, why = semantic_compatible(src.element_type, dst.element_type,
                                          explicit_conversion=explicit_conversion)
            return ok, f"list:{why}"
        return False, "list_shape_mismatch"
    if src.name == "GenericScalar" and dst.name in SENSITIVE:
        return (explicit_conversion,
                "explicit_conversion" if explicit_conversion
                else f"forbidden_GenericScalar_to_{dst.name}")
    if src.name in SENSITIVE and dst.name == "GenericScalar":
        return True, "specific_numeric_to_generic"
    if src.name in NUMERIC and dst.name in NUMERIC:
        return True, "numeric_family"
    return False, f"incompatible_{src.name}_to_{dst.name}"


def unit_compatible(src: str, dst: str, *, converts: bool = False) -> Tuple[bool, str]:
    if not src or not dst or src == dst:
        return True, "same_or_unspecified"
    return (converts, "explicit_conversion" if converts else f"unit_mismatch_{src}_{dst}")
