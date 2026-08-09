"""Semantic types for Pilot4.3 op composition.

Runtime types (number/integer/string/array/boolean/object) are far too weak to
decide whether an edge is meaningful: ``duration_hours -> length_km`` type-checks
and is nonsense. Every op therefore declares a *semantic* type per parameter and
for its output, and :func:`compatible` is the only gate the DAG builder uses.

Design rules:

* physical quantities never silently convert -- a ``DurationHours`` value may
  only reach a parameter that accepts hours (or an explicit converter),
* ``Percentage`` and ``Ratio`` are distinct: 0.2 and 20 are not the same value,
* ``GenericScalar`` is the deliberate escape hatch for pure arithmetic and
  accepts any dimensionless-or-not numeric quantity, but a *physical* parameter
  never accepts ``GenericScalar`` because that is how unit nonsense enters,
* collections carry their element type (``NumberList`` vs ``TextList``).
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Set

# ── numeric, dimensionless ───────────────────────────────────────────────
GENERIC = "GenericScalar"
COUNT = "Count"
INDEX = "Index"
RATIO = "Ratio"
PERCENTAGE = "Percentage"
SCORE = "Score"
MONEY = "Money"
QUANTITY = "Quantity"

# ── numeric, physical ────────────────────────────────────────────────────
DUR_S = "DurationSeconds"
DUR_MIN = "DurationMinutes"
DUR_H = "DurationHours"
DUR_D = "DurationDays"
LEN_M = "LengthM"
LEN_KM = "LengthKm"
MASS_KG = "MassKg"
MASS_G = "MassG"
VOL_L = "VolumeL"
VOL_ML = "VolumeMl"
TEMP_C = "TemperatureC"
TEMP_F = "TemperatureF"
AREA = "Area"
BYTES = "Bytes"

# ── text-ish ─────────────────────────────────────────────────────────────
TEXT = "Text"
NUMERIC_TEXT = "NumericText"
PATH = "Path"
URL = "Url"
DATE = "DateISO"
IDENTIFIER = "Identifier"
CATEGORY = "Category"
UNIT_NAME = "UnitName"

# ── collections / structures ─────────────────────────────────────────────
NUMBER_LIST = "NumberList"
TEXT_LIST = "TextList"
MAPPING = "Mapping"
RECORD = "Record"
RECORD_LIST = "RecordList"

FLAG = "Flag"

NUMERIC: FrozenSet[str] = frozenset({
    GENERIC, COUNT, INDEX, RATIO, PERCENTAGE, SCORE, MONEY, QUANTITY,
    DUR_S, DUR_MIN, DUR_H, DUR_D, LEN_M, LEN_KM, MASS_KG, MASS_G,
    VOL_L, VOL_ML, TEMP_C, TEMP_F, AREA, BYTES,
})
PHYSICAL: FrozenSet[str] = frozenset({
    DUR_S, DUR_MIN, DUR_H, DUR_D, LEN_M, LEN_KM, MASS_KG, MASS_G,
    VOL_L, VOL_ML, TEMP_C, TEMP_F, AREA, BYTES,
})
DIMENSIONLESS: FrozenSet[str] = NUMERIC - PHYSICAL
TEXTUAL: FrozenSet[str] = frozenset({TEXT, NUMERIC_TEXT, PATH, URL, DATE,
                                     IDENTIFIER, CATEGORY, UNIT_NAME})
COLLECTIONS: FrozenSet[str] = frozenset({NUMBER_LIST, TEXT_LIST, MAPPING,
                                         RECORD, RECORD_LIST})
ALL: FrozenSet[str] = NUMERIC | TEXTUAL | COLLECTIONS | frozenset({FLAG})

#: A parameter of the key type additionally accepts these producer types.
#: Kept explicit (never derived) so an audit can read the whole rule set.
_EXTRA_ACCEPTS: Dict[str, Set[str]] = {
    # pure arithmetic accepts any numeric quantity
    GENERIC: set(NUMERIC),
    # a count-like parameter takes counts and plain integers
    COUNT: {GENERIC, COUNT, INDEX, QUANTITY},
    INDEX: {GENERIC, COUNT, INDEX},
    QUANTITY: {GENERIC, COUNT, QUANTITY, MONEY, SCORE},
    SCORE: {GENERIC, SCORE, COUNT, QUANTITY},
    MONEY: {GENERIC, MONEY, QUANTITY},
    # a rate parameter takes ratios/percentages but never a raw amount
    RATIO: {RATIO},
    PERCENTAGE: {PERCENTAGE},
    # dates/text
    TEXT: {TEXT, NUMERIC_TEXT, IDENTIFIER, CATEGORY, UNIT_NAME, PATH, URL, DATE},
    IDENTIFIER: {IDENTIFIER, TEXT},
    CATEGORY: {CATEGORY, TEXT},
    NUMERIC_TEXT: {NUMERIC_TEXT},
    PATH: {PATH},
    URL: {URL},
    DATE: {DATE},
    UNIT_NAME: {UNIT_NAME},
    FLAG: {FLAG},
    NUMBER_LIST: {NUMBER_LIST},
    TEXT_LIST: {TEXT_LIST},
    MAPPING: {MAPPING},
    RECORD: {RECORD},
    RECORD_LIST: {RECORD_LIST},
}


def compatible(param_type: str, value_type: str) -> bool:
    """True when a value of ``value_type`` may feed a ``param_type`` parameter."""
    if param_type not in ALL or value_type not in ALL:
        raise ValueError(f"unknown semantic type {param_type!r}/{value_type!r}")
    if param_type == value_type:
        return True
    if param_type in PHYSICAL:
        # physical parameters accept only their exact quantity: this is the rule
        # that keeps "84 hours of rainfall in kilometres" out of the dataset
        return False
    return value_type in _EXTRA_ACCEPTS.get(param_type, set())


def runtime_of(sem: str) -> str:
    """Runtime JSON type advertised in the tool schema for a semantic type."""
    if sem in (COUNT, INDEX, BYTES):
        return "integer"
    if sem in NUMERIC:
        return "number"
    if sem == FLAG:
        return "boolean"
    if sem in (NUMBER_LIST, TEXT_LIST):
        return "array"
    if sem in (MAPPING, RECORD):
        return "object"
    if sem == RECORD_LIST:
        return "array"
    return "string"


def answer_type_of(sem: str) -> str:
    """Answer-type bucket used by the distribution gates."""
    if sem == FLAG:
        return "boolean"
    if sem in (COUNT, INDEX, BYTES):
        return "integer"
    if sem == CATEGORY:
        return "category"
    if sem in NUMERIC:
        return "float"
    if sem in (NUMBER_LIST, TEXT_LIST, RECORD_LIST):
        return "list"
    if sem in (MAPPING, RECORD):
        return "object"
    return "string"


def value_kind(value: object) -> str:
    """Observed kind of a concrete value (bool checked before int)."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return "unknown"


def matches_value(sem: str, value: object) -> bool:
    """Runtime check that an observed value is admissible for a semantic type."""
    kind = value_kind(value)
    expected = runtime_of(sem)
    if expected == "number":
        return kind in ("integer", "float")
    if expected == "integer":
        return kind == "integer"
    if expected == "boolean":
        return kind == "boolean"
    if expected == "array":
        return kind == "list"
    if expected == "object":
        return kind == "object"
    return kind == "string"
