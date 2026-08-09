"""Capability taxonomy over the primitive registry (Phase D).

The registry knows *how* a primitive computes; it did not know *what kind of
capability* it represents. Distractor hardness, cell design and coverage gaps
all need that second view, so this module adds an explicit, hand-auditable
semantic layer on top of the existing registry instead of forking it.

``semantic_neighbors`` and ``confusable_non_equivalents`` are computed by
differential testing on sampled inputs, never by name similarity.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import registry as reg

SCHEMA_VERSION = "ttdf.capability_registry.v1"

CAPABILITY_FAMILIES: List[str] = [
    "arithmetic.binary", "arithmetic.unary", "arithmetic.reduction",
    "comparison", "boolean.logic", "rounding", "statistics",
    "sequence.map", "sequence.filter", "sequence.reduce", "sequence.index",
    "sequence.combine", "dictionary.lookup", "dictionary.update",
    "string.parse", "string.transform", "string.format",
    "conversion.numeric", "conversion.text", "geometry", "date_time",
    "path_url", "bitwise", "validation", "classification.deterministic",
]

# Hand-maintained: a primitive's capability is a semantic judgement, not
# something to infer from its python category.
FAMILY_BY_SID: Dict[str, str] = {
    # arithmetic.binary
    "add": "arithmetic.binary", "subtract": "arithmetic.binary",
    "multiply": "arithmetic.binary", "divide": "arithmetic.binary",
    "power": "arithmetic.binary", "modulo": "arithmetic.binary",
    "floor_divide": "arithmetic.binary", "percent_of": "arithmetic.binary",
    "ratio_of": "arithmetic.binary", "increase_by_percent": "arithmetic.binary",
    "decrease_by_percent": "arithmetic.binary", "abs_difference": "arithmetic.binary",
    # arithmetic.unary
    "negate": "arithmetic.unary", "inverse": "arithmetic.unary",
    "square": "arithmetic.unary", "sqrt": "arithmetic.unary",
    "digit_sum": "arithmetic.unary", "ratio_to_percent": "arithmetic.unary",
    # arithmetic.reduction
    "sum_three": "arithmetic.reduction", "product_three": "arithmetic.reduction",
    # statistics
    "average_two": "statistics", "mean_three": "statistics",
    "range_three": "statistics", "mean_values": "statistics",
    "range_spread": "statistics", "median_values": "statistics",
    # comparison
    "max_two": "comparison", "min_two": "comparison", "is_greater": "comparison",
    # boolean.logic
    "logical_and": "boolean.logic", "logical_or": "boolean.logic",
    "logical_not": "boolean.logic", "is_divisible_by": "boolean.logic",
    # rounding
    "ceil_value": "rounding", "floor_value": "rounding",
    "round_to_int": "rounding", "round_places": "rounding",
    "round_direction": "rounding",
    # validation
    "is_within_range": "validation", "clamp": "validation",
    "is_non_negative": "validation",
    # sequence.*
    "scale_list": "sequence.map", "cumulative_sums": "sequence.map",
    "sort_values_desc": "sequence.map", "offset_list": "sequence.map",
    "filter_above": "sequence.filter", "top_k_values": "sequence.filter",
    "sum_values": "sequence.reduce", "count_values": "sequence.reduce",
    "max_values": "sequence.reduce", "min_values": "sequence.reduce",
    "index_of_max": "sequence.index", "value_at_position": "sequence.index",
    "append_value": "sequence.combine", "concat_lists": "sequence.combine",
    "join_values": "sequence.combine",
    # dictionary
    "lookup_unit_factor": "dictionary.lookup",
    "apply_rate_override": "dictionary.update",
    # string
    "parse_number": "string.parse", "file_extension": "string.parse",
    "domain_of_url": "path_url", "join_path_segments": "path_url",
    "text_length": "string.transform", "text_upper": "string.transform",
    "concat_texts": "string.format", "tag_value": "string.format",
    "format_with_unit": "string.format",
    "number_to_string": "conversion.text", "format_fixed": "conversion.text",
    # conversion.numeric
    "celsius_to_fahrenheit": "conversion.numeric",
    "fahrenheit_to_celsius": "conversion.numeric",
    "km_to_meters": "conversion.numeric", "meters_to_km": "conversion.numeric",
    "minutes_to_seconds": "conversion.numeric",
    "seconds_to_minutes": "conversion.numeric",
    # date_time
    "hours_to_minutes": "date_time", "days_to_hours": "date_time",
    "weeks_to_days": "date_time", "minutes_since_midnight": "date_time",
    # geometry / bitwise / classification
    "rectangle_area": "geometry", "rectangle_perimeter": "geometry",
    "circle_area": "geometry", "hypotenuse": "geometry",
    "bitwise_and": "bitwise", "bitwise_or": "bitwise",
    "bitwise_xor": "bitwise", "left_shift": "bitwise",
    "classify_threshold": "classification.deterministic",
    "grade_band": "classification.deterministic",
}

DIFFICULTY_TAGS: Dict[str, List[str]] = {
    "arithmetic.binary": ["order_sensitive"],
    "arithmetic.reduction": ["multi_operand"],
    "boolean.logic": ["type_transition", "non_numeric_output"],
    "comparison": ["output_equals_an_input"],
    "sequence.reduce": ["type_transition"],
    "sequence.map": ["list_output"],
    "sequence.filter": ["list_output", "length_varying"],
    "sequence.index": ["off_by_one_risk"],
    "sequence.combine": ["list_output"],
    "string.parse": ["type_transition"],
    "string.format": ["non_numeric_output"],
    "conversion.text": ["type_transition", "non_numeric_output"],
    "conversion.numeric": ["unit_sensitive"],
    "date_time": ["unit_sensitive"],
    "geometry": ["multi_operand"],
    "bitwise": ["integer_only"],
    "dictionary.lookup": ["enum_argument", "leaf_only"],
    "dictionary.update": ["enum_argument"],
    "classification.deterministic": ["non_numeric_output", "banded"],
    "validation": ["boundary_sensitive"],
    "rounding": ["precision_sensitive"],
    "statistics": ["multi_operand"],
    "path_url": ["string_structure"],
    "string.transform": ["non_numeric_output"],
    "arithmetic.unary": [],
}

# Order-sensitivity matters for distractor validity: swapping arguments of a
# non-commutative primitive changes the answer, so an otherwise identical
# surface is a legitimate hard distractor.
_COMMUTATIVE = {"add", "multiply", "max_two", "min_two", "average_two",
                "sum_three", "mean_three", "product_three", "abs_difference",
                "range_three", "hypotenuse", "logical_and", "logical_or",
                "bitwise_and", "bitwise_or", "bitwise_xor"}


@dataclass
class CapabilitySpec:
    primitive_id: str
    capability_family: str
    input_signature: List[str] = field(default_factory=list)
    output_type: str = ""
    arity: int = 0
    deterministic: bool = True
    supports_repeated_use: bool = True
    supports_fan_in: bool = True
    supports_fan_out: bool = True
    difficulty_tags: List[str] = field(default_factory=list)
    semantic_neighbors: List[str] = field(default_factory=list)
    confusable_non_equivalents: List[str] = field(default_factory=list)
    # extra factory-side metadata (not part of the required contract)
    answer_kind: str = "float"
    commutative: bool = False
    accepts_reference_args: bool = True
    n_surfaces_a: int = 0
    n_surfaces_g: int = 0


def family_of(sid: str) -> str:
    fam = FAMILY_BY_SID.get(sid)
    if fam:
        return fam
    prim = reg.all_primitives().get(sid)
    if prim is None:
        return "unknown"
    return {"arithmetic": "arithmetic.binary", "unary": "arithmetic.unary",
            "aggregate": "arithmetic.reduction", "predicate": "boolean.logic",
            "string": "string.format", "list": "sequence.reduce",
            "selection": "validation", "conversion": "conversion.numeric",
            }.get(prim.category, "unknown")


def _signature(prim: reg.Primitive) -> List[str]:
    return [t for (_n, t, _s) in prim.params]


def _accepts_reference_args(prim: reg.Primitive) -> bool:
    return any(not t.startswith("enum:") for (_n, t, _s) in prim.params)


# ── differential equivalence testing ──────────────────────────────────────
def _sample_inputs(prim: reg.Primitive, n: int, seed: int) -> List[List[Any]]:
    rng = random.Random(f"cap:{prim.sid}:{seed}")
    out = []
    for _ in range(n):
        try:
            out.append(list(prim.sampler(rng)))
        except Exception:  # noqa: BLE001 - a broken sampler is reported, not fatal
            break
    return out


def _apply(prim: reg.Primitive, args: Sequence[Any]) -> Any:
    kwargs = {name: val for (name, _t, _s), val in zip(prim.params, args)}
    return prim.fn(**kwargs)


def signatures_compatible(a: reg.Primitive, b: reg.Primitive) -> bool:
    return _signature(a) == _signature(b) and a.out_type == b.out_type


def behaviourally_equivalent(a: reg.Primitive, b: reg.Primitive, *,
                             n_samples: int = 24, seed: int = 7) -> bool:
    """True when b returns a's output on every sampled input of a's domain.

    Used to reject distractors that are only *named* differently. Lexical
    similarity is never consulted.
    """
    if not signatures_compatible(a, b):
        return False
    checked = 0
    for args in _sample_inputs(a, n_samples, seed):
        try:
            va = _apply(a, args)
        except Exception:  # noqa: BLE001
            continue
        try:
            vb = _apply(b, args)
        except Exception:  # noqa: BLE001
            return False        # b cannot even accept the domain -> not equivalent
        checked += 1
        if not _values_equal(va, vb):
            return False
    return checked > 0


def _values_equal(x: Any, y: Any) -> bool:
    if isinstance(x, bool) or isinstance(y, bool):
        return bool(x) == bool(y)
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(float(x) - float(y)) <= 1e-9
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(_values_equal(a, b) for a, b in zip(x, y))
    return x == y


def build_registry(*, n_samples: int = 24, seed: int = 7) -> Dict[str, Dict[str, Any]]:
    """Capability spec for every registered primitive."""
    prims = reg.all_primitives()
    specs: Dict[str, CapabilitySpec] = {}
    for sid, p in sorted(prims.items()):
        fam = family_of(sid)
        specs[sid] = CapabilitySpec(
            primitive_id=sid,
            capability_family=fam,
            input_signature=_signature(p),
            output_type=p.out_type,
            arity=len(p.params),
            deterministic=True,
            supports_repeated_use=True,
            supports_fan_in=_accepts_reference_args(p) and len(p.params) >= 2,
            supports_fan_out=True,
            difficulty_tags=list(DIFFICULTY_TAGS.get(fam, [])),
            answer_kind=p.answer_kind,
            commutative=sid in _COMMUTATIVE,
            accepts_reference_args=_accepts_reference_args(p),
            n_surfaces_a=len(p.surfaces_a),
            n_surfaces_g=len(p.surfaces_g),
        )
    for sid, spec in specs.items():
        a = prims[sid]
        neighbors, confusable = [], []
        for other, ospec in specs.items():
            if other == sid:
                continue
            b = prims[other]
            if not signatures_compatible(a, b):
                continue
            if behaviourally_equivalent(a, b, n_samples=n_samples, seed=seed):
                continue    # an alias, not a distractor
            if ospec.capability_family == spec.capability_family:
                neighbors.append(other)
            confusable.append(other)
        spec.semantic_neighbors = sorted(neighbors)
        spec.confusable_non_equivalents = sorted(confusable)
    return {sid: asdict(s) for sid, s in specs.items()}


def coverage(registry: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    registry = registry or build_registry()
    by_family: Dict[str, List[str]] = {f: [] for f in CAPABILITY_FAMILIES}
    unknown: List[str] = []
    for sid, spec in registry.items():
        fam = spec["capability_family"]
        by_family.setdefault(fam, []).append(sid)
        if fam not in CAPABILITY_FAMILIES:
            unknown.append(sid)
    empty = [f for f in CAPABILITY_FAMILIES if not by_family.get(f)]
    return {
        "n_primitives": len(registry),
        "n_families_declared": len(CAPABILITY_FAMILIES),
        "n_families_populated": sum(1 for f in CAPABILITY_FAMILIES if by_family.get(f)),
        "empty_families": empty,
        "primitives_outside_taxonomy": sorted(unknown),
        "by_family": {f: sorted(v) for f, v in sorted(by_family.items())},
        "family_sizes": {f: len(v) for f, v in sorted(by_family.items())},
    }


def validate(registry: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
    registry = registry or build_registry()
    errs: List[str] = []
    for sid, spec in registry.items():
        if spec["capability_family"] == "unknown":
            errs.append(f"{sid}: no capability family assigned")
        if spec["arity"] != len(spec["input_signature"]):
            errs.append(f"{sid}: arity/signature mismatch")
        if not spec["output_type"]:
            errs.append(f"{sid}: missing output type")
        if sid in spec["semantic_neighbors"]:
            errs.append(f"{sid}: listed as its own neighbor")
        for other in spec["semantic_neighbors"]:
            if other not in registry:
                errs.append(f"{sid}: unknown neighbor {other}")
    return errs


def gap_rows(registry: Optional[Dict[str, Dict[str, Any]]] = None,
             observed_families: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """Per-family supply vs (optional) observed demand."""
    registry = registry or build_registry()
    cov = coverage(registry)
    rows = []
    for fam in CAPABILITY_FAMILIES:
        members = cov["by_family"].get(fam, [])
        rows.append({
            "capability_family": fam,
            "n_primitives": len(members),
            "n_surfaces": sum(registry[s]["n_surfaces_a"] + registry[s]["n_surfaces_g"]
                              for s in members),
            "observed_uses": (observed_families or {}).get(fam, ""),
            "status": "EMPTY" if not members else ("THIN" if len(members) < 3 else "OK"),
            "example_primitives": ", ".join(members[:4]),
        })
    return rows


def markdown_report(registry: Dict[str, Dict[str, Any]], cov: Dict[str, Any],
                    errs: Sequence[str]) -> str:
    lines = [
        "# CAPABILITY_REGISTRY_AUDIT", "",
        f"- primitives: **{cov['n_primitives']}**",
        f"- declared capability families: **{cov['n_families_declared']}**",
        f"- populated families: **{cov['n_families_populated']}**",
        f"- validation errors: **{len(errs)}**", "",
        "## Family coverage", "",
        "| family | primitives | members |", "|---|---:|---|",
    ]
    for fam in CAPABILITY_FAMILIES:
        members = cov["by_family"].get(fam, [])
        lines.append(f"| `{fam}` | {len(members)} | {', '.join(members) or '—'} |")
    if cov["empty_families"]:
        lines += ["", "## Still empty", ""]
        lines += [f"- `{f}`" for f in cov["empty_families"]]
    if errs:
        lines += ["", "## Validation errors", ""] + [f"- {e}" for e in errs]
    lines += [
        "", "## Method notes", "",
        "- `semantic_neighbors` = same capability family, schema-compatible, and",
        "  proven non-equivalent by differential testing on sampled inputs.",
        "- `confusable_non_equivalents` = schema-compatible and non-equivalent,",
        "  regardless of family. Name similarity is never used.",
        "",
    ]
    return "\n".join(lines) + "\n"
