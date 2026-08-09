"""Schema-semantic distractors (Phase I).

Pilot3 ranked distractor hardness by name and description similarity, which
rewards a model for reading strings rather than schemas. A distractor is hard
here only if it is *actually substitutable*: same arity, compatible input
types, compatible output type — and provably wrong for the step it targets.

Levels, easiest first:

    EASY_TYPE_INCOMPATIBLE          cannot be called in the gold slot at all
    MEDIUM_SAME_OUTPUT_TYPE         plausible output, incompatible inputs
    HARD_SAME_ARITY_AND_TYPES       fully substitutable schema
    HARD_SAME_CAPABILITY_FAMILY     substitutable and semantically adjacent
    HARD_SEMANTIC_NEIGHBOR          registry-declared neighbour
    HARD_REPEATED_SURFACE_AMBIGUITY substitutable and surface-confusable
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .. import registry as reg
from ..capability import (behaviourally_equivalent, build_registry, family_of,
                          signatures_compatible)
from ..schemas import ToolSpec
from .surface_render import TRACK_CODE, render_tool

DISTRACTOR_LEVELS = [
    "EASY_TYPE_INCOMPATIBLE", "MEDIUM_SAME_OUTPUT_TYPE",
    "HARD_SAME_ARITY_AND_TYPES", "HARD_SAME_CAPABILITY_FAMILY",
    "HARD_SEMANTIC_NEIGHBOR", "HARD_REPEATED_SURFACE_AMBIGUITY",
]
HARD_LEVELS = {"HARD_SAME_ARITY_AND_TYPES", "HARD_SAME_CAPABILITY_FAMILY",
               "HARD_SEMANTIC_NEIGHBOR", "HARD_REPEATED_SURFACE_AMBIGUITY"}

_CAP_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def capability_registry() -> Dict[str, Dict[str, Any]]:
    global _CAP_CACHE
    if _CAP_CACHE is None:
        _CAP_CACHE = build_registry()
    return _CAP_CACHE


def _signature(sid: str) -> Tuple[Tuple[str, ...], str]:
    p = reg.get(sid)
    return tuple(t for (_n, t, _s) in p.params), p.out_type


def _share_token(a: str, b: str) -> bool:
    ta = {t for t in a.lower().split("_") if len(t) > 3}
    tb = {t for t in b.lower().split("_") if len(t) > 3}
    return bool(ta & tb)


def classify_pair(gold_sid: str, cand_sid: str, gold_name: str = "",
                  cand_name: str = "") -> Optional[str]:
    """Hardness level of ``cand`` as a distractor for ``gold``, or None."""
    if gold_sid == cand_sid:
        return None
    g_sig, g_out = _signature(gold_sid)
    c_sig, c_out = _signature(cand_sid)
    same_types = g_sig == c_sig
    same_out = g_out == c_out
    if same_types and same_out:
        cap = capability_registry()
        if cand_sid in cap.get(gold_sid, {}).get("semantic_neighbors", []):
            if gold_name and cand_name and _share_token(gold_name, cand_name):
                return "HARD_REPEATED_SURFACE_AMBIGUITY"
            return "HARD_SEMANTIC_NEIGHBOR"
        if family_of(cand_sid) == family_of(gold_sid):
            return "HARD_SAME_CAPABILITY_FAMILY"
        return "HARD_SAME_ARITY_AND_TYPES"
    if same_out:
        return "MEDIUM_SAME_OUTPUT_TYPE"
    return "EASY_TYPE_INCOMPATIBLE"


def _reason_incorrect(gold_sid: str, cand_sid: str, level: str) -> str:
    gf, cf = family_of(gold_sid), family_of(cand_sid)
    if level == "EASY_TYPE_INCOMPATIBLE":
        return (f"schema mismatch: {cand_sid} takes {_signature(cand_sid)[0]} -> "
                f"{_signature(cand_sid)[1]}, the step needs "
                f"{_signature(gold_sid)[0]} -> {_signature(gold_sid)[1]}")
    if level == "MEDIUM_SAME_OUTPUT_TYPE":
        return (f"output type matches but the inputs do not; {cand_sid} cannot "
                f"consume this step's arguments")
    if gf == cf:
        return (f"same capability family ({gf}) and same schema, but a different "
                f"operation: substituting {cand_sid} changes the oracle value")
    return (f"schema-compatible but from capability family {cf} instead of {gf}; "
            f"substituting it changes the oracle value")


def _verified_non_equivalent(gold_sid: str, cand_sid: str) -> bool:
    if not signatures_compatible(reg.get(gold_sid), reg.get(cand_sid)):
        return True
    return not behaviourally_equivalent(reg.get(gold_sid), reg.get(cand_sid))


def build_offered_set(spec: Any, gold_tools: Dict[str, ToolSpec], track: str,
                      rng: random.Random, offered_count: int,
                      distractor_profile: str = "balanced_hard"
                      ) -> Dict[str, Any]:
    """Assemble the offered tool menu with typed, verified distractors."""
    code = TRACK_CODE.get(track, "A")
    gold_sids = list(gold_tools.keys())
    gold_names = {t.name for t in gold_tools.values()}
    wanted_hard = _hard_target(distractor_profile, offered_count, len(gold_sids))

    pool: List[Tuple[str, str, reg.Surface, str]] = []   # (sid, level, surf, gold)
    prims = reg.all_primitives()
    for gold_sid in gold_sids:
        gname = gold_tools[gold_sid].name
        for cand_sid, cand in prims.items():
            if cand_sid in gold_tools:
                continue
            surfaces = cand.surfaces(code) or cand.surfaces("A" if code == "G" else "G")
            for surf in surfaces:
                if surf.name in gold_names:
                    continue
                level = classify_pair(gold_sid, cand_sid, gname, surf.name)
                if level is None:
                    continue
                if level in HARD_LEVELS and not _verified_non_equivalent(gold_sid, cand_sid):
                    continue        # hidden alias -> never a valid distractor
                pool.append((cand_sid, level, surf, gold_sid))

    rng.shuffle(pool)
    order = {lvl: i for i, lvl in enumerate(DISTRACTOR_LEVELS)}
    hard_pool = [p for p in pool if p[1] in HARD_LEVELS]
    soft_pool = [p for p in pool if p[1] not in HARD_LEVELS]
    hard_pool.sort(key=lambda p: -order[p[1]])

    chosen: List[Tuple[str, str, reg.Surface, str]] = []
    used_names: Set[str] = set(gold_names)
    used_sids: Set[str] = set(gold_sids)
    n_slots = max(offered_count - len(gold_sids), 0)

    for bucket, limit in ((hard_pool, min(wanted_hard, n_slots)),
                          (soft_pool, n_slots)):
        for cand_sid, level, surf, gold_sid in bucket:
            if len(chosen) >= limit:
                break
            if surf.name in used_names or cand_sid in used_sids:
                continue
            chosen.append((cand_sid, level, surf, gold_sid))
            used_names.add(surf.name)
            used_sids.add(cand_sid)

    records: List[Dict[str, Any]] = []
    specs: List[ToolSpec] = list(gold_tools.values())
    for cand_sid, level, surf, gold_sid in chosen:
        tool = render_tool(cand_sid, track, surf)
        tool.is_distractor = True
        tool.distractor_type = level
        g_sig, g_out = _signature(gold_sid)
        c_sig, c_out = _signature(cand_sid)
        records.append({
            "distractor_tool": tool.name,
            "distractor_primitive": cand_sid,
            "target_gold_tool": gold_tools[gold_sid].name,
            "target_gold_primitive": gold_sid,
            "difficulty_level": level,
            "arity_compatible": len(g_sig) == len(c_sig),
            "input_types_compatible": g_sig == c_sig,
            "output_type_compatible": g_out == c_out,
            "same_capability_family": family_of(gold_sid) == family_of(cand_sid),
            "semantic_neighbor": cand_sid in capability_registry().get(
                gold_sid, {}).get("semantic_neighbors", []),
            "verified_non_equivalent": _verified_non_equivalent(gold_sid, cand_sid),
            "reason_incorrect": _reason_incorrect(gold_sid, cand_sid, level),
        })
        specs.append(tool)

    rng.shuffle(specs)
    positions = [i for i, t in enumerate(specs) if not t.is_distractor]
    hard_n = sum(1 for r in records if r["difficulty_level"] in HARD_LEVELS)
    same_family_n = sum(1 for r in records if r["same_capability_family"])
    return {
        "offered_tools": specs,
        "gold_tool_positions": positions,
        "distractor_records": records,
        "hard_distractor_count": hard_n,
        "easy_distractor_count": len(records) - hard_n,
        "same_family_distractor_count": same_family_n,
        "schema_compatible_distractor_count": sum(
            1 for r in records if r["input_types_compatible"] and r["output_type_compatible"]),
        "distractor_levels": sorted({r["difficulty_level"] for r in records}),
        "requested_hard": wanted_hard,
    }


_PROFILE_HARD_SHARE = {
    "easy_only": 0.0, "mostly_easy": 0.25, "balanced_hard": 0.55,
    "hard_dense": 0.8, "schema_adversarial": 1.0,
}


def _hard_target(profile: str, offered_count: int, n_gold: int) -> int:
    share = _PROFILE_HARD_SHARE.get(profile, 0.55)
    return int(round(max(offered_count - n_gold, 0) * share))
