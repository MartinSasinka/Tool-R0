"""Generation cells v2 (Phase L).

A cell is one addressable corner of the design space. Pilot3 cells were
(track x call bucket x motif); pilot4 cells add the structural pattern, the
query mode, the capability mix and the distractor profile. The full cross
product would be tens of thousands of cells, so the design is *sparse*: the
profile decides how much mass each call bucket gets, the derived topology
constraints decide which patterns that bucket must contain, and query-mode and
distractor quotas are laid over the top.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..capability import CAPABILITY_FAMILIES
from ..profile_v2 import CALL_BUCKETS
from ..util import short_hash
from .patterns import (MIN_CALLS, PATTERN_FAMILIES, PatternError, build_shape,
                       patterns_for, shape_signature)
from .query_render import QUERY_RENDERERS
from .surface_render import SURFACE_TRACKS

SCHEMA_VERSION = "ttdf.generation_cell.v2"

DISTRACTOR_PROFILES = ["mostly_easy", "balanced_hard", "hard_dense",
                       "schema_adversarial"]

# Capability mixes are named so a cell reads as an intent, not a random subset.
CAPABILITY_MIXES: Dict[str, List[str]] = {
    "numeric_core": ["arithmetic.binary", "arithmetic.unary", "arithmetic.reduction"],
    "statistical": ["statistics", "sequence.reduce", "arithmetic.reduction"],
    "sequence_heavy": ["sequence.map", "sequence.filter", "sequence.reduce",
                       "sequence.index", "sequence.combine"],
    "predicate_logic": ["comparison", "boolean.logic", "validation"],
    "text_and_format": ["string.parse", "string.transform", "string.format",
                        "conversion.text"],
    "unit_conversion": ["conversion.numeric", "date_time", "dictionary.lookup"],
    "geometry_bitwise": ["geometry", "bitwise", "arithmetic.binary"],
    "mixed_transitions": ["arithmetic.binary", "sequence.reduce", "rounding",
                          "classification.deterministic", "conversion.text"],
}

TARGET_SKILLS = {
    "2": ("continuation_after_observation", "wrong_second_tool_after_correct_prefix"),
    "3": ("variable_planning", "too_few_calls"),
    "4": ("dependency_tracking", "wrong_reference_target"),
    "5": ("long_horizon_planning", "premature_stop"),
    "6+": ("long_horizon_planning", "premature_stop"),
}

BUCKET_CALLS: Dict[str, List[int]] = {
    "2": [2], "3": [3], "4": [4], "5": [5], "6+": [6, 7, 8],
}


@dataclass
class Cell:
    cell_id: str
    mode: str
    track: str
    query_mode: str
    call_bucket: str
    pattern_family: str
    capability_mix: List[str]
    capability_mix_name: str
    target_failure_skill: str
    target_skill: str
    offered_tool_range: Tuple[int, int]
    distractor_profile: str
    reference_profile: str
    difficulty_band: str
    target_count: int = 0
    quota_weight: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["offered_tool_range"] = list(self.offered_tool_range)
        d["schema_version"] = SCHEMA_VERSION
        return d


def _band(call_bucket: str, query_mode: str, distractor_profile: str) -> str:
    score = {"2": 0, "3": 1, "4": 1, "5": 2, "6+": 3}[call_bucket]
    score += {"PROCEDURAL_EXPLICIT": 0, "SEMI_IMPLICIT": 1,
              "GOAL_BASED_IMPLICIT": 2}[query_mode]
    score += {"mostly_easy": 0, "balanced_hard": 1, "hard_dense": 1,
              "schema_adversarial": 2}[distractor_profile]
    return "easy" if score <= 1 else ("medium" if score <= 3 else "hard")


def _offered_range(call_bucket: str, distractor_profile: str) -> Tuple[int, int]:
    base = {"2": (8, 11), "3": (9, 13), "4": (10, 14), "5": (11, 16),
            "6+": (12, 18)}[call_bucket]
    if distractor_profile in ("hard_dense", "schema_adversarial"):
        return (base[0] + 2, base[1] + 2)
    return base


def _topology_group(pattern: str, n_calls: int) -> str:
    """Patterns that reduce to the same shape at this call count share a group.

    REPEATED_PRIMITIVE and TYPE_TRANSITION_CHAIN are both chains: counting them
    as separate patterns would silently give the chain topology three times the
    mass of every other shape in the bucket.
    """
    try:
        return shape_signature(build_shape(pattern, n_calls))
    except PatternError:
        return pattern


def _pattern_weights(patterns: Sequence[str], n_calls: int,
                     max_top1_share: float) -> Dict[str, float]:
    """Split a bucket's mass over patterns, capped per topology group."""
    groups: Dict[str, List[str]] = {}
    for p in patterns:
        groups.setdefault(_topology_group(p, n_calls), []).append(p)
    share = {sig: len(ps) / len(patterns) for sig, ps in groups.items()}
    if len(groups) > 1 and 0.0 < max_top1_share < 1.0:
        excess = 0.0
        for sig, s in list(share.items()):
            if s > max_top1_share:
                excess += s - max_top1_share
                share[sig] = max_top1_share
        room = sum(s for s in share.values() if s < max_top1_share)
        if excess > 0.0 and room > 0.0:
            for sig, s in list(share.items()):
                if s < max_top1_share:
                    share[sig] = s + excess * s / room
    weights: Dict[str, float] = {}
    for sig, ps in groups.items():
        for p in ps:
            weights[p] = share[sig] / len(ps)
    return weights


def build_cells(profile: Dict[str, Any], constraints: Dict[str, Any], *,
                query_mode_shares: Optional[Dict[str, float]] = None,
                track_shares: Optional[Dict[str, float]] = None,
                distractor_shares: Optional[Dict[str, float]] = None,
                call_bucket_boosts: Optional[Dict[str, float]] = None,
                max_cell_share: float = 0.035) -> List[Cell]:
    """Sparse, profile-driven cell design."""
    call_shares = dict(profile.get("call_count_dist") or {})
    for bucket, boost in (call_bucket_boosts or {}).items():
        call_shares[bucket] = call_shares.get(bucket, 0.0) + float(boost)
    donor = max(call_shares, key=lambda k: call_shares[k])
    total_boost = sum(float(v) for v in (call_bucket_boosts or {}).values())
    if total_boost:
        call_shares[donor] = max(call_shares[donor] - total_boost, 0.05)
    ssum = sum(call_shares.values()) or 1.0
    call_shares = {k: v / ssum for k, v in call_shares.items()}

    qshares = query_mode_shares or {"PROCEDURAL_EXPLICIT": 0.2,
                                    "SEMI_IMPLICIT": 0.35,
                                    "GOAL_BASED_IMPLICIT": 0.45}
    tshares = track_shares or {"A_NATIVE": 0.55, "G_GENERAL": 0.45}
    dshares = distractor_shares or {"mostly_easy": 0.15, "balanced_hard": 0.45,
                                    "hard_dense": 0.25, "schema_adversarial": 0.15}
    mix_names = list(CAPABILITY_MIXES)

    cells: List[Cell] = []
    # cell_id -> topology signature, used to hold the per-bucket topology shares
    # after the per-cell cap has been applied
    topo_of: Dict[str, str] = {}
    for bucket in CALL_BUCKETS:
        bshare = call_shares.get(bucket, 0.0)
        if bshare <= 0:
            continue
        max_calls = max(BUCKET_CALLS[bucket])
        cons = constraints.get(bucket, {})
        allowed = cons.get("allowed_patterns")
        patterns = ([p for p in allowed if MIN_CALLS[p] <= max_calls]
                    if allowed else patterns_for(max_calls))
        need = int(cons.get("minimum_pattern_families", 1))
        # rotate deterministically per bucket so LINEAR_CHAIN does not always
        # win the head of the list and dominate every bucket
        if patterns:
            offset = int(short_hash(bucket)[:6], 16) % len(patterns)
            patterns = patterns[offset:] + patterns[:offset]
        patterns = patterns[:max(need, min(len(patterns), need + 4))] or ["LINEAR_CHAIN"]
        skill, failure = TARGET_SKILLS[bucket]
        pweights = _pattern_weights(
            patterns, max_calls,
            float(cons.get("maximum_top1_topology_share", 1.0)))

        for pi, pattern in enumerate(patterns):
            for track, tshare in tshares.items():
                for qi, (qmode, qshare) in enumerate(sorted(qshares.items())):
                    dprofile = DISTRACTOR_PROFILES[(pi + qi) % len(DISTRACTOR_PROFILES)]
                    mix_name = mix_names[(pi + qi + len(track)) % len(mix_names)]
                    w = bshare * tshare * qshare * pweights[pattern]
                    if w <= 0:
                        continue
                    cid = (f"P4_{bucket.replace('+', 'p')}_{pattern[:14]}_"
                           f"{track[0]}_{qmode.split('_')[0][:4]}_{mix_name[:8]}_"
                           f"{short_hash([bucket, pattern, track, qmode, mix_name])[:6]}")
                    cells.append(Cell(
                        cell_id=cid, mode="PROFILE_SAFE", track=track,
                        query_mode=qmode, call_bucket=bucket,
                        pattern_family=pattern,
                        capability_mix=list(CAPABILITY_MIXES[mix_name]),
                        capability_mix_name=mix_name,
                        target_failure_skill=failure, target_skill=skill,
                        offered_tool_range=_offered_range(bucket, dprofile),
                        distractor_profile=dprofile,
                        reference_profile=("nestful_like" if track == "A_NATIVE"
                                           else "generalized"),
                        difficulty_band=_band(bucket, qmode, dprofile),
                        quota_weight=w))
                    topo_of[cid] = _topology_group(pattern, max_calls)

    # Normalise inside each (call bucket, topology) group, so a bucket served by
    # few cells (the 2-call bucket only admits LINEAR_CHAIN) still receives the
    # share the profile asked for instead of losing it to the per-cell cap, and
    # the per-cell cap cannot flatten the topology shares back to "one share per
    # pattern" -- which would hand the chain topology the mass of every chain-
    # shaped pattern in the bucket.
    for bucket in CALL_BUCKETS:
        group = [c for c in cells if c.call_bucket == bucket]
        if not group:
            continue
        share = call_shares.get(bucket, 0.0)
        gsum = sum(c.quota_weight for c in group) or 1.0
        by_topo: Dict[str, List[Cell]] = {}
        for c in group:
            by_topo.setdefault(topo_of[c.cell_id], []).append(c)
        for members in by_topo.values():
            tshare = sum(c.quota_weight for c in members) / gsum * share
            # the cap is an absolute mass, floored at the uniform mass so a group
            # spread over many cells can still spend the share it was given
            cap = max(max_cell_share, tshare / len(members))
            msum = sum(c.quota_weight for c in members) or 1.0
            for c in members:
                c.quota_weight = min(c.quota_weight / msum * tshare, cap)
            msum = sum(c.quota_weight for c in members) or 1.0
            for c in members:
                c.quota_weight = c.quota_weight / msum * tshare
    total = sum(c.quota_weight for c in cells) or 1.0
    for c in cells:
        c.quota_weight = c.quota_weight / total
    return cells


def assign_targets(cells: Sequence[Cell], n_total: int) -> None:
    for c in cells:
        c.target_count = max(1, int(round(c.quota_weight * n_total)))


def cells_summary(cells: Sequence[Cell]) -> Dict[str, Any]:
    from collections import Counter

    return {
        "schema_version": SCHEMA_VERSION,
        "n_cells": len(cells),
        "by_call_bucket": dict(Counter(c.call_bucket for c in cells)),
        "by_pattern_family": dict(Counter(c.pattern_family for c in cells)),
        "by_query_mode": dict(Counter(c.query_mode for c in cells)),
        "by_track": dict(Counter(c.track for c in cells)),
        "by_difficulty_band": dict(Counter(c.difficulty_band for c in cells)),
        "by_distractor_profile": dict(Counter(c.distractor_profile for c in cells)),
        "by_capability_mix": dict(Counter(c.capability_mix_name for c in cells)),
        "max_cell_quota_weight": round(max((c.quota_weight for c in cells), default=0.0), 5),
    }
