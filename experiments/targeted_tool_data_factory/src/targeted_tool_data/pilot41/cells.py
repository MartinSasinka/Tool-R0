"""Pilot4.1 cell redesign: dense CORE cells, sparse enrichment/challenge."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..pilot4.patterns import PATTERN_FAMILIES
from ..util import short_hash
from . import CELL_TIERS, QUERY_MODES
from .workflows import get_workflows

SCHEMA_VERSION = "ttdf.generation_cell.v41"

CALL_QUOTAS_TRAIN = {"2": 330, "3": 220, "4": 135, "5": 95, "6+": 220}
QUERY_SHARES = {
    "GRAPH_EXPLICIT": 0.04,
    "OPERATION_EXPLICIT_GRAPH_IMPLICIT": 0.10,
    "SEMI_IMPLICIT": 0.15,
    "GOAL_BASED_IMPLICIT": 0.20,
    "DOMAIN_GROUNDED_IMPLICIT": 0.51,
}
TRACK_SHARES = {"A_NATIVE": 0.60, "G_GENERAL": 0.40}
TIER_SHARES = {
    "CORE_PROFILE": 0.72,
    "STRUCTURAL_ENRICHMENT": 0.15,
    "CAPABILITY_ENRICHMENT": 0.10,
    "CHALLENGE": 0.03,
}
BUCKET_CALLS = {"2": [2], "3": [3], "4": [4], "5": [5], "6+": [6, 7, 8]}


@dataclass
class Cell41:
    cell_id: str
    tier: str
    mode: str
    track: str
    query_mode: str
    call_bucket: str
    pattern_family: str
    workflow_domain: str
    workflow_ids: List[str]
    distractor_profile: str
    difficulty_band: str
    target_count: int = 0
    min_support: int = 1
    quota_weight: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d


def _band(bucket: str, qmode: str, tier: str) -> str:
    if tier == "CHALLENGE" or bucket == "6+":
        return "hard"
    if qmode in ("GOAL_BASED_IMPLICIT", "DOMAIN_GROUNDED_IMPLICIT") and bucket >= "4":
        return "medium"
    if bucket in ("2", "3") and qmode == "GRAPH_EXPLICIT":
        return "easy"
    return "medium" if bucket in ("3", "4") else "hard"


def build_cells(*, train_n: int = 1000,
                n_core_cells: int = 60) -> List[Cell41]:
    """Sparse-but-dense design: few core cells with high support."""
    workflows = get_workflows()
    by_domain: Dict[str, List[str]] = {}
    for w in workflows:
        by_domain.setdefault(w.domain, []).append(w.workflow_id)
    domains = sorted(by_domain)

    # Core: cover call buckets × domains with rotating patterns/modes/tracks
    cells: List[Cell41] = []
    core_patterns = ["LINEAR_CHAIN", "FAN_IN_SINGLE", "REUSE_EARLY_OUTPUT",
                     "DIAMOND", "PARALLEL_THEN_MERGE", "LATE_REFERENCE",
                     "MULTI_JOIN", "FAN_OUT", "TWO_STAGE_AGGREGATION",
                     "NESTED_AGGREGATION", "REPEATED_PRIMITIVE",
                     "TYPE_TRANSITION_CHAIN", "FAN_IN_MULTIPLE",
                     "ALTERNATING_BRANCH_CHAIN", "MIXED_INDEPENDENT_DEPENDENT"]
    buckets = ["2", "3", "4", "5", "6+"]
    # distribute core cells across buckets proportional to quotas
    total_q = sum(CALL_QUOTAS_TRAIN.values())
    core_per_bucket = {
        b: max(4, int(round(n_core_cells * CALL_QUOTAS_TRAIN[b] / total_q)))
        for b in buckets
    }
    # adjust to exact n_core_cells
    while sum(core_per_bucket.values()) > n_core_cells:
        b = max(core_per_bucket, key=core_per_bucket.get)
        if core_per_bucket[b] > 4:
            core_per_bucket[b] -= 1
        else:
            break
    while sum(core_per_bucket.values()) < n_core_cells:
        b = min(core_per_bucket, key=core_per_bucket.get)
        core_per_bucket[b] += 1

    idx = 0
    for bucket in buckets:
        n_b = core_per_bucket[bucket]
        for i in range(n_b):
            domain = domains[idx % len(domains)]
            pattern = core_patterns[idx % len(core_patterns)]
            # skip illegal pattern/bucket pairs lightly
            from ..pilot4.patterns import MIN_CALLS
            max_c = max(BUCKET_CALLS[bucket])
            if MIN_CALLS.get(pattern, 2) > max_c:
                pattern = "LINEAR_CHAIN"
            qmode = list(QUERY_SHARES)[idx % len(QUERY_SHARES)]
            # bias core toward domain-grounded / goal
            if i % 3 == 0:
                qmode = "DOMAIN_GROUNDED_IMPLICIT"
            elif i % 3 == 1:
                qmode = "GOAL_BASED_IMPLICIT"
            track = "A_NATIVE" if i % 5 != 4 else "G_GENERAL"
            wids = by_domain[domain][:3]
            cid = (f"P41_CORE_{bucket.replace('+','p')}_{pattern[:10]}_"
                   f"{track[0]}_{qmode.split('_')[0][:4]}_{domain[:8]}_"
                   f"{short_hash([bucket, pattern, domain, qmode, i])[:6]}")
            cells.append(Cell41(
                cell_id=cid, tier="CORE_PROFILE", mode="PROFILE_SAFE",
                track=track, query_mode=qmode, call_bucket=bucket,
                pattern_family=pattern, workflow_domain=domain,
                workflow_ids=wids,
                distractor_profile=["balanced_hard", "hard_dense",
                                    "mostly_easy"][i % 3],
                difficulty_band=_band(bucket, qmode, "CORE_PROFILE"),
                min_support=8,
            ))
            idx += 1

    # Enrichment + challenge: fewer cells, lower min_support
    for tier, n_extra, min_sup in (
            ("STRUCTURAL_ENRICHMENT", 12, 4),
            ("CAPABILITY_ENRICHMENT", 10, 4),
            ("CHALLENGE", 6, 2)):
        for i in range(n_extra):
            bucket = "6+" if tier != "CAPABILITY_ENRICHMENT" else buckets[i % 5]
            pattern = core_patterns[(idx + i) % len(core_patterns)]
            domain = domains[(idx + i) % len(domains)]
            qmode = "DOMAIN_GROUNDED_IMPLICIT" if tier != "CHALLENGE" else "GOAL_BASED_IMPLICIT"
            track = "G_GENERAL" if i % 2 else "A_NATIVE"
            cid = (f"P41_{tier[:5]}_{bucket.replace('+','p')}_{pattern[:8]}_"
                   f"{short_hash([tier, bucket, pattern, i])[:6]}")
            cells.append(Cell41(
                cell_id=cid, tier=tier, mode="PROFILE_SAFE",
                track=track, query_mode=qmode, call_bucket=bucket,
                pattern_family=pattern, workflow_domain=domain,
                workflow_ids=by_domain[domain][:2],
                distractor_profile="schema_adversarial" if tier == "CHALLENGE"
                else "balanced_hard",
                difficulty_band=_band(bucket, qmode, tier),
                min_support=min_sup,
            ))

    # Assign target counts from tier × bucket mass
    for c in cells:
        bshare = CALL_QUOTAS_TRAIN[c.call_bucket] / train_n
        tshare = TIER_SHARES[c.tier]
        # equal split among cells of same tier+bucket
        peers = sum(1 for x in cells
                    if x.tier == c.tier and x.call_bucket == c.call_bucket)
        c.quota_weight = bshare * tshare / max(peers, 1)
    # renormalize and set targets
    total = sum(c.quota_weight for c in cells) or 1.0
    for c in cells:
        c.quota_weight /= total
        raw = int(round(c.quota_weight * train_n))
        c.target_count = max(c.min_support if c.tier == "CORE_PROFILE" else 1, raw)
    # scale if overshoot
    s = sum(c.target_count for c in cells)
    if s > train_n * 1.15:
        factor = train_n / s
        for c in cells:
            c.target_count = max(c.min_support if c.tier == "CORE_PROFILE" else 1,
                                 int(round(c.target_count * factor)))
    return cells


def cells_summary(cells: Sequence[Cell41]) -> Dict[str, Any]:
    from collections import Counter
    return {
        "schema_version": SCHEMA_VERSION,
        "n_cells": len(cells),
        "by_tier": dict(Counter(c.tier for c in cells)),
        "by_call_bucket": dict(Counter(c.call_bucket for c in cells)),
        "by_query_mode": dict(Counter(c.query_mode for c in cells)),
        "by_track": dict(Counter(c.track for c in cells)),
        "core_cells": sum(1 for c in cells if c.tier == "CORE_PROFILE"),
        "min_core_support_target": min(
            (c.target_count for c in cells if c.tier == "CORE_PROFILE"),
            default=0),
        "sum_targets": sum(c.target_count for c in cells),
    }
