"""Selection v2 (Phase K) and family-safe splitting.

Pilot3 selected by marginal deficit against the profile, which let a bucket be
"on target" while being structurally monotone. Selection here is greedy over a
multi-objective score that mixes deficit terms (what the profile still wants),
novelty terms (what the pool has not shown yet) and concentration penalties
(what is already over-represented), under hard constraints that the report
always states as requested / achieved / deficit / reason.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..profile_v2 import CALL_BUCKETS, topology_signature

SCHEMA_VERSION = "ttdf.selection.v2"

DEFAULT_WEIGHTS = {
    "profile_deficit": 3.0,
    "conditional_structure_deficit": 2.5,
    "query_mode_deficit": 2.5,
    "capability_coverage_deficit": 1.5,
    "topology_novelty_in_bucket": 2.0,
    "tool_combination_novelty": 1.0,
    "schema_novelty": 0.8,
    "difficulty_bucket_deficit": 1.2,
    "program_family_concentration": -2.5,
    "query_template_concentration": -2.0,
    "primitive_chain_concentration": -1.5,
    "cell_deficit": 2.0,
}

DEFAULT_CONSTRAINTS = {
    "max_program_family_share": 0.02,
    "max_query_skeleton_share": 0.06,
    "min_goal_based_share": 0.35,
    "min_g_general_share": 0.40,
    "min_schema_compatible_distractor_share": 0.60,
    "min_topology_diversity_5call": 6,
    "min_topology_diversity_6plus": 10,
    "min_per_cell_count": 1,
    "call_count_tolerance": 0.05,
}


def _primitive_chain(rec: Dict[str, Any]) -> str:
    nodes = (rec.get("semantic_program") or {}).get("nodes") or []
    return ">".join(str(n.get("primitive_id")) for n in nodes)


def _topology_key(rec: Dict[str, Any]) -> str:
    return f"{rec.get('call_bucket')}::{rec.get('graph_template_id')}"


def _capabilities(rec: Dict[str, Any]) -> List[str]:
    return list(rec.get("capability_families") or [])


class _State:
    def __init__(self) -> None:
        self.n = 0
        self.call_bucket = Counter()
        self.query_mode = Counter()
        self.track = Counter()
        self.cell = Counter()
        self.family = Counter()
        self.skeleton = Counter()
        self.chain = Counter()
        self.tool_combo = Counter()
        self.surface_sig = Counter()
        self.capability = Counter()
        self.difficulty = Counter()
        self.topology_by_bucket: Dict[str, Set[str]] = defaultdict(set)
        self.topology_count = Counter()
        self.join_by_bucket = Counter()
        self.schema_compatible = 0

    def add(self, rec: Dict[str, Any]) -> None:
        self.n += 1
        self.call_bucket[rec["call_bucket"]] += 1
        self.query_mode[rec["classified_query_mode"]] += 1
        self.track[rec["surface_track"]] += 1
        self.cell[rec["generation_cell"]] += 1
        self.family[rec["program_family_id"]] += 1
        self.skeleton[rec["query_skeleton"]] += 1
        self.chain[_primitive_chain(rec)] += 1
        self.tool_combo[rec["tool_combination_hash"]] += 1
        self.surface_sig[rec["surface_signature"]] += 1
        self.difficulty[rec["difficulty_band"]] += 1
        for fam in _capabilities(rec):
            self.capability[fam] += 1
        self.topology_by_bucket[rec["call_bucket"]].add(rec["graph_template_id"])
        self.topology_count[_topology_key(rec)] += 1
        if (rec.get("structural_features") or {}).get("n_joins", 0):
            self.join_by_bucket[rec["call_bucket"]] += 1
        if rec.get("schema_compatible_distractor_count", 0) > 0:
            self.schema_compatible += 1


def _deficit(target: float, achieved: float) -> float:
    return max(target - achieved, 0.0)


def score_candidate(rec: Dict[str, Any], state: _State, targets: Dict[str, Any],
                    n_select: int, weights: Dict[str, float]) -> Dict[str, float]:
    n = max(state.n, 1)
    parts: Dict[str, float] = {}

    bucket = rec["call_bucket"]
    parts["profile_deficit"] = _deficit(
        targets["call_count_dist"].get(bucket, 0.0), state.call_bucket[bucket] / n)

    cond = targets.get("conditional_motif", {}).get(bucket, {})
    has_join = bool((rec.get("structural_features") or {}).get("n_joins", 0))
    motif_key = "join" if has_join else "no_join"
    seen_in_bucket = max(state.call_bucket[bucket], 1)
    achieved_cond = state.join_by_bucket[bucket] / seen_in_bucket if has_join else \
        (seen_in_bucket - state.join_by_bucket[bucket]) / seen_in_bucket
    parts["conditional_structure_deficit"] = _deficit(
        cond.get(motif_key, 0.0), achieved_cond)

    parts["query_mode_deficit"] = _deficit(
        targets["query_mode_dist"].get(rec["classified_query_mode"], 0.0),
        state.query_mode[rec["classified_query_mode"]] / n)

    caps = _capabilities(rec)
    unseen = sum(1 for c in caps if state.capability[c] == 0)
    parts["capability_coverage_deficit"] = unseen / max(len(caps), 1)

    parts["topology_novelty_in_bucket"] = (
        1.0 if rec["graph_template_id"] not in state.topology_by_bucket[bucket]
        else 1.0 / (1.0 + state.topology_count[_topology_key(rec)]))

    parts["tool_combination_novelty"] = (
        1.0 if state.tool_combo[rec["tool_combination_hash"]] == 0 else 0.0)
    parts["schema_novelty"] = (
        1.0 if state.surface_sig[rec["surface_signature"]] == 0 else 0.0)

    parts["difficulty_bucket_deficit"] = _deficit(
        targets["difficulty_dist"].get(rec["difficulty_band"], 0.0),
        state.difficulty[rec["difficulty_band"]] / n)

    parts["cell_deficit"] = _deficit(
        targets["cell_targets"].get(rec["generation_cell"], 0.0) / max(n_select, 1),
        state.cell[rec["generation_cell"]] / n)

    parts["program_family_concentration"] = state.family[rec["program_family_id"]] / n
    parts["query_template_concentration"] = state.skeleton[rec["query_skeleton"]] / n
    parts["primitive_chain_concentration"] = state.chain[_primitive_chain(rec)] / n

    total = sum(weights.get(k, 0.0) * v for k, v in parts.items())
    parts["_total"] = total
    return parts


def _violates_hard_constraints(rec: Dict[str, Any], state: _State,
                               n_select: int, cons: Dict[str, Any],
                               bucket_caps: Optional[Dict[str, int]] = None
                               ) -> Optional[str]:
    if (state.family[rec["program_family_id"]] + 1) / n_select > cons["max_program_family_share"]:
        return "max_program_family_share"
    if (state.skeleton[rec["query_skeleton"]] + 1) / n_select > cons["max_query_skeleton_share"]:
        return "max_query_skeleton_share"
    # The call-count distribution is a quota, not a preference: novelty terms
    # otherwise pull the whole selection into the 6+ bucket, which has by far
    # the most distinct topologies to reward.
    if bucket_caps:
        cap = bucket_caps.get(rec["call_bucket"])
        if cap is not None and state.call_bucket[rec["call_bucket"]] >= cap:
            return f"call_bucket_cap[{rec['call_bucket']}]"
    return None


def _bucket_caps(targets: Dict[str, Any], n_select: int,
                 cons: Dict[str, Any]) -> Dict[str, int]:
    tol = 1.0 + float(cons.get("call_count_tolerance", 0.05))
    dist = targets.get("call_count_dist") or {}
    caps = {b: int(math.ceil(share * n_select * tol))
            for b, share in dist.items() if share > 0}
    # a paired variant may be pulled in one over the cap; leave room for it
    return {b: max(c, 1) + 1 for b, c in caps.items()}


def select_records(candidates: Sequence[Dict[str, Any]], cells: Sequence[Any],
                   n_select: int, *, profile: Dict[str, Any], seed: int,
                   weights: Optional[Dict[str, float]] = None,
                   constraints: Optional[Dict[str, Any]] = None,
                   window_size: Optional[int] = None
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    cons = {**DEFAULT_CONSTRAINTS, **(constraints or {})}
    rng = random.Random(seed)

    targets = _build_targets(profile, cells, n_select)
    caps = _bucket_caps(targets, n_select, cons)
    # Sorting by task_id makes the selection independent of the order the
    # candidates arrive in; the seed only breaks exact score ties, so the same
    # pool and seed always yield the same set in the same order.
    pool = sorted((c for c in candidates
                   if c.get("validation", {}).get("V8", {}).get("passed", True)),
                  key=lambda r: r["task_id"])
    tie_break = {r["task_id"]: rng.random() for r in pool}

    state = _State()
    chosen: List[Dict[str, Any]] = []
    chosen_ids: Set[str] = set()
    rejections = Counter()
    by_id = {c["task_id"]: c for c in pool}

    # Scoring depends on the running state, so the argmax has to be recomputed
    # every pick. A deterministic random window keeps that O(n_select * window)
    # instead of O(n_select * pool) without changing the greedy character.
    window = max(int(window_size or 0), 1) if window_size else max(
        200, min(len(pool), 8 * (len(pool) // max(n_select, 1) + 1) * 25))
    window = min(window, len(pool)) or 1
    cursor = 0
    stalls = 0

    while len(chosen) < n_select:
        best = None
        best_key = (-math.inf, -math.inf)
        remaining = [r for r in pool if r["task_id"] not in chosen_ids]
        if not remaining:
            break
        if len(remaining) > window:
            cursor = (cursor + window) % len(remaining)
            view = remaining[cursor:cursor + window]
            if len(view) < window:
                view = view + remaining[:window - len(view)]
        else:
            view = remaining
        for rec in view:
            reason = _violates_hard_constraints(rec, state, n_select, cons, caps)
            if reason:
                rejections[reason] += 1
                continue
            s = score_candidate(rec, state, targets, n_select, weights)["_total"]
            s += _constraint_pull(rec, state, len(chosen), n_select, cons)
            s += _bucket_pull(rec, state, len(chosen), n_select, targets)
            key = (s, tie_break[rec["task_id"]])
            if key > best_key:
                best, best_key = rec, key
        if best is None:
            stalls += 1
            if len(view) == len(remaining) or stalls > 2 * (len(remaining) // window + 2):
                break
            continue
        stalls = 0
        chosen.append(best)
        chosen_ids.add(best["task_id"])
        state.add(best)
        # paired variants travel together so the pairing survives selection
        pair_id = best.get("paired_with")
        if pair_id and pair_id in by_id and pair_id not in chosen_ids \
                and len(chosen) < n_select:
            pair = by_id[pair_id]
            chosen.append(pair)
            chosen_ids.add(pair_id)
            state.add(pair)

    report = _selection_report(chosen, state, targets, cons, n_select,
                               len(candidates), rejections,
                               pool_by_bucket=Counter(r["call_bucket"] for r in pool))
    return chosen, report


def _constraint_pull(rec: Dict[str, Any], state: _State, n_done: int,
                     n_select: int, cons: Dict[str, Any]) -> float:
    """Extra pressure toward hard minimums that are still unmet."""
    n = max(n_done, 1)
    bonus = 0.0
    if rec["classified_query_mode"] == "GOAL_BASED_IMPLICIT":
        gap = cons["min_goal_based_share"] - state.query_mode["GOAL_BASED_IMPLICIT"] / n
        bonus += max(gap, 0.0) * 4.0
    if rec["surface_track"] == "G_GENERAL":
        gap = cons["min_g_general_share"] - state.track["G_GENERAL"] / n
        bonus += max(gap, 0.0) * 4.0
    if rec.get("schema_compatible_distractor_count", 0) > 0:
        gap = cons["min_schema_compatible_distractor_share"] - state.schema_compatible / n
        bonus += max(gap, 0.0) * 3.0
    bucket = rec["call_bucket"]
    if bucket in ("5", "6+"):
        need = (cons["min_topology_diversity_5call"] if bucket == "5"
                else cons["min_topology_diversity_6plus"])
        have = len(state.topology_by_bucket[bucket])
        if have < need and rec["graph_template_id"] not in state.topology_by_bucket[bucket]:
            bonus += 3.0
    if state.cell[rec["generation_cell"]] < cons["min_per_cell_count"]:
        bonus += 2.0
    return bonus


def _bucket_pull(rec: Dict[str, Any], state: _State, n_done: int,
                 n_select: int, targets: Dict[str, Any]) -> float:
    """Keep the call-count distribution on track while it is still fillable."""
    bucket = rec["call_bucket"]
    want = float((targets.get("call_count_dist") or {}).get(bucket, 0.0))
    if want <= 0:
        return 0.0
    have = state.call_bucket[bucket] / max(n_done, 1)
    return max(want - have, 0.0) * 8.0


def _build_targets(profile: Dict[str, Any], cells: Sequence[Any],
                   n_select: int) -> Dict[str, Any]:
    cell_targets = {}
    for c in cells:
        cid = getattr(c, "cell_id", None) or c.get("cell_id")
        w = getattr(c, "quota_weight", None)
        if w is None:
            w = c.get("quota_weight", 0.0)
        cell_targets[cid] = float(w) * n_select
    cond_motif: Dict[str, Dict[str, float]] = {}
    for bucket, d in (profile.get("topology_diversity_by_bucket") or {}).items():
        cond_motif[bucket] = {"join": d.get("join_rate", 0.0),
                              "no_join": max(1.0 - d.get("join_rate", 0.0), 0.0)}
    return {
        "call_count_dist": dict(profile.get("call_count_dist") or {}),
        "query_mode_dist": dict((profile.get("marginal") or {}).get("query_mode") or {}),
        "difficulty_dist": {"easy": 0.25, "medium": 0.45, "hard": 0.30},
        "cell_targets": cell_targets,
        "conditional_motif": cond_motif,
    }


def _selection_report(chosen: Sequence[Dict[str, Any]], state: _State,
                      targets: Dict[str, Any], cons: Dict[str, Any],
                      n_select: int, n_pool: int, rejections: Counter,
                      pool_by_bucket: Optional[Counter] = None) -> Dict[str, Any]:
    n = max(len(chosen), 1)
    rows: List[Dict[str, Any]] = []

    def add(name: str, requested: float, achieved: float, reason: str = "",
            tolerance: float = 0.0) -> None:
        rel = (max(requested - achieved, 0.0) / requested) if requested else 0.0
        met = achieved + 1e-9 >= requested or rel <= tolerance
        rows.append({
            "constraint": name,
            "requested_target": round(requested, 4),
            "achieved": round(achieved, 4),
            "absolute_deficit": round(max(requested - achieved, 0.0), 4),
            "relative_deficit": round(rel, 4),
            "tolerance": tolerance,
            "met": met,
            "reason_not_met": reason if not met else "",
        })

    pool_by_bucket = pool_by_bucket or Counter()
    for bucket in CALL_BUCKETS:
        want = targets["call_count_dist"].get(bucket, 0.0)
        have = state.call_bucket[bucket]
        available = pool_by_bucket.get(bucket, 0)
        reason = ("candidate pool exhausted for this bucket "
                  f"({available} validated candidates, {have} selectable under "
                  "the family and skeleton caps)"
                  if available <= have else
                  "displaced by higher-scoring candidates in other buckets")
        add(f"call_bucket_share[{bucket}]", want, have / n, reason,
            tolerance=float(cons.get("call_count_tolerance", 0.05)))
    add("min_goal_based_share", cons["min_goal_based_share"],
        state.query_mode["GOAL_BASED_IMPLICIT"] / n,
        "not enough candidates classified as goal-based after V7")
    add("min_g_general_share", cons["min_g_general_share"],
        state.track["G_GENERAL"] / n, "not enough G_GENERAL candidates")
    add("min_schema_compatible_distractor_share",
        cons["min_schema_compatible_distractor_share"],
        state.schema_compatible / n,
        "distractor pool lacked schema-compatible partners")
    add("min_topology_diversity_5call", cons["min_topology_diversity_5call"],
        len(state.topology_by_bucket["5"]),
        "generator produced too few distinct 5-call topologies")
    add("min_topology_diversity_6plus", cons["min_topology_diversity_6plus"],
        len(state.topology_by_bucket["6+"]),
        "generator produced too few distinct 6+ topologies")

    max_family = max(state.family.values()) / n if state.family else 0.0
    max_skeleton = max(state.skeleton.values()) / n if state.skeleton else 0.0
    empty_cells = [cid for cid, t in targets["cell_targets"].items()
                   if t >= 1 and state.cell[cid] == 0]

    return {
        "schema_version": SCHEMA_VERSION,
        "n_pool": n_pool,
        "n_requested": n_select,
        "n_selected": len(chosen),
        "constraint_rows": rows,
        "all_hard_constraints_met": all(r["met"] for r in rows),
        "max_program_family_share": round(max_family, 4),
        "max_query_skeleton_share": round(max_skeleton, 4),
        "n_cells_with_target_but_empty": len(empty_cells),
        "empty_cells": sorted(empty_cells)[:30],
        "hard_constraint_rejections": dict(rejections),
        "distributions": {
            "call_bucket": {k: round(v / n, 4) for k, v in sorted(state.call_bucket.items())},
            "query_mode": {k: round(v / n, 4) for k, v in sorted(state.query_mode.items())},
            "surface_track": {k: round(v / n, 4) for k, v in sorted(state.track.items())},
            "difficulty_band": {k: round(v / n, 4) for k, v in sorted(state.difficulty.items())},
            "capability_family": {k: v for k, v in sorted(state.capability.items())},
            "topologies_per_bucket": {k: len(v) for k, v in
                                      sorted(state.topology_by_bucket.items())},
        },
    }


# ── family-safe split ─────────────────────────────────────────────────────
# Identity keys name one program (or one paired rendering of it) and must never
# straddle two splits. A graph template is only a topology, shared by design
# across unrelated programs, so it is reported for information: requiring it to
# be split-exclusive would put every chain-shaped task in the same split.
SPLIT_IDENTITY_KEYS = ["semantic_program_id", "program_family_id"]
SPLIT_SHARED_KEYS = ["graph_template_id"]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def split_records(records: Sequence[Dict[str, Any]], sizes: Dict[str, int],
                  seed: int) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Whole program families are assigned to one split; never partially."""
    uf = _UnionFind()
    for rec in records:
        anchor = f"task::{rec['task_id']}"
        uf.find(anchor)
        for key in ("program_family_id", "semantic_program_id"):
            if rec.get(key):
                uf.union(anchor, f"{key}::{rec[key]}")
        if rec.get("paired_with"):
            uf.union(anchor, f"task::{rec['paired_with']}")

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[uf.find(f"task::{rec['task_id']}")].append(rec)

    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    rng = random.Random(seed)
    rng.shuffle(order)
    order.sort(key=lambda kv: -len(kv[1]))

    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in sizes}
    for _gid, members in order:
        target = min(sizes, key=lambda k: (len(out[k]) / max(sizes[k], 1),
                                           -sizes[k], k))
        if len(out[target]) + len(members) > sizes[target]:
            fits = [k for k in sizes if len(out[k]) + len(members) <= sizes[k]]
            if fits:
                target = min(fits, key=lambda k: len(out[k]) / max(sizes[k], 1))
            else:
                continue
        out[target].extend(members)
        for rec in members:
            rec["split"] = target

    leakage = _leakage_audit(out, SPLIT_IDENTITY_KEYS)
    leakage["paired_with"] = _paired_leakage(out)
    manifest = {
        "schema_version": "ttdf.split_manifest.v2",
        "sizes_requested": dict(sizes),
        "sizes_achieved": {k: len(v) for k, v in out.items()},
        "n_groups": len(groups),
        "largest_group": max((len(v) for v in groups.values()), default=0),
        "group_keys": SPLIT_IDENTITY_KEYS + ["paired_with"],
        "leakage": leakage,
        "leak_free": all(v == 0 for v in leakage.values()),
        "shared_by_design": _leakage_audit(out, SPLIT_SHARED_KEYS),
        "seed": seed,
    }
    return out, manifest


def _leakage_audit(splits: Dict[str, List[Dict[str, Any]]],
                   keys: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key in keys:
        seen: Dict[str, str] = {}
        collisions = 0
        for name, rows in splits.items():
            for rec in rows:
                val = rec.get(key)
                if not val:
                    continue
                if val in seen and seen[val] != name:
                    collisions += 1
                seen.setdefault(val, name)
        out[key] = collisions
    return out


def _paired_leakage(splits: Dict[str, List[Dict[str, Any]]]) -> int:
    """A rendering pair that lands in two splits would leak the oracle."""
    where = {rec["task_id"]: name for name, rows in splits.items() for rec in rows}
    return sum(1 for name, rows in splits.items() for rec in rows
               if rec.get("paired_with") in where
               and where[rec["paired_with"]] != name)
