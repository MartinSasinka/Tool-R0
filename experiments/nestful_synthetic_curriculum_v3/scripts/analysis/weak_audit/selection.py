"""Cohort selection for weak-model audit."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

from weak_audit.constants import COHORT_LIMITS, COHORT_PRIORITY, SEED


def gold_bucket(n: int) -> str:
    return str(n) if n <= 5 else "6+"


def stratified_sample(
    candidates: List[str],
    meta: Dict[str, dict],
    n: int,
    *,
    seed: int,
) -> List[str]:
    if len(candidates) <= n:
        return list(candidates)
    rng = random.Random(seed)
    buckets: Dict[tuple, List[str]] = defaultdict(list)
    for tid in candidates:
        m = meta[tid]
        key = (
            m.get("gold_call_bucket", "?"),
            m.get("motif", "?"),
            str(m.get("first_divergence_turn")),
            (m.get("c0_failure") or "?")[:20],
            bool(m.get("reward_mismatch")),
        )
        buckets[key].append(tid)
    picked: List[str] = []
    keys = list(buckets.keys())
    rng.shuffle(keys)
    while len(picked) < n:
        progressed = False
        for key in keys:
            if buckets[key]:
                picked.append(buckets[key].pop(rng.randrange(len(buckets[key]))))
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
    if len(picked) < n:
        rest = [x for x in candidates if x not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return picked[:n]


def build_pools(meta: Dict[str, dict], r0_class: Dict[str, dict]) -> Dict[str, List[str]]:
    pools: Dict[str, List[str]] = {c: [] for c in COHORT_PRIORITY}
    for tid, m in meta.items():
        w0, w1, w2 = m["w0"], m["w1"], m["w2"]
        if w0 and not w2:
            pools["c0_win_e2_loss"].append(tid)
        if not w0 and w2:
            pools["c0_loss_e2_win"].append(tid)
        if w0 and w1 and w2:
            pools["stable_win_control"].append(tid)
        if not w0 and not w1 and not w2:
            pools["stable_loss_control"].append(tid)
        if w0 and r0_class.get(tid, {}).get("class_C0") == "too_few_calls":
            pools["official_win_reward_too_few"].append(tid)
        if m.get("e2_executable_wrong") and not (w0 and not w2):
            pools["e2_executable_wrong_other"].append(tid)
    return pools


def select_tasks(
    meta: Dict[str, dict],
    r0_class: Dict[str, dict],
    *,
    seed: int = SEED,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Return cohort->task_ids and task_id->primary cohort."""
    pools = build_pools(meta, r0_class)
    assigned: Dict[str, str] = {}
    cohort_tasks: Dict[str, List[str]] = {c: [] for c in COHORT_PRIORITY}

    for cohort in COHORT_PRIORITY:
        limit = COHORT_LIMITS[cohort]
        candidates = [t for t in pools[cohort] if t not in assigned]
        if limit is None:
            chosen = candidates
        else:
            chosen = stratified_sample(
                candidates, meta, limit, seed=seed + hash(cohort) % 997
            )
        for tid in chosen:
            if tid not in assigned:
                assigned[tid] = cohort
                cohort_tasks[cohort].append(tid)
    return cohort_tasks, assigned


def task_cohorts(assigned: Dict[str, str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for tid, c in assigned.items():
        out[tid].append(c)
    return dict(out)
