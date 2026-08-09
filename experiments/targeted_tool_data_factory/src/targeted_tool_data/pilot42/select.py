"""Hard-gated deterministic Pilot4.2 selection."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from ..util import short_hash


def eligible(record: Dict[str, Any]) -> bool:
    semantic = record.get("semantic_validation") or {}
    query = record.get("query_validation") or {}
    v4 = record.get("v4_gate") or {}
    return bool(semantic.get("passed") and query.get("passed") and v4.get("passed")
                and not v4.get("has_shortcut") and not v4.get("unresolved"))


def select_records(records: List[Dict[str, Any]], n_selected: int = 4000,
                   seed: int = 20260731) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pool = [r for r in records if eligible(r)]
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pool:
        buckets[row["workflow_id"]].append(row)
    for rows in buckets.values():
        rows.sort(key=lambda r: short_hash([seed, r["semantic_program_id"]]))
    selected = []
    keys = sorted(buckets)
    cursor = 0
    while len(selected) < min(n_selected, len(pool)) and keys:
        key = keys[cursor % len(keys)]
        if buckets[key]:
            selected.append(buckets[key].pop())
        else:
            keys.remove(key)
            if not keys:
                break
            continue
        cursor += 1
    counts = Counter(r["workflow_id"] for r in selected)
    hard = len(selected) == n_selected and all(eligible(r) for r in selected)
    return selected, {
        "requested": n_selected, "selected": len(selected),
        "eligible_pool": len(pool), "workflow_support": dict(counts),
        "selection_all_hard_constraints_met": hard,
        "deficit": max(0, n_selected - len(selected)),
    }
