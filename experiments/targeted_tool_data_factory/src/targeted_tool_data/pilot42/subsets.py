"""Nested, hash-stratified training subsets."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from ..util import short_hash


def nested_stratified_subsets(records: List[Dict[str, Any]],
                              sizes=(500, 1000, 2000, 3000),
                              seed: int = 20260731) -> Dict[int, List[Dict[str, Any]]]:
    strata = defaultdict(list)
    for row in records:
        key = (row.get("workflow_id"), row.get("pattern_family"),
               row.get("requested_query_mode"))
        strata[key].append(row)
    for key, rows in strata.items():
        rows.sort(key=lambda r: short_hash([seed, key, r["semantic_program_id"]]))
    ordered, keys, index = [], sorted(strata), 0
    while len(ordered) < len(records) and keys:
        key = keys[index % len(keys)]
        if strata[key]:
            ordered.append(strata[key].pop())
            index += 1
        else:
            keys.remove(key)
    return {int(size): ordered[:min(int(size), len(ordered))] for size in sizes}


def assert_nested(subsets: Dict[int, List[Dict[str, Any]]]) -> bool:
    prior = set()
    for size in sorted(subsets):
        current = {r["task_id"] for r in subsets[size]}
        if not prior.issubset(current):
            return False
        prior = current
    return True
