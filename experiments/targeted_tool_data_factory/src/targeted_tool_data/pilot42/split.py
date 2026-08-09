"""Union-find leakage-safe splitting over hard provenance keys."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from ..util import short_hash

# Keep hard UF narrow so exact split sizes remain achievable.
HARD_KEYS = ("semantic_program_id", "workflow_instance_id")
SOFT_KEYS = ("query_template_fingerprint", "workflow_id", "program_family_id")


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def split_records(records: List[Dict[str, Any]],
                  sizes: Dict[str, int] | None = None,
                  seed: int = 20260731) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    sizes = sizes or {"train": 3000, "heldout": 500, "reserve": 500}
    uf = UnionFind(len(records))
    seen: Dict[Tuple[str, str], int] = {}
    for i, row in enumerate(records):
        for key in HARD_KEYS:
            value = str(row.get(key) or "")
            if not value:
                continue
            marker = (key, value)
            if marker in seen:
                uf.union(i, seen[marker])
            else:
                seen[marker] = i
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[uf.find(i)].append(row)
    ordered = sorted(groups.values(), key=lambda g: short_hash([seed, g[0]["task_id"]]))
    out = {name: [] for name in sizes}
    assigned = set()
    for group in ordered:
        candidates = [name for name in sizes if len(out[name]) + len(group) <= sizes[name]]
        if not candidates:
            continue
        name = max(candidates, key=lambda n: sizes[n] - len(out[n]))
        out[name].extend(group)
        for row in group:
            assigned.add(row["task_id"])
            row["split"] = name
    # Top-up underfilled splits only when hard keys do not collide
    owner: Dict[Tuple[str, str], str] = {}
    for name, rows in out.items():
        for row in rows:
            for key in HARD_KEYS:
                val = str(row.get(key) or "")
                if val:
                    owner[(key, val)] = name
    leftovers = [r for r in records if r["task_id"] not in assigned]
    leftovers.sort(key=lambda r: short_hash([seed, "leftover", r["task_id"]]))
    for row in leftovers:
        under = [n for n in sizes if len(out[n]) < sizes[n]]
        if not under:
            break
        name = max(under, key=lambda n: sizes[n] - len(out[n]))
        conflict = False
        for key in HARD_KEYS:
            val = str(row.get(key) or "")
            if val and (key, val) in owner and owner[(key, val)] != name:
                conflict = True
                break
        if conflict:
            continue
        out[name].append(row)
        row["split"] = name
        assigned.add(row["task_id"])
        for key in HARD_KEYS:
            val = str(row.get(key) or "")
            if val:
                owner[(key, val)] = name
    collisions = []
    owner2: Dict[Tuple[str, str], str] = {}
    for name, rows in out.items():
        for row in rows:
            for key in HARD_KEYS:
                marker = (key, str(row.get(key) or ""))
                if marker[1] and marker in owner2 and owner2[marker] != name:
                    collisions.append(marker)
                owner2[marker] = name
    soft_overlap = {}
    for key in SOFT_KEYS:
        seen_soft: Dict[str, str] = {}
        overlaps = 0
        for name, rows in out.items():
            for row in rows:
                val = str(row.get(key) or "")
                if not val:
                    continue
                if val in seen_soft and seen_soft[val] != name:
                    overlaps += 1
                seen_soft.setdefault(val, name)
        soft_overlap[key] = overlaps
    return out, {
        "leak_free": not collisions,
        "collisions": collisions[:20],
        "split_sizes": {k: len(v) for k, v in out.items()},
        "sizes_requested": dict(sizes),
        "hard_keys": list(HARD_KEYS),
        "soft_keys": list(SOFT_KEYS),
        "soft_key_overlap": soft_overlap,
    }
