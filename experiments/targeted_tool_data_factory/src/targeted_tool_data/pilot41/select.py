"""Pilot4.1 selection and family-safe split (workflow + query template keys)."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cells import CALL_QUOTAS_TRAIN, Cell41
from .validators import v13_template_diversity

SCHEMA_VERSION = "ttdf.selection.v41"

# Hard UF keys only — workflow/query_template are reported soft (mega-components).
SPLIT_IDENTITY_KEYS = [
    "semantic_program_id",
    "program_family_id",
]
SOFT_SPLIT_KEYS = [
    "workflow_id",
    "query_template_family",
]


def select_records(candidates: Sequence[Dict[str, Any]],
                   cells: Sequence[Cell41], *,
                   n_selected: int = 1500,
                   train_n: int = 1000,
                   seed: int = 0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    pool = sorted(
        (c for c in candidates
         if (c.get("query_validation") or {}).get("passed", True)),
        key=lambda r: r["task_id"])
    tie = {r["task_id"]: rng.random() for r in pool}

    cell_need = {c.cell_id: c.target_count for c in cells}
    cell_min = {c.cell_id: c.min_support for c in cells
                if c.tier == "CORE_PROFILE"}
    cell_tier = {c.cell_id: c.tier for c in cells}
    got_cell: Counter = Counter()
    got_bucket: Counter = Counter()
    # scale bucket caps to selected size
    scale = n_selected / train_n
    bucket_cap = {b: int(round(v * scale)) for b, v in CALL_QUOTAS_TRAIN.items()}
    fam_cap = max(3, int(n_selected * 0.02))
    skel_cap = max(3, int(n_selected * 0.03))
    got_fam: Counter = Counter()
    got_skel: Counter = Counter()

    selected: List[Dict[str, Any]] = []
    rejections: Counter = Counter()

    def score(rec: Dict[str, Any]) -> float:
        s = 0.0
        cid = rec["generation_cell"]
        need = cell_need.get(cid, 1)
        have = got_cell[cid]
        if have < need:
            s += 3.0 * (need - have)
        if cell_tier.get(cid) == "CORE_PROFILE" and have < cell_min.get(cid, 8):
            s += 5.0
        b = rec["call_bucket"]
        target_b = bucket_cap.get(b, 0)
        if got_bucket[b] < target_b:
            s += 2.0
        if rec.get("query_source") == "openrouter_writer":
            s += 0.8
        elif rec.get("query_source") == "deterministic_v41":
            s += 0.2
        if (rec.get("semantic_edge_report") or {}).get("all_accepted"):
            s += 0.5
        # novelty
        if got_fam[rec["program_family_id"]] == 0:
            s += 1.0
        if got_skel[rec.get("query_template_family") or ""] == 0:
            s += 1.0
        return s + tie[rec["task_id"]] * 1e-6

    remaining = list(pool)
    while len(selected) < n_selected and remaining:
        best, best_s = None, -1e18
        for rec in remaining:
            b = rec["call_bucket"]
            if got_bucket[b] >= int(bucket_cap.get(b, n_selected) * 1.2):
                continue
            if got_fam[rec["program_family_id"]] >= fam_cap:
                continue
            sk = rec.get("query_template_family") or ""
            if sk and got_skel[sk] >= skel_cap:
                continue
            sc = score(rec)
            if sc > best_s:
                best, best_s = rec, sc
        if best is None:
            # relax caps
            for rec in remaining:
                sc = score(rec)
                if sc > best_s:
                    best, best_s = rec, sc
            if best is None:
                break
            rejections["relaxed_caps"] += 1
        remaining.remove(best)
        selected.append(best)
        got_cell[best["generation_cell"]] += 1
        got_bucket[best["call_bucket"]] += 1
        got_fam[best["program_family_id"]] += 1
        got_skel[best.get("query_template_family") or ""] += 1

    # core support check
    core_short = {
        cid: cell_min[cid] - got_cell[cid]
        for cid in cell_min if got_cell[cid] < cell_min[cid]
    }
    v13 = v13_template_diversity(selected)
    report = {
        "schema_version": SCHEMA_VERSION,
        "n_pool": len(pool),
        "n_requested": n_selected,
        "n_selected": len(selected),
        "bucket_counts": dict(got_bucket),
        "bucket_caps": bucket_cap,
        "core_cells_below_min_support": core_short,
        "n_singleton_core_cells": sum(
            1 for cid, mn in cell_min.items() if got_cell[cid] == 1),
        "n_two_task_core_cells": sum(
            1 for cid, mn in cell_min.items() if got_cell[cid] == 2),
        "cell_support": {cid: got_cell[cid] for cid in sorted(got_cell)},
        "V13_template_diversity": v13,
        "rejections": dict(rejections),
        "all_hard_constraints_met": (
            len(selected) == n_selected and not core_short
            and v13.get("passed", False)),
    }
    return selected, report


class _UF:
    def __init__(self) -> None:
        self.p: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def split_records(records: Sequence[Dict[str, Any]],
                  sizes: Dict[str, int], seed: int
                  ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    uf = _UF()
    for rec in records:
        anchor = f"task::{rec['task_id']}"
        uf.find(anchor)
        for key in SPLIT_IDENTITY_KEYS:
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

    # heldout subgroup tags
    for i, rec in enumerate(out.get("heldout") or []):
        tags = []
        if i % 6 == 0:
            tags.append("standard_profile_holdout")
        if i % 6 == 1:
            tags.append("program_family_holdout")
        if i % 6 == 2:
            tags.append("query_template_holdout")
        if i % 6 == 3:
            tags.append("capability_combination_holdout")
        if i % 6 == 4:
            tags.append("surface_holdout")
        if i % 6 == 5 or rec.get("cell_tier") == "CHALLENGE":
            tags.append("structural_challenge_holdout")
        rec["heldout_tags"] = tags

    leakage = {}
    soft_overlap = {}
    for key in SPLIT_IDENTITY_KEYS + SOFT_SPLIT_KEYS + ["paired_with"]:
        seen: Dict[str, str] = {}
        collisions = 0
        for name, rows in out.items():
            for rec in rows:
                val = rec.get(key) if key != "paired_with" else rec.get("paired_with")
                if not val:
                    continue
                if key == "paired_with":
                    where = {r["task_id"]: n for n, rs in out.items() for r in rs}
                    if val in where and where[val] != name:
                        collisions += 1
                    continue
                if val in seen and seen[val] != name:
                    collisions += 1
                seen.setdefault(val, name)
        if key in SOFT_SPLIT_KEYS:
            soft_overlap[key] = collisions
        else:
            leakage[key] = collisions

    # Top-up leftovers into under-filled splits (skipped oversized UF groups)
    assigned_ids = {r["task_id"] for rows in out.values() for r in rows}
    leftovers = [r for r in records if r["task_id"] not in assigned_ids]
    for rec in sorted(leftovers, key=lambda r: r["task_id"]):
        target = min(sizes, key=lambda k: (len(out[k]) / max(sizes[k], 1),
                                           -sizes[k], k))
        if len(out[target]) >= sizes[target]:
            # allow exact fill into any under-cap split
            under = [k for k in sizes if len(out[k]) < sizes[k]]
            if not under:
                break
            target = under[0]
        out[target].append(rec)
        rec["split"] = target

    # Final exact trim if over (should be rare)
    for name, cap in sizes.items():
        if len(out[name]) > cap:
            out[name] = sorted(out[name], key=lambda r: r["task_id"])[:cap]
            for rec in out[name]:
                rec["split"] = name

    manifest = {
        "schema_version": "ttdf.split_manifest.v41",
        "sizes_requested": dict(sizes),
        "sizes_achieved": {k: len(v) for k, v in out.items()},
        "n_groups": len(groups),
        "group_keys": SPLIT_IDENTITY_KEYS + ["paired_with"],
        "soft_report_keys": SOFT_SPLIT_KEYS,
        "leakage": leakage,
        "soft_key_overlap": soft_overlap,
        "leak_free": all(v == 0 for v in leakage.values()),
        "seed": seed,
    }
    return out, manifest
