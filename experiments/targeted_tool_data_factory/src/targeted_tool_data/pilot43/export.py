"""Dataset export: splits, subsets, NESTFUL views, metrics tables, reports.

Every number written here is recomputed from the exported records, not carried over
from an earlier stage: the metrics functions take the JSONL rows as input. The
independent audit then repeats the exercise with code that shares nothing with this
module, and the two are compared. That double computation is the whole point --
Pilot4.2's report was a summary of its own labels.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..repro import sha256_obj
from . import (ANSWER_TYPES, CALL_BUCKETS, HELDOUT_PARTS, PROFILE_CALL_TARGETS,
               QUERY_MODES, RESERVE_TARGET, RUN_ID, SCHEMA_VERSION,
               STRUCTURAL_PATTERNS, TIERS, TIER_TARGETS, TRAIN_MASTER_TARGET)
from .blueprints import all_blueprints, export_registry, registry_hash
from .distractors import rerender_tools
from .ops import CODING_FAMILIES, build_ops, export_registry as export_ops
from .pipeline import iter_jsonl, read_jsonl, write_jsonl
from .select import (SURFACE_HOLDOUT_TRACK, Task, build_pool, cut_heldout,
                     nested_subsets, plan_heldout, select_tiers,
                     split_overlap_report)
from .tasks import nestful_compat, rebuild, task_record

TIER_FILES = {
    "PROFILE_CORE": "train_profile_core_3000.jsonl",
    "LONG_HORIZON_ENRICHMENT": "train_long_horizon_1200.jsonl",
    "CAPABILITY_ENRICHMENT": "train_capability_enrichment_600.jsonl",
    "CHALLENGE": "train_challenge_200.jsonl",
}
MASTER_FILE = "train_master_5000.jsonl"
SELECTED_FILE = "selected_all_7000.jsonl"
RESERVE_FILE = "reserve_1000.jsonl"
HELDOUT_FILE = "heldout_all.jsonl"
MIX_SIZES = (1000, 2000, 3000)
PROFILE_SIZES = (1000, 2000, 3000)


def _record(task: Task, tier: str, split: str) -> Dict[str, Any]:
    inst, bp, plan = rebuild(task.row)
    ver = task.verified
    validation = {
        "executor_replay_identical": True,
        "value_type_check_passed": True,
        "counterfactuals": ver.get("counterfactuals", {}),
        "node_necessity": ver.get("necessity", []),
        "node_necessity_summary": ver.get("necessity_summary", {}),
        "v4": ver.get("v4", {}),
        "query_checks": {
            "passed": task.query.get("passed"),
            "failed_layers": task.query.get("failed_layers", []),
            "classification": task.query.get("classification", {}),
        },
        "critic": task.query.get("critic", {"executed": False,
                                            "reason": "openrouter unavailable"}),
        "second_critic": task.query.get("critic2", {"executed": False,
                                                    "routed": False}),
    }
    query = {
        "query": task.query["query"],
        "requested_mode": task.query["requested_mode"],
        "actual_mode": task.query["actual_mode"],
        "source": task.query["query_source"],
        "renderer": task.query.get("renderer", ""),
        "fingerprints": task.query.get("fingerprints", {}),
    }
    offered = {"tools": rerender_tools(ver["offered_tools"], inst.track),
               **ver["offered"]}
    return task_record(row=task.row, inst=inst, bp=bp, plan=plan, query=query,
                       offered=offered, validation=validation,
                       verifier=ver["verifier"], tier=tier, split=split,
                       requested_skill=task.row["actual_primary_pattern"])


def export_dataset(out_dir: Path, *, seed: int = 20260731,
                   targets: Dict[str, int] | None = None) -> Dict[str, Any]:
    """Select, split, export and measure. Returns the selection report."""
    pool = build_pool(out_dir)
    keys = plan_heldout(pool, seed=seed)
    parts, train_candidates = cut_heldout(pool, keys, seed=seed)
    tiers, leftover = select_tiers(train_candidates, targets, seed=seed)

    master: List[Tuple[Task, str]] = []
    for tier in TIERS:
        for task in tiers[tier].tasks:
            master.append((task, tier))
    train_tasks = [t for t, _ in master]

    reserve = _reserve(leftover, RESERVE_TARGET, seed=seed)
    overlap = split_overlap_report(train_tasks, parts)

    # ── write splits ────────────────────────────────────────────────────
    written: Dict[str, int] = {}
    records: Dict[str, Dict[str, Any]] = {}
    for tier in TIERS:
        rows = [_record(t, tier, "train") for t in tiers[tier].tasks]
        for r in rows:
            records[r["task_id"]] = r
        written[TIER_FILES[tier]] = write_jsonl(out_dir / TIER_FILES[tier], rows)
    master_rows = [records[t.task_id] for t, _tier in master]
    written[MASTER_FILE] = write_jsonl(out_dir / MASTER_FILE, master_rows)

    heldout_rows: List[Dict[str, Any]] = []
    for name, tasks in parts.items():
        rows = [_record(t, "HELDOUT", f"heldout_{name}") for t in tasks]
        for r in rows:
            records[r["task_id"]] = r
        written[f"heldout_{name}.jsonl"] = write_jsonl(
            out_dir / f"heldout_{name}.jsonl", rows)
        heldout_rows.extend(rows)
    written[HELDOUT_FILE] = write_jsonl(out_dir / HELDOUT_FILE, heldout_rows)

    reserve_rows = [_record(t, "RESERVE", "reserve") for t in reserve]
    for r in reserve_rows:
        records[r["task_id"]] = r
    written[RESERVE_FILE] = write_jsonl(out_dir / RESERVE_FILE, reserve_rows)

    selected_rows = master_rows + heldout_rows + reserve_rows
    written[SELECTED_FILE] = write_jsonl(out_dir / SELECTED_FILE, selected_rows)

    # ── nested subsets ──────────────────────────────────────────────────
    mixes = nested_subsets(master, MIX_SIZES + (len(master),), seed=seed)
    for size in MIX_SIZES:
        ids = mixes.get(size, [])
        rows = [records[i] for i in ids if i in records]
        written[f"train_mix_{size}.jsonl"] = write_jsonl(
            out_dir / f"train_mix_{size}.jsonl", rows)
    core = [(t, "PROFILE_CORE") for t in tiers["PROFILE_CORE"].tasks]
    core_mixes = nested_subsets(core, PROFILE_SIZES, seed=seed + 1)
    for size in PROFILE_SIZES:
        ids = core_mixes.get(size, [])
        rows = [records[i] for i in ids if i in records]
        written[f"train_profile_{size}.jsonl"] = write_jsonl(
            out_dir / f"train_profile_{size}.jsonl", rows)
    nesting_ok = _check_nesting(mixes, MIX_SIZES) and _check_nesting(
        core_mixes, PROFILE_SIZES)

    # ── NESTFUL-compatible views ────────────────────────────────────────
    for name, rows in (("train_master", master_rows),
                       ("heldout", heldout_rows),
                       ("reserve", reserve_rows)):
        written[f"nestful_compat_{name}.jsonl"] = write_jsonl(
            out_dir / f"nestful_compat_{name}.jsonl",
            [nestful_compat(r) for r in rows])
    for size in MIX_SIZES:
        ids = mixes.get(size, [])
        written[f"nestful_compat_train_mix_{size}.jsonl"] = write_jsonl(
            out_dir / f"nestful_compat_train_mix_{size}.jsonl",
            [nestful_compat(records[i]) for i in ids if i in records])

    report = {
        "run_id": RUN_ID,
        "schema_version": SCHEMA_VERSION,
        "pool_size": len(pool),
        "train_candidates": len(train_candidates),
        "tiers": {tier: {"target": tiers[tier].target,
                         "selected": len(tiers[tier].tasks),
                         "met": tiers[tier].met,
                         "deficits": tiers[tier].deficits,
                         "notes": tiers[tier].notes} for tier in TIERS},
        "train_master": len(master_rows),
        "train_master_target": TRAIN_MASTER_TARGET,
        "heldout": {name: len(parts[name]) for name in HELDOUT_PARTS},
        "heldout_total": len(heldout_rows),
        "heldout_target": sum(HELDOUT_PARTS.values()),
        "reserve": len(reserve_rows),
        "reserve_target": RESERVE_TARGET,
        "selected_total": len(selected_rows),
        "leftover_after_reserve": max(0, len(leftover) - len(reserve_rows)),
        "heldout_keys": {"workflows": len(keys.workflows),
                         "plans": len(keys.plans),
                         "topologies": len(keys.topologies),
                         "templates": len(keys.templates),
                         "capability_combinations": len(keys.combos),
                         "surface_track": keys.track},
        "split_overlap": overlap,
        "nested_subsets_valid": nesting_ok,
        "files": written,
        "quotas_met": all(tiers[t].met for t in TIERS),
        "all_deficits": {t: tiers[t].deficits for t in TIERS
                         if tiers[t].deficits},
    }
    (out_dir / "selection_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_split_manifest(out_dir, tiers, parts, reserve, mixes, core_mixes)
    return report


def _reserve(leftover: Sequence[Task], want: int, seed: int) -> List[Task]:
    """The reserve is a plain stratified cut of what selection never used."""
    from .select import _stratified_take
    return _stratified_take(leftover, want, seed=seed)


def _check_nesting(mixes: Dict[int, List[str]], sizes: Sequence[int]) -> bool:
    ordered = sorted(mixes)
    for smaller, larger in zip(ordered, ordered[1:]):
        if not set(mixes[smaller]).issubset(set(mixes[larger])):
            return False
    return all(len(mixes.get(s, [])) == s for s in sizes if s in mixes)


def _write_split_manifest(out_dir: Path, tiers, parts, reserve, mixes,
                          core_mixes) -> None:
    manifest = {
        "run_id": RUN_ID,
        "train": {tier: [t.task_id for t in tiers[tier].tasks] for tier in TIERS},
        "heldout": {name: [t.task_id for t in tasks]
                    for name, tasks in parts.items()},
        "reserve": [t.task_id for t in reserve],
        "nested_train_mix": {str(k): v for k, v in mixes.items()},
        "nested_train_profile": {str(k): v for k, v in core_mixes.items()},
        "surface_holdout_track": SURFACE_HOLDOUT_TRACK,
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    ledger = {
        "run_id": RUN_ID,
        "reserve_size": len(reserve),
        "created_after": ["selection thresholds", "tier quotas", "split rules",
                          "query validation", "v4 gate", "necessity gate"],
        "used_for": [],
        "accesses": [],
        "untouched": True,
        "note": ("The reserve was cut from tasks left after every threshold in "
                 "select.py had already been applied, and nothing in this run "
                 "reads it back."),
    }
    (out_dir / "reserve_access_ledger.json").write_text(
        json.dumps(ledger, indent=1), encoding="utf-8")


# ── metrics recomputed from the exported records ─────────────────────────
def _rate(count: int, total: int) -> float:
    return round(count / total, 5) if total else 0.0


def dataset_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """All acceptance-criteria metrics, computed from exported record content."""
    ops = build_ops()
    n = len(rows)
    call_counts = Counter(str(len(r["gold_calls"])) for r in rows)
    buckets = Counter(_bucket(len(r["gold_calls"])) for r in rows)
    answer_types = Counter(r["answer_type"] for r in rows)
    patterns = Counter(r["declared"]["structural_pattern"] for r in rows)
    modes = Counter(r["actual_query_mode"] for r in rows)
    tracks = Counter(r["surface_track"] for r in rows)
    prims: Counter = Counter()
    caps: Counter = Counter()
    families: Counter = Counter()
    coding_calls = 0
    total_calls = 0
    sequences: Counter = Counter()
    norm_sequences: Counter = Counter()
    for r in rows:
        seq = []
        nseq = []
        for call in r["gold_calls"]:
            pid = call["primitive_id"]
            prims[pid] += 1
            caps[call["capability"]] += 1
            families[call["capability_family"]] += 1
            coding_calls += int(bool(call.get("coding_like")))
            total_calls += 1
            seq.append(pid)
            nseq.append(call["capability_family"])
        sequences["->".join(seq)] += 1
        norm_sequences["->".join(nseq)] += 1
    coding_tasks = sum(1 for r in rows
                       if any(c.get("coding_like") for c in r["gold_calls"]))
    booleans = [r for r in rows if r["answer_type"] == "boolean"]
    true_share = _rate(sum(1 for r in booleans if r["gold_answer"] is True),
                       len(booleans))
    v4 = [r["validation"]["v4"] for r in rows if r["validation"].get("v4")]
    nec_rows = [r["validation"].get("node_necessity") or [] for r in rows]
    nodes = sum(len(x) for x in nec_rows)
    necessary = sum(1 for x in nec_rows for row in x if row["necessary"])
    critic = [r["validation"].get("critic", {}) for r in rows]
    queries = [r["question"] for r in rows]
    from .qvalidate import diversity_report
    return {
        "n": n,
        "call_count_distribution": {k: _rate(v, n)
                                    for k, v in sorted(call_counts.items(),
                                                       key=lambda kv: int(kv[0]))},
        "call_bucket_distribution": {b: _rate(buckets.get(b, 0), n)
                                     for b in CALL_BUCKETS},
        "six_plus_share": _rate(buckets.get("6+", 0), n),
        "answer_type_distribution": {a: _rate(answer_types.get(a, 0), n)
                                     for a in ANSWER_TYPES},
        "structured_answer_share": _rate(
            sum(answer_types.get(a, 0) for a in ("string", "list", "object")), n),
        "pattern_distribution": {p: _rate(patterns.get(p, 0), n)
                                 for p in STRUCTURAL_PATTERNS
                                 if patterns.get(p)},
        "distinct_patterns": len(patterns),
        "distinct_patterns_6plus": len({r["declared"]["structural_pattern"]
                                        for r in rows
                                        if len(r["gold_calls"]) >= 6}),
        "query_mode_distribution": {m: _rate(modes.get(m, 0), n)
                                    for m in QUERY_MODES},
        "surface_track_distribution": {t: _rate(v, n) for t, v in tracks.items()},
        "actual_primitives_used": len(prims),
        "actual_capabilities_used": len(caps),
        "actual_capability_families": len(families),
        "coding_capability_families": len([f for f in families
                                           if f in CODING_FAMILIES]),
        "coding_task_share": _rate(coding_tasks, n),
        "coding_call_share": _rate(coding_calls, total_calls),
        "distinct_coding_primitives": len([p for p in prims
                                           if p in ops and ops[p].coding_like]),
        "primitives_not_in_registry": sorted(p for p in prims if p not in ops),
        "max_exact_sequence_share": _rate(max(sequences.values(), default=0), n),
        "top10_exact_sequence_share": _rate(
            sum(c for _k, c in sequences.most_common(10)), n),
        "max_normalized_sequence_share": _rate(
            max(norm_sequences.values(), default=0), n),
        "boolean_true_share": true_share,
        "boolean_count": len(booleans),
        "v4_coverage": _rate(sum(1 for g in v4 if g.get("v4_executed")), n),
        "v4_shortcuts": sum(1 for g in v4 if g.get("has_shortcut")),
        "v4_unresolved": sum(1 for g in v4 if not g.get("resolved")),
        "v4_skipped": n - len(v4),
        "node_necessity_coverage": _rate(sum(1 for x in nec_rows if x), n),
        "nodes_checked": nodes,
        "unnecessary_gold_nodes": nodes - necessary,
        "critic_coverage": _rate(sum(1 for c in critic if c.get("executed")), n),
        "llm_query_share": _rate(sum(1 for r in rows
                                     if r["query_source"] == "openrouter"), n),
        "distinct_workflows": len({r["workflow_id"] for r in rows}),
        "distinct_cells": len({r["cell_id"] for r in rows}),
        "offered_tool_count_mean": round(
            sum(r["offered_tool_count"] for r in rows) / max(1, n), 3),
        "diversity": diversity_report(queries),
    }


def _bucket(n: int) -> str:
    return str(n) if n <= 5 else "6+"


def tv_distance(observed: Dict[str, float],
                target: Dict[str, float]) -> float:
    keys = set(observed) | set(target)
    return round(0.5 * sum(abs(observed.get(k, 0.0) - target.get(k, 0.0))
                           for k in keys), 5)


def write_metric_tables(out_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """The CSV artifacts the spec names, all derived from the exported rows."""
    _csv(out_dir / "actual_graph_features.csv",
         ["task_id", "call_count", "n_nodes", "n_edges", "depth", "n_roots",
          "n_leaves", "n_join_nodes", "n_fan_out_nodes", "n_reused_outputs",
          "n_late_edges", "max_reference_distance", "n_parallel_branches",
          "critical_path", "n_type_transitions"],
         ({"task_id": r["task_id"], "call_count": len(r["gold_calls"]),
           **{k: r["declared"]["graph_features"].get(k)
              for k in ("n_nodes", "n_edges", "depth", "n_roots", "n_leaves",
                        "n_join_nodes", "n_fan_out_nodes", "n_reused_outputs",
                        "n_late_edges", "max_reference_distance",
                        "n_parallel_branches", "critical_path",
                        "n_type_transitions")}}
          for r in rows))
    _csv(out_dir / "actual_pattern_classification.csv",
         ["task_id", "call_count", "primary_pattern", "satisfied_patterns",
          "requested_skill", "matches_requested"],
         ({"task_id": r["task_id"], "call_count": len(r["gold_calls"]),
           "primary_pattern": r["declared"]["structural_pattern"],
           "satisfied_patterns": ";".join(r["declared"]["satisfied_patterns"]),
           "requested_skill": r["declared"]["requested_structural_skill"],
           "matches_requested": (r["declared"]["requested_structural_skill"]
                                 in r["declared"]["satisfied_patterns"])}
          for r in rows))
    usage: Counter = Counter()
    fam: Counter = Counter()
    for r in rows:
        for call in r["gold_calls"]:
            usage[(call["primitive_id"], call["capability"],
                   call["capability_family"],
                   bool(call.get("coding_like")))] += 1
            fam[call["capability_family"]] += 1
    _csv(out_dir / "actual_capability_usage.csv",
         ["primitive_id", "capability", "capability_family", "coding_like",
          "gold_calls", "family_gold_calls"],
         ({"primitive_id": pid, "capability": cap, "capability_family": family,
           "coding_like": coding, "gold_calls": count,
           "family_gold_calls": fam[family]}
          for (pid, cap, family, coding), count in sorted(usage.items())))
    seqs: Counter = Counter()
    for r in rows:
        seqs["->".join(c["primitive_id"] for c in r["gold_calls"])] += 1
    total = max(1, len(rows))
    _csv(out_dir / "primitive_sequence_distribution.csv",
         ["primitive_sequence", "n_tasks", "share"],
         ({"primitive_sequence": seq, "n_tasks": count,
           "share": round(count / total, 6)}
          for seq, count in seqs.most_common()))
    bool_rows: Dict[Tuple[str, str], List[bool]] = {}
    for r in rows:
        if r["answer_type"] != "boolean":
            continue
        key = (r["workflow_id"], r["cell_tier"])
        bool_rows.setdefault(key, []).append(bool(r["gold_answer"]))
    _csv(out_dir / "boolean_balance.csv",
         ["workflow_id", "tier", "n", "true_count", "true_share", "within_band"],
         ({"workflow_id": wid, "tier": tier, "n": len(vals),
           "true_count": sum(vals),
           "true_share": round(sum(vals) / len(vals), 4),
           "within_band": (0.35 <= sum(vals) / len(vals) <= 0.65
                           if len(vals) >= 20 else "n<20")}
          for (wid, tier), vals in sorted(bool_rows.items())))


def _csv(path: Path, columns: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns),
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_registries(out_dir: Path, samples: Dict[str, Any] | None = None) -> None:
    (out_dir / "workflow_registry_v3.json").write_text(
        json.dumps(export_registry(samples), indent=1, ensure_ascii=False),
        encoding="utf-8")
    (out_dir / "primitive_registry_v3.json").write_text(
        json.dumps(export_ops(), indent=1, ensure_ascii=False), encoding="utf-8")


def generation_cells(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Cell support measured on the exported master pool."""
    cells: Counter = Counter(r["cell_id"] for r in rows)
    small = {k: v for k, v in cells.items() if v < 20}
    return {
        "schema_version": "ttdf.pilot43.cells.v3",
        "n_cells": len(cells),
        "cells": [{"cell_id": k, "n_tasks": v} for k, v in cells.most_common()],
        "cells_with_min_20": sum(1 for v in cells.values() if v >= 20),
        "cells_in_preferred_band_30_80": sum(1 for v in cells.values()
                                            if 30 <= v <= 80),
        "singleton_cells": sum(1 for v in cells.values() if v == 1),
        "two_task_cells": sum(1 for v in cells.values() if v == 2),
        "under_supported_cells": dict(sorted(small.items(), key=lambda kv: kv[1])),
    }
