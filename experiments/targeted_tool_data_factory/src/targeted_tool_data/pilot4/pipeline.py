"""End-to-end PROFILE_SAFE pilot4 run: profile -> cells -> generate ->
validate -> select -> split -> export.

No model is called anywhere in this path. Everything is seeded, so the same
commit plus the same config reproduces the pool byte for byte.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .. import registry as reg
from ..executor import executor_hash
from ..profile_v2 import build_profile_v2, derive_topology_constraints
from ..repro import sha256_file, stamp, write_json, write_jsonl
from ..util import short_hash
from . import PILOT4_VERSION, SCHEMA_VERSION
from .cells import assign_targets, build_cells, cells_summary
from .generate import generate_candidates
from .select import select_records, split_records
from .validate import v5_dedup, v6_distribution, validate_record

DEFAULT_CONFIG: Dict[str, Any] = {
    "run_id": "pilot4_profile_safe",
    "mode": "PROFILE_SAFE",
    "seed": 20260730,
    "candidate_target": 5000,
    "selected_total": 1000,
    "splits": {"train": 600, "heldout": 200, "reserve": 200},
    "paired_variant_rate": 0.25,
    "run_v4_minimal_path": False,
    "profile_source": "nestful_dev_200",
}


def query_mode_shares_from_profile(profile: Dict[str, Any], *,
                                   explicit_floor: float = 0.15) -> Dict[str, float]:
    """Map the measured dev-200 query modes onto the three renderers.

    The classifier has five labels but the factory has three renderers, so
    PROCEDURAL_PARTIAL folds into SEMI_IMPLICIT and UNCLASSIFIED is spread over
    the implicit modes. A floor keeps the explicit bucket populated: pilot4 is
    meant to *quota* explicit questions, not to eliminate them.
    """
    dist = ((profile.get("query_realism") or {}).get("query_mode_distribution")
            or {})
    explicit = float(dist.get("PROCEDURAL_EXPLICIT", 0.0))
    semi = float(dist.get("SEMI_IMPLICIT", 0.0)) + float(
        dist.get("PROCEDURAL_PARTIAL", 0.0))
    goal = float(dist.get("GOAL_BASED_IMPLICIT", 0.0))
    unclassified = float(dist.get("UNCLASSIFIED", 0.0))
    if semi + goal > 0:
        semi += unclassified * semi / (semi + goal)
        goal += unclassified * goal / (semi + goal)
    if explicit + semi + goal <= 0:
        return {"PROCEDURAL_EXPLICIT": 0.20, "SEMI_IMPLICIT": 0.35,
                "GOAL_BASED_IMPLICIT": 0.45}
    explicit = max(explicit, explicit_floor)
    total = explicit + semi + goal
    return {"PROCEDURAL_EXPLICIT": round(explicit / total, 4),
            "SEMI_IMPLICIT": round(semi / total, 4),
            "GOAL_BASED_IMPLICIT": round(goal / total, 4)}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]],
               fieldnames: Optional[Sequence[str]] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (rows[0].keys() if rows else ["key"]))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def to_grpo_row(rec: Dict[str, Any], stage: str = "ttdf_pilot4_v1") -> Dict[str, Any]:
    """Same row contract the MT-GRPO trainer already consumes for pilot3."""
    return {
        "sample_id": rec["task_id"],
        "question": rec["question"],
        "tools": [{k: v for k, v in t.items()
                   if k not in ("is_distractor", "distractor_type")}
                  for t in rec["tools"]],
        "gold_calls": rec["gold_calls"],
        "gold_answer": rec["gold_answer"],
        "observations": rec["oracle_observations"],
        "num_calls": rec["call_count"],
        "stage": stage,
        "motif_type": rec["pattern_family"],
        "answer_type": rec["answer_type"],
        "generation_seed": rec.get("generation_seed", 0),
        "source": "targeted_tool_data_factory.pilot4",
        "provenance": {
            "generation_cell_id": rec["generation_cell"],
            "track": rec["track"],
            "surface_track": rec["surface_track"],
            "query_mode": rec["requested_query_mode"],
            "target_skill": rec["target_skill"],
            "target_failure_mode": rec["target_failure_mode"],
            "generator_version": rec["generator_version"],
            "registry_hash": rec["registry_hash"],
            "executor_hash": rec["executor_hash"],
            "semantic_program_family": rec["program_family_id"],
            "semantic_program_id": rec["semantic_program_id"],
            "graph_template_id": rec["graph_template_id"],
            "difficulty_band": rec["difficulty_band"],
            "difficulty_signature": rec["difficulty_signature"],
        },
    }


def to_nestful_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": rec["task_id"],
        "input": rec["question"],
        "output": rec["gold_calls"],
        "tools": [{"name": t["name"], "description": t["description"],
                   "parameters": t["parameters"],
                   "output_parameters": t["output_parameters"]}
                  for t in rec["tools"]],
        "gold_answer": rec["gold_answer"],
    }


def run_pipeline(repo_root: Path, out_dir: Path, *,
                 dev_path: Optional[Path] = None,
                 config: Optional[Dict[str, Any]] = None,
                 cli_args: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])

    # ── 1. PROFILE_SAFE target profile (dev-200 only, aggregates only)
    dev_rows = _load_jsonl(dev_path) if dev_path and dev_path.exists() else []
    profile = build_profile_v2(dev_rows, source=str(cfg["profile_source"]),
                               mode="PROFILE_SAFE")
    constraints = derive_topology_constraints(profile)
    write_json(out_dir / "target_profile_v2.json", profile)
    write_json(out_dir / "topology_constraints.json", constraints)

    # ── 2. sparse, profile-driven cell design
    qshares = cfg.get("query_mode_shares") or query_mode_shares_from_profile(profile)
    cfg["resolved_query_mode_shares"] = qshares
    cells = build_cells(profile, constraints,
                        call_bucket_boosts=cfg.get("call_bucket_boosts"),
                        query_mode_shares=qshares,
                        track_shares=cfg.get("track_shares"))
    assign_targets(cells, int(cfg["candidate_target"]))
    write_json(out_dir / "generation_cells.json", {
        "schema_version": SCHEMA_VERSION,
        "n_cells": len(cells),
        "summary": cells_summary(cells),
        "cells": [c.as_dict() for c in cells],
    })

    # ── 3. candidates
    candidates, gen_stats = generate_candidates(
        cells, int(cfg["candidate_target"]), seed,
        paired_variant_rate=float(cfg["paired_variant_rate"]))
    for rec in candidates:
        rec["generation_seed"] = seed
    write_jsonl(out_dir / "candidates.jsonl", candidates)

    # ── 4. validation V1-V8
    validated: List[Dict[str, Any]] = []
    layer_fail = Counter()
    for rec in candidates:
        report = validate_record(rec, run_v4=bool(cfg["run_v4_minimal_path"]))
        rec["validation"] = {**(rec.get("validation") or {}), **report["layers"]}
        rec["validation_passed"] = report["passed"]
        rec["v7_in_target_bucket"] = report["v7_in_target_bucket"]
        if report["passed"]:
            validated.append(rec)
        else:
            for name, layer in report["layers"].items():
                if not layer.get("passed", True):
                    layer_fail[name] += 1
    dedup = v5_dedup(validated)
    dup_ids = set(dedup["duplicate_ids"])
    if dedup["n_duplicates"]:
        seen: set = set()
        deduped: List[Dict[str, Any]] = []
        for rec in validated:
            key = short_hash([rec["question"].lower().strip(),
                              rec["tool_combination_hash"]])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rec)
        validated = deduped
    dedup_after = v5_dedup(validated)
    dist = v6_distribution(validated)
    write_jsonl(out_dir / "validated.jsonl", validated)

    validation_report = {
        "schema_version": SCHEMA_VERSION,
        "n_candidates": len(candidates),
        "n_validated": len(validated),
        "pass_rate": round(len(validated) / max(len(candidates), 1), 4),
        "per_layer_failures": dict(layer_fail),
        "V5_dedup": dedup,
        "V5_dedup_after_removal": dedup_after,
        "V6_distribution": dist,
        "V7_in_target_bucket_rate": round(
            sum(1 for r in validated if r.get("v7_in_target_bucket")) /
            max(len(validated), 1), 4),
        "V8_pass_rate": round(
            sum(1 for r in validated
                if (r.get("validation") or {}).get("V8", {}).get("passed", True)) /
            max(len(validated), 1), 4),
        "v4_minimal_path": ("bounded search enabled"
                            if cfg["run_v4_minimal_path"]
                            else "skipped: reported as a known limitation"),
        "generation": gen_stats,
    }
    write_json(out_dir / "validation_report.json", validation_report)

    # ── 5. multi-objective selection
    n_select = int(cfg["selected_total"])
    selected, sel_report = select_records(validated, cells, n_select,
                                          profile=profile, seed=seed)
    # the pool-level V5/V6 numbers describe the candidate pool; what ships is
    # the selected set, so both are stated instead of only the pool
    sel_report["V5_dedup_selected"] = v5_dedup(selected)
    sel_report["V6_distribution_selected"] = v6_distribution(selected)
    write_jsonl(out_dir / "selected.jsonl", selected)
    write_json(out_dir / "selection_report.json", sel_report)

    # ── 6. family-safe split
    splits, split_manifest = split_records(selected, dict(cfg["splits"]), seed)
    for name, rows in splits.items():
        for r in rows:
            r["split"] = name
        write_jsonl(out_dir / f"{name}.jsonl", [to_grpo_row(r) for r in rows])
    write_json(out_dir / "split_manifest.json", split_manifest)

    # ── 7. exports
    ordered = [r for name in ("train", "heldout", "reserve")
               for r in splits.get(name, [])]
    write_jsonl(out_dir / "canonical.jsonl", ordered)
    write_jsonl(out_dir / "nestful_compat.jsonl",
                [to_nestful_row(r) for r in ordered])
    _write_csv(out_dir / "pilot4_tasks.csv", [{
        "task_id": r["task_id"], "split": r["split"],
        "cell": r["generation_cell"], "call_count": r["call_count"],
        "call_bucket": r["call_bucket"], "pattern_family": r["pattern_family"],
        "query_mode": r["requested_query_mode"],
        "classified_query_mode": r["classified_query_mode"],
        "track": r["surface_track"], "difficulty_band": r["difficulty_band"],
        "difficulty_score": r["difficulty_score"],
        "offered": r["offered_tool_count"],
        "hard_distractors": r["hard_distractor_count"],
        "depth": r["structural_features"]["depth"],
        "n_joins": r["structural_features"]["n_joins"],
        "n_reuses": r["structural_features"]["n_reused_outputs"],
        "n_late_refs": r["structural_features"]["n_late_references"],
        "sequence_leakage": r["query_audit"]["sequence_leakage"],
        "exact_op_coverage": r["query_audit"]["exact_operation_coverage"],
        "family": r["program_family_id"],
    } for r in ordered])

    # ── 8. freeze + hashes
    provenance = stamp(repo_root, schema_version=SCHEMA_VERSION,
                       cli_args=cli_args, seeds={"generation": seed},
                       config=cfg,
                       input_paths=[dev_path] if dev_path else None,
                       extra={"pilot4_version": PILOT4_VERSION,
                              "registry_hash": reg.registry_hash(),
                              "executor_hash": executor_hash()})
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "run_id": cfg["run_id"],
        "mode": cfg["mode"],
        "frozen": True,
        "provenance": provenance,
        "counts": {
            "candidates": len(candidates),
            "validated": len(validated),
            "selected": len(selected),
            **{k: len(v) for k, v in splits.items()},
        },
        "ordered_sample_ids_hash": short_hash([r["task_id"] for r in ordered]),
        "deficits": sel_report.get("deficits", {}),
    }
    write_json(out_dir / "freeze_manifest.json", freeze)

    files = sorted(p for p in out_dir.iterdir() if p.is_file()
                   and p.name != "MANIFEST.sha256.json")
    write_json(out_dir / "MANIFEST.sha256.json", {
        "schema_version": SCHEMA_VERSION,
        "run_id": cfg["run_id"],
        "provenance": provenance,
        "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                  for p in files},
    })

    return {
        "out_dir": str(out_dir),
        "n_cells": len(cells),
        "n_candidates": len(candidates),
        "n_validated": len(validated),
        "n_selected": len(selected),
        "splits": {k: len(v) for k, v in splits.items()},
        "validation_report": validation_report,
        "selection_report": sel_report,
        "freeze_manifest": freeze,
    }
