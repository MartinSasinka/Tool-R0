"""Exporters: canonical JSONL, NESTFUL-compatible, GRPO train-ready,
analysis CSV, manifests with SHA256. Nothing is silently dropped —
canonical JSONL always carries the full record; parity is test-covered.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from ..render import tool_to_jsonschema, tool_to_nestful
from ..schemas import TaskRecord, ToolSpec
from ..util import sha256_file, write_json, write_jsonl


def to_nestful_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": rec["task_id"],
        "input": rec["query"],
        "output": [{"name": c["name"], "arguments": c["arguments"], "label": c["label"]}
                   for c in rec["canonical_calls"]],
        "tools": [tool_to_nestful(ToolSpec(**t)) for t in rec["offered_tools"]],
        "gold_answer": rec["gold_answer"],
    }


def to_grpo_row(rec: Dict[str, Any], stage: str = "ttdf_pilot_v1") -> Dict[str, Any]:
    """Mirrors the stage3_train_ready row contract used by the GRPO trainer."""
    return {
        "sample_id": rec["task_id"],
        "question": rec["query"],
        "tools": [tool_to_jsonschema(ToolSpec(**t)) for t in rec["offered_tools"]],
        "gold_calls": [{"name": c["name"], "arguments": c["arguments"], "label": c["label"]}
                       for c in rec["canonical_calls"]],
        "gold_answer": rec["gold_answer"],
        "observations": rec["oracle_observations"],
        "num_calls": rec["call_count"],
        "stage": stage,
        "motif_type": rec["motif"],
        "answer_type": rec["answer_type"],
        "generation_seed": rec["value_seed"],
        "source": "targeted_tool_data_factory",
        "provenance": {
            "generation_cell_id": rec["generation_cell_id"],
            "track": rec["track"],
            "target_skill": rec["target_skill"],
            "target_failure_mode": rec["target_failure_mode"],
            "generator_version": rec["generator_version"],
            "profile_version": rec["profile_version"],
            "registry_hash": rec["registry_hash"],
            "executor_hash": rec["executor_hash"],
            "semantic_program_family": rec["semantic_program_family"],
            "graph_template_id": rec["graph_template_id"],
        },
    }


def to_analysis_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    sims = rec.get("distractor_similarity") or {}
    probe = rec.get("student_probe_result") or {}
    return {
        "task_id": rec["task_id"], "track": rec["track"],
        "cell": rec["generation_cell_id"], "skill": rec["target_skill"],
        "failure_mode": rec["target_failure_mode"], "motif": rec["motif"],
        "call_count": rec["call_count"], "depth": rec["dependency_depth"],
        "minimal_valid_call_count": rec.get("minimal_valid_call_count"),
        "offered": rec["offered_tool_count"], "relevant": rec["relevant_tool_count"],
        "hard_distractors": rec["hard_distractor_count"],
        "easy_distractors": rec["easy_distractor_count"],
        "dsim_name": sims.get("name"), "dsim_desc": sims.get("description"),
        "dsim_sig": sims.get("signature"),
        "ref_arg_share": rec["reference_arg_share"],
        "numeric_string_args": rec["numeric_string_args"],
        "answer_type": rec["answer_type"], "template_id": rec["template_id"],
        "program_family": rec["semantic_program_family"],
        "split": rec.get("split"),
        "probe_status": probe.get("status"),
        "probe_difficulty": probe.get("structural_difficulty"),
        "probe_success": probe.get("success_count"),
        "q_len": len(rec["query"]),
    }


def export_all(records: List[Dict[str, Any]], out_dir: Path,
               version: str, extra_manifest: Dict[str, Any]) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Path] = {}

    files["canonical"] = out_dir / f"canonical_{version}.jsonl"
    write_jsonl(files["canonical"], records)

    files["nestful_compat"] = out_dir / f"nestful_compat_{version}.jsonl"
    write_jsonl(files["nestful_compat"], [to_nestful_row(r) for r in records])

    files["grpo_train_ready"] = out_dir / f"grpo_train_ready_{version}.jsonl"
    write_jsonl(files["grpo_train_ready"], [to_grpo_row(r) for r in records])

    files["analysis_csv"] = out_dir / f"analysis_{version}.csv"
    rows = [to_analysis_row(r) for r in records]
    with open(files["analysis_csv"], "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["task_id"])
        w.writeheader()
        w.writerows(rows)

    # per-split exports
    by_split: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        if r.get("split"):
            by_split.setdefault(r["split"], []).append(r)
    for split_name, rows_s in sorted(by_split.items()):
        for fmt, conv in (("grpo", to_grpo_row), ("nestful", to_nestful_row)):
            p = out_dir / f"{split_name}_{fmt}_{version}.jsonl"
            write_jsonl(p, [conv(r) for r in rows_s])
            files[f"{split_name}_{fmt}"] = p

    manifest = {
        "version": version,
        "n_records": len(records),
        "splits": {k: len(v) for k, v in by_split.items()},
        "files": {k: {"path": str(p), "sha256": sha256_file(p)}
                  for k, p in files.items()},
        **extra_manifest,
    }
    write_json(out_dir / f"manifest_{version}.json", manifest)
    return manifest
