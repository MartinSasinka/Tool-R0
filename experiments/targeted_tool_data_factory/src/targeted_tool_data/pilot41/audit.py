"""Offline Pilot3 / Pilot4 / Pilot4.1 comparison."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..repro import stamp, write_csv, write_json, write_text
from .graph_leak import audit_dataset
from .query_render import query_template_fingerprint
from .validators import v13_template_diversity

SCHEMA_VERSION = "ttdf.pilot41.audit.v1"


def _load(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _tv(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    return round(0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys), 4)


def _shares(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    c = Counter(str(r.get(key)) for r in rows)
    n = sum(c.values()) or 1
    return {k: round(v / n, 4) for k, v in sorted(c.items())}


def _stats(rows: Sequence[Dict[str, Any]], label: str) -> Dict[str, Any]:
    leak = audit_dataset(rows, label=label)
    v13 = v13_template_diversity(rows)
    cells = Counter(r.get("generation_cell") for r in rows)
    tops = Counter(r.get("graph_template_id") for r in rows)
    n = len(rows) or 1
    return {
        "label": label,
        "n": len(rows),
        "call_count_dist": _shares(rows, "call_bucket"),
        "query_mode_dist": _shares(
            rows, "requested_query_mode" if rows and "requested_query_mode" in rows[0]
            else "classified_query_mode"),
        "stages_related_rate": leak["stages_related_phrase_rate"],
        "mean_graph_edge_coverage": leak["mean_graph_edge_coverage"],
        "high_or_complete_graph_leak_rate": leak["high_or_complete_rate"],
        "call_count_disclosed_rate": leak["call_count_disclosed_rate"],
        "top1_skeleton_share": v13["evidence"]["top1_skeleton_share"],
        "n_distinct_skeletons": v13["evidence"]["n_distinct_skeletons"],
        "singleton_cell_rate": round(
            sum(1 for v in cells.values() if v == 1) / max(len(cells), 1), 4),
        "singleton_topology_rate": round(
            sum(1 for v in tops.values() if v == 1) / max(len(tops), 1), 4),
        "n_cells": len(cells),
        "n_topologies": len(tops),
        "mean_cell_support": round(n / max(len(cells), 1), 2),
    }


def run_pilot41_audit(repo_root: Path, out_dir: Path, *,
                      cli_args: Optional[Sequence[str]] = None
                      ) -> Dict[str, Any]:
    module = repo_root / "experiments" / "targeted_tool_data_factory"
    p3 = module / "outputs" / "selected" / "export_pilot3" / "train_grpo_pilot3.jsonl"
    p4 = module / "outputs" / "pilot4_profile_safe" / "canonical.jsonl"
    p41 = module / "outputs" / "pilot4_1_profile_safe" / "canonical.jsonl"
    dev = (repo_root / "experiments" / "nestful_mtgrpo_minimal" / "data" /
           "splits" / "nestful_dev.jsonl")

    rows3 = _load(p3)
    rows4 = [r for r in _load(p4) if r.get("split") == "train"] or _load(p4)
    rows41 = [r for r in _load(p41) if r.get("split") == "train"] or _load(p41)
    rows_dev = _load(dev)

    s3, s4, s41 = _stats(rows3, "pilot3"), _stats(rows4, "pilot4"), _stats(rows41, "pilot41")
    sdev = _stats(rows_dev, "nestful_dev") if rows_dev else {}

    metrics = []
    for key, direction, caveat in [
        ("stages_related_rate", "lower_is_better", "graph-explicit phrase"),
        ("mean_graph_edge_coverage", "lower_is_better", "dependency disclosure"),
        ("high_or_complete_graph_leak_rate", "lower_is_better", "graph leak class"),
        ("top1_skeleton_share", "lower_is_better", "template concentration"),
        ("singleton_cell_rate", "lower_is_better", "cell sparsity"),
        ("n_topologies", "higher_is_better", "topology coverage"),
        ("mean_cell_support", "higher_is_better", "repeated skill exposure"),
        ("call_count_disclosed_rate", "lower_is_better", "call-count leakage"),
    ]:
        v3, v4, v41 = s3.get(key), s4.get(key), s41.get(key)
        if direction == "lower_is_better":
            verdict = ("VERIFIED_IMPROVEMENT" if (v41 is not None and v4 is not None
                       and v41 < v4) else "UNRESOLVED")
        else:
            verdict = ("VERIFIED_IMPROVEMENT" if (v41 is not None and v4 is not None
                       and v41 > v4) else "UNRESOLVED")
        if key in ("n_topologies",) and v41 and v4 and v41 < v4:
            verdict = "PROFILE_TRADEOFF"
        metrics.append({
            "metric": key, "pilot3": v3, "pilot4": v4, "pilot41": v41,
            "dev200": sdev.get(key), "direction": direction,
            "verdict": verdict, "caveat": caveat,
        })

    # call-count TV to dev
    if sdev:
        metrics.append({
            "metric": "call_count_tv_to_dev",
            "pilot3": _tv(s3["call_count_dist"], sdev["call_count_dist"]),
            "pilot4": _tv(s4["call_count_dist"], sdev["call_count_dist"]),
            "pilot41": _tv(s41["call_count_dist"], sdev["call_count_dist"])
            if s41.get("call_count_dist") else None,
            "direction": "lower_is_better",
            "verdict": "PROFILE_TRADEOFF",
            "caveat": "profile match",
        })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "stats": {"pilot3": s3, "pilot4": s4, "pilot41": s41, "dev200": sdev},
        "metrics": metrics,
        "claim_classes": {
            "VERIFIED_IMPROVEMENT": [m["metric"] for m in metrics
                                     if m["verdict"] == "VERIFIED_IMPROVEMENT"],
            "PROFILE_TRADEOFF": [m["metric"] for m in metrics
                                 if m["verdict"] == "PROFILE_TRADEOFF"],
            "REQUIRES_TRAINING": ["NESTFUL official win", "dead_group_rate"],
            "REQUIRES_NESTFUL_EVAL": ["official win delta"],
        },
        "provenance": stamp(repo_root, schema_version=SCHEMA_VERSION,
                            cli_args=cli_args,
                            input_paths=[p for p in (p3, p4, p41, dev) if p.is_file()]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "PILOT4_VS_PILOT41_AUDIT.json", payload)
    write_csv(out_dir / "PILOT4_VS_PILOT41_METRICS.csv", metrics)
    md = ["# Pilot4 vs Pilot4.1 audit", "",
          "Claim classes separate offline verified improvements from "
          "training/eval-only claims.", ""]
    for m in metrics:
        md.append(f"- **{m['metric']}**: p4={m.get('pilot4')} → p41={m.get('pilot41')} "
                  f"[{m['verdict']}]")
    write_text(out_dir / "PILOT4_VS_PILOT41_AUDIT.md", "\n".join(md) + "\n")
    write_text(out_dir / "PILOT41_DATA_QUALITY_REPORT.md",
               "\n".join([
                   "# Pilot4.1 data quality",
                   "",
                   f"Train n: {s41.get('n')}",
                   f"Graph stages_related_rate: {s41.get('stages_related_rate')}",
                   f"High/complete graph leak: {s41.get('high_or_complete_graph_leak_rate')}",
                   f"Top1 skeleton share: {s41.get('top1_skeleton_share')}",
                   f"Singleton core-cell rate: {s41.get('singleton_cell_rate')}",
                   f"Mean cell support: {s41.get('mean_cell_support')}",
                   "",
                   "HUMAN-REVIEW REQUIRED for naturalness of LLM queries.",
                   "NOT TESTED BY TRAINING / NESTFUL.",
                   "",
               ]))
    payload["n_metrics"] = len(metrics)
    return payload
