"""Difficulty signature (Phase J).

A *description* of what makes a task demanding along four independent axes. It
is deliberately not a claim about how hard the task is for any given model —
model-relative difficulty only becomes knowable once rollout statistics exist,
and the sampler's history state is where that will live.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

SCHEMA_VERSION = "ttdf.difficulty_signature.v1"

DIFFICULTY_BANDS = ["easy", "medium", "hard"]


def build_signature(*, features: Dict[str, Any], query_audit: Dict[str, Any],
                    track: str, schema_complexity: float,
                    repeated_tool_count: int, reference_format: str,
                    offered_tool_count: int, distractor_summary: Dict[str, Any]
                    ) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "structural": {
            "call_count": int(features.get("n_nodes", 0)),
            "depth": int(features.get("depth", 0)),
            "critical_path": int(features.get("critical_path", 0)),
            "n_joins": int(features.get("n_joins", 0)),
            "n_fan_out": int(features.get("n_fan_out_nodes", 0)),
            "n_reuses": int(features.get("n_reused_outputs", 0)),
            "n_late_references": int(features.get("n_late_references", 0)),
            "n_type_transitions": int(features.get("n_type_transitions", 0)),
        },
        "query": {
            "mode": query_audit.get("query_mode", "UNCLASSIFIED"),
            "operation_explicitness": float(
                query_audit.get("lexical_operation_coverage", 0.0)),
            "sequence_leakage": float(query_audit.get("sequence_leakage", 0.0)),
            "procedural_cue_count": int(query_audit.get("procedural_cue_count", 0)),
        },
        "surface": {
            "track": track,
            "schema_complexity": round(float(schema_complexity), 4),
            "repeated_tool_count": int(repeated_tool_count),
            "reference_format": reference_format,
        },
        "environment": {
            "offered_tool_count": int(offered_tool_count),
            "distractor_count": int(distractor_summary.get("hard_distractor_count", 0))
                                + int(distractor_summary.get("easy_distractor_count", 0)),
            "hard_distractor_count": int(distractor_summary.get("hard_distractor_count", 0)),
            "same_family_distractor_count": int(
                distractor_summary.get("same_family_distractor_count", 0)),
        },
    }


def difficulty_band(sig: Dict[str, Any]) -> str:
    """Coarse band used for cell quotas and curriculum unlocking."""
    s, q, e = sig["structural"], sig["query"], sig["environment"]
    score = 0.0
    score += min(s["call_count"] / 8.0, 1.0) * 2.0
    score += min(s["depth"] / 6.0, 1.0)
    score += min((s["n_joins"] + s["n_fan_out"] + s["n_reuses"]) / 4.0, 1.0)
    score += min(s["n_late_references"] / 3.0, 1.0) * 0.5
    score += (1.0 - q["operation_explicitness"]) * 2.0
    score += (1.0 - q["sequence_leakage"]) * 1.0
    score += min(e["hard_distractor_count"] / 8.0, 1.0)
    score += min(e["offered_tool_count"] / 20.0, 1.0) * 0.5
    if score < 3.0:
        return "easy"
    if score < 5.0:
        return "medium"
    return "hard"


def difficulty_score(sig: Dict[str, Any]) -> float:
    s, q, e = sig["structural"], sig["query"], sig["environment"]
    return round(
        min(s["call_count"] / 8.0, 1.0) * 2.0
        + min(s["depth"] / 6.0, 1.0)
        + min((s["n_joins"] + s["n_fan_out"] + s["n_reuses"]) / 4.0, 1.0)
        + min(s["n_late_references"] / 3.0, 1.0) * 0.5
        + (1.0 - q["operation_explicitness"]) * 2.0
        + (1.0 - q["sequence_leakage"]) * 1.0
        + min(e["hard_distractor_count"] / 8.0, 1.0)
        + min(e["offered_tool_count"] / 20.0, 1.0) * 0.5, 4)


def flatten(sig: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for group in ("structural", "query", "surface", "environment"):
        for k, v in sig.get(group, {}).items():
            out[f"{group}.{k}"] = v
    return out
