"""Lightweight schema detection for Pilot3 forensic inputs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

TRAJ_REQUIRED = {
    "sample_id",
    "num_gold_calls",
    "_traj",
}
TRAJ_OPTIONAL_TOP = {
    "strict_gold_trace_pass",
    "solution_equivalent_pass",
    "final_answer_pass",
    "alternative_valid_solution_pass",
    "correct_answer_but_unsupported_trace",
    "internal_f1_func",
    "internal_f1_param",
}
TRAJ_INNER_REQUIRED = {
    "official_win",
    "turns",
}
TRAIN_REQUIRED = {
    "sample_id",
    "gold_calls",
    "tools",
    "question",
}
DIAG_REQUIRED = {
    "sample_id",
    "input",
    "output",
    "tools",
}


def detect_schema(rows: List[Dict[str, Any]], kind_hint: Optional[str] = None) -> str:
    if not rows:
        return "empty"
    r0 = rows[0]
    keys = set(r0.keys())
    if "_traj" in keys and "sample_id" in keys:
        return "eval_trajectory"
    if "gold_calls" in keys and "provenance" in keys:
        return "train_grpo"
    if "gold_calls" in keys and "tools" in keys:
        return "train_like"
    if "output" in keys and "input" in keys and "tools" in keys:
        return "nestful_diagnostic"
    if "generation_cell_id" in keys or ("cell_id" in keys and "call_count" in keys):
        return "generation_cell"
    if kind_hint:
        return kind_hint
    return "unknown"


def field_audit(
    rows: List[Dict[str, Any]],
    required: Set[str],
    optional: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    optional = optional or set()
    present: Set[str] = set()
    missing_counts: Dict[str, int] = {k: 0 for k in required | optional}
    invalid = 0
    for r in rows:
        keys = set(r.keys())
        present |= keys
        miss_req = required - keys
        if miss_req:
            invalid += 1
        for k in required | optional:
            if k not in keys or r.get(k) is None:
                missing_counts[k] += 1
    return {
        "present_fields": sorted(present),
        "required_fields": sorted(required),
        "optional_fields": sorted(optional),
        "missing_required_fields": sorted(required - present),
        "missing_field_row_counts": {k: v for k, v in missing_counts.items() if v},
        "invalid_rows": invalid,
        "n_rows": len(rows),
    }


def schema_for_kind(kind: str) -> Tuple[Set[str], Set[str]]:
    if kind in ("c0_trajectories", "d1_trajectories", "c0_hf_trajectories", "eval_trajectory"):
        return TRAJ_REQUIRED, TRAJ_OPTIONAL_TOP
    if kind in ("train_data", "full_train_data", "heldout_data", "reserve_data", "train_grpo"):
        return TRAIN_REQUIRED, {"provenance", "motif_type", "answer_type", "gold_answer"}
    if kind in ("diagnostic_data", "nestful_diagnostic"):
        return DIAG_REQUIRED, {"gold_answer"}
    return set(), set()
