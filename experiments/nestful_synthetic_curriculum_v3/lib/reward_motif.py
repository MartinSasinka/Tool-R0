"""execution_aware_v2_1_motif reward (v3 experiment).

Skeleton implementation — wired via v3/run.py before training on pod.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RewardResult:
    reward: float
    diagnostics: Dict[str, Any]


DEFAULT_WEIGHTS = {
    "tool_final_answer_pass": 0.30,
    "executable_trajectory": 0.20,
    "valid_references": 0.15,
    "motif_trace_consistency": 0.15,
    "tool_use_completeness": 0.10,
    "gold_or_equivalent_trace_progress": 0.10,
}

DEFAULT_CAPS = {
    "parse_error": 0.0,
    "clipped": 0.0,
    "no_tool_call": 0.0,
    "terminal_before_first_successful_tool": 0.0,
    "not_executable": 0.25,
    "invalid_reference": 0.25,
    "too_few_calls_without_final": 0.20,
    "final_pass_but_severe_short_trace": 0.65,
    "final_pass_but_low_motif_consistency": 0.75,
}

FLOOR = 0.85


def motif_trace_consistency(trajectory, task: Dict[str, Any]) -> float:
    """Check structural motif requirements from task metadata."""
    motif = task.get("motif_type", "linear_dependency")
    calls = [t for t in trajectory.turns if getattr(t, "parsed_call", None)]
    n = len(calls)
    gold_n = int(task.get("num_calls") or len(task.get("gold_calls", [])))

    if motif in ("fan_in", "simple_fan_in") and n < 3:
        return 0.0
    if motif == "fan_out" and n < 3:
        return 0.0
    if motif == "reference_reuse" and n < 2:
        return 0.0
    if motif == "long_chain" and n < 4:
        return 0.0
    if n < max(1, gold_n - 1):
        return 0.3
    return 1.0


def execution_aware_v2_1_motif(
    trajectory,
    task: Dict[str, Any],
    gold_observations: Optional[List[Any]] = None,
    *,
    predicates: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
    caps: Optional[Dict[str, float]] = None,
) -> RewardResult:
    """Motif-aware reward — delegates predicates to nestful_core when available."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    c = dict(DEFAULT_CAPS)
    if caps:
        c.update(caps)

    try:
        import sys
        from pathlib import Path
        exp = Path(__file__).resolve().parents[2]
        if str(exp) not in sys.path:
            sys.path.insert(0, str(exp))
        from nestful_core import rewards as R
        pred = predicates or {}
        final_pass = pred.get("final_pass")
        if final_pass is None:
            final_pass = R.tool_final_answer_pass(trajectory, task)
        executable = pred.get("executable")
        if executable is None:
            executable = float(R.is_executable_trajectory(trajectory))
        refs = pred.get("refs")
        if refs is None:
            rf = R.valid_references_fraction(trajectory)
            refs = 1.0 if rf is None else rf
        completeness = pred.get("completeness")
        if completeness is None:
            completeness = R.tool_use_completeness(trajectory, task)
        gold_prog = pred.get("gold_prog")
        if gold_prog is None:
            gold_prog = R.gold_trace_progress(trajectory, task, gold_observations)
        parse_err = R.has_parse_error(trajectory)
        clipped = bool(trajectory.clipped_any)
        no_tool = R.has_no_tool_call(trajectory)
        term_before = R.terminal_before_first_successful_tool(trajectory)
        invalid_ref = R.has_invalid_reference(trajectory)
        few = R.too_few_calls(trajectory, task)
    except Exception:
        final_pass = executable = refs = completeness = gold_prog = 0.0
        parse_err = clipped = no_tool = term_before = invalid_ref = few = False

    motif_cons = motif_trace_consistency(trajectory, task)
    R_val = (
        w["tool_final_answer_pass"] * (1.0 if final_pass else 0.0)
        + w["executable_trajectory"] * executable
        + w["valid_references"] * refs
        + w["motif_trace_consistency"] * motif_cons
        + w["tool_use_completeness"] * completeness
        + w["gold_or_equivalent_trace_progress"] * gold_prog
    )
    R_val = max(0.0, min(1.0, R_val))
    cap_applied = None

    if parse_err:
        R_val = c["parse_error"]; cap_applied = "parse_error"
    elif clipped:
        R_val = c["clipped"]; cap_applied = "clipped"
    elif no_tool:
        R_val = c["no_tool_call"]; cap_applied = "no_tool_call"
    elif term_before:
        R_val = c["terminal_before_first_successful_tool"]; cap_applied = "terminal_before_first_successful_tool"
    elif not executable and R_val > c["not_executable"]:
        R_val = c["not_executable"]; cap_applied = "not_executable"
    elif invalid_ref and R_val > c["invalid_reference"]:
        R_val = c["invalid_reference"]; cap_applied = "invalid_reference"
    elif few and not final_pass and R_val > c["too_few_calls_without_final"]:
        R_val = c["too_few_calls_without_final"]; cap_applied = "too_few_calls_without_final"
    elif final_pass and motif_cons < 0.5 and R_val > c["final_pass_but_low_motif_consistency"]:
        R_val = c["final_pass_but_low_motif_consistency"]; cap_applied = "final_pass_but_low_motif_consistency"
    elif final_pass and few and R_val > c["final_pass_but_severe_short_trace"]:
        R_val = c["final_pass_but_severe_short_trace"]; cap_applied = "final_pass_but_severe_short_trace"
    elif final_pass and executable and motif_cons >= 0.75 and R_val < FLOOR:
        R_val = FLOOR
        cap_applied = cap_applied or "floor_executable_final_motif_consistent"

    diag = {
        "reward_type": "execution_aware_v2_1_motif",
        "reward": R_val,
        "motif_trace_consistency": motif_cons,
        "tool_final_answer_pass": 1.0 if final_pass else 0.0,
        "cap_applied": cap_applied,
    }
    return RewardResult(R_val, diag)
