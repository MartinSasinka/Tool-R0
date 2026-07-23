"""Offline reward variants for counterfactual audit (no GPU / no LLM).

R0 = current execution_aware_v3_2_dense
R1 = terminal outcome only (official scorer authority; path-invariant)
R2 = R1 + epsilon * process tie-breaker (no call-count / gold-length penalty)
R3 = R2 with wider official_success vs executable_wrong_outcome gap
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lib.reward_v3_1 import (  # type: ignore
    _dependency_use_fraction,
    _has_final_answer,
    _is_ref,
    _per_call_analysis,
    _predicates,
    detect_stage,
    expected_calls,
)
from lib.reward_v3_2_dense import execution_aware_v3_2_dense  # type: ignore

# Terminal bands: (lo, hi) midpoints used as scalar unless quality tie-break inside band.
OUTCOME_BANDS_R1: Dict[str, tuple] = {
    "official_success": (0.92, 1.00),
    "executable_wrong_outcome": (0.48, 0.58),
    "non_executable_failure": (0.08, 0.22),
    "parse_or_no_tool": (0.00, 0.04),
}

OUTCOME_BANDS_R3: Dict[str, tuple] = {
    "official_success": (0.94, 1.00),
    "executable_wrong_outcome": (0.12, 0.28),
    "non_executable_failure": (0.05, 0.18),
    "parse_or_no_tool": (0.00, 0.04),
}

PROCESS_WEIGHTS = {
    "format_score": 0.15,
    "per_call_tool_score": 0.25,
    "per_call_argument_score": 0.25,
    "reference_score": 0.20,
    "execution_score": 0.15,
}

DEFAULT_EPS_R2 = 0.05
DEFAULT_EPS_R3 = 0.04


@dataclass
class VariantScore:
    variant: str
    terminal_class: str
    process_score: float
    terminal_reward: float
    total_reward: float
    epsilon: float
    components: Dict[str, Any]


def _mid(lo: float, hi: float) -> float:
    return round((lo + hi) / 2.0, 6)


def _official_win(row: dict) -> bool:
    v = row.get("official_win")
    if v is None:
        v = (row.get("_traj") or {}).get("official_win")
    return bool(v)


def _outcome_terminal_class(row: dict, pred: Dict[str, Any]) -> str:
    if _official_win(row):
        return "official_success"
    if pred.get("parse_err") or pred.get("clipped"):
        return "parse_or_no_tool"
    if pred.get("no_tool"):
        return "parse_or_no_tool"
    traj = row.get("_traj") or {}
    if traj.get("executable") and pred.get("is_executable"):
        return "executable_wrong_outcome"
    return "non_executable_failure"


def _process_score_no_length(
    pred: Dict[str, Any],
    calls_info: List[Dict[str, Any]],
    gold_n: int,
    gold_calls: List[Dict[str, Any]],
    trajectory,
) -> Dict[str, float]:
    format_score = 1.0 if (not pred["parse_err"] and not pred["clipped"]) else 0.0
    name_fracs = [1.0 if c["name_ok"] else 0.0 for c in calls_info[:gold_n]]
    val_fracs = [c["val_frac"] for c in calls_info[:gold_n]]
    per_call_tool_score = (sum(name_fracs) / gold_n) if gold_n else 0.0
    per_call_argument_score = (sum(val_fracs) / gold_n) if gold_n else 0.0
    refs = pred["refs"]
    gold_has_refs = any(
        _is_ref(v) for g in gold_calls for v in (g.get("arguments") or {}).values()
    )
    reference_score = (0.0 if gold_has_refs else 1.0) if refs is None else float(refs)
    dep_use = _dependency_use_fraction(trajectory, gold_calls)
    if dep_use is not None:
        reference_score = 0.5 * reference_score + 0.5 * float(dep_use)
    execution_score = float(pred["executable_frac"])
    components = {
        "format_score": format_score,
        "per_call_tool_score": per_call_tool_score,
        "per_call_argument_score": per_call_argument_score,
        "reference_score": reference_score,
        "execution_score": execution_score,
    }
    q = sum(PROCESS_WEIGHTS[k] * components[k] for k in PROCESS_WEIGHTS)
    components["process_score"] = round(q, 6)
    return components


def _terminal_scalar(
    terminal_class: str,
    bands: Dict[str, tuple],
    process_q: float,
    *,
    use_process_in_band: bool,
) -> float:
    lo, hi = bands.get(terminal_class, (0.0, 0.0))
    if use_process_in_band and hi > lo:
        return round(lo + (hi - lo) * process_q, 6)
    return _mid(lo, hi)


def score_variants(
    trajectory,
    task: Dict[str, Any],
    row: dict,
    *,
    eps_r2: float = DEFAULT_EPS_R2,
    eps_r3: float = DEFAULT_EPS_R3,
    train_stage: Optional[int] = 3,
) -> Dict[str, VariantScore]:
    os.environ.setdefault("TRAIN_STAGE", str(train_stage or 3))
    r0 = execution_aware_v3_2_dense(trajectory, task, train_stage=train_stage)
    d0 = r0.diagnostics

    pred = _predicates(trajectory, task)
    gold_calls = list(task.get("gold_calls") or [])
    gold_n = expected_calls(detect_stage(task, train_stage), task)
    calls_info = _per_call_analysis(trajectory, gold_calls)
    proc = _process_score_no_length(pred, calls_info, gold_n, gold_calls, trajectory)
    process_q = proc["process_score"]
    terminal = _outcome_terminal_class(row, pred)

    r1_term = _terminal_scalar(terminal, OUTCOME_BANDS_R1, process_q, use_process_in_band=False)
    r1_total = r1_term

    r2_term = r1_term
    r2_total = round(r1_term + eps_r2 * process_q, 6)

    r3_term = _terminal_scalar(terminal, OUTCOME_BANDS_R3, process_q, use_process_in_band=False)
    r3_total = round(r3_term + eps_r3 * process_q, 6)

    r0_components = {
        k: d0.get(k)
        for k in (
            "reward_class", "quality_score", "format_score", "call_count_progress",
            "per_call_tool_score", "per_call_argument_score", "reference_score",
            "execution_score", "final_answer_score", "too_few_calls", "fully_correct",
        )
    }

    return {
        "R0": VariantScore(
            variant="R0",
            terminal_class=str(d0.get("reward_class") or ""),
            process_score=round(float(d0.get("quality_score") or 0.0), 6),
            terminal_reward=float(r0.reward),
            total_reward=float(r0.reward),
            epsilon=0.0,
            components=r0_components,
        ),
        "R1": VariantScore(
            variant="R1",
            terminal_class=terminal,
            process_score=process_q,
            terminal_reward=r1_term,
            total_reward=r1_total,
            epsilon=0.0,
            components={**proc, "outcome_band": list(OUTCOME_BANDS_R1[terminal])},
        ),
        "R2": VariantScore(
            variant="R2",
            terminal_class=terminal,
            process_score=process_q,
            terminal_reward=r2_term,
            total_reward=r2_total,
            epsilon=eps_r2,
            components={**proc, "outcome_band": list(OUTCOME_BANDS_R1[terminal])},
        ),
        "R3": VariantScore(
            variant="R3",
            terminal_class=terminal,
            process_score=process_q,
            terminal_reward=r3_term,
            total_reward=r3_total,
            epsilon=eps_r3,
            components={**proc, "outcome_band": list(OUTCOME_BANDS_R3[terminal])},
        ),
    }


def variant_to_dict(v: VariantScore) -> dict:
    return {
        "variant": v.variant,
        "terminal_class": v.terminal_class,
        "process_score": v.process_score,
        "terminal_reward": v.terminal_reward,
        "total_reward": v.total_reward,
        "epsilon": v.epsilon,
        "components": v.components,
    }
