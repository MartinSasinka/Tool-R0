"""execution_aware_v3_2_dense — densified within-band reward for GRPO.

Motivation (audits/REWARD_AUDIT.md, FAILURE_MODE_AUDIT.md): v3.1's hard caps
collapse most completions of a group onto a handful of identical reward values
(dead groups, no within-group variance, no GRPO gradient). v3.2 keeps the SAME
failure-class ordering as v3.1 but maps each completion into a CLASS BAND
``[lo, hi]`` and positions it inside the band with a continuous quality score,
so two different completions in the same class almost never tie exactly.

Class bands (monotone; a too-few completion can never outscore a complete
executable correct trace):

    parse_error / clipped              0.00        (flat — nothing to grade)
    no_tool_call                       0.02        (distinct from parse error)
    premature final w/o required calls 0.04
    invalid_reference                  [0.05, 0.15]
    too_few_calls                      [0.10, 0.45]   (one-correct-call-then-stop
                                                       lands high in this band)
    wrong_tool                         [0.10, 0.35]
    correct tool, wrong args           [0.35, 0.60]
    too_many_calls                     [0.45, 0.70]
    executable, wrong final answer     [0.60, 0.80]
    fully correct                      [0.90, 1.00]

v3.1 (lib/reward_v3_1.py) is FROZEN as the baseline — this module imports its
helpers but never modifies it. Dispatch name: ``execution_aware_v3_2_dense``.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Frozen v3.1 helpers (read-only import; reward_v3_1 itself is never edited).
from lib.reward_v3_1 import (  # type: ignore
    RewardResult,
    _dependency_use_fraction,
    _has_final_answer,
    _is_ref,
    _per_call_analysis,
    _predicates,
    _turn_scores,
    detect_stage,
    expected_calls,
)

REWARD_POLICY_NAME = "execution_aware_v3_2_dense"

# (lo, hi) per failure class — hi of a worse class stays below the "fully
# correct" floor, preserving monotonicity of class ordering.
BANDS: Dict[str, tuple] = {
    "parse_error": (0.0, 0.0),
    "clipped": (0.0, 0.0),
    "no_tool_call": (0.02, 0.02),
    "premature_final_nonterminal": (0.04, 0.04),
    "invalid_reference": (0.05, 0.15),
    "wrong_tool": (0.10, 0.35),
    "too_few_calls": (0.10, 0.45),
    "correct_tool_wrong_args": (0.35, 0.60),
    "too_many_calls": (0.45, 0.70),
    "executable_wrong_final": (0.60, 0.80),
    "fully_correct": (0.90, 1.00),
    # fallthrough for partially-wrong multi-call traces that fit no class above
    "partial_progress": (0.10, 0.55),
}

# Within-band quality weights (continuous components; sum = 1.0).
QUALITY_WEIGHTS = {
    "format_score": 0.10,
    "call_count_progress": 0.20,
    "per_call_tool_score": 0.20,
    "per_call_argument_score": 0.20,
    "reference_score": 0.10,
    "execution_score": 0.10,
    "final_answer_score": 0.10,
}


def _band_reward(cls: str, q: float) -> float:
    lo, hi = BANDS[cls]
    q = max(0.0, min(1.0, q))
    return round(lo + (hi - lo) * q, 6)


def execution_aware_v3_2_dense(
    trajectory,
    task: Dict[str, Any],
    gold_observations: Optional[List[Any]] = None,  # noqa: ARG001 (interface parity)
    *,
    train_stage: Optional[int] = None,
) -> RewardResult:
    stage = detect_stage(task, train_stage)  # raises RewardError when unknown
    gold_calls = list(task.get("gold_calls") or [])
    gold_n = expected_calls(stage, task)
    terminal = bool(task.get("terminal_stage", True))

    try:
        pred = _predicates(trajectory, task)
    except Exception as exc:  # noqa: BLE001 — surface, never fake a value
        msg = f"{type(exc).__name__}: {exc}"
        print(f"[reward_v3_2] PREDICATES_ERROR task={task.get('task_id')}: {msg}",
              flush=True)
        return RewardResult(0.0, {
            "reward_type": REWARD_POLICY_NAME,
            "reward_total": 0.0,
            "predicates_error": msg,
            "reward_cap_reason": "predicates_error",
            "cap_applied": "predicates_error",
            "stage": stage,
            "turn_scores": [0.0] * len(getattr(trajectory, "turns", [])),
        })

    calls_info = _per_call_analysis(trajectory, gold_calls)
    n_pred = len(calls_info)
    has_final = _has_final_answer(trajectory)
    final_ok = bool(pred["final_pass"])

    # ── continuous subcomponents (all in [0,1]) ─────────────────────────────
    format_score = 1.0 if (not pred["parse_err"] and not pred["clipped"]) else 0.0
    call_count_progress = min(1.0, n_pred / gold_n) if gold_n else 0.0
    # emitted-call quality is graded over GOLD length so missing calls hurt,
    # but each emitted call contributes its own fraction (dense, not binary)
    name_fracs = [1.0 if c["name_ok"] else 0.0 for c in calls_info[:gold_n]]
    val_fracs = [c["val_frac"] for c in calls_info[:gold_n]]
    per_call_tool_score = (sum(name_fracs) / gold_n) if gold_n else 0.0
    per_call_argument_score = (sum(val_fracs) / gold_n) if gold_n else 0.0

    refs = pred["refs"]
    gold_has_refs = any(_is_ref(v) for g in gold_calls
                        for v in (g.get("arguments") or {}).values())
    reference_score = (0.0 if gold_has_refs else 1.0) if refs is None else float(refs)
    dep_use = _dependency_use_fraction(trajectory, gold_calls)
    if dep_use is not None:
        reference_score = 0.5 * reference_score + 0.5 * float(dep_use)

    execution_score = float(pred["executable_frac"])
    if terminal:
        final_answer_score = 1.0 if final_ok else 0.0
    else:
        final_answer_score = 1.0 if not has_final else 0.0

    components = {
        "format_score": format_score,
        "call_count_progress": call_count_progress,
        "per_call_tool_score": per_call_tool_score,
        "per_call_argument_score": per_call_argument_score,
        "reference_score": reference_score,
        "execution_score": execution_score,
        "final_answer_score": final_answer_score,
    }
    q = sum(QUALITY_WEIGHTS[k] * v for k, v in components.items())

    # ── failure-class selection (same priority ordering as v3.1) ────────────
    keys_oks = [c["keys_ok"] for c in calls_info[:gold_n]]
    matchable = min(n_pred, gold_n)
    tools_full = bool(gold_n and matchable == gold_n
                      and all(f >= 0.999 for f in name_fracs))
    args_full = bool(gold_n and matchable == gold_n and all(keys_oks)
                     and all(v >= 0.999 for v in val_fracs))
    too_few = n_pred < gold_n
    too_many = n_pred > gold_n
    premature_final = (not terminal) and has_final
    wrong_tool = (bool(calls_info) and not calls_info[0]["name_ok"]) if stage == "stage1" \
        else (matchable > 0 and not tools_full)
    fully_correct = (tools_full and args_full and pred["is_executable"]
                     and not too_few and not too_many
                     and (final_ok if terminal else not has_final))

    if pred["parse_err"]:
        cls = "parse_error"
    elif pred["clipped"]:
        cls = "clipped"
    elif pred["no_tool"]:
        cls = "no_tool_call"
    elif premature_final:
        cls = "premature_final_nonterminal"
    elif pred["invalid_ref"]:
        cls = "invalid_reference"
    elif fully_correct:
        cls = "fully_correct"
    elif too_few:
        cls = "too_few_calls"
    elif wrong_tool:
        cls = "wrong_tool"
    elif tools_full and not args_full:
        cls = "correct_tool_wrong_args"
    elif too_many:
        cls = "too_many_calls"
    elif tools_full and args_full and pred["is_executable"] and terminal and not final_ok:
        cls = "executable_wrong_final"
    else:
        cls = "partial_progress"

    R_val = _band_reward(cls, q)

    too_few_penalty = round(max(0.0, q - R_val), 6) if cls == "too_few_calls" else 0.0
    too_many_penalty = round(max(0.0, q - R_val), 6) if cls == "too_many_calls" else 0.0

    turn_scores = _turn_scores(trajectory, calls_info, final_ok, terminal)

    diag = {
        "reward_type": REWARD_POLICY_NAME,
        "reward_policy": REWARD_POLICY_NAME,
        "reward": R_val,
        "reward_total": R_val,
        "reward_class": cls,
        "reward_band": list(BANDS[cls]),
        "quality_score": round(q, 6),
        # required subcomponent diagnostics
        "format_score": format_score,
        "call_count_progress": round(call_count_progress, 6),
        "per_call_tool_score": round(per_call_tool_score, 6),
        "per_call_argument_score": round(per_call_argument_score, 6),
        "reference_score": round(reference_score, 6),
        "execution_score": round(execution_score, 6),
        "final_answer_score": final_answer_score,
        "too_few_penalty": too_few_penalty,
        "too_many_penalty": too_many_penalty,
        "cap_reason": cls if cls != "fully_correct" else None,
        "reward_cap_reason": cls if cls != "fully_correct" else None,
        "cap_applied": cls if cls != "fully_correct" else None,
        # v3.1-compatible failure flags (probe/trainer diagnostics read these)
        "stage": stage,
        "n_pred_calls": n_pred,
        "gold_n_calls": gold_n,
        "predicted_num_calls": n_pred,
        "too_few_calls": bool(too_few),
        "too_many_calls": bool(too_many),
        "wrong_tool": bool(wrong_tool),
        "wrong_args": bool(tools_full and not args_full),
        "parse_error": bool(pred["parse_err"]),
        "no_tool_call": bool(pred["no_tool"]),
        "invalid_reference": bool(pred["invalid_ref"]),
        "premature_final": bool(premature_final),
        "fully_correct": bool(fully_correct),
        "predicates_error": None,
        "turn_scores": turn_scores,
    }
    return RewardResult(R_val, diag)


# ─────────────────────────────────────────────────────────────────────────────
#  Trainer adapter — same contract as reward_v3_1.episode_turn_reward_seq
# ─────────────────────────────────────────────────────────────────────────────

def _env_train_stage() -> Optional[int]:
    v = os.environ.get("TRAIN_STAGE", "").strip()
    if not v:
        return None
    try:
        return int(v) or None
    except ValueError:
        return None


def episode_turn_reward_seq(trajectory, task: Dict[str, Any],
                            gold_observations=None) -> Dict[str, Any]:
    res = execution_aware_v3_2_dense(
        trajectory, task, gold_observations, train_stage=_env_train_stage())
    r_seq = res.diagnostics.get("turn_scores") or [0.0] * len(trajectory.turns)
    return {
        "r_seq": [float(x) for x in r_seq],
        "episode_reward": float(res.reward),
        "diagnostics": res.diagnostics,
    }


episode_turn_reward_seq.reward_policy = REWARD_POLICY_NAME  # type: ignore[attr-defined]
