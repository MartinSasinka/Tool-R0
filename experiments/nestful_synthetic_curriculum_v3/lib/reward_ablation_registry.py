"""Centralized reward registry for the reward-only ablation
(reports/reward_ablation/ABLATION_PLAN.md).

Five arms, one shared unified terminal taxonomy, one output contract:

    {
      "reward_id": str,
      "terminal_class": str,
      "terminal_score": float,
      "process_score": float,
      "epsilon": float,
      "total_reward": float,
      "components": dict,
    }

Unified terminal taxonomy (best -> worst; TESTED in
tests/test_reward_ablation.py::test_terminal_ordering):

    official_success > executable_wrong_result > executable_partial
        > execution_failure > parse_or_no_call

"official_success" means: the NESTFUL official scorer's `official_win`
when evaluating on real NESTFUL tasks (row/official_win given), OR — when
no official scorer is available (synthetic curriculum TRAINING tasks) —
the deterministic, executor-verified path-invariant success check
`nestful_core.rewards.tool_final_answer_pass` against the task's own gold
answer. Both are "the trajectory is verifiably correct end-to-end"; which
one applies is a function of what ground truth exists for that task, not a
free choice.

Arms
----
A0_R0_CURRENT          — current production reward, UNCHANGED
                         (`lib/reward_v3_2_dense.execution_aware_v3_2_dense`).
                         Its own 11-way internal taxonomy/bands are frozen
                         and not touched here; `terminal_class` below is
                         only a read-only PROJECTION into the unified
                         5-class taxonomy for cross-arm reporting.
A1_OUTCOME_ONLY        — path-invariant terminal outcome only. No process
                         tie-break (epsilon = 0). Bands = the already
                         audited `OUTCOME_BANDS_R1` from
                         `lib/reward_variants_offline.py` (imported, not
                         copied).
A2_R3_OUTCOME_FIRST    — the already-implemented-and-audited R3: outcome
                         bands = `OUTCOME_BANDS_R3`, process tie-break =
                         `_process_score_no_length` (gold-trace-aware; this
                         is the ONE arm allowed to use gold-similarity in
                         its process component, because that is literally
                         what R3 is and was already audited as such),
                         epsilon = `DEFAULT_EPS_R3` — all imported verbatim.
A3_VERIFIABLE_PROCESS  — same terminal bands as A2, but the process
                         tie-break is `verifiable_process_reward` (NO gold
                         comparison: parse validity, tool existence, schema
                         validity, type/range validity, reference
                         resolvability, execution success, execution
                         integrity/state-transition validity).
A4_GATED_VERIFIABLE    — same as A3, but the process tie-break is gated to
                         0 unless the trajectory is fully executable (no
                         parse/no-call failure, no fully-failed execution).

Hard invariant (tested): for every arm, no process contribution can lift a
lower terminal class above a higher one:

    epsilon * (P_max - P_min) < min adjacent gap between terminal scalars

`predicted_call_count < gold_call_count` is NEVER penalized directly by
A1-A4 (no call-count / gold-length term exists in any of their formulas).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_EXP = Path(__file__).resolve().parents[2]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from lib.reward_v3_1 import RewardResult, _emitted_calls, _predicates, detect_stage, expected_calls  # noqa: E402
from lib.reward_v3_2_dense import execution_aware_v3_2_dense  # noqa: E402
from lib.reward_variants_offline import (  # noqa: E402
    DEFAULT_EPS_R2,
    DEFAULT_EPS_R3,
    OUTCOME_BANDS_R1,
    OUTCOME_BANDS_R3,
    PROCESS_WEIGHTS,
    _process_score_no_length,
)
from lib.verifiable_process_reward import (  # noqa: E402
    gate_open,
    verifiable_process_components,
    verifiable_process_score,
)

# ─────────────────────────────────────────────────────────────────────────
# Unified terminal taxonomy
# ─────────────────────────────────────────────────────────────────────────

TERMINAL_CLASSES: tuple = (
    "official_success",
    "executable_wrong_result",
    "executable_partial",
    "execution_failure",
    "parse_or_no_call",
)
TERMINAL_RANK: Dict[str, int] = {c: i for i, c in enumerate(TERMINAL_CLASSES)}

ARM_IDS: tuple = (
    "A0_R0_CURRENT",
    "A1_OUTCOME_ONLY",
    "A2_R3_OUTCOME_FIRST",
    "A3_VERIFIABLE_PROCESS",
    "A4_GATED_VERIFIABLE",
)

ARM_LABELS: Dict[str, str] = {
    "A0_R0_CURRENT": "Current production reward (execution_aware_v3_2_dense) — control",
    "A1_OUTCOME_ONLY": "Path-invariant terminal outcome only, no process tie-break",
    "A2_R3_OUTCOME_FIRST": "Outcome-first + epsilon gold-aware process tie-break (= R3, audited)",
    "A3_VERIFIABLE_PROCESS": "Outcome-first + epsilon deterministically-verifiable process tie-break",
    "A4_GATED_VERIFIABLE": "Same as A3, process tie-break gated to fully-executable trajectories only",
}


def _mid(band: tuple) -> float:
    lo, hi = band
    return round((lo + hi) / 2.0, 6)


# Terminal scalars derived FROM the already-audited bands (not hand copied
# numbers): A1 <- OUTCOME_BANDS_R1, A2/A3/A4 <- OUTCOME_BANDS_R3.
_R1_OFFICIAL = _mid(OUTCOME_BANDS_R1["official_success"])
_R1_EXEC_WRONG = _mid(OUTCOME_BANDS_R1["executable_wrong_outcome"])
_R1_NON_EXEC = _mid(OUTCOME_BANDS_R1["non_executable_failure"])
_R1_PARSE = _mid(OUTCOME_BANDS_R1["parse_or_no_tool"])

_R3_OFFICIAL = _mid(OUTCOME_BANDS_R3["official_success"])
_R3_EXEC_WRONG = _mid(OUTCOME_BANDS_R3["executable_wrong_outcome"])
_R3_NON_EXEC = _mid(OUTCOME_BANDS_R3["non_executable_failure"])
_R3_PARSE = _mid(OUTCOME_BANDS_R3["parse_or_no_tool"])

TERMINAL_SCALARS: Dict[str, Dict[str, float]] = {
    "A1_OUTCOME_ONLY": {
        "official_success": _R1_OFFICIAL,
        "executable_wrong_result": _R1_EXEC_WRONG,
        # R1/R3 only distinguish 4 classes; "partial" is interpolated
        # strictly between wrong-result and full failure so unified
        # ordering still holds without inventing a new independent number.
        "executable_partial": round((_R1_EXEC_WRONG + _R1_NON_EXEC) / 2.0, 6),
        "execution_failure": _R1_NON_EXEC,
        "parse_or_no_call": _R1_PARSE,
    },
    "A2_R3_OUTCOME_FIRST": {
        "official_success": _R3_OFFICIAL,
        "executable_wrong_result": _R3_EXEC_WRONG,
        "executable_partial": round((_R3_EXEC_WRONG + _R3_NON_EXEC) / 2.0, 6),
        "execution_failure": _R3_NON_EXEC,
        "parse_or_no_call": _R3_PARSE,
    },
}
TERMINAL_SCALARS["A3_VERIFIABLE_PROCESS"] = dict(TERMINAL_SCALARS["A2_R3_OUTCOME_FIRST"])
TERMINAL_SCALARS["A4_GATED_VERIFIABLE"] = dict(TERMINAL_SCALARS["A2_R3_OUTCOME_FIRST"])

EPSILONS: Dict[str, float] = {
    "A1_OUTCOME_ONLY": 0.0,
    # NOTE: R3's own audited epsilon (DEFAULT_EPS_R3=0.04) was calibrated for
    # R3's original 4-class taxonomy. Extending R3's bands with a 5th
    # ("executable_partial") class for the unified taxonomy shrinks the
    # tightest adjacent gap to ~0.0425 (see min_adjacent_gap below), which
    # leaves DEFAULT_EPS_R3 with almost no safety margin. We therefore use a
    # smaller epsilon here — still "a small process tie-breaker" per the
    # ablation spec — chosen so `verify_epsilon_safety` holds with margin.
    # The terminal BANDS themselves (OUTCOME_BANDS_R3) are untouched/reused
    # verbatim; only this epsilon differs from `DEFAULT_EPS_R3`.
    "A2_R3_OUTCOME_FIRST": min(DEFAULT_EPS_R3, 0.02),
    "A3_VERIFIABLE_PROCESS": min(DEFAULT_EPS_R3, 0.02),
    "A4_GATED_VERIFIABLE": min(DEFAULT_EPS_R3, 0.02),
}


def min_adjacent_gap(scalars: Dict[str, float]) -> float:
    ordered = [scalars[c] for c in TERMINAL_CLASSES]
    return min(ordered[i] - ordered[i + 1] for i in range(len(ordered) - 1))


def verify_epsilon_safety(arm_id: str) -> bool:
    """epsilon * (P_max - P_min) < smallest gap between adjacent terminal
    scalars. P is normalized to [0, 1], so P_max - P_min == 1."""
    if arm_id not in TERMINAL_SCALARS:
        return True  # A0 uses its own frozen v3.2 bands, out of scope here
    gap = min_adjacent_gap(TERMINAL_SCALARS[arm_id])
    return EPSILONS[arm_id] * 1.0 < gap


for _arm in ("A1_OUTCOME_ONLY", "A2_R3_OUTCOME_FIRST", "A3_VERIFIABLE_PROCESS", "A4_GATED_VERIFIABLE"):
    assert verify_epsilon_safety(_arm), f"epsilon-band-safety violated for {_arm}"
    assert list(TERMINAL_SCALARS[_arm].values()) == sorted(TERMINAL_SCALARS[_arm].values(), reverse=True), (
        f"terminal scalars not strictly ordered for {_arm}"
    )


@dataclass
class ArmScore:
    reward_id: str
    terminal_class: str
    terminal_score: float
    process_score: float
    epsilon: float
    total_reward: float
    components: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward_id": self.reward_id,
            "terminal_class": self.terminal_class,
            "terminal_score": self.terminal_score,
            "process_score": self.process_score,
            "epsilon": self.epsilon,
            "total_reward": self.total_reward,
            "components": self.components,
        }


def _official_win_from_row(row: Optional[dict]) -> Optional[bool]:
    if row is None:
        return None
    v = row.get("official_win")
    if v is None:
        v = (row.get("_traj") or {}).get("official_win")
    return None if v is None else bool(v)


def unified_terminal_class(pred: Dict[str, Any], is_success: bool) -> str:
    """Classify into the 5-class unified taxonomy from gold-free predicates.

    `is_success` is the outcome ground truth: NESTFUL `official_win` when
    available (eval), else the synthetic path-invariant `final_pass` check
    (training) — see module docstring.
    """
    if is_success:
        return "official_success"
    if pred["parse_err"] or pred["clipped"] or pred["no_tool"]:
        return "parse_or_no_call"
    ef = float(pred["executable_frac"])
    if pred["is_executable"] and ef >= 0.999:
        return "executable_wrong_result"
    if ef > 0.0:
        return "executable_partial"
    return "execution_failure"


def _is_success(trajectory, task: Dict[str, Any], pred: Dict[str, Any], official_win: Optional[bool]) -> bool:
    if official_win is not None:
        return bool(official_win)
    return bool(pred["final_pass"])


def _v0_to_unified(v0_class: str) -> str:
    """Project the frozen v3.2 11-way class onto the unified 5-class
    taxonomy, for reporting only. Does not change A0's actual reward."""
    if v0_class == "fully_correct":
        return "official_success"
    if v0_class == "executable_wrong_final":
        return "executable_wrong_result"
    if v0_class in ("too_many_calls", "correct_tool_wrong_args", "partial_progress"):
        return "executable_partial"
    if v0_class in ("too_few_calls", "wrong_tool", "invalid_reference"):
        return "execution_failure"
    return "parse_or_no_call"  # parse_error, clipped, no_tool_call, premature_final_nonterminal


def score_arm(
    arm_id: str,
    trajectory,
    task: Dict[str, Any],
    *,
    official_win: Optional[bool] = None,
    train_stage: Optional[int] = 3,
) -> ArmScore:
    if arm_id not in ARM_IDS:
        raise ValueError(f"Unknown reward ablation arm: {arm_id!r}. Known: {ARM_IDS}")

    os.environ.setdefault("TRAIN_STAGE", str(train_stage or 3))

    if arm_id == "A0_R0_CURRENT":
        r0 = execution_aware_v3_2_dense(trajectory, task, train_stage=train_stage)
        v0_class = str(r0.diagnostics.get("reward_class") or "")
        return ArmScore(
            reward_id=arm_id,
            terminal_class=_v0_to_unified(v0_class),
            terminal_score=float(r0.reward),
            process_score=float(r0.diagnostics.get("quality_score") or 0.0),
            epsilon=0.0,
            total_reward=float(r0.reward),
            components={"v0_reward_class": v0_class, **{
                k: r0.diagnostics.get(k) for k in (
                    "format_score", "call_count_progress", "per_call_tool_score",
                    "per_call_argument_score", "reference_score", "execution_score",
                    "final_answer_score",
                )
            }},
        )

    pred = _predicates(trajectory, task)
    is_success = _is_success(trajectory, task, pred, official_win)
    terminal = unified_terminal_class(pred, is_success)
    terminal_scalar = TERMINAL_SCALARS[arm_id][terminal]
    epsilon = EPSILONS[arm_id]

    if arm_id == "A1_OUTCOME_ONLY":
        process_score = 0.0
        components: Dict[str, Any] = {"note": "no process tie-break"}
    elif arm_id == "A2_R3_OUTCOME_FIRST":
        gold_calls = list(task.get("gold_calls") or [])
        gold_n = expected_calls(detect_stage(task, train_stage), task)
        calls_info = _per_call_analysis_safe(trajectory, gold_calls)
        proc = _process_score_no_length(pred, calls_info, gold_n, gold_calls, trajectory)
        process_score = float(proc["process_score"])
        components = proc
    else:  # A3 / A4 verifiable
        comps = verifiable_process_components(trajectory, task, pred)
        raw_score = verifiable_process_score(comps)
        if arm_id == "A4_GATED_VERIFIABLE" and not gate_open(pred):
            process_score = 0.0
            comps = {**comps, "gate_open": False}
        else:
            process_score = raw_score
            comps = {**comps, "gate_open": True}
        components = comps

    total = round(terminal_scalar + epsilon * process_score, 6)

    return ArmScore(
        reward_id=arm_id,
        terminal_class=terminal,
        terminal_score=terminal_scalar,
        process_score=round(process_score, 6),
        epsilon=epsilon,
        total_reward=total,
        components=components,
    )


def _per_call_analysis_safe(trajectory, gold_calls):
    from lib.reward_v3_1 import _per_call_analysis  # local import to avoid cycle at module load
    return _per_call_analysis(trajectory, gold_calls)


# ─────────────────────────────────────────────────────────────────────────
# Training adapter: episode_turn_reward_seq contract expected by
# vllm_dp_pool.resolve_reward_info / grpo_train.py. r_seq is sparse
# (all zero) for A1-A4 so the ONLY experimental variable is the terminal
# scalar the trainer sees, and turn-level credit assignment (`_turn_returns`
# + `compute_group_stats`) is byte-for-byte identical across arms. A0 keeps
# its existing dense per-turn shaping (`_turn_scores`), unchanged.
# ─────────────────────────────────────────────────────────────────────────

def make_episode_turn_reward_seq(arm_id: str) -> Callable[..., Dict[str, Any]]:
    if arm_id == "A0_R0_CURRENT":
        from lib.reward_v3_2_dense import episode_turn_reward_seq as _r0_seq
        return _r0_seq

    def _fn(trajectory, task: Dict[str, Any], gold_observations=None) -> Dict[str, Any]:
        score = score_arm(arm_id, trajectory, task, official_win=None)
        n_turns = len(getattr(trajectory, "turns", []) or [])
        return {
            "r_seq": [0.0] * n_turns,
            "episode_reward": float(score.total_reward),
            "diagnostics": {
                "reward_type": f"reward_ablation_{arm_id}",
                "reward_policy": f"reward_ablation_{arm_id}",
                **score.to_dict(),
            },
        }

    _fn.reward_policy = f"reward_ablation_{arm_id}"  # type: ignore[attr-defined]
    _fn.__name__ = f"episode_turn_reward_seq_{arm_id}"
    return _fn
