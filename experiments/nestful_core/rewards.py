"""Unified reward registry: explicit predicates + legacy policies + v2 rewards.

This is the only genuinely new logic in ``nestful_core``. It hosts:

* Explicit trajectory predicates (``has_parse_error``, ``has_invalid_reference`` …)
  that classify a trajectory DIRECTLY rather than inferring failures from
  observations. These are the single source of truth for the v2 rewards, the v2
  unit tests and the component logging.
* LEGACY policies (``strict_gold_trace_legacy``, ``partial_gold_trace_legacy``,
  ``execution_aware_v1_legacy``) that DELEGATE to the frozen
  ``reward.py`` / ``partial_reward.py`` / ``execution_reward.py`` modules, so the
  original numbers reproduce bit-for-bit.
* NEW policies ``partial_gold_trace_v2`` and ``execution_aware_v2`` with fixed
  edge cases and full per-component breakdowns.
* A registry (``get_episode_reward`` / ``get_episode_reward_seq``) so the trainer
  and DP-pool workers can pick a policy by name.

Nothing here uses ``solution_equivalent`` / Win Rate / any eval-only signal:
every reward is derived ONLY from the gold trace, the gold answer, and the
model's OWN executed trajectory.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from . import ensure_paths

ensure_paths()

# Frozen legacy implementations (single source for delegation).
from reward import (  # noqa: E402
    RewardResult,
    _turn_executed_cleanly,
    strict_gold_trace_reward,
    episode_turn_reward_seq as _strict_seq,
)
from partial_reward import (  # noqa: E402
    _turn_component_scores,
    partial_gold_trace_reward,
    episode_turn_reward_seq as _partial_seq,
)
from execution_reward import (  # noqa: E402
    execution_aware_reward as _execution_aware_v1,
    episode_turn_reward_seq as _execution_v1_seq,
)
from executor import matches_gold  # noqa: E402

_VAR_REF_RE = re.compile(r"^\$([A-Za-z_][\w]*)(?:\.([A-Za-z_][\w]*))?\$$")


# ===========================================================================
#  Explicit trajectory predicates (single source of truth)
# ===========================================================================

def has_parse_error(traj) -> bool:
    """True if ANY turn failed to parse a valid tool call."""
    return any(t.fail_reason and t.fail_reason.startswith("parse:") for t in traj.turns)


def has_no_tool_call(traj) -> bool:
    """True if the trajectory emitted ZERO parseable tool calls."""
    return num_emitted_calls(traj) == 0


def num_emitted_calls(traj) -> int:
    """Turns that emitted a parseable tool call (regardless of execution)."""
    return sum(1 for t in traj.turns if t.parsed_call is not None)


def num_successful_calls(traj) -> int:
    """Turns that emitted a parsed call which executed WITHOUT error."""
    return sum(1 for t in traj.turns if t.parsed_call is not None and t.fail_reason is None)


def terminal_before_first_successful_tool(traj) -> bool:
    """True if the episode ended on a terminal ``[]`` before any clean tool call.

    Covers both "[] on the very first turn" and "emitted call(s) that all errored,
    then gave up": in every such case no tool ever produced a usable observation.
    """
    if num_successful_calls(traj) > 0:
        return False
    # An explicit terminal turn, or simply never having a successful call while
    # the episode stopped for a non-clipped/non-parse reason.
    return any(t.is_terminal for t in traj.turns) or traj.stop_reason == "terminal"


def has_executor_error(traj) -> bool:
    """True if any emitted call raised an executor error (``exec:``)."""
    return any(t.fail_reason and t.fail_reason.startswith("exec:") for t in traj.turns)


def has_invalid_reference(traj) -> bool:
    """True if a variable reference pointed to a non-existent earlier output.

    Detected two ways (explicitly, not just via observation):
      1. an executor error of the ``unresolved_variable`` family, OR
      2. a static scan: a ``$varN…$`` arg whose label was never produced by an
         earlier call in this trajectory.
    """
    for t in traj.turns:
        if t.fail_reason and "unresolved_variable" in t.fail_reason:
            return True
    seen: set = set()
    for idx, t in enumerate(traj.turns):
        call = t.parsed_call
        if call is None:
            continue
        for val in (call.get("arguments") or {}).values():
            if isinstance(val, str):
                m = _VAR_REF_RE.match(val.strip())
                if m and m.group(1) not in seen:
                    return True
        label = call.get("label") or f"$var{idx + 1}"
        seen.add(label.lstrip("$"))
        seen.add(label)
    return False


def valid_references_fraction(traj) -> Optional[float]:
    """Fraction of variable references resolving to an earlier label.

    Returns None when the trajectory has NO references (caller awards full credit).
    """
    seen: set = set()
    total = 0
    valid = 0
    for idx, t in enumerate(traj.turns):
        call = t.parsed_call
        if call is None:
            continue
        for val in (call.get("arguments") or {}).values():
            if isinstance(val, str):
                m = _VAR_REF_RE.match(val.strip())
                if m:
                    total += 1
                    if m.group(1) in seen:
                        valid += 1
        label = call.get("label") or f"$var{idx + 1}"
        seen.add(label.lstrip("$"))
        seen.add(label)
    if total == 0:
        return None
    return valid / total


def executable_fraction(traj) -> float:
    """Fraction of EMITTED calls that executed cleanly (0 when none emitted)."""
    emitted = [t for t in traj.turns if t.parsed_call is not None]
    if not emitted:
        return 0.0
    return sum(1 for t in emitted if t.fail_reason is None) / len(emitted)


def is_executable_trajectory(traj) -> bool:
    """True if at least one call ran and EVERY emitted call executed cleanly.

    Requires: not clipped, no parse error, >=1 successful call, and no emitted
    call errored (executable_fraction == 1.0).
    """
    if traj.clipped_any or has_parse_error(traj):
        return False
    if num_successful_calls(traj) < 1:
        return False
    return executable_fraction(traj) >= 1.0 - 1e-9


def tool_final_answer_pass(traj, task: Dict[str, Any]) -> bool:
    """Final observation matches the gold answer AND came from a tool call."""
    if num_successful_calls(traj) < 1:
        return False
    return bool(matches_gold(traj.final_observation, task.get("gold_answer")))


def tool_use_completeness(traj, task: Dict[str, Any]) -> float:
    """min(successful_calls, gold_n) / gold_n — penalises finishing too early.

    This is NOT a reward for longer trajectories: it saturates at 1.0 once the
    model has made as many successful calls as the gold trace needs.
    """
    gold_n = len(task.get("gold_calls", []))
    if gold_n <= 0:
        return 0.0
    return min(num_successful_calls(traj), gold_n) / gold_n


def gold_trace_progress(traj, task: Dict[str, Any], gold_observations=None) -> float:
    """Mean graded gold-trace turn score (weak auxiliary signal in [0, 1])."""
    gold_n = len(task.get("gold_calls", []))
    if gold_n == 0:
        return 0.0
    scores, _ = _turn_component_scores(traj, task, gold_observations)
    return sum(scores) / gold_n


def too_few_calls(traj, task: Dict[str, Any]) -> bool:
    gold_n = len(task.get("gold_calls", []))
    return gold_n > 0 and num_successful_calls(traj) < gold_n


def num_extra_calls(traj, task: Dict[str, Any]) -> int:
    gold_n = len(task.get("gold_calls", []))
    return max(0, num_emitted_calls(traj) - gold_n)


# ===========================================================================
#  execution_aware_v2 — new primary training reward
# ===========================================================================

# Hardened v2.1 defaults (anti trace-drift). Rationale: the previous weights let
# a CORRECT final answer reached via the model's OWN shorter (non-gold) trace earn
# ~0.85+ (w_final 0.55 dominated + a high unconditional floor), which decoupled the
# reward from the gold trace. NESTFUL Win Rate then collapsed while the model
# learned to emit fewer, non-gold calls. We lower w_final, raise w_gold_trace /
# w_completeness, gate the floor on real gold-trace match, add an explicit cap for
# "correct answer but drifted trace", and penalize too-short traces.
_V2_EXEC_DEFAULTS: Dict[str, float] = {
    "w_final": 0.35,
    "w_executable": 0.15,
    "w_completeness": 0.20,
    "w_references": 0.10,
    "w_gold_trace": 0.20,
    # caps
    "cap_executor_error": 0.25,
    "cap_invalid_reference": 0.30,
    "cap_not_executable": 0.25,
    "cap_too_few_wrong": 0.25,
    "cap_not_final": 0.35,
    # Correct final answer but the trace does NOT follow gold => trace drift. When
    # gold_trace_progress < gold_trace_min_for_final, cap the reward so a lucky
    # short path can never beat a faithful gold-aligned trajectory.
    "cap_final_no_gold_trace": 0.55,
    "gold_trace_min_for_final": 0.50,
    # Floor for a fully-executable correct answer, now GATED on real gold-trace
    # progress (was an unconditional 0.85 — the main gaming vector).
    "floor_executable_final": 0.70,
    "floor_gold_trace_min": 0.75,
    # mild penalty per extra call when the final answer IS correct
    "extra_call_penalty": 0.02,
    "extra_call_penalty_max": 0.10,
    # penalty per MISSING call vs gold (discourages too-short/collapsed traces),
    # applied regardless of final_pass to fight trace drift toward 1-2 calls.
    "short_trace_penalty": 0.10,
    "short_trace_penalty_max": 0.40,
}

_V2_EXEC_WEIGHTS: Dict[str, float] = dict(_V2_EXEC_DEFAULTS)


def set_execution_v2_weights_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    """Load execution_aware_v2 weights from ``config['execution_reward_v2']``."""
    global _V2_EXEC_WEIGHTS
    _V2_EXEC_WEIGHTS = dict(_V2_EXEC_DEFAULTS)
    block = (config or {}).get("execution_reward_v2", {}) or {}
    _V2_EXEC_WEIGHTS.update({k: float(v) for k, v in block.items() if k in _V2_EXEC_DEFAULTS})
    print(f"[execution_aware_v2] weights = {_V2_EXEC_WEIGHTS}", flush=True)
    return dict(_V2_EXEC_WEIGHTS)


def get_execution_v2_weights() -> Dict[str, float]:
    return dict(_V2_EXEC_WEIGHTS)


def execution_aware_v2(
    trajectory,
    task: Dict[str, Any],
    gold_observations: Optional[List[Any]] = None,
) -> RewardResult:
    """Execution-aware reward v2 in [0, 1] with explicit edge-case caps.

        R = 0.35*final + 0.15*executable + 0.20*completeness
          + 0.10*valid_refs + 0.20*gold_progress          (hardened v2.1 defaults)

    Extra-call and MISSING-call (short-trace) penalties are then applied, followed
    by the soft caps in order (incl. the anti trace-drift cap for a correct answer
    whose trace does not follow gold), and finally a gold-trace-gated floor.
    Returns a ``RewardResult`` whose ``diagnostics`` carry the FULL component
    breakdown and ``cap_applied``.
    """
    w = _V2_EXEC_WEIGHTS
    gold_n = len(task.get("gold_calls", []))

    # Predicates / components.
    parse_err = has_parse_error(trajectory)
    clipped = bool(trajectory.clipped_any)
    no_tool = has_no_tool_call(trajectory)
    term_before = terminal_before_first_successful_tool(trajectory)
    exec_err = has_executor_error(trajectory)
    invalid_ref = has_invalid_reference(trajectory)
    executable_b = is_executable_trajectory(trajectory)
    final_pass = tool_final_answer_pass(trajectory, task)
    few = too_few_calls(trajectory, task)

    exec_frac = executable_fraction(trajectory)
    completeness = tool_use_completeness(trajectory, task)
    refs = valid_references_fraction(trajectory)
    ref_component = 1.0 if refs is None else refs
    gold_prog = gold_trace_progress(trajectory, task, gold_observations)
    final_component = 1.0 if final_pass else 0.0
    extra = num_extra_calls(trajectory, task)

    diag: Dict[str, Any] = {
        "reward_type": "execution_aware_v2",
        "reward": 0.0,
        "episode_reward": 0.0,
        "tool_final_answer_pass": final_component,
        "executable_trajectory": exec_frac,
        "tool_use_completeness": completeness,
        "valid_references": ref_component,
        "valid_references_present": refs is not None,
        "small_gold_trace_progress": gold_prog,
        "parse_error": parse_err,
        "clipped": clipped,
        "no_tool_call": no_tool,
        "terminal_before_first_successful_tool": term_before,
        "too_few_calls": few,
        "invalid_reference": invalid_ref,
        "executor_error": exec_err,
        "is_executable_trajectory": executable_b,
        "num_emitted_calls": num_emitted_calls(trajectory),
        "num_successful_calls": num_successful_calls(trajectory),
        "num_extra_calls": extra,
        "stop_reason": trajectory.stop_reason,
        "cap_applied": None,
    }

    # ── Hard zero caps ─────────────────────────────────────────────────────
    if parse_err:
        diag["cap_applied"] = "parse_error"
        return _finish(0.0, diag, task, trajectory, gold_observations)
    if clipped:
        diag["cap_applied"] = "clipped"
        return _finish(0.0, diag, task, trajectory, gold_observations)
    if no_tool:
        diag["cap_applied"] = "no_tool_call"
        return _finish(0.0, diag, task, trajectory, gold_observations)
    if term_before:
        diag["cap_applied"] = "terminal_before_first_successful_tool"
        return _finish(0.0, diag, task, trajectory, gold_observations)

    R = (w["w_final"] * final_component
         + w["w_executable"] * exec_frac
         + w["w_completeness"] * completeness
         + w["w_references"] * ref_component
         + w["w_gold_trace"] * gold_prog)
    R = max(0.0, min(1.0, R))

    # Mild penalty for extra calls ONLY when the final answer is correct
    # (harmful extra calls with a wrong answer are handled by the caps below).
    if extra > 0 and final_pass:
        pen = min(w["extra_call_penalty"] * extra, w["extra_call_penalty_max"])
        if pen > 0:
            R = max(0.0, R - pen)
            diag["extra_call_penalty_applied"] = pen

    # Penalty for MISSING calls vs gold (too-short / collapsed trace). Applied
    # even when the answer is correct so the model cannot game Win with a 1-call
    # shortcut on a multi-call task.
    missing = max(0, gold_n - num_successful_calls(trajectory))
    diag["num_missing_calls"] = missing
    if missing > 0:
        spen = min(w["short_trace_penalty"] * missing, w["short_trace_penalty_max"])
        if spen > 0:
            R = max(0.0, R - spen)
            diag["short_trace_penalty_applied"] = spen

    # ── Soft caps (applied in order; most binding wins) ────────────────────
    caps_fired: List[str] = []
    if exec_err and R > w["cap_executor_error"]:
        R = w["cap_executor_error"]; caps_fired.append("executor_error")
    if invalid_ref and R > w["cap_invalid_reference"]:
        R = w["cap_invalid_reference"]; caps_fired.append("invalid_reference")
    if not executable_b and R > w["cap_not_executable"]:
        R = w["cap_not_executable"]; caps_fired.append("not_executable")
    if few and not final_pass and R > w["cap_too_few_wrong"]:
        R = w["cap_too_few_wrong"]; caps_fired.append("too_few_calls_wrong_answer")
    if not final_pass and R > w["cap_not_final"]:
        R = w["cap_not_final"]; caps_fired.append("not_final")
    # Anti trace-drift: correct answer but the trace diverges from gold.
    if final_pass and gold_prog < w["gold_trace_min_for_final"] \
            and R > w["cap_final_no_gold_trace"]:
        R = w["cap_final_no_gold_trace"]; caps_fired.append("final_no_gold_trace")

    # ── Floor: executable + correct answer AND the trace actually follows gold ─
    if executable_b and final_pass and gold_prog >= w["floor_gold_trace_min"] \
            and R < w["floor_executable_final"]:
        R = w["floor_executable_final"]
        diag["floor_applied"] = "executable_final_gold_trace"

    if caps_fired:
        diag["cap_applied"] = caps_fired[-1]
        diag["caps_fired"] = caps_fired
    return _finish(R, diag, task, trajectory, gold_observations)


def _finish(R: float, diag: Dict[str, Any], task, trajectory, gold_observations) -> RewardResult:
    diag["reward"] = float(R)
    diag["episode_reward"] = float(R)
    scores, _ = _turn_component_scores(trajectory, task, gold_observations)
    diag.setdefault("turn_rewards", scores)
    return RewardResult(float(R), diag)


# ===========================================================================
#  partial_gold_trace_v2 — fixed graded baseline (not the primary reward)
# ===========================================================================

_V2_PARTIAL_DEFAULTS: Dict[str, float] = {
    "w_name": 0.4, "w_keys": 0.3, "w_exec": 0.3,
    "w_trace": 0.7, "w_final": 0.3,
    "cap_executor_error": 0.40,
    "cap_invalid_reference": 0.40,
    "cap_not_final": 0.60,
    "cap_extra_calls_wrong": 0.50,
}
_V2_PARTIAL_WEIGHTS: Dict[str, float] = dict(_V2_PARTIAL_DEFAULTS)


def set_partial_v2_weights_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    global _V2_PARTIAL_WEIGHTS
    _V2_PARTIAL_WEIGHTS = dict(_V2_PARTIAL_DEFAULTS)
    block = (config or {}).get("partial_reward_v2", {}) or {}
    _V2_PARTIAL_WEIGHTS.update({k: float(v) for k, v in block.items() if k in _V2_PARTIAL_DEFAULTS})
    print(f"[partial_gold_trace_v2] weights = {_V2_PARTIAL_WEIGHTS}", flush=True)
    return dict(_V2_PARTIAL_WEIGHTS)


def partial_gold_trace_v2(
    trajectory,
    task: Dict[str, Any],
    gold_observations: Optional[List[Any]] = None,
) -> RewardResult:
    """Graded gold-trace reward with v2 edge-case fixes + component breakdown."""
    w = _V2_PARTIAL_WEIGHTS
    gold_n = len(task.get("gold_calls", []))
    final_pass = tool_final_answer_pass(trajectory, task)

    parse_err = has_parse_error(trajectory)
    clipped = bool(trajectory.clipped_any)
    no_tool = has_no_tool_call(trajectory)
    term_before = terminal_before_first_successful_tool(trajectory)
    exec_err = has_executor_error(trajectory)
    invalid_ref = has_invalid_reference(trajectory)
    extra = num_extra_calls(trajectory, task)

    scores, components = _turn_component_scores(trajectory, task, gold_observations)
    trace_score = (sum(scores) / gold_n) if gold_n else 0.0

    diag: Dict[str, Any] = {
        "reward_type": "partial_gold_trace_v2",
        "reward": 0.0,
        "episode_reward": 0.0,
        "final_answer_pass": final_pass,
        "trace_score": trace_score,
        "turn_rewards": scores,
        "turn_components": components,
        "parse_error": parse_err,
        "clipped": clipped,
        "no_tool_call": no_tool,
        "terminal_before_first_successful_tool": term_before,
        "executor_error": exec_err,
        "invalid_reference": invalid_ref,
        "num_extra_calls": extra,
        "stop_reason": trajectory.stop_reason,
        "cap_applied": None,
    }

    if parse_err:
        diag["cap_applied"] = "parse_error"; diag["reward"] = diag["episode_reward"] = 0.0
        return RewardResult(0.0, diag)
    if clipped:
        diag["cap_applied"] = "clipped"; diag["reward"] = diag["episode_reward"] = 0.0
        return RewardResult(0.0, diag)
    if no_tool:
        diag["cap_applied"] = "no_tool_call"; diag["reward"] = diag["episode_reward"] = 0.0
        return RewardResult(0.0, diag)
    if term_before:
        diag["cap_applied"] = "terminal_before_first_successful_tool"
        diag["reward"] = diag["episode_reward"] = 0.0
        return RewardResult(0.0, diag)

    R = w["w_trace"] * trace_score + w["w_final"] * (1.0 if final_pass else 0.0)
    R = max(0.0, min(1.0, R))

    caps_fired: List[str] = []
    if exec_err and R > w["cap_executor_error"]:
        R = w["cap_executor_error"]; caps_fired.append("executor_error")
    if invalid_ref and R > w["cap_invalid_reference"]:
        R = w["cap_invalid_reference"]; caps_fired.append("invalid_reference")
    if not final_pass and R > w["cap_not_final"]:
        R = w["cap_not_final"]; caps_fired.append("not_final")
    if extra > 0 and not final_pass and R > w["cap_extra_calls_wrong"]:
        R = w["cap_extra_calls_wrong"]; caps_fired.append("extra_calls_wrong_answer")

    if caps_fired:
        diag["cap_applied"] = caps_fired[-1]
        diag["caps_fired"] = caps_fired
    diag["reward"] = diag["episode_reward"] = float(R)
    return RewardResult(float(R), diag)


# ===========================================================================
#  Legacy delegations (frozen behaviour for reproducibility)
# ===========================================================================

def strict_gold_trace_legacy(trajectory, task, gold_observations=None) -> RewardResult:
    return strict_gold_trace_reward(trajectory, task, gold_observations)


def partial_gold_trace_legacy(trajectory, task, gold_observations=None) -> RewardResult:
    return partial_gold_trace_reward(trajectory, task, gold_observations)


def execution_aware_v1_legacy(trajectory, task, gold_observations=None) -> RewardResult:
    return _execution_aware_v1(trajectory, task, gold_observations)


# ===========================================================================
#  Per-generated-turn reward sequences for the v2 policies
# ===========================================================================

def _v2_seq(reward_fn, trajectory, task, gold_observations) -> Dict[str, Any]:
    rr = reward_fn(trajectory, task, gold_observations)
    gold_turn_scores = rr.diagnostics.get("turn_rewards", [])
    r_seq: List[float] = []
    success_idx = 0
    for t in trajectory.turns:
        if t.parsed_call is not None and t.fail_reason is None:
            gold_part = (gold_turn_scores[success_idx]
                         if success_idx < len(gold_turn_scores) else 0.0)
            exec_part = 1.0 if _turn_executed_cleanly(trajectory, success_idx) else 0.0
            r_seq.append(float(0.5 * exec_part + 0.5 * gold_part))
            success_idx += 1
        else:
            r_seq.append(0.0)
    return {"r_seq": r_seq, "episode_reward": float(rr.reward), "diagnostics": rr.diagnostics}


def execution_aware_v2_seq(trajectory, task, gold_observations=None) -> Dict[str, Any]:
    return _v2_seq(execution_aware_v2, trajectory, task, gold_observations)


def partial_gold_trace_v2_seq(trajectory, task, gold_observations=None) -> Dict[str, Any]:
    return _v2_seq(partial_gold_trace_v2, trajectory, task, gold_observations)


# ===========================================================================
#  Registry
# ===========================================================================

_EPISODE_REWARD: Dict[str, Callable] = {
    "strict_gold_trace": strict_gold_trace_reward,
    "strict_gold_trace_legacy": strict_gold_trace_legacy,
    "partial_gold_trace": partial_gold_trace_reward,
    "partial_gold_trace_legacy": partial_gold_trace_legacy,
    "partial_gold_trace_v2": partial_gold_trace_v2,
    "execution_aware": _execution_aware_v1,
    "execution_aware_v1_legacy": execution_aware_v1_legacy,
    "execution_aware_v2": execution_aware_v2,
}

_EPISODE_SEQ: Dict[str, Callable] = {
    "strict_gold_trace": _strict_seq,
    "strict_gold_trace_legacy": _strict_seq,
    "partial_gold_trace": _partial_seq,
    "partial_gold_trace_legacy": _partial_seq,
    "partial_gold_trace_v2": partial_gold_trace_v2_seq,
    "execution_aware": _execution_v1_seq,
    "execution_aware_v1_legacy": _execution_v1_seq,
    "execution_aware_v2": execution_aware_v2_seq,
}


def available_policies() -> List[str]:
    return sorted(_EPISODE_REWARD)


def get_episode_reward(policy: str) -> Callable:
    key = str(policy).lower()
    if key not in _EPISODE_REWARD:
        raise KeyError(f"unknown reward policy {policy!r}; valid: {available_policies()}")
    return _EPISODE_REWARD[key]


def get_episode_reward_seq(policy: str) -> Callable:
    key = str(policy).lower()
    if key not in _EPISODE_SEQ:
        raise KeyError(f"unknown reward policy {policy!r}; valid: {available_policies()}")
    return _EPISODE_SEQ[key]


def set_weights_from_config(config: Dict[str, Any]) -> None:
    """Load v2 weights from config (both exec v2 + partial v2 blocks)."""
    set_execution_v2_weights_from_config(config)
    set_partial_v2_weights_from_config(config)
