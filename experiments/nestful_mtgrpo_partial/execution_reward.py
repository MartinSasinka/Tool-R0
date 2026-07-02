"""Execution-aware reward for MT-GRPO (v1, NEW — alongside strict/partial).

Motivation (RQ3 — execution alignment)
--------------------------------------
``reward.py`` (strict) and ``partial_reward.py`` (graded) both measure mainly
**gold-trace fidelity**: how closely the model reproduces the reference call
sequence. The official NESTFUL metric we actually care about is **Win Rate**, an
*execution* metric: does the executed tool trajectory lead to the correct answer?

This reward shifts the main signal from "looks like the gold trace" to "the tool
trajectory executes and yields the correct answer", while keeping gold-trace only
as a weak auxiliary nudge. It is a SEPARATE module — strict and partial are left
untouched, so all three are directly comparable.

Episode reward (clipped to [0, 1])
----------------------------------
    R = 0.45 * tool_final_answer_pass
      + 0.20 * executable_trajectory
      + 0.15 * tool_use_completeness
      + 0.10 * valid_references
      + 0.10 * small_gold_trace_progress

Components
  tool_final_answer_pass     final answer is correct AND was produced after at
                             least one successfully executed tool call.
  executable_trajectory      fraction of emitted tool calls that parsed, were
                             schema-valid and executed without error.
  tool_use_completeness      min(successful_calls, gold_n) / gold_n — penalises
                             finishing the ReAct trajectory too early.
  valid_references           fraction of variable references ($varN.field$) that
                             point to an output produced by an earlier call
                             (1.0 when the episode needs no references).
  small_gold_trace_progress  mean graded gold-trace turn score (reused from the
                             partial reward) — weak auxiliary signal only.

Caps against shortcuts (applied AFTER the weighted sum)
  parse_error anywhere                       -> R = 0
  no_tool_call (zero successful calls)       -> R = 0
  terminal before first successful tool      -> R = 0
  clipped rollout                            -> R = 0
  not executable (all calls errored)         -> R <= cap_not_executable (0.10)
  too_few_calls AND wrong answer             -> R <= cap_incomplete_wrong (0.50)

Nothing here uses ``solution_equivalent`` / Win Rate / any eval-only signal: the
reward is still derived ONLY from the gold trace, the gold answer and the
model's OWN executed trajectory. Evaluation stays strict + official.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Reuse validated primitives from the sibling strict reward and the partial one
# so this module can never drift on shared building blocks.
from reward import RewardResult, _turn_executed_cleanly
from partial_reward import _turn_component_scores
from executor import matches_gold

# Matches a NESTFUL variable reference, e.g. "$var1.result$" or "$var2$".
_VAR_REF_RE = re.compile(r"^\$([A-Za-z_][\w]*)(?:\.([A-Za-z_][\w]*))?\$$")


# ---------------------------------------------------------------------------
#  Weights (tunable via config["execution_reward"]; see set_weights_from_config)
# ---------------------------------------------------------------------------
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "w_final": 0.45,        # tool_final_answer_pass
    "w_executable": 0.20,   # executable_trajectory
    "w_completeness": 0.15, # tool_use_completeness
    "w_references": 0.10,   # valid_references
    "w_gold_trace": 0.10,   # small_gold_trace_progress
    # Caps against degenerate shortcuts.
    "cap_not_executable": 0.10,
    "cap_incomplete_wrong": 0.50,
}

_WEIGHTS: Dict[str, float] = dict(_DEFAULT_WEIGHTS)


def set_weights(**kwargs: float) -> None:
    """Override one or more reward weights in-place (process-global)."""
    for k, v in kwargs.items():
        if k not in _DEFAULT_WEIGHTS:
            raise KeyError(f"unknown execution_reward weight: {k!r} "
                           f"(valid: {sorted(_DEFAULT_WEIGHTS)})")
        _WEIGHTS[k] = float(v)
    _warn_if_unnormalized()


def set_weights_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    """Load weights from ``config['execution_reward']`` (missing keys keep defaults)."""
    global _WEIGHTS
    _WEIGHTS = dict(_DEFAULT_WEIGHTS)
    block = (config or {}).get("execution_reward", {}) or {}
    overrides = {k: float(v) for k, v in block.items() if k in _DEFAULT_WEIGHTS}
    _WEIGHTS.update(overrides)
    _warn_if_unnormalized()
    print(f"[execution_reward] weights = {_WEIGHTS}", flush=True)
    return dict(_WEIGHTS)


def get_weights() -> Dict[str, float]:
    return dict(_WEIGHTS)


def _warn_if_unnormalized() -> None:
    component_sum = (_WEIGHTS["w_final"] + _WEIGHTS["w_executable"]
                     + _WEIGHTS["w_completeness"] + _WEIGHTS["w_references"]
                     + _WEIGHTS["w_gold_trace"])
    if abs(component_sum - 1.0) > 1e-6:
        print(f"[execution_reward] WARNING: component weights sum to {component_sum:.3f} "
              "!= 1.0 (a perfect episode will not score exactly 1.0)", flush=True)


# ---------------------------------------------------------------------------
#  Per-episode helpers
# ---------------------------------------------------------------------------

def _any_parse_fail(trajectory) -> bool:
    return any(
        t.fail_reason and t.fail_reason.startswith("parse:") for t in trajectory.turns
    )


def _emitted_calls(trajectory) -> int:
    """Number of turns that emitted a parseable tool call (regardless of exec)."""
    return sum(1 for t in trajectory.turns if t.parsed_call is not None)


def _successful_calls(trajectory) -> int:
    """Number of turns that emitted a parsed call which executed without error."""
    return sum(
        1 for t in trajectory.turns
        if t.parsed_call is not None and t.fail_reason is None
    )


def _executable_fraction(trajectory) -> float:
    """Fraction of EMITTED tool calls (parsed attempts) that executed cleanly.

    A turn that produced a parsed call counts in the denominator; it counts in
    the numerator only if it executed without an executor error. Turns with no
    parsed call (pure parse failure / terminal) are ignored here — parse failures
    are handled separately as a hard cap.
    """
    emitted = [t for t in trajectory.turns if t.parsed_call is not None]
    if not emitted:
        return 0.0
    ok = sum(1 for t in emitted if t.fail_reason is None)
    return ok / len(emitted)


def _reference_validity(trajectory) -> Optional[float]:
    """Fraction of variable references that point to an earlier call's label.

    Returns None when the trajectory contains NO references at all (so the
    caller can award full credit — a single-call task needs no references).
    """
    seen_labels: set = set()
    total_refs = 0
    valid_refs = 0
    for idx, t in enumerate(trajectory.turns):
        call = t.parsed_call
        if call is None:
            continue
        args = call.get("arguments") or {}
        for val in args.values():
            if isinstance(val, str):
                m = _VAR_REF_RE.match(val.strip())
                if m:
                    total_refs += 1
                    if m.group(1) in seen_labels:
                        valid_refs += 1
        # Register this call's label AFTER scanning its args (a call cannot
        # reference its own output). Fall back to the positional $varN label.
        label = call.get("label") or f"$var{idx + 1}"
        norm = label.lstrip("$")
        seen_labels.add(norm)
        seen_labels.add(label)
    if total_refs == 0:
        return None
    return valid_refs / total_refs


def _gold_trace_progress(trajectory, task, gold_observations) -> float:
    """Mean graded gold-trace turn score (weak auxiliary signal)."""
    gold_calls = task.get("gold_calls", [])
    gold_n = len(gold_calls)
    if gold_n == 0:
        return 0.0
    scores, _ = _turn_component_scores(trajectory, task, gold_observations)
    return sum(scores) / gold_n


# ---------------------------------------------------------------------------
#  Episode reward
# ---------------------------------------------------------------------------

def execution_aware_reward(
    trajectory,
    task: Dict[str, Any],
    gold_observations: Optional[List[Any]] = None,
) -> RewardResult:
    """Execution-aware reward in [0, 1]. See module docstring for the formula."""
    gold_calls = task.get("gold_calls", [])
    gold_n = len(gold_calls)

    final_answer_pass = bool(matches_gold(trajectory.final_observation,
                                          task.get("gold_answer")))
    n_emitted = _emitted_calls(trajectory)
    n_success = _successful_calls(trajectory)
    parse_fail = _any_parse_fail(trajectory)
    # Gave up (terminal) after emitting calls but none ever executed cleanly.
    terminal_before_tool = (
        trajectory.stop_reason == "terminal" and n_emitted > 0 and n_success == 0
    )

    executable = _executable_fraction(trajectory)
    completeness = (min(n_success, gold_n) / gold_n) if gold_n else 0.0
    ref_validity = _reference_validity(trajectory)
    ref_component = 1.0 if ref_validity is None else ref_validity
    gold_progress = _gold_trace_progress(trajectory, task, gold_observations)
    # Final answer only counts if it was produced via executed tool calls.
    final_component = 1.0 if (final_answer_pass and n_success >= 1) else 0.0

    diag: Dict[str, Any] = {
        "reward_type": "execution_aware",
        "final_answer_pass": final_answer_pass,
        "tool_final_answer_pass": final_component,
        "executable_trajectory": executable,
        "tool_use_completeness": completeness,
        "valid_references": ref_component,
        "valid_references_present": ref_validity is not None,
        "small_gold_trace_progress": gold_progress,
        "num_emitted_calls": n_emitted,
        "num_successful_calls": n_success,
        "num_tool_calls": trajectory.num_tool_calls,
        "parse_fail": parse_fail,
        "terminal_before_tool": terminal_before_tool,
        "too_few_calls": gold_n > 0 and n_success < gold_n,
        "clipped": trajectory.clipped_any,
        "stop_reason": trajectory.stop_reason,
        "turn_rewards": [0.0] * gold_n,
        "episode_reward": 0.0,
        "cap_applied": None,
    }

    # ── Hard caps (shortcut guards), R = 0 ─────────────────────────────────
    if trajectory.clipped_any:
        diag["cap_applied"] = "clipped"
        return RewardResult(0.0, diag)
    if parse_fail:
        diag["cap_applied"] = "parse_error"
        return RewardResult(0.0, diag)
    if n_emitted == 0:
        diag["cap_applied"] = "no_tool_call"
        return RewardResult(0.0, diag)
    if terminal_before_tool:
        diag["cap_applied"] = "terminal_before_first_tool"
        return RewardResult(0.0, diag)

    reward = (_WEIGHTS["w_final"] * final_component
              + _WEIGHTS["w_executable"] * executable
              + _WEIGHTS["w_completeness"] * completeness
              + _WEIGHTS["w_references"] * ref_component
              + _WEIGHTS["w_gold_trace"] * gold_progress)
    reward = max(0.0, min(1.0, reward))

    # ── Soft caps (bound, don't zero), only recorded when they bite ────────
    if executable == 0.0 and reward > _WEIGHTS["cap_not_executable"]:
        reward = _WEIGHTS["cap_not_executable"]
        diag["cap_applied"] = "not_executable"
    elif (diag["too_few_calls"] and not final_answer_pass
          and reward > _WEIGHTS["cap_incomplete_wrong"]):
        reward = _WEIGHTS["cap_incomplete_wrong"]
        diag["cap_applied"] = "too_few_calls_wrong_answer"

    # Per-turn graded signal for MT-GRPO credit (reuse gold-trace turn scores,
    # scaled by whether the turn executed cleanly so execution matters per step).
    scores, _ = _turn_component_scores(trajectory, task, gold_observations)
    diag["turn_rewards"] = scores
    diag["episode_reward"] = reward
    return RewardResult(reward, diag)


# Alias mirroring the strict/partial module naming.
execution_aware_episode_reward = execution_aware_reward


def episode_turn_reward_seq(
    trajectory,
    task: Dict[str, Any],
    gold_observations: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Per-GENERATED-turn reward sequence + R_episode for the trainer.

    Drop-in replacement for ``reward.episode_turn_reward_seq`` (same return keys
    ``r_seq`` / ``episode_reward`` / ``diagnostics``). The s-th successful tool
    call receives a graded per-turn score that blends gold-trace credit with
    clean execution; parse-fail / terminal / clipped turns receive 0. R_episode
    is the execution-aware episode reward.
    """
    rr = execution_aware_reward(trajectory, task, gold_observations)
    gold_turn_scores = rr.diagnostics.get("turn_rewards", [])
    r_seq: List[float] = []
    success_idx = 0
    for t in trajectory.turns:
        if t.parsed_call is not None and t.fail_reason is None:
            gold_part = (gold_turn_scores[success_idx]
                         if success_idx < len(gold_turn_scores) else 0.0)
            exec_part = 1.0 if _turn_executed_cleanly(trajectory, success_idx) else 0.0
            # Blend: execution success carries the per-turn signal, gold trace nudges.
            r_seq.append(float(0.5 * exec_part + 0.5 * gold_part))
            success_idx += 1
        else:
            r_seq.append(0.0)
    return {
        "r_seq": r_seq,
        "episode_reward": float(rr.reward),
        "diagnostics": rr.diagnostics,
    }
