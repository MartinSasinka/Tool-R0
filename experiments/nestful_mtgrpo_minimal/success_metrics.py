"""Explicit success flags for P43 training/eval logging.

Definitions (do not alias continuous reward):

* ``tool_trace_success`` — trajectory executed tools without parse/exec failure
  and completed required semantic subgoals when available (else: all executed
  tool calls succeeded and count ≥ 1).
* ``full_sequence_match`` — grounded canonical gold sequence match (NESTFUL-style
  full sequence), or strict gold-trace success when that is already in diag.
* ``final_answer_present`` — model emitted terminal ``[]`` (or stop_reason
  terminal_answer).
* ``final_answer_correct`` — final_answer_present AND last tool observation
  matches gold (protocol: ``[]`` only when latest obs IS the answer).
* ``official_win`` — NESTFUL-compatible internal win from ``metrics``:
  executable trajectory AND final observation matches gold
  (same criterion as ``metrics.compute_nestful_style_scores`` / paper internal
  win). This is **not** ``reward >= threshold``. At corpus eval time, prefer
  ``nestful_official_score.calculate_win_score`` when IBM funcs are available.

``win_rate`` in trainers MUST be ``mean(official_win)``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def _matches_gold(obs: Any, gold: Any) -> bool:
    try:
        from executor import matches_gold
        return bool(matches_gold(obs, gold))
    except Exception:
        return obs == gold


def compute_rollout_success_flags(
    trajectory,
    task: Dict[str, Any],
    diag: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    diag = dict(diag or {})
    gold = task.get("gold_answer")
    # DP pool parent-side traj (_PoolTraj) has no turns/observations — prefer
    # worker reward_diag / interaction_meta fields already merged into ``diag``.
    meta = getattr(trajectory, "interaction_meta", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    if not meta:
        meta = {k: diag[k] for k in (
            "final_answer_present", "final_response_turn_attempted",
            "final_turn_tool_attempt", "tool_budget", "assistant_turn_budget",
            "tool_calls_executed", "assistant_turns", "tool_calls_emitted",
        ) if k in diag}

    turns = getattr(trajectory, "turns", None)
    stop_reason = getattr(trajectory, "stop_reason", None) or diag.get("stop_reason")

    final_present = bool(
        meta.get("final_answer_present")
        or diag.get("final_answer_present")
        or (turns is not None and any(
            getattr(t, "is_terminal", False) for t in turns))
        or stop_reason in ("terminal", "terminal_answer")
    )

    parse_fail = stop_reason == "parse_fail"
    exec_fail = bool(
        getattr(trajectory, "executor_error", False)
        or diag.get("executor_error")
        or int(diag.get("execfail_total") or 0) > 0
    )
    clipped = bool(getattr(trajectory, "clipped_any", False))
    zero = bool(getattr(trajectory, "zero_tool_calls", False)
                or diag.get("zero_tool_calls"))

    # Prefer reward diag semantic completeness when present (p43).
    if "required_subgoals_completed" in diag and "required_subgoals_total" in diag:
        total = int(diag.get("required_subgoals_total") or 0)
        done = int(diag.get("required_subgoals_completed") or 0)
        tool_trace_success = (not parse_fail and not exec_fail and not zero
                              and total > 0 and done >= total)
    elif turns is not None:
        n_ok = sum(
            1 for t in turns
            if t.parsed_call is not None and not (t.fail_reason or "").startswith("exec:")
            and not (t.fail_reason or "").startswith("parse:")
            and not t.is_terminal
        )
        tool_trace_success = (not parse_fail and not exec_fail and not zero
                              and n_ok >= int(task.get("num_calls")
                                              or len(task.get("gold_calls") or [])
                                              or 1))
    else:
        n_calls = int(getattr(trajectory, "num_tool_calls", 0)
                      or diag.get("tool_calls_executed")
                      or meta.get("tool_calls_executed") or 0)
        need = int(task.get("num_calls") or len(task.get("gold_calls") or []) or 1)
        tool_trace_success = (not parse_fail and not exec_fail and not zero
                              and n_calls >= need)

    if "full_sequence_match" in diag and isinstance(diag["full_sequence_match"], bool):
        full_sequence_match = bool(diag["full_sequence_match"])
    elif diag.get("strict_gold_trace_success") is not None:
        full_sequence_match = bool(diag.get("strict_gold_trace_success"))
    elif hasattr(trajectory, "predicted_calls") and turns is not None:
        try:
            from metrics import compute_nestful_official_metrics
            sc = compute_nestful_official_metrics(
                trajectory.predicted_calls, task.get("gold_calls") or [],
                trajectory=trajectory, task=task)
            full_sequence_match = float(sc.get("full_sequence_accuracy") or 0.0) >= 1.0 - 1e-9
        except Exception:
            full_sequence_match = False
    else:
        full_sequence_match = False

    final_obs = getattr(trajectory, "final_observation", None)
    if final_obs is not None:
        obs_match = _matches_gold(final_obs, gold)
    else:
        # Pool path: worker already scored answer match into diag.
        obs_match = bool(
            diag.get("final_answer_pass")
            or diag.get("tool_final_answer_pass")
            or diag.get("terminal_success")
        )
    final_answer_correct = bool(final_present and obs_match)

    executable = (not parse_fail and not exec_fail and not zero
                  and stop_reason not in ("parse_fail",))
    # Internal NESTFUL-compatible win (metrics.py). Not reward-threshold.
    official_win = bool(executable and obs_match and not clipped)

    # Boolean strict pass (never a continuous reward float).
    if "strict_gold_trace_success" in diag:
        strict_pass = bool(diag.get("strict_gold_trace_success"))
    else:
        v = diag.get("strict_gold_trace_pass")
        if isinstance(v, bool):
            strict_pass = v
        elif isinstance(v, (int, float)) and float(v) in (0.0, 1.0):
            strict_pass = bool(int(v))
        else:
            strict_pass = False

    return {
        "tool_trace_success": bool(tool_trace_success),
        "full_sequence_match": bool(full_sequence_match),
        "final_answer_present": bool(final_present),
        "final_answer_correct": bool(final_answer_correct),
        "official_win": bool(official_win),
        "terminal_success": bool(final_answer_correct),
        "strict_gold_trace_pass": bool(strict_pass),
        "strict_gold_trace_success": bool(strict_pass),
        "final_response_turn_attempted": bool(
            meta.get("final_response_turn_attempted", False)),
        "final_turn_tool_attempt": bool(meta.get("final_turn_tool_attempt", False)),
        "tool_budget": int(meta.get("tool_budget")
                           or getattr(trajectory, "tool_budget", 0) or 0),
        "assistant_turn_budget": int(
            meta.get("assistant_turn_budget")
            or getattr(trajectory, "assistant_turn_budget", 0) or 0),
        "tool_calls_executed": int(meta.get("tool_calls_executed")
                                   or getattr(trajectory, "num_tool_calls", 0)
                                   or 0),
        "assistant_turns": int(meta.get("assistant_turns") or 0),
        "stop_reason": stop_reason,
        "win_rate_definition": "mean(official_win); official_win=executable AND "
                               "final_observation matches gold (NESTFUL-internal)",
    }
