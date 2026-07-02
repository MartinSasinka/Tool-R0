#!/usr/bin/env python3
"""Shared reward v2 edge-case table (used by tests AND the audit CSV).

Builds synthetic trajectories covering every edge case in request §3/§4 and
records, for each, the legacy strict/partial rewards and the v2 rewards plus the
cap that fired. Run directly to (re)generate ``reward_v2_audit_cases.csv``.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.dirname(_HERE)
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)

from nestful_core import rewards as R  # noqa: E402
from nestful_core.rollout import Trajectory, Turn  # noqa: E402
from nestful_core.logging_utils import write_csv  # noqa: E402

GOLD = [
    {"name": "add", "arguments": {"arg_0": 1, "arg_1": 2}, "label": "$var1"},
    {"name": "multiply", "arguments": {"arg_0": "$var1.result$", "arg_1": 3}, "label": "$var2"},
]
GOLD_ANSWER = 9


def task() -> Dict[str, Any]:
    return {"task_id": "t", "gold_calls": GOLD, "gold_answer": GOLD_ANSWER,
            "num_calls": len(GOLD), "tools": []}


def _traj(turns_spec, final_obs, stop_reason, *, clipped=False) -> Trajectory:
    tr = Trajectory(task_id="t", stage=2, gold_num_turns=2, executor_mode="full")
    tr.clipped_any = clipped
    tr.stop_reason = stop_reason
    tr.final_observation = final_obs
    for i, (call, fail, obs) in enumerate(turns_spec):
        t = Turn(turn_idx=i, model_text="")
        t.parsed_call = call
        t.fail_reason = fail
        t.observation = obs
        if call is None and fail is None and obs is None:
            t.is_terminal = True
        if fail == "clipped_completion":
            t.clipped_completion = True
        tr.turns.append(t)
    return tr


def build_cases() -> List[Tuple[str, Trajectory, Dict[str, Any], Dict[str, Any]]]:
    """Return (name, traj, task, expectations) tuples.

    ``expectations`` keys: ``exec_max`` / ``exec_min`` bound execution_aware_v2,
    ``partial_max`` bounds partial_gold_trace_v2 where meaningful, and
    ``flags`` is a dict of diagnostic flags that must hold on the v2 exec diag.
    """
    tk = task()
    alt = [
        {"name": "sum", "arguments": {"arg_0": 1, "arg_1": 2}, "label": "$var1"},
        {"name": "prod", "arguments": {"arg_0": "$var1.result$", "arg_1": 3}, "label": "$var2"},
    ]
    bad_ref = {"name": "multiply", "arguments": {"arg_0": "$var5.result$", "arg_1": 3}, "label": "$var2"}
    wrong_name = {"name": "subtract", "arguments": {"arg_0": 1, "arg_1": 2}, "label": "$var1"}
    wrong_keys = {"name": "add", "arguments": {"x": 1, "y": 2}, "label": "$var1"}
    extra = {"name": "identity", "arguments": {"arg_0": "$var2.result$"}, "label": "$var3"}

    cases = []
    cases.append(("perfect_gold_trace",
                  _traj([(GOLD[0], None, 3), (GOLD[1], None, 9)], 9, "max_turns"),
                  tk, {"exec_min": 0.85, "flags": {"tool_final_answer_pass": 1.0}}))
    # v2.1 anti trace-drift: correct answer via non-gold tools is capped at
    # cap_final_no_gold_trace (0.55), NOT floored to 0.85 like the old reward.
    cases.append(("correct_final_answer_via_alternative_path",
                  _traj([(alt[0], None, 3), (alt[1], None, 9)], 9, "max_turns"),
                  tk, {"exec_min": 0.55, "exec_max": 0.55,
                       "flags": {"cap_applied": "final_no_gold_trace",
                                 "tool_final_answer_pass": 1.0}}))
    cases.append(("no_tool_call",
                  _traj([(None, None, None)], None, "terminal"),
                  tk, {"exec_max": 0.0, "flags": {"no_tool_call": True}}))
    cases.append(("terminal_before_first_successful_tool",
                  _traj([(GOLD[0], "exec:runtime_error:boom", None), (None, None, None)],
                        None, "terminal"),
                  tk, {"exec_max": 0.0, "flags": {"terminal_before_first_successful_tool": True}}))
    cases.append(("too_few_calls_wrong_answer",
                  _traj([(GOLD[0], None, 3)], 3, "max_turns"),
                  tk, {"exec_max": 0.25, "flags": {"too_few_calls": True}}))
    cases.append(("valid_executable_trajectory_wrong_answer",
                  _traj([(GOLD[0], None, 3), (GOLD[1], None, 99)], 99, "max_turns"),
                  tk, {"exec_max": 0.35, "flags": {"is_executable_trajectory": True}}))
    cases.append(("invalid_reference",
                  _traj([(GOLD[0], None, 3), (bad_ref, "exec:unresolved_variable:var5", None)],
                        3, "executor_error"),
                  tk, {"exec_max": 0.30, "flags": {"invalid_reference": True}}))
    cases.append(("wrong_tool_name",
                  _traj([(wrong_name, None, -1)], -1, "max_turns"),
                  tk, {"exec_max": 0.35}))
    cases.append(("wrong_argument_keys",
                  _traj([(wrong_keys, None, 3)], 3, "max_turns"),
                  tk, {"exec_max": 0.35}))
    cases.append(("extra_calls_correct_answer",
                  _traj([(GOLD[0], None, 3), (GOLD[1], None, 9), (extra, None, 9)], 9, "max_turns"),
                  tk, {"exec_min": 0.85, "flags": {"num_extra_calls": 1}}))
    cases.append(("extra_calls_wrong_answer",
                  _traj([(GOLD[0], None, 3), (GOLD[1], None, 99), (extra, None, 99)], 99, "max_turns"),
                  tk, {"exec_max": 0.35}))
    cases.append(("parse_error",
                  _traj([(None, "parse:invalid_json", None)], None, "parse_fail"),
                  tk, {"exec_max": 0.0, "flags": {"parse_error": True}}))
    cases.append(("clipped_rollout",
                  _traj([(None, "clipped_completion", None)], None, "clipped", clipped=True),
                  tk, {"exec_max": 0.0, "flags": {"clipped": True}}))
    cases.append(("executor_error",
                  _traj([(GOLD[0], None, 3), (GOLD[1], "exec:runtime_error:boom", None)],
                        3, "executor_error"),
                  tk, {"exec_max": 0.25, "flags": {"executor_error": True}}))
    return cases


def main() -> int:
    rows = []
    for name, tr, tk, _exp in build_cases():
        strict = R.strict_gold_trace_legacy(tr, tk).reward
        partial_legacy = R.partial_gold_trace_legacy(tr, tk).reward
        partial_v2 = R.partial_gold_trace_v2(tr, tk).reward
        ev1 = R.execution_aware_v1_legacy(tr, tk).reward
        ev2_res = R.execution_aware_v2(tr, tk)
        rows.append({
            "case": name,
            "strict_legacy": round(strict, 3),
            "partial_legacy": round(partial_legacy, 3),
            "partial_v2": round(partial_v2, 3),
            "execution_v1": round(ev1, 3),
            "execution_v2": round(ev2_res.reward, 3),
            "exec_v2_cap": ev2_res.diagnostics.get("cap_applied"),
        })
    out = os.path.join(_HERE, "reward_v2_audit_cases.csv")
    write_csv(out, rows, fieldnames=[
        "case", "strict_legacy", "partial_legacy", "partial_v2",
        "execution_v1", "execution_v2", "exec_v2_cap",
    ])
    print(f"wrote {out} ({len(rows)} cases)")
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
