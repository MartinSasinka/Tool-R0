"""Score agentic weak-solver completions with the SAME reward dispatch as GRPO.

The rollout GRPO-signal probe previously used ``score_prediction()`` from
``solvers.py`` (coarse 0.5-prefix bands). Training uses
``vllm_dp_pool.resolve_reward_info`` → ``episode_turn_reward_seq`` (e.g.
``execution_aware_v3_2_dense``). This module bridges the two so generation
gates predict the reward landscape the trainer will actually see.

Policy selection (first match wins):
  * ``AGENTIC_REWARD_POLICY``
  * ``REWARD_POLICY``
  * default ``execution_aware_v3_2_dense``
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..nestful_like_generator import TOOLS, execute_call

V3_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = V3_ROOT.parent
MINIMAL = EXPERIMENTS / "nestful_mtgrpo_minimal"
PARTIAL = EXPERIMENTS / "nestful_mtgrpo_partial"

_REWARD_FN = None
_REWARD_INFO: Optional[Dict[str, Any]] = None

_EXEC_ERR_MAP = {
    "wrong_tool": "exec:unknown_tool",
    "wrong_args": "exec:wrong_args",
    "invalid_reference": "exec:unresolved_variable",
    "execution_error": "exec:error",
}


def configured_reward_policy() -> str:
    return (os.environ.get("AGENTIC_REWARD_POLICY")
            or os.environ.get("REWARD_POLICY")
            or "execution_aware_v3_2_dense")


def _ensure_training_import_paths() -> None:
    for p in (str(MINIMAL), str(PARTIAL), str(V3_ROOT), str(EXPERIMENTS)):
        if p not in sys.path:
            sys.path.insert(0, p)


def get_training_reward_fn() -> Tuple[Any, Dict[str, Any]]:
    """Lazy-resolve the training reward fn (same path as probe / GRPO workers)."""
    global _REWARD_FN, _REWARD_INFO
    if _REWARD_FN is None:
        _ensure_training_import_paths()
        from vllm_dp_pool import resolve_reward_info  # noqa: E402
        policy = configured_reward_policy()
        _REWARD_FN, _REWARD_INFO = resolve_reward_info(
            {"reward": {"train_policy": policy}})
    return _REWARD_FN, _REWARD_INFO or {}


def reset_reward_cache() -> None:
    """Test helper — force re-resolve on next call."""
    global _REWARD_FN, _REWARD_INFO
    _REWARD_FN = None
    _REWARD_INFO = None


def build_task_dict(*, gold_calls: List[Dict[str, Any]], gold_answer: Any,
                    stage: str, question: str,
                    task_id: str = "agentic_probe",
                    num_calls: Optional[int] = None,
                    tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    n = num_calls if num_calls is not None else len(gold_calls)
    task: Dict[str, Any] = {
        "task_id": task_id,
        "stage": stage,
        "num_calls": n,
        "gold_calls": gold_calls,
        "gold_answer": gold_answer,
        "question": question,
        "terminal_stage": True,
    }
    if tools is not None:
        _ensure_training_import_paths()
        from data import _normalize_tool_schema  # noqa: E402
        task["tools"] = _normalize_tool_schema(tools)
    return task


def score_episode_trajectory(traj: Any, task: Dict[str, Any],
                             gold_observations: List[Any]) -> Dict[str, Any]:
    """Score a multi-turn ``run_episode`` trajectory with training reward."""
    reward_fn, info = get_training_reward_fn()
    rr = reward_fn(traj, task, gold_observations)
    episode = float(rr["episode_reward"])
    diag = rr.get("diagnostics") or {}
    reward_class = str(diag.get("reward_class") or diag.get("reward_cap_reason")
                       or "unknown")
    n_pred = int(diag.get("n_pred_calls") or diag.get("predicted_num_calls")
                or len([t for t in traj.turns
                        if getattr(t, "parsed_call", None)]))
    return {
        "score": episode,
        "episode_reward": episode,
        "status": reward_class,
        "reward_class": reward_class,
        "n_calls": n_pred,
        "diagnostics": diag,
        "reward_policy": info.get("resolved_policy")
        or info.get("configured_policy"),
    }


def _execute_predicted(calls: List[Dict[str, Any]]
                       ) -> Tuple[List[Any], Optional[str]]:
    scope: Dict[str, Any] = {}
    observations: List[Any] = []
    for i, call in enumerate(calls):
        name = call["name"]
        if name not in TOOLS:
            return observations, "wrong_tool"
        expected = set(TOOLS[name]["params"].keys())
        if set(call["arguments"].keys()) != expected:
            return observations, "wrong_args"
        try:
            obs = execute_call(name, call["arguments"], scope)
        except KeyError:
            return observations, "invalid_reference"
        except Exception:  # noqa: BLE001
            return observations, "execution_error"
        scope[str(call.get("label", f"$var{i + 1}")).lstrip("$")] = obs
        observations.append(obs)
    return observations, None


def build_trajectory_from_completion(
        parsed: Optional[Dict[str, Any]],
        calls: Optional[List[Dict[str, Any]]],
        final_answer: Any,
        *,
        task_id: str,
        gold_num_turns: int,
        raw_text: str = "",
) -> Any:
    """Map a single-shot agentic JSON completion → training ``Trajectory``.

    Each emitted tool call becomes its own ``Turn`` (sequential execution with
    scope), matching how the reward predicates count ``parsed_call`` turns.
    """
    _ensure_training_import_paths()
    from rollout import Trajectory, Turn  # noqa: E402

    traj = Trajectory(task_id, gold_num_turns, gold_num_turns)
    if parsed is None:
        turn = Turn(0, raw_text or "", fail_reason="parse:json_decode")
        traj.turns.append(turn)
        traj.stop_reason = "parse_fail"
        return traj

    if not calls:
        turn = Turn(0, raw_text or json.dumps(parsed), is_terminal=True)
        traj.turns.append(turn)
        traj.stop_reason = "terminal"
        if final_answer is not None:
            traj.final_observation = final_answer
        return traj

    obs, err = _execute_predicted(calls)
    for i, call in enumerate(calls):
        turn = Turn(i, raw_text or json.dumps(call), parsed_call=call)
        if i < len(obs):
            turn.observation = obs[i]
        if err is not None and i == len(obs):
            turn.fail_reason = _EXEC_ERR_MAP.get(err, f"exec:{err}")
            traj.turns.append(turn)
            traj.stop_reason = "executor_error"
            return traj
        traj.turns.append(turn)

    if err is None and obs:
        traj.final_observation = obs[-1]
    elif final_answer is not None:
        traj.final_observation = final_answer
    traj.stop_reason = "terminal"
    return traj


def score_with_training_reward(
        calls: Optional[List[Dict[str, Any]]],
        final_answer: Any,
        task: Dict[str, Any],
        gold_observations: List[Any],
        *,
        parsed: Optional[Dict[str, Any]] = None,
        raw_text: str = "",
) -> Dict[str, Any]:
    """Return a rollout score dict using the configured training reward policy."""
    reward_fn, info = get_training_reward_fn()
    gold_n = int(task.get("num_calls") or len(task.get("gold_calls") or []))
    traj = build_trajectory_from_completion(
        parsed, calls, final_answer,
        task_id=str(task.get("task_id") or "agentic_probe"),
        gold_num_turns=gold_n,
        raw_text=raw_text,
    )
    rr = reward_fn(traj, task, gold_observations)
    episode = float(rr["episode_reward"])
    diag = rr.get("diagnostics") or {}
    reward_class = str(diag.get("reward_class") or diag.get("reward_cap_reason")
                       or "unknown")
    n_pred = int(diag.get("n_pred_calls") or diag.get("predicted_num_calls")
                or len([t for t in traj.turns
                        if getattr(t, "parsed_call", None)]))
    return {
        "score": episode,
        "episode_reward": episode,
        "status": reward_class,
        "reward_class": reward_class,
        "n_calls": n_pred,
        "diagnostics": diag,
        "reward_policy": info.get("resolved_policy")
        or info.get("configured_policy"),
    }
