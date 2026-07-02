"""Rewards and per-episode diagnostics.

  - strict_gold_trace_reward(...)         TRAINING reward (binary, strict).
  - compute_gold_observations(...)        helper to get per-turn gold observations.

The solution-equivalent / NESTFUL metrics live in metrics.py and are EVAL ONLY.
solution_equivalent is NEVER used as a training reward.

This file is a minimal standalone reimplementation; it imports nothing from
curricullum/ or nestful_evaluation/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from executor import ToolExecutor, matches_gold


@dataclass
class RewardResult:
    reward: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def compute_gold_observations(
    task: Dict[str, Any], registry=None
) -> Optional[List[Any]]:
    """Execute the gold call sequence to obtain per-turn gold observations.

    Returns a list of observations (one per gold call) in `full` executor mode,
    or None if execution is not possible (gold_replay mode, or any gold call
    errors). When None, per-turn observation matching is skipped and strict
    correctness falls back to name + argument-key + final-answer checks.
    """
    gold_calls = task.get("gold_calls", [])
    if not gold_calls:
        return None
    ex = ToolExecutor(task, registry=registry, mode="auto")
    if ex.mode != "full":
        return None
    observations: List[Any] = []
    for call in gold_calls:
        res = ex.execute(call)
        if res.error is not None:
            return None
        observations.append(res.observation)
    return observations


def strict_gold_trace_reward(
    trajectory,
    task: Dict[str, Any],
    gold_observations: Optional[List[Any]] = None,
) -> RewardResult:
    """Binary strict reward. R=1 iff the model reproduces the gold trace AND
    reaches the gold answer; else 0. Format is a gate, not a partial reward.
    """
    gold_calls = task.get("gold_calls", [])
    gold_n = len(gold_calls)
    gold_answer = task.get("gold_answer")
    pred_calls = trajectory.predicted_calls

    diag: Dict[str, Any] = {
        "parse_ok": True,
        "tool_name_ok": False,
        "argument_keys_ok": False,
        "executor_ok": False,
        "turn_pass": [],
        "turn_rewards": [],
        "episode_reward": 0.0,
        "first_error_turn": None,
        "correct_prefix_len": 0,
        "zero_tool_calls": trajectory.zero_tool_calls,
        "too_many_turns": trajectory.num_tool_calls > gold_n,
        "final_answer_pass": False,
        "answer_correct_wrong_path": False,
        "clipped": trajectory.clipped_any,
        "stop_reason": trajectory.stop_reason,
    }

    # Final-answer check is independent of path correctness.
    final_answer_pass = matches_gold(trajectory.final_observation, gold_answer)
    diag["final_answer_pass"] = final_answer_pass

    # A parse failure anywhere breaks strictness.
    any_parse_fail = any(
        t.fail_reason and t.fail_reason.startswith("parse:") for t in trajectory.turns
    )
    if any_parse_fail:
        diag["parse_ok"] = False

    # Clipped episodes get reward 0 (and are masked from updates upstream).
    if trajectory.clipped_any:
        diag["first_error_turn"] = next(
            (t.turn_idx for t in trajectory.turns if t.clipped_completion), None
        )
        return RewardResult(0.0, diag)

    # Turn count must match gold exactly.
    count_ok = trajectory.num_tool_calls == gold_n and gold_n > 0

    turn_pass: List[bool] = []
    correct_prefix = 0
    first_error: Optional[int] = None
    all_name_ok = True
    all_keys_ok = True
    all_exec_ok = True

    for i in range(gold_n):
        if i >= len(pred_calls):
            turn_pass.append(False)
            if first_error is None:
                first_error = i
            all_name_ok = all_keys_ok = all_exec_ok = False
            continue
        pred = pred_calls[i]
        gold = gold_calls[i]
        name_ok = (pred.get("name") or "") == (gold.get("name") or "")
        keys_ok = set((pred.get("arguments") or {}).keys()) == set(
            (gold.get("arguments") or {}).keys()
        )
        # Per-turn observation match (only when gold observations are available).
        exec_ok = True
        if gold_observations is not None and i < len(gold_observations):
            model_obs = _turn_observation(trajectory, i)
            exec_ok = matches_gold(model_obs, gold_observations[i])
        else:
            # No gold observations: trust that the executor ran without error.
            exec_ok = _turn_executed_cleanly(trajectory, i)

        passed = bool(name_ok and keys_ok and exec_ok)
        turn_pass.append(passed)
        all_name_ok = all_name_ok and name_ok
        all_keys_ok = all_keys_ok and keys_ok
        all_exec_ok = all_exec_ok and exec_ok
        if passed and first_error is None:
            correct_prefix += 1
        if not passed and first_error is None:
            first_error = i

    diag["turn_pass"] = turn_pass
    diag["turn_rewards"] = [1.0 if p else 0.0 for p in turn_pass]
    diag["tool_name_ok"] = all_name_ok
    diag["argument_keys_ok"] = all_keys_ok
    diag["executor_ok"] = all_exec_ok
    diag["correct_prefix_len"] = correct_prefix
    diag["first_error_turn"] = first_error

    trace_ok = bool(
        count_ok and all_name_ok and all_keys_ok and all_exec_ok
        and not any_parse_fail
    )
    reward = 1.0 if (trace_ok and final_answer_pass) else 0.0

    if final_answer_pass and not trace_ok:
        diag["answer_correct_wrong_path"] = True

    diag["episode_reward"] = reward
    return RewardResult(reward, diag)


# Alias: episode reward is the strict gold-trace reward (R_episode).
strict_gold_trace_episode_reward = strict_gold_trace_reward


def strict_gold_turn_rewards(
    trajectory,
    task: Dict[str, Any],
    gold_observations: Optional[List[Any]] = None,
) -> List[float]:
    """Per-turn strict rewards r_t aligned to the GOLD turn positions.

    r_t = 1 iff (at gold position t) the model emitted exactly one valid tool
    call, the tool name matches gold, the argument keys match gold, and the
    executor result matches the gold observation/result for that turn.
    Otherwise r_t = 0.

    Returned list has length == number of gold calls. This is derived ONLY from
    the gold trace; it never uses solution_equivalent / win_rate / any soft
    signal. Used by the turn-level MT-GRPO trainer for credit assignment.
    """
    rr = strict_gold_trace_reward(trajectory, task, gold_observations)
    return list(rr.diagnostics.get("turn_rewards", []))


def episode_turn_reward_seq(
    trajectory,
    task: Dict[str, Any],
    gold_observations: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Build the per-GENERATED-turn reward sequence + R_episode for the trainer.

    The trajectory's generated turns map 1:1 (by index) to the trainer's
    per-turn token tensors. A generated turn that is the s-th successful tool
    call receives r = gold turn_rewards[s] (0 if beyond gold length). Terminal /
    parse-fail / clipped turns receive r = 0. R_episode is the strict episode
    reward. All values are gold-trace-derived.
    """
    rr = strict_gold_trace_reward(trajectory, task, gold_observations)
    gold_turn_rewards = rr.diagnostics.get("turn_rewards", [])
    r_seq: List[float] = []
    success_idx = 0
    for t in trajectory.turns:
        if t.parsed_call is not None and t.fail_reason is None:
            r = gold_turn_rewards[success_idx] if success_idx < len(gold_turn_rewards) else 0.0
            r_seq.append(float(r))
            success_idx += 1
        else:
            r_seq.append(0.0)
    return {
        "r_seq": r_seq,
        "episode_reward": float(rr.reward),
        "diagnostics": rr.diagnostics,
    }


def _turn_observation(trajectory, gold_idx: int) -> Any:
    """Observation for the gold_idx-th successful tool call in the trajectory."""
    count = 0
    for t in trajectory.turns:
        if t.parsed_call is not None and t.fail_reason is None:
            if count == gold_idx:
                return t.observation
            count += 1
    return None


def _turn_executed_cleanly(trajectory, gold_idx: int) -> bool:
    count = 0
    for t in trajectory.turns:
        if t.parsed_call is not None:
            if count == gold_idx:
                return t.fail_reason is None
            count += 1
    return False
