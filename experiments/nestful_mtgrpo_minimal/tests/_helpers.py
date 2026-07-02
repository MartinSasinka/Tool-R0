"""Shared test helpers: build trajectories without a real model."""
from rollout import Trajectory, Turn


def make_trajectory(
    task_id,
    calls,
    observations,
    *,
    gold_num_turns,
    final_observation,
    executor_mode="full",
    stop_reason="terminal",
    fail_reasons=None,
):
    """calls/observations are aligned lists; fail_reasons optional per-turn."""
    fail_reasons = fail_reasons or [None] * len(calls)
    traj = Trajectory(task_id, gold_num_turns, gold_num_turns,
                      executor_mode=executor_mode, stop_reason=stop_reason)
    for i, (c, o, fr) in enumerate(zip(calls, observations, fail_reasons)):
        traj.turns.append(Turn(
            turn_idx=i, model_text="<stub>", parsed_call=c,
            observation=o, fail_reason=fr,
        ))
    traj.final_observation = final_observation
    return traj
