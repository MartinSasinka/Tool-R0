"""Partial (graded) gold-trace reward for MT-GRPO.

This is the ONLY difference from the sibling ``nestful_mtgrpo_minimal``
experiment: instead of a binary all-or-nothing strict reward, the model is
rewarded PARTIALLY for each tool call it gets right and for reaching the gold
answer.

Why a separate reward (research motivation, RQ2)
------------------------------------------------
The strict reward gives R=1 only when the WHOLE gold trace is reproduced (every
tool name + argument keys + executor observation) AND the final answer matches.
A single early mistake zeroes the whole episode — even if the model got 3 of 4
calls right. That is a very sparse signal.

The partial reward decomposes the episode into:
  * a per-turn component  — graded credit for each gold position, and
  * a final-answer component — credit for reaching the gold answer.

It is a *denser* training signal. Evaluation is intentionally left UNCHANGED
(strict ``strict_gold_trace_pass`` + official NESTFUL Win Rate) so the two
experiments are directly comparable: does partial-credit training improve the
execution-based metrics, or does it just inflate the (now matching) train
reward without transferring? — that is exactly the RQ2 question.

Reward shape (per episode)
--------------------------
For each gold position ``i`` (0..gold_n-1) we compute a graded turn score::

    turn_score_i = w_name * 1{name_ok}
                 + w_keys * 1{name_ok AND keys_ok}
                 + w_exec * 1{name_ok AND keys_ok AND exec_ok}

``keys`` and ``exec`` credit is GATED on the tool name being correct — matching
argument keys against the wrong tool is meaningless, so it earns nothing. With
the default weights (0.4 / 0.3 / 0.3) a fully-correct turn scores 1.0.

The episode reward combines the mean turn score with the final-answer signal::

    R = w_trace * mean(turn_score) + w_final * 1{final_answer_pass}
        - length_penalty * max(0, num_calls - gold_n) / gold_n
    R = clip(R, 0.0, 1.0)

With default weights (w_trace=0.7, w_final=0.3) a perfect episode scores 1.0,
identical to the strict reward's R=1, so the two are on the same [0, 1] scale.

Turn-level credit for MT-GRPO
-----------------------------
``episode_turn_reward_seq`` maps the graded turn scores onto the model's
generated turns (same contract as the strict version), so the existing
turn-level MT-GRPO advantage computation in ``grpo_train.py`` works unchanged —
it simply receives graded floats in ``r_seq`` instead of 0/1.

Nothing here uses ``solution_equivalent`` / Win Rate / any eval-only signal.
The reward is still derived ONLY from the gold trace + gold answer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Reuse the validated primitives from the sibling experiment so this reward can
# never drift from the strict one on the shared building blocks.
from reward import RewardResult, _turn_executed_cleanly, _turn_observation
from executor import matches_gold

# ---------------------------------------------------------------------------
#  Weights (tunable via config["partial_reward"]; see set_weights_from_config)
# ---------------------------------------------------------------------------
_DEFAULT_WEIGHTS: Dict[str, float] = {
    # Per-turn component (sum should be 1.0 so a perfect turn scores 1.0).
    "w_name": 0.4,
    "w_keys": 0.3,
    "w_exec": 0.3,
    # Episode combination (sum should be 1.0 so a perfect episode scores 1.0).
    "w_trace": 0.7,
    "w_final": 0.3,
    # Optional penalty for emitting MORE tool calls than the gold trace has.
    # 0.0 = off (default). Extra calls beyond gold are otherwise ignored;
    # missing gold positions already score 0 via the per-turn component.
    "length_penalty": 0.0,
}

_WEIGHTS: Dict[str, float] = dict(_DEFAULT_WEIGHTS)


def set_weights(**kwargs: float) -> None:
    """Override one or more reward weights in-place (process-global)."""
    for k, v in kwargs.items():
        if k not in _DEFAULT_WEIGHTS:
            raise KeyError(f"unknown partial_reward weight: {k!r} "
                           f"(valid: {sorted(_DEFAULT_WEIGHTS)})")
        _WEIGHTS[k] = float(v)
    _warn_if_unnormalized()


def set_weights_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    """Load weights from ``config['partial_reward']`` (missing keys keep defaults)."""
    global _WEIGHTS
    _WEIGHTS = dict(_DEFAULT_WEIGHTS)
    block = (config or {}).get("partial_reward", {}) or {}
    overrides = {k: float(v) for k, v in block.items() if k in _DEFAULT_WEIGHTS}
    _WEIGHTS.update(overrides)
    _warn_if_unnormalized()
    print(f"[partial_reward] weights = {_WEIGHTS}", flush=True)
    return dict(_WEIGHTS)


def get_weights() -> Dict[str, float]:
    return dict(_WEIGHTS)


def _warn_if_unnormalized() -> None:
    turn_sum = _WEIGHTS["w_name"] + _WEIGHTS["w_keys"] + _WEIGHTS["w_exec"]
    ep_sum = _WEIGHTS["w_trace"] + _WEIGHTS["w_final"]
    if abs(turn_sum - 1.0) > 1e-6:
        print(f"[partial_reward] WARNING: w_name+w_keys+w_exec={turn_sum:.3f} != 1.0 "
              "(a perfect turn will not score exactly 1.0)", flush=True)
    if abs(ep_sum - 1.0) > 1e-6:
        print(f"[partial_reward] WARNING: w_trace+w_final={ep_sum:.3f} != 1.0 "
              "(a perfect episode will not score exactly 1.0)", flush=True)


# ---------------------------------------------------------------------------
#  Per-turn graded scoring
# ---------------------------------------------------------------------------

def _turn_component_scores(
    trajectory, task: Dict[str, Any], gold_observations: Optional[List[Any]],
):
    """Return (scores, components) aligned to gold positions.

    scores[i]      = graded turn score in [0, 1]
    components[i]  = (name_ok, keys_ok, exec_ok) booleans (keys/exec gated on name)
    """
    gold_calls = task.get("gold_calls", [])
    gold_n = len(gold_calls)
    pred = trajectory.predicted_calls

    w_name, w_keys, w_exec = _WEIGHTS["w_name"], _WEIGHTS["w_keys"], _WEIGHTS["w_exec"]
    scores: List[float] = []
    components: List[tuple] = []

    for i in range(gold_n):
        if i >= len(pred):
            scores.append(0.0)
            components.append((False, False, False))
            continue
        p, g = pred[i], gold_calls[i]
        name_ok = (p.get("name") or "") == (g.get("name") or "")
        keys_ok = name_ok and (
            set((p.get("arguments") or {}).keys())
            == set((g.get("arguments") or {}).keys())
        )
        if keys_ok and gold_observations is not None and i < len(gold_observations):
            exec_ok = bool(matches_gold(_turn_observation(trajectory, i),
                                        gold_observations[i]))
        elif keys_ok:
            exec_ok = bool(_turn_executed_cleanly(trajectory, i))
        else:
            exec_ok = False
        score = (w_name * float(name_ok)
                 + w_keys * float(keys_ok)
                 + w_exec * float(exec_ok))
        scores.append(float(score))
        components.append((name_ok, keys_ok, exec_ok))
    return scores, components


# ---------------------------------------------------------------------------
#  Episode reward
# ---------------------------------------------------------------------------

def partial_gold_trace_reward(
    trajectory,
    task: Dict[str, Any],
    gold_observations: Optional[List[Any]] = None,
) -> RewardResult:
    """Graded gold-trace reward in [0, 1]. See module docstring for the formula."""
    gold_calls = task.get("gold_calls", [])
    gold_n = len(gold_calls)

    final_answer_pass = bool(matches_gold(trajectory.final_observation,
                                          task.get("gold_answer")))

    diag: Dict[str, Any] = {
        "reward_type": "partial_gold_trace",
        "final_answer_pass": final_answer_pass,
        "turn_rewards": [0.0] * gold_n,
        "trace_score": 0.0,
        "episode_reward": 0.0,
        "strict_gold_trace_pass": False,
        "answer_correct_wrong_path": False,
        "correct_prefix_len": 0,
        "clipped": trajectory.clipped_any,
        "stop_reason": trajectory.stop_reason,
    }

    # Clipped episodes get reward 0 (and are masked from updates upstream),
    # exactly like the strict reward — a truncated rollout is not a valid sample.
    if trajectory.clipped_any:
        return RewardResult(0.0, diag)

    scores, components = _turn_component_scores(trajectory, task, gold_observations)
    trace_score = (sum(scores) / gold_n) if gold_n else 0.0

    # Correct prefix length (consecutive fully-correct turns from the start).
    prefix = 0
    for comp in components:
        if comp == (True, True, True):
            prefix += 1
        else:
            break

    extra_calls = max(0, trajectory.num_tool_calls - gold_n)
    pen = _WEIGHTS["length_penalty"] * ((extra_calls / gold_n) if gold_n else 0.0)

    reward = (_WEIGHTS["w_trace"] * trace_score
              + _WEIGHTS["w_final"] * (1.0 if final_answer_pass else 0.0)
              - pen)
    reward = max(0.0, min(1.0, reward))

    full_trace = (gold_n > 0
                  and trajectory.num_tool_calls == gold_n
                  and all(c == (True, True, True) for c in components))

    diag.update({
        "turn_rewards": scores,
        "turn_components": components,
        "trace_score": trace_score,
        "episode_reward": reward,
        "strict_gold_trace_pass": bool(full_trace and final_answer_pass),
        "answer_correct_wrong_path": bool(final_answer_pass and not full_trace),
        "correct_prefix_len": prefix,
        "length_penalty_applied": pen,
    })
    return RewardResult(reward, diag)


# Alias mirroring the strict module's naming.
partial_gold_trace_episode_reward = partial_gold_trace_reward


def episode_turn_reward_seq(
    trajectory,
    task: Dict[str, Any],
    gold_observations: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Per-GENERATED-turn graded reward sequence + R_episode for the trainer.

    Drop-in replacement for ``reward.episode_turn_reward_seq`` (same return keys
    ``r_seq`` / ``episode_reward`` / ``diagnostics``). The s-th successful tool
    call gets the graded gold turn score for gold position s (0 beyond gold
    length). Parse-fail / terminal / clipped turns get 0.
    """
    rr = partial_gold_trace_reward(trajectory, task, gold_observations)
    gold_turn_scores = rr.diagnostics.get("turn_rewards", [])
    r_seq: List[float] = []
    success_idx = 0
    for t in trajectory.turns:
        if t.parsed_call is not None and t.fail_reason is None:
            r = gold_turn_scores[success_idx] if success_idx < len(gold_turn_scores) else 0.0
            r_seq.append(float(r))
            success_idx += 1
        else:
            r_seq.append(0.0)
    return {
        "r_seq": r_seq,
        "episode_reward": float(rr.reward),
        "diagnostics": rr.diagnostics,
    }
