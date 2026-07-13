"""Multi-turn solver-gap scoring (same ``run_episode`` path as MT-GRPO).

Replaces the legacy single-shot JSON weak/strong probe when
``AGENTIC_SOLVER_GAP_MODE=multiturn`` (default when ``WEAK_SOLVER_BACKEND=local``).

Weak uses the local Qwen checkpoint; strong uses OpenRouter with the same
multi-turn prompt/parse loop. Scores come from the configured training reward
(``execution_aware_v3_2_dense`` by default) so solver-gap thresholds align with
the rollout gate and trainer.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from .rollout_signal import (_make_local_generate_fn, load_rollout_config,
                             target_is_local)
from .solvers import STRONG_ATTEMPTS, best_of
from .training_reward import build_task_dict, score_episode_trajectory

SOLVER_GAP_MODES = ("multiturn", "single_shot")
from .env_defaults import (
    SOLVER_MT_STRONG_TEMPERATURE as _SOLVER_MT_STRONG_TEMPERATURE_DEFAULT,
    SOLVER_MT_WEAK_TEMPERATURE as _SOLVER_MT_WEAK_TEMPERATURE_DEFAULT,
    env_float,
)

SOLVER_MT_WEAK_TEMPERATURE = env_float(
    "SOLVER_MT_WEAK_TEMPERATURE", _SOLVER_MT_WEAK_TEMPERATURE_DEFAULT)
SOLVER_MT_STRONG_TEMPERATURE = env_float(
    "SOLVER_MT_STRONG_TEMPERATURE", _SOLVER_MT_STRONG_TEMPERATURE_DEFAULT)

_WIN_STATUSES = frozenset({"fully_correct", "win", "solution_equivalent"})
_CONFIG_CACHE: Dict[int, Tuple[Dict[str, Any], Any]] = {}


def solver_gap_mode() -> str:
    """``multiturn`` (default when local weak) or legacy ``single_shot``."""
    raw = os.environ.get("AGENTIC_SOLVER_GAP_MODE", "").strip().lower()
    if raw in ("single_shot", "single-shot", "singleshot", "json"):
        return "single_shot"
    if raw in ("multiturn", "mt", "multi-turn", "multi_turn"):
        return "multiturn"
    return "multiturn" if target_is_local() else "single_shot"


def _context(num_calls: int) -> Tuple[Dict[str, Any], Any]:
    if num_calls not in _CONFIG_CACHE:
        _CONFIG_CACHE[num_calls] = load_rollout_config(num_calls)
    return _CONFIG_CACHE[num_calls]


def _gap_score(episode_reward: float, reward_class: str,
               diagnostics: Optional[Dict[str, Any]] = None) -> float:
    """Map dense MT reward to solver-gap scale (exact_win expects ~1.0).

    On agentic synthetic tools, IBM execution often caps ``execution_score``
    even when calls/args are perfect — treat structural gold-trace match as a
    win for the solver-gap filter (same intent as legacy ``score_prediction``).
    """
    score = float(episode_reward)
    diag = diagnostics or {}
    if reward_class in _WIN_STATUSES or score >= 0.999:
        return max(score, 1.0)
    n_pred = int(diag.get("n_pred_calls") or diag.get("predicted_num_calls") or 0)
    n_gold = int(diag.get("gold_n_calls") or 0)
    if (n_gold > 0 and n_pred >= n_gold
            and float(diag.get("per_call_tool_score") or 0) >= 0.99
            and float(diag.get("per_call_argument_score") or 0) >= 0.99
            and float(diag.get("call_count_progress") or 0) >= 0.99):
        return max(score, 1.0)
    return score


def _to_solver_result(scored: Dict[str, Any], *, solver_mode: str) -> Dict[str, Any]:
    reward_class = str(scored.get("reward_class") or scored.get("status") or "unknown")
    episode = float(scored.get("episode_reward", scored.get("score", 0.0)))
    diag = scored.get("diagnostics") or {}
    return {
        "score": _gap_score(episode, reward_class, diag),
        "status": reward_class,
        "n_calls": int(scored.get("n_calls") or 0),
        "episode_reward": episode,
        "reward_class": reward_class,
        "solver_mode": solver_mode,
        "diagnostics": diag,
    }


def run_solver_episode(
        question: str, tools: List[Dict[str, Any]],
        gold_calls: List[Dict[str, Any]], gold_observations: List[Any],
        gold_answer: Any, *, stage: str,
        generate_fn: Callable[[List[Dict[str, str]], int], Dict[str, Any]],
        seed: Optional[int] = None,
) -> Dict[str, Any]:
    """One multi-turn episode scored with the training reward dispatch."""
    from .training_reward import _ensure_training_import_paths
    _ensure_training_import_paths()
    from rollout import run_episode  # noqa: E402

    num_calls = len(gold_calls)
    task = build_task_dict(
        gold_calls=gold_calls, gold_answer=gold_answer,
        stage=stage, question=question, tools=tools)
    config, registry = _context(num_calls)
    traj = run_episode(
        None, None, task, config,
        registry=registry, mode="train", generate_fn=generate_fn)
    return _to_solver_result(
        score_episode_trajectory(traj, task, gold_observations),
        solver_mode="multiturn")


def solve_weak_multiturn(
        question: str, tools: List[Dict[str, Any]],
        gold_calls: List[Dict[str, Any]], gold_observations: List[Any],
        gold_answer: Any, *, stage: str,
        seed: Optional[int] = None,
) -> Dict[str, Any]:
    from .local_llm import get_local_weak_solver
    solver = get_local_weak_solver()
    gen_fn = _make_local_generate_fn(
        solver, temperature=SOLVER_MT_WEAK_TEMPERATURE, rollout_seed=seed)
    return run_solver_episode(
        question, tools, gold_calls, gold_observations, gold_answer,
        stage=stage, generate_fn=gen_fn, seed=seed)


def _make_api_generate_fn(
        api_chat: Callable[..., Optional[Dict[str, Any]]],
        *, model: str, role: str, temperature: float,
        attempt_seed: Optional[int],
) -> Callable[[List[Dict[str, str]], int], Dict[str, Any]]:
    state = {"turn": 0}

    def generate_fn(messages: List[Dict[str, str]], max_new_tokens: int) -> Dict[str, Any]:
        call_seed = None
        if attempt_seed is not None:
            call_seed = int(attempt_seed) + state["turn"]
        state["turn"] += 1
        resp = api_chat(
            role=role, model=model, messages=messages,
            temperature=temperature, max_tokens=max_new_tokens,
            json_mode=False, seed=call_seed)
        if resp is None:
            return {"text": "", "prompt_tokens": 0, "completion_tokens": 0,
                    "clipped": False, "prompt_overflow": False}
        text = str(resp.get("text") or "")
        clipped = len(text) >= max_new_tokens if max_new_tokens > 0 else False
        return {
            "text": text,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "clipped": clipped,
            "prompt_overflow": False,
        }

    return generate_fn


def solve_strong_multiturn(
        api_chat: Callable[..., Optional[Dict[str, Any]]],
        model: str,
        question: str, tools: List[Dict[str, Any]],
        gold_calls: List[Dict[str, Any]], gold_observations: List[Any],
        gold_answer: Any, *, stage: str,
        seed: Optional[int] = None,
        attempts: int = STRONG_ATTEMPTS,
) -> Dict[str, Any]:
    scores: List[Dict[str, Any]] = []
    for a in range(attempts):
        gen_fn = _make_api_generate_fn(
            api_chat, model=model, role="strong_solver",
            temperature=SOLVER_MT_STRONG_TEMPERATURE,
            attempt_seed=(int(seed) + a) if seed is not None else None)
        try:
            scores.append(run_solver_episode(
                question, tools, gold_calls, gold_observations, gold_answer,
                stage=stage, generate_fn=gen_fn,
                seed=(int(seed) + a) if seed is not None else None))
        except Exception:  # noqa: BLE001
            scores.append({"score": 0.0, "status": "api_error", "n_calls": 0,
                           "solver_mode": "multiturn"})
        if scores[-1].get("status") == "api_error":
            return scores[-1]
    return best_of(scores)


def reset_solver_context_cache() -> None:
    """Test helper."""
    _CONFIG_CACHE.clear()
