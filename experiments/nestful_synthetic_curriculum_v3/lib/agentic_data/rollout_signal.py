"""Multi-rollout GRPO-signal probe against the exact target Qwen weak-solver
setup (local HF backend only — this IS the training-target checkpoint, not
an API proxy).

For every candidate we sample ``ROLLOUT_N`` (default 8) independent
completions from the SAME prompt the weak solver used, score each with the
**same training reward dispatch** as GRPO (``vllm_dp_pool.resolve_reward_info``,
default ``execution_aware_v3_2_dense`` via ``AGENTIC_REWARD_POLICY`` /
``REWARD_POLICY``), and summarize whether the task carries usable training
signal.

A task is GRPO-signal-positive when, across the rollouts:
  * unique_rewards >= 2                    (not degenerate: some spread)
  * reward_variance > 0
  * not all rollouts are a full win        (task is not trivially easy)
  * not all rollouts are identically wrong (some spread in HOW it fails)
  * at least one rollout produced a valid partial or complete trace
  * failures are not EXCLUSIVELY parse_error / no_tool_call

The orchestrator hard-rejects candidates whose probe is not
``grpo_signal_positive`` (when the local weak backend is available).

Skipped entirely when ``WEAK_SOLVER_BACKEND != local``, because the exact
target Qwen3-4B setup is only available locally.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from .solvers import parse_solver_output, solver_messages
from .training_reward import (build_task_dict, configured_reward_policy,
                              get_training_reward_fn,
                              score_with_training_reward)

ROLLOUT_N = int(os.environ.get("ROLLOUT_N", "8"))
ROLLOUT_TEMPERATURE = float(os.environ.get("ROLLOUT_TEMPERATURE", "0.8"))
# Deliberately smaller than the single-shot weak-solver budget (700): Stage 2
# gold traces are 2 short tool calls, and this budget is paid 8x per accepted
# row (batched — see LocalWeakSolver.generate_n) so wall-clock matters.
ROLLOUT_MAX_TOKENS = int(os.environ.get("ROLLOUT_MAX_TOKENS", "260"))

DEGENERATE_STATUSES = {"parse_error", "no_tool_call", "clipped"}

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def target_is_local() -> bool:
    """True iff the weak solver IS the exact target Qwen checkpoint running
    locally (not an OpenRouter proxy for a different model)."""
    return os.environ.get("WEAK_SOLVER_BACKEND", "openrouter") == "local"


def _lenient_json(text: str) -> Optional[Any]:
    """Best-effort JSON extraction for a rollout probe (no paid retries — a
    genuinely unparseable rollout should just score as parse_error)."""
    if not text:
        return None
    candidates = [text.strip()]
    m = _FENCE_RE.search(text)
    if m:
        candidates.insert(0, m.group(1).strip())
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            pass
        start, end = cand.find("{"), cand.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cand[start:end + 1])
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def run_rollouts(question: str, tools: List[Dict[str, Any]],
                 gold_calls: List[Dict[str, Any]], gold_observations: List[Any],
                 gold_answer: Any, *, stage: str,
                 n: Optional[int] = None,
                 seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Sample ``n`` independent local-weak-solver completions; return training-
    reward scores (same dispatch as GRPO / stage probe).

    Uses ``generate_n()`` (one batched forward pass, shared prefill) when the
    solver supports it — 8 rollouts then cost roughly one generation instead
    of eight sequential ones. Falls back to ``n`` sequential ``generate()`` calls
    for any solver stub that only implements the single-completion API."""
    from .local_llm import get_local_weak_solver
    solver = get_local_weak_solver()
    n = ROLLOUT_N if n is None else n
    messages = solver_messages(question, tools, strong=False)
    if hasattr(solver, "generate_n"):
        texts = solver.generate_n(messages, temperature=ROLLOUT_TEMPERATURE,
                                  max_tokens=ROLLOUT_MAX_TOKENS, n=n, seed=seed)
    else:
        texts = [solver.generate(messages, temperature=ROLLOUT_TEMPERATURE,
                                 max_tokens=ROLLOUT_MAX_TOKENS,
                                 seed=(seed + i) if seed is not None else None)
                for i in range(n)]
    task = build_task_dict(
        gold_calls=gold_calls, gold_answer=gold_answer,
        stage=stage, question=question)
    scored: List[Dict[str, Any]] = []
    for text in texts:
        parsed = _lenient_json(text)
        calls = parse_solver_output(parsed) if parsed is not None else None
        final = parsed.get("final_answer") if isinstance(parsed, dict) else None
        score = score_with_training_reward(
            calls, final, task, gold_observations,
            parsed=parsed, raw_text=text or "")
        scored.append(score)
    return scored


def _episode_reward(entry: Dict[str, Any]) -> float:
    return float(entry.get("episode_reward", entry.get("score", 0.0)))


def _status(entry: Dict[str, Any]) -> str:
    return str(entry.get("reward_class") or entry.get("status") or "unknown")


def summarize_rollouts(scored: List[Dict[str, Any]], n_gold_calls: int,
                       *, reward_policy: Optional[str] = None
                       ) -> Dict[str, Any]:
    """Aggregate metrics + GRPO-signal-positive verdict over rollout scores."""
    n = len(scored)
    if n == 0:
        return {"n": 0, "skipped": True}
    if reward_policy is None:
        try:
            _, info = get_training_reward_fn()
            reward_policy = info.get("resolved_policy") or configured_reward_policy()
        except Exception:  # noqa: BLE001
            reward_policy = configured_reward_policy()
    rewards = [_episode_reward(s) for s in scored]
    statuses = [_status(s) for s in scored]
    unique_rewards = len({round(r, 6) for r in rewards})
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    full_success_rate = round(
        sum(1 for r, st in zip(rewards, statuses)
            if st in ("fully_correct", "win", "solution_equivalent") or r >= 0.999)
        / n, 3)
    correct_prefix_rate = round(
        sum(1 for s in statuses
            if s in ("too_few_calls", "partial_progress", "executable_wrong_final",
                     "correct_tool_wrong_args", "too_many_calls"))
        / n, 3)
    too_few_call_rate = round(
        sum(1 for s in scored if (s.get("n_calls") or 0) < n_gold_calls) / n, 3)
    call_dist: Dict[str, int] = {}
    for s in scored:
        k = str(s.get("n_calls", 0))
        call_dist[k] = call_dist.get(k, 0) + 1
    failure_dist: Dict[str, int] = {}
    for st in statuses:
        if st != "fully_correct":
            failure_dist[st] = failure_dist.get(st, 0) + 1
    has_valid_trace = any(
        st not in DEGENERATE_STATUSES and (sc.get("n_calls") or 0) >= 1
        for sc, st in zip(scored, statuses))
    all_degenerate = all(st in DEGENERATE_STATUSES for st in statuses)
    all_correct = full_success_rate >= 0.999
    all_identically_wrong = unique_rewards == 1 and full_success_rate == 0.0
    grpo_signal_positive = (
        unique_rewards >= 2 and variance > 0.0 and not all_correct
        and not all_identically_wrong and has_valid_trace and not all_degenerate)
    return {
        "n": n,
        "skipped": False,
        "reward_policy": reward_policy,
        "rewards": [round(r, 6) for r in rewards],
        "unique_rewards": unique_rewards,
        "reward_variance": round(variance, 8),
        "reward_mean": round(mean, 6),
        "full_success_rate": full_success_rate,
        "correct_prefix_rate": correct_prefix_rate,
        "too_few_call_rate": too_few_call_rate,
        "predicted_call_distribution": call_dist,
        "failure_type_distribution": failure_dist,
        "has_valid_trace": has_valid_trace,
        "all_degenerate": all_degenerate,
        "grpo_signal_positive": grpo_signal_positive,
    }


def probe_rollout_signal(question: str, tools: List[Dict[str, Any]],
                         gold_calls: List[Dict[str, Any]],
                         gold_observations: List[Any], gold_answer: Any,
                         *, stage: str,
                         n: Optional[int] = None, seed: Optional[int] = None
                         ) -> Dict[str, Any]:
    """Full probe: skip cleanly when the target-local backend is unavailable."""
    if not target_is_local():
        return {"skipped": True, "reason": "WEAK_SOLVER_BACKEND != local — "
                "exact target Qwen3-4B setup unavailable"}
    scored = run_rollouts(question, tools, gold_calls, gold_observations,
                          gold_answer, stage=stage, n=n, seed=seed)
    return summarize_rollouts(scored, len(gold_calls))
