"""Multi-rollout GRPO-signal probe against the exact target Qwen weak-solver
setup (local HF backend only — this IS the training-target checkpoint, not
an API proxy).

For every candidate we sample ``ROLLOUT_N`` (default 8) independent
**multi-turn** episodes via ``run_episode(mode="train")`` — the same rollout
path as ``probe_stage.py`` and MT-GRPO training — score each with the
**same training reward dispatch** as GRPO (``vllm_dp_pool.resolve_reward_info``,
default ``execution_aware_v3_2_dense`` via ``AGENTIC_REWARD_POLICY`` /
``REWARD_POLICY``), and summarize whether the task carries usable training
signal.

Legacy single-shot JSON probing (``solver_messages`` + one JSON blob) is
available only when ``AGENTIC_ROLLOUT_MODE=single_shot``; it is **not**
aligned with MT-GRPO and must not be used for accept/reject decisions.

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

import importlib.util
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .solvers import parse_solver_output, solver_messages
from .training_reward import (PARTIAL, MINIMAL, build_task_dict,
                              configured_reward_policy,
                              get_training_reward_fn, score_episode_trajectory,
                              score_with_training_reward)

ROLLOUT_N = int(os.environ.get("ROLLOUT_N", "8"))
ROLLOUT_TEMPERATURE = float(os.environ.get("ROLLOUT_TEMPERATURE", "0.8"))
ROLLOUT_TOP_P = float(os.environ.get("ROLLOUT_TOP_P", "0.95"))
# Optional per-turn cap override for multi-turn rollouts (0 = use training
# config stage_defaults, which is what probe/GRPO use).
ROLLOUT_MAX_TOKENS = int(os.environ.get("ROLLOUT_MAX_TOKENS", "0"))
ROLLOUT_MODE = os.environ.get("AGENTIC_ROLLOUT_MODE", "multiturn").strip().lower()

DEGENERATE_STATUSES = {"parse_error", "no_tool_call", "clipped"}

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_DEFAULT_CONFIG = PARTIAL / "config.yaml"


def target_is_local() -> bool:
    """True iff the weak solver IS the exact target Qwen checkpoint running
    locally (not an OpenRouter proxy for a different model)."""
    return os.environ.get("WEAK_SOLVER_BACKEND", "openrouter") == "local"


def rollout_mode() -> str:
    """``multiturn`` (default) or legacy ``single_shot``."""
    return "single_shot" if ROLLOUT_MODE in ("single_shot", "single-shot",
                                             "singleshot") else "multiturn"


def _load_base_run():
    path = str(MINIMAL / "run.py")
    spec = importlib.util.spec_from_file_location("mtgrpo_base_run_rollout", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_rollout_config(num_calls: int, *,
                        temperature: Optional[float] = None,
                        top_p: Optional[float] = None) -> Tuple[Dict[str, Any], Any]:
    """Training-aligned config + tool registry (same loader as probe_stage)."""
    from .training_reward import _ensure_training_import_paths
    _ensure_training_import_paths()
    base_run = _load_base_run()
    config = base_run.load_config(str(_DEFAULT_CONFIG))
    overrides = [
        f"reward.train_policy={configured_reward_policy()}",
        f"generation.temperature={temperature if temperature is not None else ROLLOUT_TEMPERATURE}",
        f"generation.top_p={top_p if top_p is not None else ROLLOUT_TOP_P}",
        "hardware.use_vllm=false",
    ]
    if ROLLOUT_MAX_TOKENS > 0:
        overrides.append(f"generation.max_new_tokens_train={ROLLOUT_MAX_TOKENS}")
        sd = (config.get("token_budget", {}) or {}).get("stage_defaults", {}) or {}
        stage_key = str(num_calls)
        if stage_key in sd:
            sd[stage_key] = dict(sd[stage_key])
            sd[stage_key]["max_new_tokens"] = ROLLOUT_MAX_TOKENS
            config.setdefault("token_budget", {})["stage_defaults"] = sd
    base_run._apply_overrides(config, overrides)
    base_run._normalize_config_paths(config)
    registry = base_run.build_registry(config)
    return config, registry


def _make_local_generate_fn(solver, *, temperature: float,
                            rollout_seed: Optional[int]) -> Any:
    """Wrap LocalWeakSolver for ``run_episode``'s generate_fn contract."""
    state = {"turn": 0}

    def generate_fn(messages: List[Dict[str, str]], max_new_tokens: int) -> Dict[str, Any]:
        call_seed = None
        if rollout_seed is not None:
            call_seed = int(rollout_seed) + state["turn"]
        state["turn"] += 1
        text = solver.generate(
            messages, temperature=temperature, max_tokens=max_new_tokens, seed=call_seed)
        clipped = len(text) >= max_new_tokens if max_new_tokens > 0 else False
        return {
            "text": text,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "clipped": clipped,
            "prompt_overflow": False,
        }

    return generate_fn


def run_multiturn_rollouts(question: str, tools: List[Dict[str, Any]],
                           gold_calls: List[Dict[str, Any]],
                           gold_observations: List[Any], gold_answer: Any,
                           *, stage: str,
                           n: Optional[int] = None,
                           seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Sample ``n`` independent multi-turn episodes; score with training reward."""
    from .local_llm import get_local_weak_solver
    from .training_reward import _ensure_training_import_paths
    _ensure_training_import_paths()
    from rollout import run_episode  # noqa: E402

    solver = get_local_weak_solver()
    n = ROLLOUT_N if n is None else n
    num_calls = len(gold_calls)
    task = build_task_dict(
        gold_calls=gold_calls, gold_answer=gold_answer,
        stage=stage, question=question, tools=tools)
    config, registry = load_rollout_config(num_calls)
    from reward import compute_gold_observations  # noqa: E402
    gold_obs = compute_gold_observations(task, registry)

    scored: List[Dict[str, Any]] = []
    for i in range(n):
        rollout_seed = (int(seed) + i) if seed is not None else None
        gen_fn = _make_local_generate_fn(
            solver, temperature=ROLLOUT_TEMPERATURE, rollout_seed=rollout_seed)
        traj = run_episode(
            None, None, task, config,
            registry=registry, mode="train", generate_fn=gen_fn)
        scored.append(score_episode_trajectory(traj, task, gold_obs))
    return scored


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


def run_single_shot_rollouts(question: str, tools: List[Dict[str, Any]],
                             gold_calls: List[Dict[str, Any]],
                             gold_observations: List[Any], gold_answer: Any,
                             *, stage: str,
                             n: Optional[int] = None,
                             seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Legacy single-shot JSON rollouts (NOT aligned with MT-GRPO)."""
    from .local_llm import get_local_weak_solver
    solver = get_local_weak_solver()
    n = ROLLOUT_N if n is None else n
    max_tokens = ROLLOUT_MAX_TOKENS if ROLLOUT_MAX_TOKENS > 0 else 260
    messages = solver_messages(question, tools, strong=False)
    if hasattr(solver, "generate_n"):
        texts = solver.generate_n(messages, temperature=ROLLOUT_TEMPERATURE,
                                  max_tokens=max_tokens, n=n, seed=seed)
    else:
        texts = [solver.generate(messages, temperature=ROLLOUT_TEMPERATURE,
                                 max_tokens=max_tokens,
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


def run_rollouts(question: str, tools: List[Dict[str, Any]],
                 gold_calls: List[Dict[str, Any]], gold_observations: List[Any],
                 gold_answer: Any, *, stage: str,
                 n: Optional[int] = None,
                 seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Dispatch to multi-turn (default) or legacy single-shot rollouts."""
    if rollout_mode() == "single_shot":
        return run_single_shot_rollouts(
            question, tools, gold_calls, gold_observations, gold_answer,
            stage=stage, n=n, seed=seed)
    return run_multiturn_rollouts(
        question, tools, gold_calls, gold_observations, gold_answer,
        stage=stage, n=n, seed=seed)


def _episode_reward(entry: Dict[str, Any]) -> float:
    return float(entry.get("episode_reward", entry.get("score", 0.0)))


def _status(entry: Dict[str, Any]) -> str:
    return str(entry.get("reward_class") or entry.get("status") or "unknown")


def summarize_rollouts(scored: List[Dict[str, Any]], n_gold_calls: int,
                       *, reward_policy: Optional[str] = None,
                       mode: Optional[str] = None
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
        "rollout_mode": mode or rollout_mode(),
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
