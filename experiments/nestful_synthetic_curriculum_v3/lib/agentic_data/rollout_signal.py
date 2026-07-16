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
  * parse/clipped rollouts are not dominant (<= ROLLOUT_MAX_PARSE_CLIP_RATE)
  * meaningful training signal: partial win, reward_range >= min, multiple
    failure classes, or different call-count strategies — not just numeric
    noise within one failure bucket

Quality tiers on acceptance:
  * frontier — partial/full wins in the sweet spot (primary training data)
  * partial_frontier — no full win but meaningfully different failures
  * easy_anchor — mostly correct (retain sparingly)
  * weak_partial / degenerate — rejected

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
from .env_defaults import (
    ROLLOUT_BORDERLINE_CONFIRM as _ROLLOUT_BORDERLINE_CONFIRM_DEFAULT,
    ROLLOUT_MAX_PARSE_CLIP_RATE as _ROLLOUT_MAX_PARSE_CLIP_RATE_DEFAULT,
    ROLLOUT_MAX_TOKENS as _ROLLOUT_MAX_TOKENS_DEFAULT,
    ROLLOUT_MIN_REWARD_RANGE as _ROLLOUT_MIN_REWARD_RANGE_DEFAULT,
    ROLLOUT_N as _ROLLOUT_N_DEFAULT,
    ROLLOUT_REQUIRE_ACHIEVABLE_WIN as _ROLLOUT_REQUIRE_ACHIEVABLE_WIN_DEFAULT,
    ROLLOUT_TEMPERATURE as _ROLLOUT_TEMPERATURE_DEFAULT,
    ROLLOUT_TOP_P as _ROLLOUT_TOP_P_DEFAULT,
    ROLLOUT_UNIVERSAL_MIN_REWARD_RANGE as _ROLLOUT_UNIVERSAL_MIN_REWARD_RANGE_DEFAULT,
    env_bool,
    env_float,
    env_int,
)

ROLLOUT_N = env_int("ROLLOUT_N", _ROLLOUT_N_DEFAULT)
ROLLOUT_TEMPERATURE = env_float("ROLLOUT_TEMPERATURE", _ROLLOUT_TEMPERATURE_DEFAULT)
ROLLOUT_TOP_P = env_float("ROLLOUT_TOP_P", _ROLLOUT_TOP_P_DEFAULT)
# Optional per-turn cap override for multi-turn rollouts (0 = use training
# config stage_defaults, which is what probe/GRPO use).
ROLLOUT_MAX_TOKENS = env_int("ROLLOUT_MAX_TOKENS", _ROLLOUT_MAX_TOKENS_DEFAULT)
ROLLOUT_MODE = os.environ.get("AGENTIC_ROLLOUT_MODE", "multiturn").strip().lower()

DEGENERATE_STATUSES = {"parse_error", "no_tool_call", "clipped"}
WIN_STATUSES = {"fully_correct", "win", "solution_equivalent"}
PARTIAL_SUCCESS_STATUSES = {
    "too_few_calls", "partial_progress", "executable_wrong_final",
    "correct_tool_wrong_args", "too_many_calls", "wrong_tool",
    "correct_prefix_then_stop", "partial_prefix",
}


def min_reward_range() -> float:
    """Meaningful reward spread for spread-only acceptance."""
    return env_float("ROLLOUT_MIN_REWARD_RANGE", _ROLLOUT_MIN_REWARD_RANGE_DEFAULT)


def universal_min_reward_range() -> float:
    """Hard floor — reject numeric micro-variance."""
    return env_float("ROLLOUT_UNIVERSAL_MIN_REWARD_RANGE",
                     _ROLLOUT_UNIVERSAL_MIN_REWARD_RANGE_DEFAULT)


def borderline_confirm_enabled() -> bool:
    return env_bool("ROLLOUT_BORDERLINE_CONFIRM", _ROLLOUT_BORDERLINE_CONFIRM_DEFAULT)


def max_parse_clip_rate() -> float:
    return env_float("ROLLOUT_MAX_PARSE_CLIP_RATE",
                     _ROLLOUT_MAX_PARSE_CLIP_RATE_DEFAULT)


def require_achievable_win() -> bool:
    """When True, accept only tasks where the weak model sometimes fully wins
    but not always — the sweet spot for GRPO (positive exemplars + variance).

    Rejects spread-only tasks where every rollout is partial failure / parse
    noise (still has unique_rewards >= 2 but the model never demonstrates a
    complete solution in any sample).
    """
    return env_bool("ROLLOUT_REQUIRE_ACHIEVABLE_WIN",
                    _ROLLOUT_REQUIRE_ACHIEVABLE_WIN_DEFAULT)

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
    # Agentic tools are synthetic — score rollouts with the SAME real
    # executor the trainer uses by default (executor.mode=synthetic): a
    # wrong predicted argument value executes for real and never falls back
    # to a gold observation, so the pre-training probe and GRPO training see
    # the identical reward landscape.
    overrides.append("executor.mode=synthetic")
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
    exec_mode = (config.get("executor", {}) or {}).get("mode", "auto")
    gold_obs = compute_gold_observations(task, registry, mode=exec_mode)

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


def _meaningful_failure_classes(failure_dist: Dict[str, int]) -> set:
    return {
        st for st in failure_dist
        if st not in WIN_STATUSES and st not in DEGENERATE_STATUSES
    }


def grpo_advantage_preview(
        rewards: List[float],
        statuses: Optional[List[str]] = None,
        *,
        mask_degenerate: bool = True,
) -> Dict[str, float]:
    """Trainer-aligned episode-group preview using ``group_stats.compute_group_stats``.

    Probes only have episode scalars (no turn ``r_seq``), so each rollout is
    modeled as a single-turn return ``[R_episode]`` — matching how
    ``turn_level_minimal`` behaves when every completion is one scalar reward.
    Clipped / parse / no-tool rollouts are masked out like ``mask_clipped``.
    """
    if not rewards:
        return {}
    from .training_reward import _ensure_training_import_paths
    _ensure_training_import_paths()
    from group_stats import compute_group_stats  # noqa: E402

    if statuses is None:
        included = [True] * len(rewards)
    else:
        included = [
            (not mask_degenerate) or (st not in DEGENERATE_STATUSES)
            for st in statuses
        ]
    ep_returns = [[float(r)] for r in rewards]
    gstats = compute_group_stats(ep_returns, [float(r) for r in rewards], included)
    flat_adv = [a for row in gstats.advantages for a in row]
    nonzero = sum(1 for a in flat_adv if abs(a) > 1e-9)
    n_adv = len(flat_adv) or 1
    return {
        "advantage_std": round(gstats.episode_reward_std, 8),
        "reward_std_episode": round(gstats.episode_reward_std, 8),
        "reward_std_between_completion": round(
            gstats.between_completion_std_max, 8),
        "max_abs_advantage": round(max((abs(a) for a in flat_adv), default=0.0), 8),
        "nonzero_advantage_fraction": round(nonzero / n_adv, 4),
        "dead_group_corrected": gstats.dead_corrected,
        "advantage_preview_mode": "group_stats_single_turn_proxy",
    }


def dominant_rollout_failure(failure_dist: Dict[str, int]) -> str:
    meaningful = {
        st: c for st, c in failure_dist.items()
        if st not in WIN_STATUSES and st not in DEGENERATE_STATUSES
    }
    if not meaningful:
        return "unknown"
    return max(meaningful, key=meaningful.get)


def is_borderline_probe(signal: Dict[str, Any]) -> bool:
    """True when an independent confirm probe is warranted."""
    if not signal or signal.get("skipped") or not signal.get("grpo_signal_positive"):
        return False
    rr = float(signal.get("reward_range") or 0.0)
    fsr = float(signal.get("full_success_rate") or 0.0)
    parse_rate = float(signal.get("parse_or_clipped_rate") or 0.0)
    unique = int(signal.get("unique_rewards") or 0)
    classes = signal.get("meaningful_failure_classes") or []
    if 0.01 <= rr < min_reward_range() and fsr == 0.0:
        return True
    if unique == 2 and rr < min_reward_range() and len(classes) <= 1:
        return True
    if parse_rate >= 0.20 and fsr == 0.0:
        return True
    if abs(parse_rate - max_parse_clip_rate()) < 0.001:
        return True
    return False


def _failure_pattern_stable(primary: Dict[str, Any],
                            secondary: Dict[str, Any],
                            combined: Dict[str, Any]) -> bool:
    pa = primary.get("dominant_rollout_failure")
    pb = secondary.get("dominant_rollout_failure")
    if pa and pb and pa == pb and pa != "unknown":
        return True
    ca = set(primary.get("meaningful_failure_classes") or [])
    cb = set(secondary.get("meaningful_failure_classes") or [])
    if ca & cb:
        return True
    return len(combined.get("meaningful_failure_classes") or []) >= 2


def _is_outlier_driven_borderline(primary: Dict[str, Any],
                                  secondary: Dict[str, Any],
                                  combined: Dict[str, Any]) -> bool:
    """Reject when contrast likely comes from one parse/outlier in one probe."""
    for probe in (primary, secondary):
        if probe.get("unique_rewards", 0) != 2:
            continue
        if (probe.get("parse_or_clipped_rate") or 0) < 0.125:
            continue
        rewards = probe.get("rewards") or []
        if not rewards:
            continue
        # One degenerate outlier + seven identical valid rewards.
        rounded = [round(r, 6) for r in rewards]
        counts: Dict[float, int] = {}
        for r in rounded:
            counts[r] = counts.get(r, 0) + 1
        if max(counts.values()) >= 6 and len(counts) == 2:
            return True
    # Both probes flat; combined spread only from pooling two flat modes.
    if (primary.get("unique_rewards", 0) == 1
            and secondary.get("unique_rewards", 0) == 1):
        pr = primary.get("reward_range") or 0.0
        sr = secondary.get("reward_range") or 0.0
        cr = combined.get("reward_range") or 0.0
        if cr > max(pr, sr) * 1.5 + 1e-6 and cr < min_reward_range():
            return True
    return False


def evaluate_grpo_signal(
    *,
    rewards: List[float],
    statuses: List[str],
    scored: List[Dict[str, Any]],
    n_gold_calls: int,
    unique_rewards: int,
    variance: float,
    full_success_rate: float,
    failure_dist: Dict[str, int],
    call_dist: Dict[str, int],
    has_valid_trace: bool,
    all_degenerate: bool,
) -> Tuple[bool, Optional[str], str]:
    """Return (grpo_signal_positive, grpo_sub_reason, quality_tier)."""
    n = len(rewards)
    if n == 0:
        return False, "empty_probe", "degenerate"

    reward_range = max(rewards) - min(rewards) if rewards else 0.0
    parse_or_clipped_rate = sum(
        1 for st in statuses if st in DEGENERATE_STATUSES) / n
    meaningful_classes = _meaningful_failure_classes(failure_dist)
    multiple_failure_classes = len(meaningful_classes) >= 2
    nonzero_call_buckets = {k for k in call_dist if int(k) > 0}
    different_call_strategies = len(nonzero_call_buckets) >= 2 or (
        len(call_dist) >= 2 and len({int(k) for k in call_dist}) >= 2)
    has_partial_success = 0.0 < full_success_rate < 0.999
    meaningful_signal = (
        has_partial_success
        or reward_range >= min_reward_range()
        or multiple_failure_classes
        or different_call_strategies
    )
    achievable_win = 0.0 < full_success_rate < 0.999
    all_correct = full_success_rate >= 0.999

    if unique_rewards < 2:
        return False, "all_same_reward", "degenerate"
    if variance <= 0.0:
        return False, "variance_below_threshold", "degenerate"
    if reward_range < universal_min_reward_range():
        return False, "reward_range_below_universal_min", "weak_partial"
    if all_correct:
        return False, "all_correct_trivial", "degenerate"
    if all_degenerate or not has_valid_trace:
        if parse_or_clipped_rate >= 1.0:
            if failure_dist.get("parse_error", 0) == n:
                return False, "all_parse_fail", "degenerate"
            if failure_dist.get("no_tool_call", 0) == n:
                return False, "all_no_tool", "degenerate"
        return False, "all_degenerate", "degenerate"
    if parse_or_clipped_rate > max_parse_clip_rate():
        return False, "parse_dominated", "degenerate"

    if require_achievable_win() and not achievable_win:
        sub = "no_full_success" if full_success_rate == 0.0 else "no_achievable_win_band"
        tier = "partial_frontier" if meaningful_signal else "weak_partial"
        return False, sub, tier

    if not meaningful_signal:
        if len(meaningful_classes) <= 1 and reward_range < min_reward_range():
            return False, "weak_partial_low_range", "weak_partial"
        return False, "no_meaningful_signal", "weak_partial"

    if 0.125 <= full_success_rate <= 0.75:
        tier = "frontier"
    elif full_success_rate > 0.75:
        tier = "easy_anchor"
    elif full_success_rate == 0.0:
        tier = "partial_frontier"
    else:
        tier = "partial_frontier"
    return True, None, tier


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
    reward_range = max(rewards) - min(rewards) if rewards else 0.0
    full_success_rate = round(
        sum(1 for r, st in zip(rewards, statuses)
            if st in WIN_STATUSES or r >= 0.999)
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
        if st not in WIN_STATUSES:
            failure_dist[st] = failure_dist.get(st, 0) + 1
    has_valid_trace = any(
        st not in DEGENERATE_STATUSES and (sc.get("n_calls") or 0) >= 1
        for sc, st in zip(scored, statuses))
    all_degenerate = all(st in DEGENERATE_STATUSES for st in statuses)
    parse_or_clipped_rate = round(
        sum(1 for st in statuses if st in DEGENERATE_STATUSES) / n, 3)
    meaningful_classes = sorted(_meaningful_failure_classes(failure_dist))
    achievable_win = 0.0 < full_success_rate < 0.999
    needs_achievable = require_achievable_win()
    grpo_signal_positive, grpo_sub_reason, quality_tier = evaluate_grpo_signal(
        rewards=rewards,
        statuses=statuses,
        scored=scored,
        n_gold_calls=n_gold_calls,
        unique_rewards=unique_rewards,
        variance=variance,
        full_success_rate=full_success_rate,
        failure_dist=failure_dist,
        call_dist=call_dist,
        has_valid_trace=has_valid_trace,
        all_degenerate=all_degenerate,
    )
    advantage_preview = grpo_advantage_preview(rewards, statuses)
    dom_failure = dominant_rollout_failure(failure_dist)
    borderline = (
        grpo_signal_positive and is_borderline_probe({
            "grpo_signal_positive": grpo_signal_positive,
            "reward_range": reward_range,
            "full_success_rate": full_success_rate,
            "parse_or_clipped_rate": parse_or_clipped_rate,
            "unique_rewards": unique_rewards,
            "meaningful_failure_classes": meaningful_classes,
        }))
    return {
        "n": n,
        "skipped": False,
        "rollout_mode": mode or rollout_mode(),
        "reward_policy": reward_policy,
        "rewards": [round(r, 6) for r in rewards],
        "unique_rewards": unique_rewards,
        "reward_variance": round(variance, 8),
        "reward_range": round(reward_range, 6),
        "reward_mean": round(mean, 6),
        "full_success_rate": full_success_rate,
        "correct_prefix_rate": correct_prefix_rate,
        "too_few_call_rate": too_few_call_rate,
        "parse_or_clipped_rate": parse_or_clipped_rate,
        "meaningful_failure_classes": meaningful_classes,
        "predicted_call_distribution": call_dist,
        "failure_type_distribution": failure_dist,
        "has_valid_trace": has_valid_trace,
        "all_degenerate": all_degenerate,
        "achievable_win": achievable_win,
        "requires_achievable_win": needs_achievable,
        "grpo_signal_positive": grpo_signal_positive,
        "grpo_sub_reason": grpo_sub_reason,
        "quality_tier": quality_tier,
        "universal_min_reward_range": universal_min_reward_range(),
        "min_reward_range": min_reward_range(),
        "max_parse_clip_rate": max_parse_clip_rate(),
        "dominant_rollout_failure": dom_failure,
        "borderline_probe": borderline,
        **advantage_preview,
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


def probe_rollout_signal_confirmed(
        question: str, tools: List[Dict[str, Any]],
        gold_calls: List[Dict[str, Any]],
        gold_observations: List[Any], gold_answer: Any,
        *, stage: str,
        n: Optional[int] = None, seed: Optional[int] = None,
        confirm: Optional[bool] = None) -> Dict[str, Any]:
    """Primary 8-rollout probe; borderline cases merge A+B (16) and re-gate."""
    if not target_is_local():
        return {"skipped": True, "reason": "WEAK_SOLVER_BACKEND != local — "
                "exact target Qwen3-4B setup unavailable"}
    scored_a = run_rollouts(
        question, tools, gold_calls, gold_observations, gold_answer,
        stage=stage, n=n, seed=seed)
    primary = summarize_rollouts(scored_a, len(gold_calls))
    if primary.get("skipped"):
        return primary
    do_confirm = borderline_confirm_enabled() if confirm is None else confirm
    if not do_confirm or not primary.get("borderline_probe"):
        primary["confirm_probe_skipped"] = not do_confirm
        return primary

    confirm_seed = (int(seed) + 100_003) if seed is not None else None
    scored_b = run_rollouts(
        question, tools, gold_calls, gold_observations, gold_answer,
        stage=stage, n=n, seed=confirm_seed)
    secondary = summarize_rollouts(scored_b, len(gold_calls))
    combined = summarize_rollouts(scored_a + scored_b, len(gold_calls))

    combined["borderline_probe"] = True
    combined["confirm_probe"] = {
        "seed": confirm_seed,
        "n_primary": primary.get("n"),
        "n_confirm": secondary.get("n"),
        "n_combined": combined.get("n"),
        "primary_grpo_ok": primary.get("grpo_signal_positive"),
        "confirm_grpo_ok": secondary.get("grpo_signal_positive"),
        "primary_sub_reason": primary.get("grpo_sub_reason"),
        "confirm_sub_reason": secondary.get("grpo_sub_reason"),
        "primary_dominant_failure": primary.get("dominant_rollout_failure"),
        "confirm_dominant_failure": secondary.get("dominant_rollout_failure"),
        "failure_pattern_stable": _failure_pattern_stable(
            primary, secondary, combined),
        "outlier_driven": _is_outlier_driven_borderline(
            primary, secondary, combined),
    }

    reject_reason: Optional[str] = None
    if not combined.get("grpo_signal_positive"):
        reject_reason = "borderline_confirm_failed"
    elif primary.get("all_degenerate") and secondary.get("all_degenerate"):
        reject_reason = "borderline_both_degenerate"
    elif _is_outlier_driven_borderline(primary, secondary, combined):
        reject_reason = "borderline_outlier_driven"
    elif not _failure_pattern_stable(primary, secondary, combined):
        reject_reason = "borderline_unstable_failure_pattern"

    if reject_reason:
        combined["grpo_signal_positive"] = False
        combined["grpo_sub_reason"] = reject_reason
        combined["quality_tier"] = "weak_partial"
    else:
        combined["borderline_confirmed"] = True
    return combined
