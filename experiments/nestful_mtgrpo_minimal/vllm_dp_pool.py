"""Data-parallel rollout pool for vLLM-accelerated MT-GRPO training.

Motivation
----------
During training the HF QLoRA learner lives on GPU 0. Rollout *generation* (the
dominant cost: num_generations episodes per task, each multi-turn) is fully
INDEPENDENT of the HF model when vLLM is used — it needs only the tokenizer, the
executor and a vLLM engine. So we can run rollouts in parallel on the OTHER
GPUs, each worker owning a single vLLM engine (tensor_parallel_size=1), and feed
the results back to the learner on GPU 0.

Design (one worker per GPU, whole-episode in worker)
----------------------------------------------------
* Each worker process pins itself to ONE GPU (CUDA_VISIBLE_DEVICES set BEFORE
  importing torch/vllm), builds a single vLLM engine, and runs ENTIRE episodes
  (generate + tool-execute loop) — never touching the HF model. Episodes queued
  on the same worker are interleaved so each assistant-turn wave is a real vLLM
  prompt batch instead of several serial batch-size-one calls.
* The worker also computes the training reward (strict OR partial, selected from
  ``config['reward']['train_policy']``) so that raw tool observations — which can
  be arbitrary, non-picklable Python objects — NEVER cross the process boundary.
* The worker returns a small, fully-picklable :class:`RolloutResult`: per-turn
  token-id lists (for the parent's log-prob pass), the episode reward, the
  per-turn reward sequence, and a few diagnostic scalars. The parent re-wraps the
  token-id lists as tensors and runs the existing GRPO update on GPU 0.

Reward policy across processes
------------------------------
The partial experiment selects its reward by monkeypatching
``grpo_train.episode_turn_reward_seq`` in the PARENT. Spawned workers would not
see that. Instead the worker picks the reward function explicitly from
``config['reward']['train_policy']`` (``partial_gold_trace`` → partial_reward,
else strict reward) and loads partial weights from ``config['partial_reward']``.
The parent's snapshot of ``sys.path`` is forwarded so ``partial_reward`` (which
lives in the sibling folder) is importable in the worker.

Opt-in
------
This whole machinery is OFF by default. It is only constructed when the caller
passes a non-empty GPU list (driven by ``hardware.rollout_data_parallel_gpus`` /
the ``ROLLOUT_DP_GPUS`` env var). The single-engine path is unchanged.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  Serializable result (worker -> parent). Plain Python types only.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RolloutResult:
    """Everything the trainer needs from one episode, fully picklable.

    ``turn_token_ids`` is a list of (prompt_ids, completion_ids) Python int lists;
    the parent re-wraps them as 1-D LongTensors for the log-prob computation.
    """
    turn_token_ids: List[Tuple[List[int], List[int]]] = field(default_factory=list)
    episode_reward: float = 0.0
    r_seq: List[float] = field(default_factory=list)
    clipped_any: bool = False
    prompt_overflow: bool = False
    zero_tool_calls: bool = False
    num_tool_calls: int = 0
    stop_reason: Optional[str] = None
    first_error_turn: Optional[int] = None
    error: Optional[str] = None  # set if the episode crashed in the worker
    # Scalar-only reward diagnostics (sanitized in the worker) for group logging.
    reward_diag: Dict[str, Any] = field(default_factory=dict)
    # Optional full trajectory summary for dispatch-canary audits
    # (set only when CANARY_TRAJ_LOG=1). Kept off the hot path otherwise.
    canary_traj: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Reward-policy resolution (runs in the worker, no monkeypatch dependency)
# ─────────────────────────────────────────────────────────────────────────────

_STRICT_POLICY_ALIASES = ("strict", "strict_gold_trace", "strict_gold_trace_legacy")


def resolve_reward_info(config: Dict[str, Any]) -> Tuple[Callable, Dict[str, Any]]:
    """Resolve config['reward']['train_policy'] to a reward function.

    Returns (fn, info) where info records exactly what was resolved:
        configured_policy / resolved_policy / reward_fn_module /
        reward_fn_name / fallback_used

    HARD-FAILS (ValueError) on an unknown policy unless the environment
    explicitly allows the strict fallback via ALLOW_STRICT_REWARD_FALLBACK=1.
    This replaces the previous SILENT fallback that invalidated the v3/v3.1
    pilots (audit Bug 1).
    """
    configured = str((config.get("reward", {}) or {}).get("train_policy", "strict"))
    policy = configured.lower()
    fallback_used = False

    if policy in ("partial_gold_trace", "partial"):
        import partial_reward
        partial_reward.set_weights_from_config(config)
        fn = partial_reward.episode_turn_reward_seq
    elif policy in ("execution_aware_v2_p43", "execution_v2_p43", "p43"):
        # No silent fallback to execution_aware / execution_aware_v2.
        import execution_reward_v2_p43
        execution_reward_v2_p43.set_weights_from_config(config)
        fn = execution_reward_v2_p43.episode_turn_reward_seq
        cfg_variant = str((config.get("reward") or {}).get("p43_reward_variant") or "").upper()
        impl_variant = str(getattr(fn, "p43_reward_variant", "") or "").upper()
        if cfg_variant and impl_variant and cfg_variant != impl_variant:
            raise ValueError(
                f"[reward_dispatch] P43 variant mismatch: config={cfg_variant!r} "
                f"implementation={impl_variant!r}. Freeze one variant in YAML.")
        if not cfg_variant:
            raise ValueError(
                "[reward_dispatch] reward.train_policy=execution_aware_v2_p43 requires "
                "reward.p43_reward_variant explicitly set to A or B (no auto-select).")
    elif policy in ("execution_aware_v2", "execution_v2"):
        import execution_reward_v2
        execution_reward_v2.set_weights_from_config(config)
        fn = execution_reward_v2.episode_turn_reward_seq
    elif policy in ("partial_gold_trace_v2", "partial_v2"):
        import partial_reward_v2
        partial_reward_v2.set_weights_from_config(config)
        fn = partial_reward_v2.episode_turn_reward_seq
    elif policy in ("execution_aware", "execution"):
        import execution_reward
        execution_reward.set_weights_from_config(config)
        fn = execution_reward.episode_turn_reward_seq
    elif policy in _STRICT_POLICY_ALIASES:
        from reward import episode_turn_reward_seq as strict_seq
        fn = strict_seq
    else:
        if os.environ.get("ALLOW_STRICT_REWARD_FALLBACK", "0") == "1":
            print(f"[reward_dispatch] WARNING: unknown reward policy '{configured}' — "
                  f"falling back to STRICT gold-trace reward because "
                  f"ALLOW_STRICT_REWARD_FALLBACK=1", flush=True)
            from reward import episode_turn_reward_seq as strict_seq
            fn = strict_seq
            fallback_used = True
        else:
            raise ValueError(
                f"[reward_dispatch] Unknown reward policy '{configured}'. "
                f"Known: execution_aware_v2_p43, partial_gold_trace, execution_aware_v2, "
                f"partial_gold_trace_v2, execution_aware, strict. "
                f"Refusing to silently fall back to the strict binary reward "
                f"(set ALLOW_STRICT_REWARD_FALLBACK=1 to override — NOT recommended)."
            )

    resolved_policy = getattr(fn, "reward_policy", None) or (
        "strict" if fallback_used or policy in _STRICT_POLICY_ALIASES else configured)
    # Normalize aliases so requested==resolved checks are stable.
    if str(resolved_policy).lower() in ("execution_v2_p43", "p43"):
        resolved_policy = "execution_aware_v2_p43"
    if str(configured).lower() in ("execution_v2_p43", "p43"):
        configured_norm = "execution_aware_v2_p43"
    else:
        configured_norm = configured
    dispatch_cfg = (config.get("reward") or {}).get("dispatch") or {}
    require_exact = bool(dispatch_cfg.get("require_exact_policy", False))
    allow_fallback = bool(dispatch_cfg.get("allow_fallback", True))
    if require_exact and not allow_fallback:
        if str(configured_norm).lower() != str(resolved_policy).lower():
            raise ValueError(
                f"[reward_dispatch] HARD ERROR: requested={configured_norm!r} "
                f"!= resolved={resolved_policy!r}. No fallback allowed.")
    info = {
        "configured_policy": configured_norm,
        "resolved_policy": resolved_policy,
        "reward_fn_module": getattr(fn, "__module__", "?"),
        "reward_fn_name": getattr(fn, "__name__", "?"),
        "fallback_used": fallback_used,
        "p43_reward_variant": (config.get("reward") or {}).get("p43_reward_variant"),
        "reward_implementation_module": getattr(fn, "__module__", "?"),
    }
    return fn, info


def _resolve_reward_fn(config: Dict[str, Any]) -> Callable:
    """Return the episode_turn_reward_seq matching config['reward']['train_policy']."""
    fn, _ = resolve_reward_info(config)
    return fn


def _encode_for_logprob(tokenizer, messages, completion_text: str) -> Tuple[List[int], List[int]]:
    """Re-tokenise (messages, completion) as plain int lists.

    Mirrors grpo_train._retokenize_for_logprob token IDs exactly (chat-template
    tokenisation is deterministic), but returns lists so the result is cheap and
    safe to ship across the process boundary.
    """
    enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    # Some templates return a nested [[...]] when batched; flatten one level.
    if enc and isinstance(enc[0], (list, tuple)):
        enc = enc[0]
    prompt_ids = [int(x) for x in enc]
    comp_ids = [int(x) for x in tokenizer.encode(completion_text, add_special_tokens=False)]
    return prompt_ids, comp_ids


# ─────────────────────────────────────────────────────────────────────────────
#  Per-episode rollout (worker side). Testable in-process via generate_fn inject.
# ─────────────────────────────────────────────────────────────────────────────

def run_episode_collect(
    *,
    tokenizer,
    task: Dict[str, Any],
    config: Dict[str, Any],
    registry,
    generate_fn: Callable[[List[Dict[str, str]], int], Dict[str, Any]],
    reward_fn: Callable,
    gold_obs=None,
) -> RolloutResult:
    """Run ONE training episode with vLLM-style generation and collect a
    fully-picklable RolloutResult (token-id lists + reward + diagnostics).

    Mirrors the vLLM branch of grpo_train._rollout_episode_for_train, but the
    reward is computed here (worker side) and observations stay local.
    """
    from rollout import (
        Trajectory, Turn, get_stage_token_budget,
        resolve_teacher_forced_prefix_n, build_teacher_forced_prefix,
    )
    from executor import ToolExecutor
    from reward import compute_gold_observations, strict_gold_trace_reward
    from interaction_loop import derive_interaction_budget, run_tool_agent_loop

    exec_cfg = config.get("executor", {})
    gold_n = int(task.get("num_calls") or len(task.get("gold_calls", [])))
    token_budget = get_stage_token_budget(config, gold_n, "train")
    max_new_tokens = token_budget["max_new_tokens"]

    if gold_obs is None:
        gold_obs = compute_gold_observations(
            task, registry, mode=exec_cfg.get("mode", "auto"))

    executor = ToolExecutor(
        task, registry=registry, mode=exec_cfg.get("mode", "auto"),
        ibm_call_timeout=float(exec_cfg.get("ibm_call_timeout", 30.0)),
    )
    traj = Trajectory(task["task_id"], gold_n, gold_n, executor_mode=executor.mode)
    turn_token_ids: List[Tuple[List[int], List[int]]] = []
    history: List[Dict[str, str]] = []

    # ── Teacher-forced continuation prefix (opt-in, off by default) ─────────
    configured_prefix = int((config.get("train", {}) or {}).get(
        "teacher_forced_prefix_calls", 0) or 0)
    n_forced = resolve_teacher_forced_prefix_n(
        task, configured_prefix, executor.mode, gold_obs)
    if n_forced > 0:
        forced_turns, forced_history = build_teacher_forced_prefix(
            task, executor, n_forced)
        traj.turns.extend(forced_turns)
        history.extend(forced_history)
        traj.final_observation = forced_turns[-1].observation

    ibudget = derive_interaction_budget(gold_n, config, mode="train")

    def _encode(messages, text):
        return _encode_for_logprob(tokenizer, messages, text)

    def _on_turn(turn: Turn):
        from prompt import build_messages as _bm
        hist_wo = list(history)
        if hist_wo and hist_wo[-1].get("role") == "assistant":
            hist_wo = hist_wo[:-1]
        msgs = _bm(task, hist_wo)
        p_ids, c_ids = _encode(msgs, turn.model_text)
        turn_token_ids.append((p_ids, c_ids))

    run_tool_agent_loop(
        task=task, config=config, executor=executor, traj=traj,
        history=history, generate_fn=generate_fn,
        max_new_tokens=max_new_tokens, budget=ibudget, n_forced=n_forced,
        lenient_parse=False, eval_hardening=False,
        encode_fn=_encode, on_generated_turn=_on_turn,
    )

    # Reward (policy-selected) computed HERE so observations never leave the worker.
    rinfo = reward_fn(traj, task, gold_obs)
    # strict first_error_turn for logging parity with the single-engine path.
    strict_diag = strict_gold_trace_reward(traj, task, gold_obs).diagnostics

    # ── r_seq alignment (Bug: teacher-forced turns must NOT get a gradient) ──
    # Every reward's `_turn_scores`-style helper emits exactly one score per
    # `trajectory.turns` entry, in order. Forced turns are always the FIRST
    # `n_forced` entries (built before any generated turn), so dropping them
    # from the front keeps r_seq aligned 1:1 with turn_token_ids (generated
    # turns only). Hard-fail rather than silently mis-aligning if a reward
    # function ever violates the 1:1-with-turns contract under forcing.
    r_seq_full = [float(x) for x in rinfo["r_seq"]]
    if len(r_seq_full) != len(traj.turns):
        raise RuntimeError(
            f"[teacher_forced] reward r_seq length {len(r_seq_full)} != "
            f"len(trajectory.turns) {len(traj.turns)} for task "
            f"{task.get('task_id')} (n_forced={n_forced}); refusing to guess "
            f"turn alignment.")
    r_seq = r_seq_full[n_forced:]
    if len(r_seq) != len(turn_token_ids):
        raise RuntimeError(
            f"[teacher_forced] post-slice r_seq length {len(r_seq)} != "
            f"turn_token_ids length {len(turn_token_ids)} for task "
            f"{task.get('task_id')} (n_forced={n_forced}).")

    reward_diag = dict(rinfo.get("diagnostics") or {})
    reward_diag["teacher_forced_prefix_calls"] = n_forced
    # Executor-failure categories (argument/reference/runtime errors) for
    # train-log aggregation — scalars only, survives _sanitize_diag.
    from rollout import exec_failure_categories
    reward_diag.update(exec_failure_categories(traj))
    meta = getattr(traj, "interaction_meta", None) or {}
    if isinstance(meta, dict):
        reward_diag.update(meta)
    from success_metrics import compute_rollout_success_flags
    reward_diag.update(compute_rollout_success_flags(traj, task, reward_diag))

    canary_traj = None
    if os.environ.get("CANARY_TRAJ_LOG", "").strip().lower() in ("1", "true", "yes"):
        canary_traj = _build_canary_traj(traj, rinfo)

    return RolloutResult(
        turn_token_ids=turn_token_ids,
        episode_reward=float(rinfo["episode_reward"]),
        r_seq=r_seq,
        clipped_any=bool(traj.clipped_any),
        prompt_overflow=bool(traj.prompt_overflow),
        zero_tool_calls=bool(traj.zero_tool_calls),
        num_tool_calls=int(traj.num_tool_calls),
        stop_reason=traj.stop_reason,
        first_error_turn=strict_diag.get("first_error_turn"),
        reward_diag=_sanitize_diag(reward_diag),
        canary_traj=canary_traj,
    )


@dataclass
class _BatchedEpisodeState:
    """Mutable state for one episode in a worker-local vLLM batch."""

    task: Dict[str, Any]
    gold_obs: Any
    executor: Any
    traj: Any
    turn_token_ids: List[Tuple[List[int], List[int]]]
    history: List[Dict[str, str]]
    budget: Any
    meta: Any
    n_forced: int
    remaining: int
    max_new_tokens: int
    rollout_seed: int
    step: int = 0
    done: bool = False
    error: Optional[str] = None


def _prepare_batched_episode(
    *, tokenizer, task: Dict[str, Any], config: Dict[str, Any], registry,
    gold_obs,
) -> _BatchedEpisodeState:
    """Create the same initial state as :func:`run_episode_collect`."""
    from executor import ToolExecutor
    from interaction_loop import InteractionMeta, derive_interaction_budget
    from rollout import (
        Trajectory, build_teacher_forced_prefix, get_stage_token_budget,
        resolve_teacher_forced_prefix_n,
    )

    exec_cfg = config.get("executor", {})
    gold_n = int(task.get("num_calls") or len(task.get("gold_calls", [])))
    token_budget = get_stage_token_budget(config, gold_n, "train")
    executor = ToolExecutor(
        task, registry=registry, mode=exec_cfg.get("mode", "auto"),
        ibm_call_timeout=float(exec_cfg.get("ibm_call_timeout", 30.0)),
    )
    traj = Trajectory(task["task_id"], gold_n, gold_n, executor_mode=executor.mode)
    history: List[Dict[str, str]] = []
    configured_prefix = int((config.get("train", {}) or {}).get(
        "teacher_forced_prefix_calls", 0) or 0)
    n_forced = resolve_teacher_forced_prefix_n(
        task, configured_prefix, executor.mode, gold_obs)
    if n_forced > 0:
        forced_turns, forced_history = build_teacher_forced_prefix(
            task, executor, n_forced)
        traj.turns.extend(forced_turns)
        history.extend(forced_history)
        traj.final_observation = forced_turns[-1].observation

    budget = derive_interaction_budget(gold_n, config, mode="train")
    return _BatchedEpisodeState(
        task=task,
        gold_obs=gold_obs,
        executor=executor,
        traj=traj,
        turn_token_ids=[],
        history=history,
        budget=budget,
        meta=InteractionMeta(
            tool_budget=budget.max_tool_calls,
            assistant_turn_budget=budget.max_assistant_turns,
        ),
        n_forced=n_forced,
        remaining=max(1, budget.max_assistant_turns - n_forced),
        max_new_tokens=int(token_budget["max_new_tokens"]),
        rollout_seed=int(task.get("_rollout_seed") or 0),
    )


def _finish_interaction_state(state: _BatchedEpisodeState) -> None:
    """Apply the sequential interaction loop's end-of-budget bookkeeping."""
    from interaction_loop import attach_budget_fields

    if state.traj.stop_reason is None:
        state.traj.stop_reason = "max_assistant_turns"
        state.meta.stop_reason = "max_assistant_turns"
    if state.traj.stop_reason == "terminal":
        state.traj.stop_reason = "terminal_answer"
        state.meta.stop_reason = "terminal_answer"
    state.meta.final_answer_present = bool(
        state.meta.final_answer_present
        or any(getattr(t, "is_terminal", False) for t in state.traj.turns)
    )
    attach_budget_fields(state.traj, state.budget, state.meta)


def _advance_batched_episode(
    state: _BatchedEpisodeState,
    messages: List[Dict[str, str]],
    generation: Dict[str, Any],
    tokenizer,
) -> None:
    """Consume one generated assistant turn, matching run_tool_agent_loop."""
    from parser import parse_tool_call
    from prompt import format_tool_response
    from rollout import Turn

    turn_idx = state.n_forced + state.step
    state.step += 1
    state.meta.assistant_turns += 1

    if generation.get("prompt_overflow"):
        state.traj.prompt_overflow = True
        state.traj.clipped_any = True
        state.traj.stop_reason = "prompt_overflow"
        state.meta.stop_reason = "prompt_overflow"
        state.done = True
        return

    text = generation.get("text") or ""
    clipped = bool(generation.get("clipped", False))
    p_ids, c_ids = _encode_for_logprob(tokenizer, messages, text)
    turn = Turn(
        turn_idx, text, prompt_tokens=len(p_ids), completion_tokens=len(c_ids),
        clipped_completion=clipped,
    )
    state.turn_token_ids.append((p_ids, c_ids))
    state.history.append({"role": "assistant", "content": text})

    if clipped:
        state.traj.clipped_any = True
        turn.fail_reason = "clipped_completion"
        state.traj.turns.append(turn)
        state.traj.stop_reason = "clipped"
        state.meta.stop_reason = "clipped"
        state.done = True
        return

    in_final_phase = (
        state.meta.tool_calls_executed >= state.budget.max_tool_calls)
    if in_final_phase:
        state.meta.final_response_turn_attempted = True

    parsed = parse_tool_call(text, lenient=False)
    if parsed.is_terminal:
        turn.is_terminal = True
        state.traj.turns.append(turn)
        state.traj.stop_reason = "terminal_answer"
        state.meta.stop_reason = "terminal_answer"
        state.meta.final_answer_present = True
        state.done = True
        return

    if not parsed.ok:
        turn.fail_reason = f"parse:{parsed.reason}"
        state.traj.turns.append(turn)
        state.traj.stop_reason = "parse_fail"
        state.meta.stop_reason = "parse_fail"
        state.done = True
        return

    call = parsed.call
    turn.parsed_call = call
    state.meta.tool_calls_emitted += 1
    if in_final_phase:
        turn.fail_reason = "final_turn_tool_attempt"
        state.meta.final_turn_tool_attempt = True
        state.traj.turns.append(turn)
        state.traj.stop_reason = "final_turn_tool_attempt"
        state.meta.stop_reason = "final_turn_tool_attempt"
        state.done = True
        return

    if state.meta.tool_calls_executed >= state.budget.max_tool_calls:
        turn.fail_reason = "max_tool_calls"
        state.meta.final_turn_tool_attempt = True
        state.meta.final_response_turn_attempted = True
        state.traj.turns.append(turn)
        state.traj.stop_reason = "max_tool_calls"
        state.meta.stop_reason = "max_tool_calls"
        state.done = True
        return

    execution = state.executor.execute(call)
    turn.observation = execution.observation
    if execution.error is not None:
        turn.fail_reason = f"exec:{execution.error}"
        state.traj.turns.append(turn)
        state.traj.stop_reason = "executor_error"
        state.meta.stop_reason = "executor_error"
        state.done = True
        return

    state.meta.tool_calls_executed += 1
    state.traj.final_observation = execution.observation
    state.traj.turns.append(turn)
    state.history.append({
        "role": "user",
        "content": format_tool_response(call, execution.observation),
    })
    if state.step >= state.remaining:
        state.done = True


def _result_from_batched_episode(
    state: _BatchedEpisodeState, reward_fn: Callable,
) -> RolloutResult:
    """Score a completed batched episode and build its picklable result."""
    from reward import strict_gold_trace_reward
    from rollout import exec_failure_categories
    from success_metrics import compute_rollout_success_flags

    _finish_interaction_state(state)
    rinfo = reward_fn(state.traj, state.task, state.gold_obs)
    strict_diag = strict_gold_trace_reward(
        state.traj, state.task, state.gold_obs).diagnostics
    r_seq_full = [float(x) for x in rinfo["r_seq"]]
    if len(r_seq_full) != len(state.traj.turns):
        raise RuntimeError(
            f"[teacher_forced] reward r_seq length {len(r_seq_full)} != "
            f"len(trajectory.turns) {len(state.traj.turns)} for task "
            f"{state.task.get('task_id')} (n_forced={state.n_forced})")
    r_seq = r_seq_full[state.n_forced:]
    if len(r_seq) != len(state.turn_token_ids):
        raise RuntimeError(
            f"[teacher_forced] post-slice r_seq length {len(r_seq)} != "
            f"turn_token_ids length {len(state.turn_token_ids)} for task "
            f"{state.task.get('task_id')} (n_forced={state.n_forced})")

    reward_diag = dict(rinfo.get("diagnostics") or {})
    reward_diag["teacher_forced_prefix_calls"] = state.n_forced
    reward_diag.update(exec_failure_categories(state.traj))
    meta = getattr(state.traj, "interaction_meta", None) or {}
    if isinstance(meta, dict):
        reward_diag.update(meta)
    reward_diag.update(compute_rollout_success_flags(
        state.traj, state.task, reward_diag))
    canary_traj = None
    if os.environ.get("CANARY_TRAJ_LOG", "").strip().lower() in (
            "1", "true", "yes"):
        canary_traj = _build_canary_traj(state.traj, rinfo)
    return RolloutResult(
        turn_token_ids=state.turn_token_ids,
        episode_reward=float(rinfo["episode_reward"]),
        r_seq=r_seq,
        clipped_any=bool(state.traj.clipped_any),
        prompt_overflow=bool(state.traj.prompt_overflow),
        zero_tool_calls=bool(state.traj.zero_tool_calls),
        num_tool_calls=int(state.traj.num_tool_calls),
        stop_reason=state.traj.stop_reason,
        first_error_turn=strict_diag.get("first_error_turn"),
        reward_diag=_sanitize_diag(reward_diag),
        canary_traj=canary_traj,
    )


def run_episodes_collect_batch(
    *, tokenizer, tasks: List[Dict[str, Any]], config: Dict[str, Any], registry,
    generate_batch_fn: Callable[
        [List[Tuple[List[Dict[str, str]], int, Optional[int]]]],
        List[Dict[str, Any]]],
    reward_fn: Callable,
) -> List[RolloutResult]:
    """Interleave several tool episodes and batch every assistant-turn wave.

    Tasks remain independent (separate executor/history/RNG seed), but active
    prompts are submitted to vLLM together.  This is the worker-side throughput
    fix: a worker that owns 2-3 GRPO completions now decodes them concurrently
    instead of making 2-3 serial ``LLM.generate([prompt])`` calls per turn.
    """
    import traceback

    from prompt import build_messages
    from reward import compute_gold_observations
    from rollout_sampling import derive_turn_seed

    results: List[Optional[RolloutResult]] = [None] * len(tasks)
    states: Dict[int, _BatchedEpisodeState] = {}
    gold_cache: Dict[str, Any] = {}
    exec_mode = (config.get("executor", {}) or {}).get("mode", "auto")

    for idx, task in enumerate(tasks):
        try:
            cache_key = str(task.get("task_id") or idx)
            if cache_key not in gold_cache:
                gold_cache[cache_key] = compute_gold_observations(
                    task, registry, mode=exec_mode)
            states[idx] = _prepare_batched_episode(
                tokenizer=tokenizer, task=task, config=config, registry=registry,
                gold_obs=gold_cache[cache_key],
            )
        except Exception as exc:  # isolate setup failure to one rollout
            results[idx] = RolloutResult(
                error=f"{exc}\n{traceback.format_exc()}")

    while True:
        active: List[Tuple[int, _BatchedEpisodeState, List[Dict[str, str]]]] = []
        requests = []
        for idx, state in states.items():
            if state.done:
                continue
            if state.step >= state.remaining:
                state.done = True
                continue
            messages = build_messages(state.task, state.history)
            seed = (derive_turn_seed(state.rollout_seed, state.step)
                    if state.rollout_seed else None)
            active.append((idx, state, messages))
            requests.append((messages, state.max_new_tokens, seed))
        if not active:
            break

        generations = generate_batch_fn(requests)
        if len(generations) != len(active):
            raise RuntimeError(
                f"batch generator returned {len(generations)} results for "
                f"{len(active)} active episodes")
        for (idx, state, messages), generation in zip(active, generations):
            try:
                _advance_batched_episode(state, messages, generation, tokenizer)
            except Exception as exc:  # isolate executor/parser failure
                state.error = f"{exc}\n{traceback.format_exc()}"
                state.done = True

    for idx, state in states.items():
        if state.error:
            results[idx] = RolloutResult(error=state.error)
            continue
        try:
            results[idx] = _result_from_batched_episode(state, reward_fn)
        except Exception as exc:
            results[idx] = RolloutResult(
                error=f"{exc}\n{traceback.format_exc()}")

    return [r if r is not None else RolloutResult(
        error="batch rollout produced no result") for r in results]


def _sanitize_diag(diag: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only cheap picklable scalars (and short float lists) for transport."""
    out: Dict[str, Any] = {}
    for k, v in diag.items():
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        elif isinstance(v, list) and len(v) <= 16 and all(
                isinstance(x, (bool, int, float)) for x in v):
            out[k] = v
    return out


def _jsonable(obj: Any, *, max_str: int = 4000) -> Any:
    """Best-effort JSON-safe truncation for canary trajectory dumps."""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= max_str else obj[:max_str] + "…[truncated]"
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, max_str=max_str) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x, max_str=max_str) for x in obj]
    return _jsonable(str(obj), max_str=max_str)


def _build_canary_traj(traj, rinfo: Dict[str, Any]) -> Dict[str, Any]:
    """Compact per-rollout trajectory payload for credit / dispatch audits."""
    diag = dict(rinfo.get("diagnostics") or {})
    turns_out = []
    for t in traj.turns:
        turns_out.append({
            "turn_idx": int(getattr(t, "turn_idx", 0) or 0),
            "raw_output": _jsonable(getattr(t, "model_text", "") or ""),
            "parsed_call": _jsonable(getattr(t, "parsed_call", None)),
            "observation": _jsonable(getattr(t, "observation", None)),
            "fail_reason": getattr(t, "fail_reason", None),
            "is_terminal": bool(getattr(t, "is_terminal", False)),
            "teacher_forced": bool(getattr(t, "teacher_forced", False)),
        })
    return {
        "raw_outputs": [x["raw_output"] for x in turns_out],
        "parsed_calls": [x["parsed_call"] for x in turns_out if x["parsed_call"] is not None],
        "observations": [x["observation"] for x in turns_out],
        "turns": turns_out,
        "stop_reason": traj.stop_reason,
        "executor_outcome": {
            "num_tool_calls": int(traj.num_tool_calls),
            "zero_tool_calls": bool(traj.zero_tool_calls),
            "clipped_any": bool(traj.clipped_any),
            "executor_error": bool(traj.executor_error),
            "stop_reason": traj.stop_reason,
        },
        "terminal_class": diag.get("terminal_class"),
        "failure_class": diag.get("failure_class") or diag.get("reward_class"),
        "reward_diag": _sanitize_diag(diag),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Worker process main loop
# ─────────────────────────────────────────────────────────────────────────────

def _worker_main(worker_id: int, gpu: int, config: Dict[str, Any],
                 adapter_path: Optional[str], extra_sys_path: List[str],
                 in_q, out_q) -> None:
    """Worker entry point. Pins to ONE GPU, builds a vLLM engine, serves rollouts.

    Protocol (messages on in_q):
        ("rollout", (req_id, task))   -> out_q.put((req_id, RolloutResult))
        ("rollout_batch", [(req_id, task), ...]) -> one result per request
        ("sync",    adapter_path)     -> out_q.put((("__ack__", worker_id), "sync"))
        ("ping",    None)             -> out_q.put((("__ack__", worker_id), "ready"))
        ("stop",    None)             -> exits
    """
    # MUST happen before importing torch / vllm so the worker sees only its GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for p in reversed(extra_sys_path):
        if p and p not in sys.path:
            sys.path.insert(0, p)

    vgen = None
    try:
        # Resolve the reward FIRST so a dispatch failure aborts before any
        # engine is built and before any rollout happens (audit Bug 1).
        reward_fn, reward_info = resolve_reward_info(config)
        print(
            f"[dp_worker {worker_id}] reward.train_policy={reward_info['configured_policy']} "
            f"resolved_reward_fn={reward_info['reward_fn_module']}."
            f"{reward_info['reward_fn_name']} "
            f"resolved_policy={reward_info['resolved_policy']} "
            f"fallback_used={str(reward_info['fallback_used']).lower()}",
            flush=True,
        )

        from transformers import AutoTokenizer
        from vllm_generate import build_vllm_generator

        base_model = config["model"]["base_model"]
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # No HF model shares this GPU -> the engine may use a high memory fraction.
        # Diversify LLM engine seed per worker (defense in depth). Per-request
        # SamplingParams.seed from task["_rollout_seed"] is the primary fix for
        # DP round-robin duplicate pairs (0==1, 3==4, 6==7).
        vgen = build_vllm_generator(
            config, tokenizer,
            adapter_path=adapter_path, mode="rollout_worker",
            engine_seed=int(worker_id) + 1,
        )
        out_q.put((("__ack__", worker_id), "ready"))
    except Exception as exc:  # noqa: BLE001 — report init failure, do not hang parent
        import traceback
        out_q.put((("__ack__", worker_id), f"init_error: {exc}\n{traceback.format_exc()}"))
        return

    if vgen is None:  # defensive; successful init always assigns it
        out_q.put((("__ack__", worker_id), "init_error: vLLM generator missing"))
        return

    while True:
        cmd, payload = in_q.get()
        if cmd == "stop":
            break
        if cmd == "ping":
            out_q.put((("__ack__", worker_id), "ready"))
            continue
        if cmd == "sync":
            try:
                vgen.sync_adapter(payload)
            except Exception as exc:  # noqa: BLE001
                out_q.put((("__ack__", worker_id), f"sync_error: {exc}"))
                continue
            out_q.put((("__ack__", worker_id), "sync"))
            continue
        if cmd == "rollout_batch":
            request_items = list(payload)
            try:
                from rollout_sampling import derive_turn_seed

                def _generate_wave(requests):
                    try:
                        return vgen.generate_batch(requests)
                    except Exception as batch_exc:  # preserve a compatibility path
                        print(
                            f"[dp_worker {worker_id}] batched generation failed "
                            f"({batch_exc}); retrying this wave request-by-request",
                            flush=True,
                        )
                        return [vgen.generate_fn(messages, max_new, seed=seed)
                                for messages, max_new, seed in requests]

                batch_results = run_episodes_collect_batch(
                    tokenizer=tokenizer,
                    tasks=[task for _, task in request_items],
                    config=config,
                    registry=_worker_registry(config),
                    generate_batch_fn=_generate_wave,
                    reward_fn=reward_fn,
                )
                for (req_id, task), res in zip(request_items, batch_results):
                    rollout_seed = int(task.get("_rollout_seed") or 0)
                    if isinstance(getattr(res, "reward_diag", None), dict):
                        res.reward_diag.update({
                            "rollout_index": task.get("_rollout_index"),
                            "rollout_seed": rollout_seed or None,
                            "actual_generation_seed": (
                                derive_turn_seed(rollout_seed, 0)
                                if rollout_seed else None),
                            "rollout_sampling_version": task.get(
                                "_rollout_sampling_version"),
                            "dp_worker_id": int(worker_id),
                            "dp_gpu": int(gpu),
                            "request_id": req_id,
                            "worker_batch_size": len(request_items),
                        })
                    out_q.put((req_id, res))
            except Exception as exc:  # never strand the parent waiting for results
                import traceback
                error = f"{exc}\n{traceback.format_exc()}"
                for req_id, _task in request_items:
                    out_q.put((req_id, RolloutResult(error=error)))
            continue
        if cmd == "rollout":
            req_id, task = payload
            try:
                from rollout_sampling import derive_turn_seed

                rollout_seed = int(task.get("_rollout_seed") or 0)
                turn_counter = {"i": 0}

                def _seeded_generate(messages, max_new_tokens, seed=None):
                    # Prefer explicit seed; else derive per-turn from rollout seed.
                    if seed is None and rollout_seed:
                        seed = derive_turn_seed(rollout_seed, turn_counter["i"])
                    turn_counter["i"] += 1
                    return vgen.generate_fn(messages, max_new_tokens, seed=seed)

                res = run_episode_collect(
                    tokenizer=tokenizer, task=task, config=config,
                    registry=_worker_registry(config), generate_fn=_seeded_generate,
                    reward_fn=reward_fn, gold_obs=None,
                )
                # Provenance for duplication audits (survives sanitize).
                if isinstance(getattr(res, "reward_diag", None), dict):
                    res.reward_diag.update({
                        "rollout_index": task.get("_rollout_index"),
                        "rollout_seed": rollout_seed or None,
                        "actual_generation_seed": (
                            derive_turn_seed(rollout_seed, 0) if rollout_seed else None),
                        "rollout_sampling_version": task.get("_rollout_sampling_version"),
                        "dp_worker_id": int(worker_id),
                        "dp_gpu": int(gpu),
                        "request_id": req_id,
                    })
            except Exception as exc:  # noqa: BLE001 — never kill the worker on one task
                import traceback
                res = RolloutResult(error=f"{exc}\n{traceback.format_exc()}")
            out_q.put((req_id, res))
            continue
        # Unknown command — ignore.

    # graceful engine teardown best-effort
    try:
        # Dropping the last reference triggers the same teardown as ``del`` and
        # keeps static scope analysis of the nested generation callbacks sound.
        vgen = None
    except Exception:
        pass


_WORKER_REGISTRY_CACHE = {"reg": None, "built": False}


def _worker_registry(config: Dict[str, Any]):
    """Build (once per worker) the executor registry from config paths.

    Replicates ``run.build_registry`` WITHOUT importing run.py — the partial
    experiment's run.py does not define build_registry, and importing either
    run.py inside a worker would needlessly re-run its heavy module body.
    """
    if not _WORKER_REGISTRY_CACHE["built"]:
        try:
            import executor as _ex
            from executor import IBMFunctionRegistry, detect_ibm_functions_dir
            paths = config.get("paths", {}) or {}
            funcs_dir = detect_ibm_functions_dir(
                explicit=paths.get("ibm_functions_dir"),
                repo_root=os.path.dirname(os.path.abspath(_ex.__file__)),
            )
            _WORKER_REGISTRY_CACHE["reg"] = (
                IBMFunctionRegistry(funcs_dir) if funcs_dir else None
            )
        except Exception:
            _WORKER_REGISTRY_CACHE["reg"] = None
        _WORKER_REGISTRY_CACHE["built"] = True
    return _WORKER_REGISTRY_CACHE["reg"]


# ─────────────────────────────────────────────────────────────────────────────
#  Parent-side pool
# ─────────────────────────────────────────────────────────────────────────────

class DataParallelRolloutPool:
    """Manages N worker processes (one vLLM engine per GPU) for parallel rollouts."""

    def __init__(self, config: Dict[str, Any], gpus: List[int],
                 adapter_path: Optional[str] = None, *, start_timeout: float = 1800.0):
        import multiprocessing as mp

        self.gpus = list(gpus)
        if not self.gpus:
            raise ValueError("DataParallelRolloutPool requires a non-empty GPU list")

        # Parent-side reward-dispatch assertion: resolve with the EXACT same
        # resolver the workers use, and abort BEFORE any rollout when the
        # configured policy cannot be honoured (audit Bug 1). resolve_reward_info
        # raises on unknown policies unless ALLOW_STRICT_REWARD_FALLBACK=1.
        _fn, self.reward_info = resolve_reward_info(config)
        configured = self.reward_info["configured_policy"]
        resolved = self.reward_info["resolved_policy"]
        print(f"[dp_pool] parent reward check: configured={configured} "
              f"resolved={resolved} "
              f"fn={self.reward_info['reward_fn_module']}."
              f"{self.reward_info['reward_fn_name']} "
              f"fallback_used={str(self.reward_info['fallback_used']).lower()}",
              flush=True)
        if self.reward_info["fallback_used"] and \
                os.environ.get("ALLOW_STRICT_REWARD_FALLBACK", "0") != "1":
            raise RuntimeError(
                f"[dp_pool] reward fallback engaged for policy '{configured}' but "
                f"ALLOW_STRICT_REWARD_FALLBACK != 1 — aborting before any rollout.")
        _is_strict = (self.reward_info["reward_fn_module"] == "reward")
        _strict_requested = configured.lower() in _STRICT_POLICY_ALIASES
        if _is_strict and not _strict_requested and not self.reward_info["fallback_used"]:
            raise RuntimeError(
                f"[dp_pool] configured reward '{configured}' resolved to the STRICT "
                f"gold-trace reward without an explicit request — aborting.")
        self._ctx = mp.get_context("spawn")
        self._in_qs = []
        self._out_q = self._ctx.Queue()
        self._procs = []
        extra_sys_path = list(sys.path)

        for wid, gpu in enumerate(self.gpus):
            in_q = self._ctx.Queue()
            p = self._ctx.Process(
                target=_worker_main,
                args=(wid, gpu, config, adapter_path, extra_sys_path, in_q, self._out_q),
                # MUST be non-daemonic: vLLM v1 spawns its own EngineCore subprocess
                # inside each worker, and Python forbids a daemonic process from
                # having children (AssertionError). close() handles teardown.
                daemon=False,
            )
            p.start()
            self._in_qs.append(in_q)
            self._procs.append(p)

        # Wait for every worker to report readiness (or surface an init error).
        self._await_acks(len(self.gpus), timeout=start_timeout, what="startup")
        print(f"[dp_pool] {len(self.gpus)} rollout workers ready on GPUs {self.gpus}",
              flush=True)

    @property
    def worker_pids(self) -> List[int]:
        """PIDs of rollout worker processes (for targeted cleanup / logging)."""
        return [int(p.pid) for p in self._procs if p.pid is not None]

    # ── public API ──────────────────────────────────────────────────────────

    def rollout_many(self, tasks: List[Dict[str, Any]]) -> List[RolloutResult]:
        """Run episodes across workers, batching each worker's local share.

        Round-robin assignment is retained, but each worker receives one batch
        command so it can interleave the multi-turn episodes and submit every
        active turn wave to vLLM in a single scheduler call.
        """
        n = len(tasks)
        if n == 0:
            return []
        assignments: List[List[Tuple[int, Dict[str, Any]]]] = [
            [] for _ in self._in_qs]
        for i, task in enumerate(tasks):
            wid = i % len(self._in_qs)
            assignments[wid].append((i, task))
        for wid, batch in enumerate(assignments):
            if batch:
                self._in_qs[wid].put(("rollout_batch", batch))
        results: Dict[int, RolloutResult] = {}
        while len(results) < n:
            req_id, res = self._out_q.get()
            if isinstance(req_id, tuple):  # stray ack — ignore
                continue
            results[req_id] = res
        return [results[i] for i in range(n)]

    def sync_adapter(self, adapter_path: Optional[str], timeout: float = 600.0) -> None:
        for in_q in self._in_qs:
            in_q.put(("sync", adapter_path))
        self._await_acks(len(self._in_qs), timeout=timeout, what="sync")

    def close(self) -> None:
        # Ask workers to exit cleanly (lets vLLM tear down its EngineCore subprocess).
        for in_q in self._in_qs:
            try:
                in_q.put(("stop", None))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
                p.join(timeout=15)
            if p.is_alive():  # last resort
                try:
                    p.kill()
                except Exception:
                    pass

    # ── internals ─────────────────────────────────────────────────────────────

    def _await_acks(self, count: int, *, timeout: float, what: str) -> None:
        deadline = time.time() + timeout
        got = 0
        while got < count:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"[dp_pool] timed out waiting for {what} acks "
                                   f"({got}/{count})")
            try:
                tag, status = self._out_q.get(timeout=min(remaining, 30))
            except Exception:
                continue
            if isinstance(tag, tuple) and tag[0] == "__ack__":
                if isinstance(status, str) and status.startswith(("init_error", "sync_error")):
                    raise RuntimeError(f"[dp_pool] worker {tag[1]} {what} failed: {status}")
                got += 1
            # Non-ack messages during startup/sync are unexpected; ignore.
