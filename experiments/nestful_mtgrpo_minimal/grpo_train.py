"""Minimal episode-level GRPO (EXPERIMENTAL pilot).

Honest scope:
  - This is a correct-but-minimal group-relative policy-gradient (GRPO-style)
    update for FULL EPISODES (one binary reward per multi-turn rollout).
  - No critic. LoRA/QLoRA trainable params only. Group-relative advantage.
  - Optional KL-to-reference (k3 estimator) using the frozen base model via
    PEFT `disable_adapter()`. Set training.kl_beta = 0 to skip the reference pass.
  - Dead groups (reward std == 0) are skipped and logged, not faked.
  - Clipped episodes (hit max_new_tokens) get reward 0 and are masked from the
    update (training.mask_clipped_from_update).

It is validated only at small scale (the smoke/pilot defaults). It is NOT a
drop-in replacement for a hardened TRL GRPOTrainer. See README "Known limitations".

This file imports nothing from curricullum/ or nestful_evaluation/.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from parser import parse_tool_call
from prompt import build_messages, format_tool_response
from executor import ToolExecutor
from reward import (
    strict_gold_trace_reward,
    compute_gold_observations,
    episode_turn_reward_seq,
)
from rollout import (
    Trajectory, Turn, get_stage_token_budget,
    resolve_teacher_forced_prefix_n, build_teacher_forced_prefix,
)
from group_stats import compute_group_stats

_STRICT_POLICY_ALIASES = ("strict", "strict_gold_trace", "strict_gold_trace_legacy")


def _policy_is_graded(policy: Optional[str]) -> bool:
    return (policy or "strict").lower() not in _STRICT_POLICY_ALIASES


def _verify_reward_dispatch(config: Dict[str, Any], rollout_pool) -> Dict[str, Any]:
    """Assert (in the PARENT, before any rollout) that the configured reward
    policy is actually the one that will be used (audit Bug 1).

    Pool path:     use the pool's already-resolved reward info (same resolver
                   the workers run; resolve_reward_info raises on unknown).
    Non-pool path: inspect the (possibly monkeypatched) module-global
                   episode_turn_reward_seq that the loop will call.
    """
    configured = str((config.get("reward", {}) or {}).get("train_policy", "strict"))
    if rollout_pool is not None and getattr(rollout_pool, "reward_info", None):
        info = dict(rollout_pool.reward_info)
    elif rollout_pool is not None:
        from vllm_dp_pool import resolve_reward_info
        _fn, info = resolve_reward_info(config)
    else:
        fn = episode_turn_reward_seq  # module global — monkeypatch target
        mod = getattr(fn, "__module__", "?")
        resolved_policy = getattr(fn, "reward_policy", None) or (
            "strict" if mod == "reward" else configured)
        info = {
            "configured_policy": configured,
            "resolved_policy": resolved_policy,
            "reward_fn_module": mod,
            "reward_fn_name": getattr(fn, "__name__", "?"),
            "fallback_used": False,
        }

    allow_fb = os.environ.get("ALLOW_STRICT_REWARD_FALLBACK", "0") == "1"
    is_strict = info.get("reward_fn_module") == "reward"
    strict_requested = configured.lower() in _STRICT_POLICY_ALIASES
    print(f"[train] reward dispatch: configured={info['configured_policy']} "
          f"resolved={info['resolved_policy']} "
          f"fn={info['reward_fn_module']}.{info['reward_fn_name']} "
          f"fallback_used={str(info.get('fallback_used', False)).lower()}",
          flush=True)
    if info.get("fallback_used") and not allow_fb:
        raise RuntimeError(
            f"[train] ABORT: reward fallback engaged for policy '{configured}' "
            f"without ALLOW_STRICT_REWARD_FALLBACK=1.")
    if is_strict and not strict_requested and not info.get("fallback_used") and not allow_fb:
        raise RuntimeError(
            f"[train] ABORT: configured reward policy '{configured}' resolved to the "
            f"STRICT gold-trace reward. The graded reward was NOT dispatched — this is "
            f"exactly the failure mode that invalidated the previous pilots. "
            f"Fix reward wiring (vllm_dp_pool.resolve_reward_info / run.py monkeypatch) "
            f"before training.")
    return info


def _completion_hash(ep: "Episode") -> str:
    h = hashlib.sha1()
    for tt in ep.turn_tokens:
        try:
            ids = tt.completion_ids.tolist()
        except AttributeError:
            ids = list(tt.completion_ids)
        h.update(str(ids).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:12]


def _diag_failure_counts(ep_diags: List[Dict[str, Any]]) -> Dict[str, int]:
    """Failure-mode counts from per-episode reward diagnostics (best-effort:
    supports both graded-reward diags and strict-reward diags)."""
    def _flag(d: Dict[str, Any], key: str, strict_key: Optional[str] = None,
              strict_invert: bool = False) -> bool:
        if key in d:
            return bool(d.get(key))
        if strict_key is not None and strict_key in d:
            v = bool(d.get(strict_key))
            return (not v) if strict_invert else v
        return False

    return {
        "parse_error_count": sum(
            1 for d in ep_diags if _flag(d, "parse_error", "parse_ok", True)),
        "no_tool_call_count": sum(
            1 for d in ep_diags if _flag(d, "no_tool_call", "zero_tool_calls")),
        "wrong_tool_count": sum(1 for d in ep_diags if _flag(d, "wrong_tool")),
        "wrong_arg_count": sum(1 for d in ep_diags if _flag(d, "wrong_args")),
        "invalid_ref_count": sum(
            1 for d in ep_diags if _flag(d, "invalid_reference")),
        "premature_final_count": sum(
            1 for d in ep_diags if _flag(d, "premature_final")),
        "too_few_calls_count": sum(
            1 for d in ep_diags if _flag(d, "too_few_calls")),
        "predicates_error_count": sum(
            1 for d in ep_diags if d.get("predicates_error")),
    }


def _write_checkpoint_sidecars(
    adapter_dir: str,
    config: Dict[str, Any],
    *,
    stage: int,
    epoch: int,
    lr: float,
    kl_beta: float,
    num_gen: int,
    grad_accum: int,
    global_step: int,
    wandb_run=None,
    log=None,
    train_stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Write reproducibility sidecars next to a saved adapter.

    Files written into ``adapter_dir``:
      - config_used.json / config_used.yaml : exact resolved config for this run
      - trainer_state.json                  : stage/epoch/lr/kl/step + init source
      - wandb_run_id.txt                    : W&B run id (if logging is active)

    Best-effort: a sidecar failure must never abort training.
    """
    runtime = config.get("_runtime", {}) or {}
    init_ckpt = runtime.get("init_checkpoint")
    try:
        with open(os.path.join(adapter_dir, "config_used.json"), "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, ensure_ascii=False, default=str)
    except Exception as exc:  # noqa: BLE001
        if log:
            log({"epoch": epoch - 1, "sidecar_config_json_error": str(exc)})
    try:
        import yaml
        with open(os.path.join(adapter_dir, "config_used.yaml"), "w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)
    except Exception as exc:  # noqa: BLE001 - yaml optional / non-serializable values
        if log:
            log({"epoch": epoch - 1, "sidecar_config_yaml_error": str(exc)})

    trainer_state = {
        "stage": stage,
        "epoch": epoch,
        "learning_rate": lr,
        "kl_beta": kl_beta,
        "num_generations": num_gen,
        "gradient_accumulation_steps": grad_accum,
        "global_step": global_step,
        "init_checkpoint": init_ckpt,
        "resumed_from_checkpoint": bool(init_ckpt),
        "mixed_replay": bool(config.get("data", {}).get("mixed_replay")),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if train_stats:
        trainer_state.update(train_stats)
    try:
        with open(os.path.join(adapter_dir, "trainer_state.json"), "w", encoding="utf-8") as fh:
            json.dump(trainer_state, fh, indent=2, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        if log:
            log({"epoch": epoch - 1, "sidecar_trainer_state_error": str(exc)})

    run_id = None
    try:
        run_id = getattr(wandb_run, "id", None)
    except Exception:  # noqa: BLE001
        run_id = None
    if run_id:
        try:
            with open(os.path.join(adapter_dir, "wandb_run_id.txt"), "w", encoding="utf-8") as fh:
                fh.write(str(run_id) + "\n")
        except Exception:  # noqa: BLE001
            pass
    if log:
        log({"epoch": epoch - 1, "sidecars_written": adapter_dir,
             "resumed_from_checkpoint": bool(init_ckpt)})


@dataclass
class TurnTokens:
    prompt_ids: Any        # 1D LongTensor (no batch dim)
    completion_ids: Any    # 1D LongTensor


@dataclass
class Episode:
    trajectory: Trajectory
    turn_tokens: List[TurnTokens]
    reward: float = 0.0
    # Number of leading gold calls teacher-forced (not generated) into this
    # episode. r_seq returned by the reward fn is aligned 1:1 with
    # trajectory.turns (forced + generated); callers MUST drop the first
    # `n_forced_turns` entries before pairing per-turn returns with
    # `turn_tokens` (generated turns only — see train()).
    n_forced_turns: int = 0


@dataclass
class _PoolTraj:
    """Minimal stand-in for Trajectory holding only the fields the GRPO update
    loop reads. Used when rollouts come back from a data-parallel worker pool,
    where the full Trajectory (with raw tool observations) stays in the worker."""
    clipped_any: bool = False
    zero_tool_calls: bool = False
    num_tool_calls: int = 0
    stop_reason: Optional[str] = None


def _episode_from_pool_result(res) -> Episode:
    """Re-wrap a worker :class:`RolloutResult` as a parent-side :class:`Episode`.

    Token-id lists become 1-D LongTensors for the log-prob pass; the reward and
    flags are taken verbatim from the worker (which already applied the correct
    strict/partial policy)."""
    import torch
    tts = [
        TurnTokens(
            torch.tensor(p_ids, dtype=torch.long),
            torch.tensor(c_ids, dtype=torch.long),
        )
        for (p_ids, c_ids) in res.turn_token_ids
    ]
    traj = _PoolTraj(clipped_any=bool(res.clipped_any),
                     zero_tool_calls=bool(res.zero_tool_calls),
                     num_tool_calls=int(getattr(res, "num_tool_calls", 0)),
                     stop_reason=getattr(res, "stop_reason", None))
    return Episode(trajectory=traj, turn_tokens=tts, reward=float(res.episode_reward))


def _generate_with_ids(model, tokenizer, messages, max_new_tokens, temperature, top_p,
                       max_prompt_tokens=0):
    import torch
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    prompt_len = int(prompt_ids.shape[1])
    # Guard: a runaway multi-turn history must not be fed to model.generate()
    # (slow + can OOM the HF model). Signal overflow so the caller ends the
    # episode and masks it from the GRPO update — same contract as the vLLM path.
    if max_prompt_tokens and max_prompt_tokens > 0 and prompt_len > max_prompt_tokens:
        print(f"[train] prompt_overflow (HF): {prompt_len} tokens > "
              f"max_prompt_tokens {max_prompt_tokens} — skipping episode", flush=True)
        empty = prompt_ids[0][:0]
        return "", prompt_ids[0], empty, prompt_len, 0, False, True
    device = getattr(model, "device", None) or next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    attn = torch.ones_like(prompt_ids)
    do_sample = temperature is not None and temperature > 0
    with torch.no_grad():
        out = model.generate(
            input_ids=prompt_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    comp_ids = out[0][prompt_len:]
    text = tokenizer.decode(comp_ids, skip_special_tokens=True)
    clipped = int(comp_ids.shape[0]) >= max_new_tokens
    return text, prompt_ids[0], comp_ids, prompt_len, int(comp_ids.shape[0]), clipped, False


def _rollout_episode_for_train(
    model, tokenizer, task, config, registry, max_turns, *, vllm_gen_fn=None,
    gold_obs=None,
) -> Episode:
    """Run one episode for GRPO.

    When ``vllm_gen_fn`` is provided (opt-in, hardware.use_vllm: true):
    - vLLM handles fast forward generation (no gradients needed here).
    - The completion text is re-tokenised to obtain TurnTokens for the
      subsequent _sequence_logprob() call which still uses the HF model.

    Without vLLM (default): _generate_with_ids() uses the HF model.

    ``gold_obs`` (optional): precomputed gold observations, used ONLY to gate
    teacher-forced continuation training (``train.teacher_forced_prefix_calls``
    > 0) — see rollout.resolve_teacher_forced_prefix_n.
    """
    gen = config.get("generation", {})
    exec_cfg = config.get("executor", {})
    gold_n = int(task.get("num_calls") or len(task.get("gold_calls", [])))
    budget = get_stage_token_budget(config, gold_n, "train")
    max_new_tokens = budget["max_new_tokens"]
    # Prompt budget for the HF generation path = context window minus the room
    # reserved for the completion. Used to skip overlong-prompt episodes.
    hf_prompt_budget = max(0, int(budget.get("max_model_length", 0)) - int(max_new_tokens))
    temperature = float(gen.get("temperature", 0.7))
    top_p = float(gen.get("top_p", 0.95))

    executor = ToolExecutor(
        task, registry=registry, mode=exec_cfg.get("mode", "auto"),
        ibm_call_timeout=float(exec_cfg.get("ibm_call_timeout", 30.0)),
    )
    traj = Trajectory(task["task_id"], gold_n, gold_n, executor_mode=executor.mode)
    turn_tokens: List[TurnTokens] = []
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
    max_turns = max(1, max_turns - n_forced)

    for _step in range(max_turns):
        turn_idx = n_forced + _step
        messages = build_messages(task, history)
        if vllm_gen_fn is not None:
            # vLLM generates text; re-tokenise to get token IDs for log-probs.
            g = vllm_gen_fn(messages, max_new_tokens)
            if g.get("prompt_overflow"):
                # Prompt exceeded vLLM context window — treat as a clipped episode
                # so it is masked from GRPO updates (same as HF prompt_overflow).
                traj.prompt_overflow = True
                traj.clipped_any = True
                traj.stop_reason = "prompt_overflow"
                break
            text = g["text"]
            c_len = g["completion_tokens"]
            clipped = g["clipped"]
            p_ids, c_ids = _retokenize_for_logprob(tokenizer, messages, text)
            p_len = int(p_ids.shape[0])
        else:
            text, p_ids, c_ids, p_len, c_len, clipped, overflow = _generate_with_ids(
                model, tokenizer, messages, max_new_tokens, temperature, top_p,
                max_prompt_tokens=hf_prompt_budget,
            )
            if overflow:
                traj.prompt_overflow = True
                traj.clipped_any = True
                traj.stop_reason = "prompt_overflow"
                break
        turn = Turn(turn_idx, text, prompt_tokens=p_len,
                    completion_tokens=c_len, clipped_completion=clipped)
        turn_tokens.append(TurnTokens(p_ids.detach().cpu(), c_ids.detach().cpu()))
        history.append({"role": "assistant", "content": text})

        if clipped:
            traj.clipped_any = True
            turn.fail_reason = "clipped_completion"
            traj.turns.append(turn)
            traj.stop_reason = "clipped"
            break

        pr = parse_tool_call(text)
        if pr.is_terminal:
            turn.is_terminal = True
            traj.turns.append(turn)
            traj.stop_reason = "terminal"
            break
        if not pr.ok:
            turn.fail_reason = f"parse:{pr.reason}"
            traj.turns.append(turn)
            traj.stop_reason = "parse_fail"
            break

        call = pr.call
        turn.parsed_call = call
        res = executor.execute(call)
        turn.observation = res.observation
        if res.error is not None:
            turn.fail_reason = f"exec:{res.error}"
            traj.turns.append(turn)
            traj.stop_reason = "executor_error"
            break
        traj.final_observation = res.observation
        traj.turns.append(turn)
        history.append({"role": "user", "content": format_tool_response(call, res.observation)})

    if traj.stop_reason is None:
        traj.stop_reason = "max_turns"
    return Episode(trajectory=traj, turn_tokens=turn_tokens, n_forced_turns=n_forced)


def _retokenize_for_logprob(tokenizer, messages, completion_text: str):
    """Re-tokenise a (messages, completion) pair as 1-D CPU LongTensors.

    Used when vLLM generates text and we need token IDs for _sequence_logprob().
    The cost is a second tokeniser pass (no model inference), which is cheap.
    """
    import torch
    _p = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    p_ids = (_p.input_ids if hasattr(_p, "input_ids") else _p)[0]
    c_ids = tokenizer.encode(
        completion_text, add_special_tokens=False, return_tensors="pt"
    )[0]
    return p_ids, c_ids


def _sequence_logprob(model, prompt_ids, completion_ids, *, with_grad: bool):
    """Sum of log p(completion_token | prefix) under `model`. Returns (sum_logp, n_tokens)."""
    import torch
    device = next(model.parameters()).device
    p = prompt_ids.to(device)
    c = completion_ids.to(device)
    if c.numel() == 0:
        return torch.zeros((), device=device), 0
    input_ids = torch.cat([p, c]).unsqueeze(0)
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        logits = model(input_ids=input_ids).logits[0]  # [T, V]
        # Predict token t from logits at t-1.
        start = p.numel()
        target = input_ids[0, start:]
        pred_logits = logits[start - 1: -1, :]
        logprobs = torch.log_softmax(pred_logits.float(), dim=-1)
        token_logp = logprobs.gather(1, target.unsqueeze(1)).squeeze(1)
    return token_logp.sum(), int(target.numel())


def train(
    config,
    model,
    tokenizer,
    registry,
    tasks,
    log_path: str,
    *,
    vllm_gen=None,
    rollout_pool=None,
    wandb_run=None,
) -> Dict[str, Any]:
    """Run the GRPO training loop.

    Args:
        vllm_gen: optional VLLMGenerator for fast rollout generation.  When
                  provided the HF model is used only for log-prob computation
                  and gradient steps; rollout text is generated by vLLM.
                  After each epoch the saved adapter is synced to vLLM via
                  ``vllm_gen.sync_adapter(adapter_dir)`` so that the next
                  epoch's rollouts use the updated weights.
                  Set ``hardware.use_vllm: true`` in config to activate.
        rollout_pool: optional DataParallelRolloutPool. When provided, the
                  per-task ``num_generations`` rollouts (and their reward) are
                  computed by worker processes — one vLLM engine per GPU — while
                  the HF learner stays on its own GPU. The pool returns
                  token-id lists + reward + r_seq; the parent only runs the
                  log-prob/GRPO update. Mutually exclusive with ``vllm_gen``;
                  the adapter is synced via ``rollout_pool.sync_adapter`` each
                  epoch. Opt-in via ``hardware.rollout_data_parallel_gpus``.
        wandb_run: optional W&B run object for online logging.  Activated
                   automatically when WANDB_PROJECT env var is set.
    """
    import torch

    tr = config.get("training", {})
    gen = config.get("generation", {})
    epochs = int(tr.get("epochs", 1))
    num_gen = int(gen.get("num_generations", 4))
    lr = float(tr.get("learning_rate", 1e-6))
    grad_accum = int(tr.get("gradient_accumulation_steps", 4))
    kl_beta = float(tr.get("kl_beta", 0.02))
    max_grad_norm = float(tr.get("max_grad_norm", 1.0))
    mask_clipped = bool(tr.get("mask_clipped_from_update", True))
    gold_n_default = int(config.get("data", {}).get("train_stage", 3))

    # Turn-level MT-GRPO settings (training rewards are gold-trace-derived only).
    mt = config.get("mt_grpo", {}) or {}
    use_turn_level = bool(mt.get("enabled", True)) and \
        mt.get("mode", "turn_level_minimal") == "turn_level_minimal"
    gamma = float(mt.get("gamma", 1.0))
    lambda_episode = float(mt.get("lambda_episode", 1.0))
    normalize_advantage = bool(mt.get("normalize_advantage", True))
    allow_fallback = bool(mt.get("fallback_episode_level_if_needed", True))
    reward_policy = str(config.get("reward", {}).get("train_policy", "strict"))

    # ── Reward-dispatch verification (audit Bug 1) — aborts BEFORE any rollout
    # when the configured graded reward would not actually run.
    dispatch_info = _verify_reward_dispatch(config, rollout_pool)
    graded_reward = _policy_is_graded(reward_policy) and not dispatch_info.get("fallback_used")

    # ── Early-abort bookkeeping (audit Bug 10) ───────────────────────────────
    early_abort_enabled = bool(tr.get("early_abort_checks", True))
    early_dead_thresh_50 = float(tr.get("early_abort_dead_rate_first_50", 0.90))
    groups_seen = 0
    first50_dead = 0
    first50_reward_values: set = set()
    all_reward_values: set = set()
    total_contributing = 0
    total_dead_groups = 0
    total_groups = 0
    position_artifact_groups = 0
    agg_no_tool = 0
    agg_too_few = 0
    agg_episodes = 0
    agg_pred_calls = 0

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    has_ref = hasattr(model, "disable_adapter") and kl_beta > 0.0

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")

    def _log(rec: Dict[str, Any]) -> None:
        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log_f.flush()

    vllm_gen_fn = vllm_gen.generate_fn if vllm_gen is not None else None

    global_step = 0
    stage = int(config.get("data", {}).get("train_stage", 0))
    summary = {
        "epochs": epochs, "num_tasks": len(tasks), "steps": 0,
        "experimental": True,
        "reward_train_policy": reward_policy,
        "reward_policy_configured": dispatch_info["configured_policy"],
        "reward_policy_resolved": dispatch_info["resolved_policy"],
        "reward_fn_module": dispatch_info["reward_fn_module"],
        "reward_fn_name": dispatch_info["reward_fn_name"],
        "reward_fallback_used": bool(dispatch_info.get("fallback_used", False)),
        "mt_grpo_mode": "turn_level_minimal" if use_turn_level else "episode_level",
        "gamma": gamma, "lambda_episode": lambda_episode,
        "fallback_used": False,
        "vllm_rollout": (vllm_gen is not None) or (rollout_pool is not None),
        "data_parallel_rollout": rollout_pool is not None,
    }
    _log({"reward_dispatch": dispatch_info})

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        epoch_rewards: List[float] = []
        epoch_dead_groups = 0
        epoch_task_groups = 0
        for ti, task in enumerate(tasks):
            gold_n = int(task.get("num_calls") or gold_n_default)
            episodes: List[Episode] = []
            ep_r_seqs: List[List[float]] = []
            pool_first_errors: List[int] = []  # only populated on the pool path

            ep_diags: List[Dict[str, Any]] = []
            if rollout_pool is not None:
                # Data-parallel: workers run the full episode AND apply the correct
                # reward policy, returning token-id lists + reward + r_seq. Raw tool
                # observations stay in the workers (never serialized).
                results = rollout_pool.rollout_many([task] * num_gen)
                for res in results:
                    if getattr(res, "error", None):
                        _log({"epoch": epoch, "task_idx": ti,
                              "task_id": task["task_id"], "rollout_error": res.error})
                    episodes.append(_episode_from_pool_result(res))
                    ep_r_seqs.append(list(res.r_seq))
                    ep_diags.append(dict(getattr(res, "reward_diag", None) or {}))
                    if res.first_error_turn is not None:
                        pool_first_errors.append(int(res.first_error_turn))
            else:
                gold_obs = compute_gold_observations(task, registry)
                # v2: train turn budget = gold_n + max_extra_turns_train (cap +4).
                # Default 0 reproduces the legacy max_turns_train = gold_n exactly.
                _extra = int(config.get("train", {}).get("max_extra_turns_train", 0))
                _train_max_turns = max(1, min(gold_n + _extra, gold_n + 4))
                for _ in range(num_gen):
                    ep = _rollout_episode_for_train(
                        model, tokenizer, task, config, registry,
                        max_turns=_train_max_turns,
                        vllm_gen_fn=vllm_gen_fn,
                        gold_obs=gold_obs,
                    )
                    rinfo = episode_turn_reward_seq(ep.trajectory, task, gold_obs)
                    ep.reward = rinfo["episode_reward"]
                    # r_seq is aligned 1:1 with ep.trajectory.turns (forced +
                    # generated); drop the forced-prefix entries so it matches
                    # ep.turn_tokens (generated turns only — no gradient on
                    # teacher-forced text). See Episode.n_forced_turns.
                    r_seq_full = [float(x) for x in rinfo["r_seq"]]
                    if len(r_seq_full) != len(ep.trajectory.turns):
                        raise RuntimeError(
                            f"[teacher_forced] reward r_seq length "
                            f"{len(r_seq_full)} != len(trajectory.turns) "
                            f"{len(ep.trajectory.turns)} for task "
                            f"{task.get('task_id')} "
                            f"(n_forced={ep.n_forced_turns}); refusing to "
                            f"guess turn alignment.")
                    r_seq = r_seq_full[ep.n_forced_turns:]
                    if len(r_seq) != len(ep.turn_tokens):
                        raise RuntimeError(
                            f"[teacher_forced] post-slice r_seq length "
                            f"{len(r_seq)} != turn_tokens length "
                            f"{len(ep.turn_tokens)} for task "
                            f"{task.get('task_id')} "
                            f"(n_forced={ep.n_forced_turns}).")
                    diag = dict(rinfo.get("diagnostics") or {})
                    diag["teacher_forced_prefix_calls"] = ep.n_forced_turns
                    episodes.append(ep)
                    ep_r_seqs.append(r_seq)
                    ep_diags.append(diag)

            rewards = [e.reward for e in episodes]
            mean_r = sum(rewards) / len(rewards)

            # Per-episode turn-level returns: G_t = sum_{k>=t} gamma^(k-t) r_k
            #   + lambda_episode * gamma^(T-t+1) * R_episode
            ep_returns: List[List[float]] = []
            for ep, r_seq in zip(episodes, ep_r_seqs):
                ep_returns.append(_turn_returns(r_seq, ep.reward, gamma, lambda_episode))

            # ── Corrected group statistics (audit Bug 3) ──────────────────────
            # Advantages are computed PER TURN POSITION across completions;
            # a group is dead iff NO position has between-completion variance.
            # The old flattened std is kept only for logging / artifact detection.
            included = [not (mask_clipped and ep.trajectory.clipped_any)
                        for ep in episodes]
            gstats = compute_group_stats(ep_returns, rewards, included)
            dead = gstats.dead_corrected
            gstd = gstats.flat_std  # legacy field, logged as flattened std

            group_all_zero = all(r == 0.0 for r in rewards)
            group_all_one = all(r == 1.0 for r in rewards)
            if gstats.position_artifact_detected:
                _log({"epoch": epoch, "task_idx": ti, "task_id": task["task_id"],
                      "position_artifact_detected": True,
                      "flat_std": gstats.flat_std,
                      "between_completion_std_max": gstats.between_completion_std_max,
                      "note": "alive under OLD flattened logic, dead under corrected "
                              "between-completion logic — old logic would have trained "
                              "on a pure turn-position artifact"})

            comp_hashes = [_completion_hash(e) for e in episodes]
            pred_calls = [int(getattr(e.trajectory, "num_tool_calls", 0) or 0)
                          for e in episodes]
            turn_reward_values = sorted({float(x) for seq in ep_r_seqs for x in seq})
            episode_reward_values = sorted({float(r) for r in rewards})
            fail_counts = _diag_failure_counts(ep_diags)

            rec = {
                "epoch": epoch, "task_idx": ti, "task_id": task["task_id"],
                "reward_train_policy": reward_policy,
                "reward_policy_configured": dispatch_info["configured_policy"],
                "reward_policy_resolved": dispatch_info["resolved_policy"],
                "reward_fn_module": dispatch_info["reward_fn_module"],
                "reward_fn_name": dispatch_info["reward_fn_name"],
                "mt_grpo_mode": summary["mt_grpo_mode"],
                "mean_reward": mean_r,
                "episode_rewards": rewards,
                "raw_episode_rewards": rewards,
                "turn_rewards": ep_r_seqs,
                "unique_episode_rewards": episode_reward_values,
                "unique_turn_rewards": turn_reward_values,
                "n_unique_episode_rewards": len(episode_reward_values),
                "n_unique_turn_rewards": len(turn_reward_values),
                "reward_std_episode": gstats.episode_reward_std,
                "reward_std_turn_flattened": gstats.flat_std,
                "reward_std_between_completion": gstats.between_completion_std_max,
                "return_std": gstd,
                "group_all_zero": group_all_zero, "group_all_one": group_all_one,
                "group_mixed": (not group_all_zero) and (not group_all_one),
                "dead_group": dead,
                "dead_group_old_flattened": gstats.dead_flattened,
                "dead_group_corrected": gstats.dead_corrected,
                "position_artifact_detected": gstats.position_artifact_detected,
                "completion_hashes": comp_hashes,
                "n_unique_completion_hashes": len(set(comp_hashes)),
                "predicted_num_calls": pred_calls,
                "gold_num_calls": gold_n,
                **fail_counts,
                "strict_gold_trace_pass": mean_r,
                "zero_tool_calls": sum(
                    1 for e in episodes if e.trajectory.zero_tool_calls
                ) / len(episodes),
                "first_error_turn_mean": (
                    (sum(pool_first_errors) / len(pool_first_errors))
                    if pool_first_errors else None
                ) if rollout_pool is not None
                else _first_error_mean(episodes, task, gold_obs),
                "clipped_rate": sum(
                    1 for e in episodes if e.trajectory.clipped_any
                ) / len(episodes),
                "learning_rate": lr,
                "kl_beta": kl_beta,
                **_reward_component_rates(episodes, task),
            }

            epoch_rewards.append(mean_r)
            epoch_task_groups += 1
            if dead:
                epoch_dead_groups += 1

            # ── Signal-collapse bookkeeping + early aborts (audit Bug 10) ─────
            groups_seen += 1
            total_groups += 1
            if dead:
                total_dead_groups += 1
            if gstats.position_artifact_detected:
                position_artifact_groups += 1
            for v in rewards:
                all_reward_values.add(round(float(v), 6))
            for seq in ep_r_seqs:
                for v in seq:
                    all_reward_values.add(round(float(v), 6))
            agg_episodes += len(episodes)
            agg_no_tool += fail_counts["no_tool_call_count"]
            agg_too_few += fail_counts["too_few_calls_count"]
            agg_pred_calls += sum(pred_calls)
            if groups_seen <= 50:
                if dead:
                    first50_dead += 1
                first50_reward_values.update(round(float(r), 6) for r in rewards)
            if early_abort_enabled and groups_seen == 50:
                d50 = first50_dead / 50.0
                summary["dead_group_rate_first_50"] = d50
                _log({"epoch": epoch, "first_50_dead_group_rate": d50,
                      "first_50_unique_episode_rewards": sorted(first50_reward_values)})
                if d50 > early_dead_thresh_50:
                    raise RuntimeError(
                        f"[train] EARLY ABORT: dead_group_rate over first 50 groups = "
                        f"{d50:.2f} > {early_dead_thresh_50}. No usable learning signal "
                        f"— inspect reward variance before burning more compute.")
                if graded_reward and first50_reward_values <= {0.0, 1.0}:
                    raise RuntimeError(
                        f"[train] EARLY ABORT: graded reward "
                        f"'{dispatch_info['resolved_policy']}' produced ONLY binary "
                        f"{{0,1}} episode rewards over the first 50 groups — this "
                        f"matches the strict-fallback failure mode. Fix reward "
                        f"dispatch / grading before training.")
            if early_abort_enabled and groups_seen == 100 and global_step == 0 \
                    and total_contributing == 0:
                raise RuntimeError(
                    "[train] EARLY ABORT: 0 optimizer steps and 0 contributing turns "
                    "after 100 groups — training is not learning anything.")

            if dead:
                _log({**rec, "update": "skipped_dead_group",
                      "optimizer_step_executed": False, "contributing_turns": 0})
                _wandb_log_task(wandb_run, rec, stage, global_step)
                continue

            step_loss = 0.0
            kl_sum = 0.0
            kl_count = 0
            logp_sum = 0.0
            logp_count = 0
            contributing = 0
            for ei, (ep, gs) in enumerate(zip(episodes, ep_returns)):
                if mask_clipped and ep.trajectory.clipped_any:
                    continue
                adv_row = gstats.advantages[ei]
                for j, tt in enumerate(ep.turn_tokens):
                    if j >= len(gs) or tt.completion_ids.numel() == 0:
                        continue
                    if use_turn_level:
                        # Per-position between-completion advantage (audit Bug 3):
                        # centered/normalized ACROSS completions at the same turn
                        # position, so mechanical position offsets cancel out.
                        if normalize_advantage:
                            adv = adv_row[j] if j < len(adv_row) else 0.0
                        else:
                            _pm = gstats.position_means[j] if j < len(gstats.position_means) else 0.0
                            adv = gs[j] - _pm
                    else:
                        # episode_level: single advantage from R_episode group.
                        adv = (ep.reward - mean_r)
                    if adv == 0.0:
                        continue
                    cur, n = _sequence_logprob(
                        model, tt.prompt_ids, tt.completion_ids, with_grad=True
                    )
                    mean_logp = cur / max(1, n)
                    pg_loss = -(adv * mean_logp)
                    if has_ref:
                        with model.disable_adapter():
                            ref, _ = _sequence_logprob(
                                model, tt.prompt_ids, tt.completion_ids, with_grad=False
                            )
                        diff = (ref / max(1, n)) - mean_logp
                        kl_term = (diff.exp() - diff - 1.0)
                        pg_loss = pg_loss + kl_beta * kl_term
                        kl_sum += float(kl_term.detach())
                        kl_count += 1
                    (pg_loss / grad_accum).backward()
                    step_loss += float(pg_loss.detach())
                    logp_sum += float(mean_logp.detach())
                    logp_count += 1
                    contributing += 1

            if contributing == 0 and allow_fallback:
                summary["fallback_used"] = True

            total_contributing += contributing
            accum += 1
            rec["loss"] = step_loss
            rec["contributing_turns"] = contributing
            rec["kl"] = (kl_sum / kl_count) if kl_count else 0.0
            rec["mean_logprob"] = (logp_sum / logp_count) if logp_count else 0.0
            rec["optimizer_step_executed"] = (accum % grad_accum == 0)
            _log({**rec, "update": "accumulated"})
            _wandb_log_task(wandb_run, rec, stage, global_step)

            if accum % grad_accum == 0:
                import torch as _t
                gnorm = _t.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                _log({"epoch": epoch, "task_idx": ti, "grad_norm": float(gnorm),
                      "update": "optimizer_step", "global_step": global_step})
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        # Flush any remaining grads at epoch end.
        if accum % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if epoch_task_groups:
            dgr = epoch_dead_groups / epoch_task_groups
            summary["dead_group_rate_last_epoch"] = dgr
            _log({"epoch": epoch, "dead_group_rate": dgr,
                  "dead_groups": epoch_dead_groups, "task_groups": epoch_task_groups})

        _wandb_log_epoch(wandb_run, epoch, stage, len(tasks), epoch_rewards, summary["fallback_used"])

        if tr.get("save_every_epoch", True):
            out_dir = config["model"]["output_adapter_dir"]
            adapter_dir = os.path.join(out_dir, f"adapter_epoch_{epoch + 1}")
            os.makedirs(adapter_dir, exist_ok=True)
            try:
                model.save_pretrained(adapter_dir)
                tokenizer.save_pretrained(adapter_dir)
                _log({"epoch": epoch, "saved_adapter": adapter_dir})
                # Reproducibility sidecars next to the adapter (config, trainer
                # state, wandb id) so any checkpoint can be resumed / audited.
                _write_checkpoint_sidecars(
                    adapter_dir, config,
                    stage=stage, epoch=epoch + 1, lr=lr, kl_beta=kl_beta,
                    num_gen=num_gen, grad_accum=grad_accum, global_step=global_step,
                    wandb_run=wandb_run, log=_log,
                    train_stats={
                        "steps": global_step,
                        "contributing_turns": total_contributing,
                        "dead_group_rate": (total_dead_groups / total_groups)
                        if total_groups else None,
                        "position_artifact_group_rate": (
                            position_artifact_groups / total_groups)
                        if total_groups else None,
                        "fractional_rewards_present": any(
                            0.0 < v < 1.0 for v in all_reward_values),
                        "reward_policy_configured": dispatch_info["configured_policy"],
                        "reward_policy_resolved": dispatch_info["resolved_policy"],
                        "reward_fn_module": dispatch_info["reward_fn_module"],
                        "reward_fn_name": dispatch_info["reward_fn_name"],
                        "reward_fallback_used": bool(
                            dispatch_info.get("fallback_used", False)),
                        # A checkpoint with 0 optimizer steps is NOT a trained
                        # model — it must never be crowned best (audit Bug 4).
                        "trained": global_step > 0,
                        "eligible_for_best": (
                            global_step > 0 and total_contributing > 0
                            and (total_groups == 0
                                 or total_dead_groups / total_groups < 0.95)),
                    },
                )
            except Exception as exc:  # pragma: no cover
                _log({"epoch": epoch, "save_error": str(exc)})
                adapter_dir = None

            # Sync updated adapter to vLLM so that the NEXT epoch's rollouts use
            # the freshly trained weights (per-epoch lag is intentional and standard
            # for vLLM-accelerated GRPO; it is cheaper than per-step sync).
            if adapter_dir:
                hw = config.get("hardware", {})
                sync_mode = hw.get("vllm_weight_sync", "after_epoch")
                if sync_mode == "after_epoch":
                    if rollout_pool is not None:
                        rollout_pool.sync_adapter(adapter_dir)
                        _log({"epoch": epoch, "dp_pool_adapter_synced": adapter_dir})
                    elif vllm_gen is not None:
                        vllm_gen.sync_adapter(adapter_dir)
                        _log({"epoch": epoch, "vllm_adapter_synced": adapter_dir})

    summary["steps"] = global_step
    summary["contributing_turns_total"] = total_contributing
    summary["trained"] = global_step > 0
    summary["dead_group_rate"] = (total_dead_groups / total_groups) if total_groups else None
    if "dead_group_rate_first_50" not in summary and total_groups:
        summary["dead_group_rate_first_50"] = (
            first50_dead / min(50, total_groups))
    summary["position_artifact_group_rate"] = (
        position_artifact_groups / total_groups) if total_groups else None
    summary["fractional_rewards_present"] = any(
        0.0 < v < 1.0 for v in all_reward_values)
    summary["n_unique_reward_values"] = len(all_reward_values)
    summary["unique_reward_values_sample"] = sorted(all_reward_values)[:50]
    summary["no_tool_call_rate"] = (agg_no_tool / agg_episodes) if agg_episodes else None
    summary["too_few_calls_rate"] = (agg_too_few / agg_episodes) if agg_episodes else None
    summary["avg_predicted_calls"] = (agg_pred_calls / agg_episodes) if agg_episodes else None
    summary["eligible_for_best"] = bool(
        global_step > 0 and total_contributing > 0
        and (total_groups == 0 or total_dead_groups / total_groups < 0.95))
    if not summary["eligible_for_best"]:
        summary["ineligible_reason"] = (
            "steps==0" if global_step == 0 else
            "contributing_turns==0" if total_contributing == 0 else
            "dead_group_rate>=0.95")
    log_f.close()
    return summary


def _turn_returns(
    r_seq: List[float], episode_reward: float, gamma: float, lambda_episode: float
) -> List[float]:
    """G_t = sum_{k=t}^{T} gamma^(k-t) r_k + lambda_episode * gamma^(T-t+1) * R_episode.

    T = last generated turn index (len(r_seq) - 1).
    """
    T = len(r_seq) - 1
    returns: List[float] = []
    for t in range(len(r_seq)):
        disc = 0.0
        for k in range(t, len(r_seq)):
            disc += (gamma ** (k - t)) * r_seq[k]
        disc += lambda_episode * (gamma ** (T - t + 1)) * episode_reward
        returns.append(disc)
    return returns


def _wandb_log_task(wandb_run, rec: Dict[str, Any], stage: int, global_step: int) -> None:
    """Log per-task training metrics to W&B (no-op if wandb_run is None)."""
    if wandb_run is None:
        return
    try:
        wandb_run.log({
            "train/mean_reward":       rec.get("mean_reward", 0.0),
            "train/strict_pass":       rec.get("strict_gold_trace_pass", 0.0),
            "train/loss":              rec.get("loss", 0.0),
            "train/contributing_turns": rec.get("contributing_turns", 0),
            "train/clipped_rate":      rec.get("clipped_rate", 0.0),
            "train/dead_group":        1.0 if rec.get("dead_group") else 0.0,
            "train/group_mixed":       1.0 if rec.get("group_mixed") else 0.0,
            "train/zero_tool_calls":   rec.get("zero_tool_calls", 0.0),
            "train/return_std":        rec.get("return_std", 0.0),
            "train/kl":                rec.get("kl", 0.0),
            "train/mean_logprob":      rec.get("mean_logprob", 0.0),
            "train/parse_error_rate":  rec.get("parse_error_rate", 0.0),
            "train/no_tool_call_rate": rec.get("no_tool_call_rate", 0.0),
            "train/too_few_calls_rate": rec.get("too_few_calls_rate", 0.0),
            "train/invalid_reference_rate": rec.get("invalid_reference_rate", 0.0),
            "train/executor_error_rate": rec.get("executor_error_rate", 0.0),
            "train/executable_trajectory_rate": rec.get("executable_trajectory_rate", 0.0),
            "train/tool_final_answer_pass_rate": rec.get("tool_final_answer_pass_rate", 0.0),
            "train/rollout_length_mean": rec.get("rollout_length_mean", 0.0),
            "train/stage":             stage,
            "train/epoch":             rec.get("epoch", 0),
        }, step=global_step)
    except Exception:
        pass


def _wandb_log_epoch(wandb_run, epoch: int, stage: int, tasks_seen: int,
                     rewards: List[float], fallback_used: bool) -> None:
    """Log end-of-epoch summary to W&B."""
    if wandb_run is None or not rewards:
        return
    try:
        wandb_run.log({
            "epoch/mean_reward":    sum(rewards) / len(rewards),
            "epoch/strict_pass":    sum(1 for r in rewards if r == 1.0) / len(rewards),
            "epoch/tasks_seen":     tasks_seen,
            "epoch/fallback_used":  1.0 if fallback_used else 0.0,
            "epoch/stage":          stage,
            "epoch/epoch":          epoch,
        })
    except Exception:
        pass


def _reward_component_rates(episodes: List[Episode], task: Dict[str, Any]) -> Dict[str, Any]:
    """v2 reward-component / failure-mode rates over a rollout group.

    Best-effort and fully guarded: if nestful_core is unavailable this returns {}
    and never interferes with training.
    """
    try:
        import os as _os
        import sys as _sys
        _exp = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _exp not in _sys.path:
            _sys.path.insert(0, _exp)
        from nestful_core import rewards as _R
    except Exception:
        return {}
    trajs = [e.trajectory for e in episodes]
    n = len(trajs) or 1

    def _rate(fn) -> float:
        return sum(1 for t in trajs if fn(t)) / n

    try:
        return {
            "parse_error_rate": _rate(_R.has_parse_error),
            "no_tool_call_rate": _rate(_R.has_no_tool_call),
            "too_few_calls_rate": sum(1 for t in trajs if _R.too_few_calls(t, task)) / n,
            "invalid_reference_rate": _rate(_R.has_invalid_reference),
            "executor_error_rate": _rate(_R.has_executor_error),
            "executable_trajectory_rate": _rate(_R.is_executable_trajectory),
            "tool_final_answer_pass_rate": sum(
                1 for t in trajs if _R.tool_final_answer_pass(t, task)) / n,
            "num_successful_calls_mean": sum(
                _R.num_successful_calls(t) for t in trajs) / n,
            "rollout_length_mean": sum(len(t.turns) for t in trajs) / n,
        }
    except Exception:
        return {}


def _first_error_mean(
    episodes: List[Episode], task: Dict[str, Any], gold_obs=None
) -> Optional[float]:
    vals = []
    for e in episodes:
        rr = strict_gold_trace_reward(e.trajectory, task, gold_obs)
        fe = rr.diagnostics.get("first_error_turn")
        if fe is not None:
            vals.append(fe)
    return (sum(vals) / len(vals)) if vals else None
