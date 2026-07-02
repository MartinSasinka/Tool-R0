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
from rollout import Trajectory, Turn, get_stage_token_budget


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


@dataclass
class _PoolTraj:
    """Minimal stand-in for Trajectory holding only the fields the GRPO update
    loop reads. Used when rollouts come back from a data-parallel worker pool,
    where the full Trajectory (with raw tool observations) stays in the worker."""
    clipped_any: bool = False
    zero_tool_calls: bool = False


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
                     zero_tool_calls=bool(res.zero_tool_calls))
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
    model, tokenizer, task, config, registry, max_turns, *, vllm_gen_fn=None
) -> Episode:
    """Run one episode for GRPO.

    When ``vllm_gen_fn`` is provided (opt-in, hardware.use_vllm: true):
    - vLLM handles fast forward generation (no gradients needed here).
    - The completion text is re-tokenised to obtain TurnTokens for the
      subsequent _sequence_logprob() call which still uses the HF model.

    Without vLLM (default): _generate_with_ids() uses the HF model.
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

    for turn_idx in range(max_turns):
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
    return Episode(trajectory=traj, turn_tokens=turn_tokens)


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
        "mt_grpo_mode": "turn_level_minimal" if use_turn_level else "episode_level",
        "gamma": gamma, "lambda_episode": lambda_episode,
        "fallback_used": False,
        "vllm_rollout": (vllm_gen is not None) or (rollout_pool is not None),
        "data_parallel_rollout": rollout_pool is not None,
    }

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
                    )
                    rinfo = episode_turn_reward_seq(ep.trajectory, task, gold_obs)
                    ep.reward = rinfo["episode_reward"]
                    episodes.append(ep)
                    ep_r_seqs.append(rinfo["r_seq"])

            rewards = [e.reward for e in episodes]
            mean_r = sum(rewards) / len(rewards)

            # Per-episode turn-level returns: G_t = sum_{k>=t} gamma^(k-t) r_k
            #   + lambda_episode * gamma^(T-t+1) * R_episode
            ep_returns: List[List[float]] = []
            for ep, r_seq in zip(episodes, ep_r_seqs):
                ep_returns.append(_turn_returns(r_seq, ep.reward, gamma, lambda_episode))

            # Group-relative advantage over all (episode, turn) returns in the group.
            flat = [g for ep, gs in zip(episodes, ep_returns)
                    for g in gs
                    if not (mask_clipped and ep.trajectory.clipped_any)]
            if flat:
                gmean = sum(flat) / len(flat)
                gvar = sum((g - gmean) ** 2 for g in flat) / len(flat)
                gstd = gvar ** 0.5
            else:
                gmean, gstd = 0.0, 0.0

            group_all_zero = all(r == 0.0 for r in rewards)
            group_all_one = all(r == 1.0 for r in rewards)
            dead = gstd == 0.0  # no turn-level signal to learn from

            rec = {
                "epoch": epoch, "task_idx": ti, "task_id": task["task_id"],
                "reward_train_policy": reward_policy,
                "mt_grpo_mode": summary["mt_grpo_mode"],
                "mean_reward": mean_r,
                "episode_rewards": rewards,
                "turn_rewards": ep_r_seqs,
                "return_std": gstd,
                "group_all_zero": group_all_zero, "group_all_one": group_all_one,
                "group_mixed": (not group_all_zero) and (not group_all_one),
                "dead_group": dead,
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

            if dead:
                _log({**rec, "update": "skipped_dead_group"})
                _wandb_log_task(wandb_run, rec, stage, global_step)
                continue

            step_loss = 0.0
            kl_sum = 0.0
            kl_count = 0
            logp_sum = 0.0
            logp_count = 0
            contributing = 0
            for ep, gs in zip(episodes, ep_returns):
                if mask_clipped and ep.trajectory.clipped_any:
                    continue
                for j, tt in enumerate(ep.turn_tokens):
                    if j >= len(gs) or tt.completion_ids.numel() == 0:
                        continue
                    if use_turn_level:
                        adv = (gs[j] - gmean) / (gstd + 1e-8) if normalize_advantage else gs[j]
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

            accum += 1
            rec["loss"] = step_loss
            rec["contributing_turns"] = contributing
            rec["kl"] = (kl_sum / kl_count) if kl_count else 0.0
            rec["mean_logprob"] = (logp_sum / logp_count) if logp_count else 0.0
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
