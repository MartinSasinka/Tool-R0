#!/usr/bin/env python3
"""Pre-training base-model probe for Qwen3-4B on PROFILE_1000.

Selects exactly 200 prompts by NESTFUL call profile (66/44/27/19/44),
runs 8 rollouts each (=1600 trajectories) with the SAME stack as GRPO,
classifies groups, initialises sampler history, and writes:

  QWEN3_PRETRAIN_PROBE.md
  QWEN3_PRETRAIN_PROBE.json
  QWEN3_PRETRAIN_PROBE_GROUPS.csv

Does NOT filter dead tasks out of the training pool.
Does NOT start GRPO training.

Usage (from nestful_mtgrpo_minimal/):
  python scripts/run_qwen3_pretrain_probe.py \\
      --config configs/qwen3_p43_profile1000_dynamic.yaml \\
      [--dry-run]   # select+plan only, no GPU
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_FACTORY = _HERE.parent / "targeted_tool_data_factory" / "src"
if str(_FACTORY) not in sys.path:
    sys.path.insert(0, str(_FACTORY))

PROBE_QUOTA = {"2": 66, "3": 44, "4": 27, "5": 19, "6+": 44}
GROUP_SIZE = 8
OUT_DIR = (_HERE.parent / "targeted_tool_data_factory" / "outputs"
           / "pilot4_3_nestful_profile_1000")


def select_stratified(rows, seed=42):
    by_b = defaultdict(list)
    for r in rows:
        n = len(r.get("gold_calls") or [])
        b = "6+" if n >= 6 else str(n)
        by_b[b].append(r)
    rng = random.Random(seed)
    selected = []
    for b, need in PROBE_QUOTA.items():
        pool = list(by_b[b])
        rng.shuffle(pool)
        if len(pool) < need:
            raise RuntimeError(f"bucket {b}: need {need}, have {len(pool)}")
        selected.extend(pool[:need])
    rng.shuffle(selected)
    return selected


def classify_rewards(rewards, terminals, eps=1e-6):
    from targeted_tool_data.sampling.nestful_profile import (
        GroupObservation, classify_group, map_group_class, is_effective_nestful,
    )
    o = GroupObservation(
        global_step=0, prompt_id="x", group_size=len(rewards),
        terminal_rewards=list(terminals),
        process_rewards=list(rewards),
        total_rewards=list(rewards),
        parse_flags=[True] * len(rewards),
    )
    cls = classify_group(o, eps_reward=eps, eps_process=eps)
    user = map_group_class(cls, o, eps)
    eff = is_effective_nestful(o, {"reward_variance_epsilon": eps,
                                   "drop_low_variance": True})
    return user, eff, o.reward_std, o.reward_mean


def dry_run_report(selected, out_dir: Path):
    """Write a probe plan + synthetic placeholder when GPU unavailable."""
    by_b = Counter()
    by_qm = Counter()
    by_diff = Counter()
    for r in selected:
        n = len(r["gold_calls"])
        b = "6+" if n >= 6 else str(n)
        by_b[b] += 1
        by_qm[r.get("actual_query_mode") or "?"] += 1
        by_diff[r.get("difficulty_band") or "?"] += 1

    payload = {
        "status": "PROBE_PLANNED_NOT_EXECUTED",
        "reason": "dry-run or inference backend unavailable",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "n_prompts": len(selected),
        "group_size": GROUP_SIZE,
        "n_trajectories_planned": len(selected) * GROUP_SIZE,
        "call_bucket_selection": dict(by_b),
        "query_mode_selection": dict(by_qm),
        "difficulty_selection": dict(by_diff),
        "selected_task_ids": [r["task_id"] for r in selected],
        "group_class_counts": None,
        "base_model_success_rate": None,
        "estimated_effective_group_rate": None,
        "note": "Run without --dry-run on a GPU host to fill group classes.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "QWEN3_PRETRAIN_PROBE.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    ids_path = out_dir / "QWEN3_PRETRAIN_PROBE_SELECTION.json"
    ids_path.write_text(json.dumps({
        "task_ids": [r["task_id"] for r in selected],
        "quota": PROBE_QUOTA,
    }, indent=2), encoding="utf-8")

    md = [
        "# QWEN3 Pre-train Probe",
        "",
        f"**Status:** `{payload['status']}`",
        "",
        f"- Model: `{payload['model']}`",
        f"- Prompts: {payload['n_prompts']} (quota {PROBE_QUOTA})",
        f"- Rollouts/prompt: {GROUP_SIZE} → {payload['n_trajectories_planned']} trajectories",
        "",
        "## Selection (executed)",
        f"- call buckets: `{dict(by_b)}`",
        f"- query modes: `{dict(by_qm)}`",
        f"- difficulty: `{dict(by_diff)}`",
        "",
        "## Group classes",
        "Not yet measured — inference backend was not run.",
        "",
        "## Next step",
        "On a GPU host with the GRPO stack:",
        "```bash",
        "cd experiments/nestful_mtgrpo_minimal",
        "python scripts/run_qwen3_pretrain_probe.py "
        "--config configs/qwen3_p43_profile1000_dynamic.yaml",
        "```",
        "",
        "Do **not** start full GRPO until this probe is analysed.",
    ]
    (out_dir / "QWEN3_PRETRAIN_PROBE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    # empty groups csv header
    with open(out_dir / "QWEN3_PRETRAIN_PROBE_GROUPS.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "task_id", "call_bucket", "query_mode", "difficulty",
            "structural_pattern", "n_terminal_success", "reward_mean",
            "reward_std", "group_class", "effective",
        ])
    print(f"[probe] dry-run artifacts -> {out_dir}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(
        _HERE / "configs" / "qwen3_p43_profile1000_dynamic.yaml"))
    ap.add_argument("--profile-jsonl", default=None)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-prompts", type=int, default=200)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    profile_path = args.profile_jsonl or cfg["paths"]["train_jsonl"]
    profile_path = Path(profile_path)
    if not profile_path.is_absolute():
        cand = (_HERE / profile_path).resolve()
        if not cand.exists():
            cand = (_HERE.parent / "targeted_tool_data_factory" /
                    str(profile_path).replace("../targeted_tool_data_factory/", "")
                    ).resolve()
        # also try relative to factory
        if not cand.exists():
            cand = (_HERE.parent / "targeted_tool_data_factory" / "outputs"
                    / "pilot4_3_nestful_profile_1000"
                    / "train_nestful_profile_1000.jsonl")
        profile_path = cand

    rows = [json.loads(l) for l in profile_path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    selected = select_stratified(rows, seed=args.seed)[: args.max_prompts]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect GPU / refuse expensive path unless available
    gpu_ok = False
    if not args.dry_run:
        try:
            import torch
            gpu_ok = bool(torch.cuda.is_available())
        except Exception:
            gpu_ok = False

    if args.dry_run or not gpu_ok:
        if not args.dry_run and not gpu_ok:
            print("[probe] CUDA unavailable — writing PLAN artifacts only "
                  "(re-run on GPU host for real rollouts).", flush=True)
        dry_run_report(selected, out_dir)
        return 0

    # ── Live probe path (GPU) ──────────────────────────────────────────────
    # Reuse trainer stack: load model once, roll out 8 gens per prompt.
    os.chdir(_HERE)
    from run import load_config, build_registry, _normalize_config_paths
    from data import load_tasks
    from model_load import load_model_and_tokenizer
    from grpo_train import _rollout_episode_for_train
    from reward import compute_gold_observations
    from vllm_dp_pool import resolve_reward_fn  # may fail; fallback below

    config = load_config(args.config)
    _normalize_config_paths(config)
    config["generation"]["num_generations"] = GROUP_SIZE
    registry = build_registry(config)

    # Prefer vLLM if configured
    model, tokenizer = load_model_and_tokenizer(config, None, for_training=False)
    vllm_gen = None
    if config.get("hardware", {}).get("use_vllm"):
        try:
            from vllm_generate import build_vllm_generator
            vllm_gen = build_vllm_generator(config, tokenizer, adapter_path=None,
                                            mode="eval")
        except Exception as exc:
            print(f"[probe] vLLM unavailable ({exc}); using HF generate", flush=True)

    # reward
    try:
        from reward import episode_turn_reward_seq
    except ImportError:
        raise

    # Monkeypatch episode_turn_reward_seq if execution_aware is configured
    policy = (config.get("reward") or {}).get("train_policy", "strict")
    if policy in ("execution_aware", "execution"):
        try:
            sys.path.insert(0, str(_HERE.parent / "nestful_mtgrpo_partial"))
            import execution_reward as er
            episode_turn_reward_seq = er.episode_turn_reward_seq  # noqa: F841
            import grpo_train as gt
            gt.episode_turn_reward_seq = er.episode_turn_reward_seq
        except Exception as exc:
            print(f"[probe] WARNING: could not enable execution_aware ({exc})",
                  flush=True)

    from targeted_tool_data.sampling.nestful_profile import NestfulProfileSampler
    from targeted_tool_data.sampling.nestful_profile import load_profile_enrichment_refs
    from nestful_sampler_bridge import save_sampler_artifacts

    prof_refs, _ = load_profile_enrichment_refs(str(profile_path), None)
    sampler = NestfulProfileSampler(
        prof_refs, config={"sampler_mode": "dynamic_profile"}, seed=args.seed)

    groups = []
    class_counts = Counter()
    success_rates = []
    by_bucket_cls = defaultdict(Counter)
    by_qmode_cls = defaultdict(Counter)
    by_diff_cls = defaultdict(Counter)

    for i, task in enumerate(selected):
        gold_n = len(task.get("gold_calls") or [])
        gold_obs = compute_gold_observations(
            task, registry, mode=(config.get("executor") or {}).get("mode", "auto"))
        rewards, terminals = [], []
        for _ in range(GROUP_SIZE):
            ep = _rollout_episode_for_train(
                model, tokenizer, task, config, registry,
                max_turns=gold_n, vllm_gen_fn=vllm_gen, gold_obs=gold_obs,
            )
            from reward import episode_turn_reward_seq as _rseq
            rinfo = _rseq(ep.trajectory, task, gold_obs)
            r = float(rinfo["episode_reward"])
            rewards.append(r)
            diag = rinfo.get("diagnostics") or {}
            terminals.append(1.0 if diag.get("final_answer_pass") or r >= 0.99 else 0.0)

        bucket = "6+" if gold_n >= 6 else str(gold_n)
        user, eff, rstd, rmean = classify_rewards(rewards, terminals)
        class_counts[user] += 1
        success_rates.append(sum(terminals) / len(terminals))
        qm = task.get("actual_query_mode") or "?"
        diff = task.get("difficulty_band") or "?"
        pat = (task.get("declared") or {}).get("structural_pattern") or "?"
        by_bucket_cls[bucket][user] += 1
        by_qmode_cls[qm][user] += 1
        by_diff_cls[diff][user] += 1

        # init sampler history (never drop task from pool)
        from targeted_tool_data.sampling.nestful_profile import GroupObservation
        gobs = GroupObservation(
            global_step=0, prompt_id=str(task["task_id"]), group_size=GROUP_SIZE,
            terminal_rewards=terminals, process_rewards=rewards,
            total_rewards=rewards, parse_flags=[True] * GROUP_SIZE,
            call_bucket=bucket, query_mode=qm, difficulty_band=diff,
        )
        sampler.observe_group(gobs)

        groups.append({
            "task_id": task["task_id"],
            "call_bucket": bucket,
            "query_mode": qm,
            "difficulty": diff,
            "structural_pattern": pat,
            "n_terminal_success": int(sum(terminals)),
            "reward_mean": round(rmean, 6),
            "reward_std": round(rstd, 6),
            "group_class": user,
            "effective": eff,
        })
        print(f"[probe] {i+1}/{len(selected)} {task['task_id']} "
              f"bucket={bucket} class={user} mean={rmean:.3f}", flush=True)

    n = len(groups)
    eff_n = sum(1 for g in groups if g["effective"])
    payload = {
        "status": "PROBE_COMPLETE",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "n_prompts": n,
        "group_size": GROUP_SIZE,
        "n_trajectories": n * GROUP_SIZE,
        "base_model_success_rate": round(sum(success_rates) / max(n, 1), 6),
        "group_class_counts": dict(class_counts),
        "by_call_bucket": {k: dict(v) for k, v in by_bucket_cls.items()},
        "by_query_mode": {k: dict(v) for k, v in by_qmode_cls.items()},
        "by_difficulty": {k: dict(v) for k, v in by_diff_cls.items()},
        "effective_groups": eff_n,
        "estimated_effective_group_rate": round(eff_n / max(n, 1), 4),
        "oversample_for_16": round(16 / max(eff_n / max(n, 1), 1e-3), 2),
        "oversample_for_32": round(32 / max(eff_n / max(n, 1), 1e-3), 2),
    }
    (out_dir / "QWEN3_PRETRAIN_PROBE.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    with open(out_dir / "QWEN3_PRETRAIN_PROBE_GROUPS.csv", "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(groups[0].keys()))
        w.writeheader()
        w.writerows(groups)

    save_sampler_artifacts(sampler, out_dir / "probe_sampler_state")

    md = [
        "# QWEN3 Pre-train Probe",
        "",
        f"**Status:** `{payload['status']}`",
        "",
        f"- Model: `{payload['model']}`",
        f"- Base-model terminal success rate: **{payload['base_model_success_rate']}**",
        f"- Effective group rate: **{payload['estimated_effective_group_rate']}** "
        f"({eff_n}/{n})",
        f"- Oversample × for 16/32 effective: "
        f"{payload['oversample_for_16']} / {payload['oversample_for_32']}",
        "",
        "## Group classes",
        "```",
        json.dumps(payload["group_class_counts"], indent=2),
        "```",
        "",
        "## By call bucket",
        "```",
        json.dumps(payload["by_call_bucket"], indent=2),
        "```",
        "",
        "Sampler history initialised under `probe_sampler_state/` "
        "(tasks NOT removed from pool).",
    ]
    (out_dir / "QWEN3_PRETRAIN_PROBE.md").write_text("\n".join(md) + "\n",
                                                      encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
