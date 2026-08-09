#!/usr/bin/env python3
"""P43 rollout diversity smoke (NO GRPO training).

Default: structural seed + fixed-pair analysis (CPU).
With --gpu: run 20 PROFILE prompts × 8 rollouts through the same DP/vLLM
stack as training (requires CUDA + model weights).

Outputs (under pilot4_3 outputs dir or --out-dir):
  P43_ROLLOUT_DIVERSITY_SMOKE.json
  P43_ROLLOUT_DIVERSITY_SMOKE.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACTORY = ROOT.parent / "targeted_tool_data_factory"
OUT_DEFAULT = FACTORY / "outputs" / "pilot4_3_nestful_profile_1000"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FACTORY / "src"))

from rollout_sampling import (  # noqa: E402
    ROLLOUT_SAMPLING_VERSION,
    derive_turn_seed,
    sampling_source_hash,
    stamp_rollout_tasks,
)


def _pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] * (c - k) + s[c] * (k - f))


def _summary(xs):
    if not xs:
        return {}
    return {
        "mean": float(statistics.mean(xs)),
        "median": float(statistics.median(xs)),
        "p10": _pct(xs, 10),
        "p25": _pct(xs, 25),
        "p50": _pct(xs, 50),
        "p75": _pct(xs, 75),
        "p90": _pct(xs, 90),
    }


def fixed_pair_rates(groups):
    """groups: list of list[str] completion hashes length 8."""
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    out = {}
    n = len(groups)
    for i, j in pairs:
        matches = sum(1 for g in groups if g[i] == g[j])
        out[f"{i}=={j}"] = {
            "exact_completion_match_rate": matches / n if n else 0.0,
            "n_match": matches,
            "n_groups": n,
        }
    return out


def structural_seed_smoke(n_prompts: int, group_size: int, base_seed: int):
    groups = []
    seed_groups = []
    for pi in range(n_prompts):
        tid = f"smoke_task_{pi:04d}"
        stamped = stamp_rollout_tasks(
            {"task_id": tid}, num_generations=group_size,
            base_seed=base_seed, global_step=0, epoch=0)
        seeds = [t["_rollout_seed"] for t in stamped]
        # Proxy "completion" = hash of seed stream (structural uniqueness)
        hashes = [
            hashlib.sha256(f"{s}:{derive_turn_seed(s, 0)}".encode()).hexdigest()[:16]
            for s in seeds
        ]
        groups.append(hashes)
        seed_groups.append(seeds)
    pair_rates = fixed_pair_rates(groups)
    suspect = {k: v for k, v in pair_rates.items()
               if k in ("0==1", "3==4", "6==7") and v["exact_completion_match_rate"] > 0.05}
    return {
        "mode": "structural_seeds",
        "n_prompts": n_prompts,
        "group_size": group_size,
        "rollout_sampling_version": ROLLOUT_SAMPLING_VERSION,
        "unique_seeds_per_group": [len(set(s)) for s in seed_groups],
        "n_unique_completions_per_group": [len(set(g)) for g in groups],
        "unique_completions_summary": _summary([len(set(g)) for g in groups]),
        "fixed_pair_match_rates": pair_rates,
        "suspect_fixed_pairs": suspect,
        "fixed_position_duplication_gone": len(suspect) == 0,
        "all_seeds_unique_within_group": all(len(set(s)) == group_size for s in seed_groups),
    }


def gpu_smoke(config_path: Path, n_prompts: int, out_dir: Path):
    import yaml
    from vllm_dp_pool import DataParallelRolloutPool

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # Keep training hypers identical; only limit prompts.
    train_path = (ROOT / cfg["paths"]["train_jsonl"]).resolve()
    if not train_path.is_file():
        train_path = Path(cfg["paths"]["train_jsonl"]).resolve()
    tasks = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            tasks.append(json.loads(line))
            if len(tasks) >= n_prompts:
                break
    assert len(tasks) >= 1

    os.environ.setdefault("SYNTHETIC_TOOLS_DIR",
                          str((ROOT / cfg["paths"]["synthetic_tools_dir"]).resolve()))

    hw = cfg.get("hardware") or {}
    dp = hw.get("rollout_data_parallel_gpus")
    gpus = []
    if dp:
        gpus = [int(x) for x in str(dp).replace(" ", "").split(",") if x != ""]
    if not gpus:
        # Single-GPU fallback: still exercise seeded generate_fn path
        gpus = [0]

    base_seed = int((cfg.get("experiment") or {}).get("seed") or 42)
    num_gen = int(cfg["generation"]["num_generations"])
    assert num_gen == 8

    pool = DataParallelRolloutPool(cfg, gpus=gpus, adapter_path=None)
    try:
        groups = []
        group_rows = []
        for gi, task in enumerate(tasks[:n_prompts]):
            stamped = stamp_rollout_tasks(
                task, num_generations=num_gen, base_seed=base_seed,
                global_step=0, epoch=0)
            results = pool.rollout_many(stamped)
            hashes = []
            rewards = []
            seeds = []
            for ri, (t, res) in enumerate(zip(stamped, results)):
                text_parts = []
                for p_ids, c_ids in (res.turn_token_ids or []):
                    text_parts.append(",".join(map(str, c_ids[:64])))
                ch = hashlib.sha256("|".join(text_parts).encode()).hexdigest()[:16]
                hashes.append(ch)
                rewards.append(float(res.episode_reward))
                seeds.append(int(t["_rollout_seed"]))
                # Per-rollout identity log
                row = {
                    "global_step": 0,
                    "group_id": f"smoke:{gi}",
                    "task_id": task.get("task_id"),
                    "rollout_index": ri,
                    "dp_worker_id": (res.reward_diag or {}).get("dp_worker_id"),
                    "request_id": (res.reward_diag or {}).get("request_id", ri),
                    "rollout_seed": t["_rollout_seed"],
                    "actual_generation_seed": (res.reward_diag or {}).get(
                        "actual_generation_seed"),
                    "temperature": cfg["generation"]["temperature"],
                    "top_p": cfg["generation"]["top_p"],
                    "completion_hash": ch,
                    "episode_reward": float(res.episode_reward),
                    "error": res.error,
                }
                group_rows.append(row)
            rstd = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
            # Lightweight group class from rewards only
            all_high = all(r >= 0.99 for r in rewards)
            all_zero = all(r <= 1e-12 for r in rewards)
            if all_high:
                gcls = "ALL_CORRECT"
            elif all_zero and rstd <= 1e-6:
                gcls = "ALL_FAIL_NO_PROGRESS"
            elif rstd > 1e-6:
                gcls = "MIXED_OR_PROGRESS"
            else:
                gcls = "LOW_VARIANCE"
            groups.append({
                "task_id": task.get("task_id"),
                "completion_hashes": hashes,
                "n_unique_completions": len(set(hashes)),
                "rollout_seeds": seeds,
                "reward_values": rewards,
                "n_unique_rewards": len({round(r, 6) for r in rewards}),
                "reward_std": rstd,
                "group_class": gcls,
                "worker_ids": [r.get("dp_worker_id") for r in group_rows[-num_gen:]],
            })
    finally:
        pool.close()

    hash_groups = [g["completion_hashes"] for g in groups]
    pair_rates = fixed_pair_rates(hash_groups)
    suspect = {k: v for k, v in pair_rates.items()
               if k in ("0==1", "3==4", "6==7")
               and v["exact_completion_match_rate"] >= 0.5}
    class_counts = Counter(g["group_class"] for g in groups)
    return {
        "mode": "gpu_vllm_dp",
        "n_prompts": len(groups),
        "group_size": num_gen,
        "gpus": gpus,
        "model": cfg["model"]["base_model"],
        "temperature": cfg["generation"]["temperature"],
        "top_p": cfg["generation"]["top_p"],
        "reward_policy": cfg["reward"]["train_policy"],
        "rollout_sampling_version": ROLLOUT_SAMPLING_VERSION,
        "rollout_sampling_source_hash": sampling_source_hash(),
        "groups": groups,
        "rollout_identity_log": group_rows,
        "n_unique_completions_per_group": [g["n_unique_completions"] for g in groups],
        "unique_completions_summary": _summary(
            [g["n_unique_completions"] for g in groups]),
        "reward_std_per_group": [g["reward_std"] for g in groups],
        "reward_std_summary": _summary([g["reward_std"] for g in groups]),
        "n_unique_rewards_per_group": [g["n_unique_rewards"] for g in groups],
        "group_class_counts": dict(class_counts),
        "fixed_pair_match_rates": pair_rates,
        "suspect_fixed_pairs": suspect,
        "fixed_position_duplication_gone": len(suspect) == 0,
        "mean_exact_dup_rate_all_pairs": (
            statistics.mean(v["exact_completion_match_rate"]
                            for v in pair_rates.values()) if pair_rates else 0.0),
    }


def write_reports(payload: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "P43_ROLLOUT_DIVERSITY_SMOKE.json"
    mp = out_dir / "P43_ROLLOUT_DIVERSITY_SMOKE.md"
    jp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                  encoding="utf-8")
    sus = payload.get("suspect_fixed_pairs") or {}
    uc = payload.get("unique_completions_summary") or {}
    lines = [
        "# P43 Rollout Diversity Smoke",
        "",
        f"- mode: `{payload.get('mode')}`",
        f"- rollout_sampling_version: `{payload.get('rollout_sampling_version')}`",
        f"- n_prompts × group_size: {payload.get('n_prompts')} × {payload.get('group_size')}",
        f"- fixed_position_duplication_gone: **{payload.get('fixed_position_duplication_gone')}**",
        f"- suspect fixed pairs (0==1 / 3==4 / 6==7 at ≥50% or structural >5%): "
        f"`{json.dumps(sus)}`",
        f"- unique completions/group summary: `{json.dumps(uc)}`",
        f"- group_class_counts: `{json.dumps(payload.get('group_class_counts'))}`",
        f"- reward_std summary: `{json.dumps(payload.get('reward_std_summary'))}`",
        "",
        "## Fixed-pair exact match rates (selected)",
        "",
    ]
    fpr = payload.get("fixed_pair_match_rates") or {}
    for key in ("0==1", "0==2", "3==4", "6==7", "1==2", "2==3"):
        if key in fpr:
            lines.append(
                f"- `{key}`: {fpr[key]['exact_completion_match_rate']:.4f} "
                f"({fpr[key]['n_match']}/{fpr[key]['n_groups']})")
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jp, mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true",
                    help="Run real vLLM DP smoke (20×8)")
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--config", type=Path,
                    default=ROOT / "configs" / "qwen3_p43_profile1000_dynamic_online.yaml")
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    if args.gpu:
        payload = gpu_smoke(args.config, args.n_prompts, args.out_dir)
    else:
        payload = structural_seed_smoke(args.n_prompts, 8, 42)
        payload["note"] = (
            "CPU structural smoke only. Re-run with --gpu on the training pod "
            "for completion-hash diversity through the real vLLM DP stack.")
    jp, mp = write_reports(payload, args.out_dir)
    print(json.dumps({
        "wrote": [str(jp), str(mp)],
        "fixed_position_duplication_gone": payload.get("fixed_position_duplication_gone"),
        "mode": payload.get("mode"),
    }, indent=2))
    if not payload.get("fixed_position_duplication_gone", False):
        sys.exit(2)


if __name__ == "__main__":
    main()
