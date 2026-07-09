"""Stage probe — forward-only GRPO signal check (Phase 1c / RESEARCH_FIX_PLAN E0).

Rolls out ``num_generations`` completions per task with the SAME rollout code
(`rollout.run_episode`, mode="train") and the SAME reward dispatch
(`vllm_dp_pool.resolve_reward_info`) as training, but runs NO optimizer step and
writes NO adapter. Answers, for the price of a forward pass: "would GRPO get any
gradient signal from this stage with this reward and this checkpoint?"

Outputs (in --output-dir):
    PROBE_REPORT.md / PROBE_REPORT.json   aggregate + per-motif/per-tool stats
    signal_positive_tasks.jsonl           raw rows of tasks with >=2 unique rewards
    dead_low_tasks.jsonl                  raw rows of dead groups with low reward
    motif_signal_table.csv                per-motif dead rates
    manifest.json                         provenance (git, dataset SHA, seed, ...)

Backends:
    vllm / hf   real model rollouts (GPU pod) — used for calibration
    stub        deterministic fake completions (CPU) — pipeline self-test ONLY;
                never use stub numbers to make training decisions.

Usage examples (repo root):
    # dry run: resolve everything, print plan, execute nothing
    python .../probe_stage.py --stage 2 --dry-run

    # CPU pipeline self-test
    python .../probe_stage.py --stage 2 --num-tasks 8 --backend stub --output-dir /tmp/probe

    # pod calibration run (stage 1 must reproduce dead rate ~1.0)
    python .../probe_stage.py --stage 1 --num-tasks 50 --num-generations 8 --backend vllm
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "lib"))

from paths import (  # noqa: E402
    CANONICAL_STAGE_FILES, MINIMAL_ROOT, REPO_ROOT, V3_ROOT,
    dataset_info, is_legacy_dataset_path,
)
from run_manifest import build_manifest, write_manifest  # noqa: E402

if MINIMAL_ROOT not in sys.path:
    sys.path.insert(0, MINIMAL_ROOT)

DEFAULT_CONFIG = os.path.join(REPO_ROOT, "experiments", "nestful_mtgrpo_partial", "config.yaml")


# ---------------------------------------------------------------------------
# stub backend (CPU pipeline self-test — NOT for training decisions)
# ---------------------------------------------------------------------------

_STUB_BEHAVIORS = ("perfect", "too_few", "wrong_tool", "wrong_args", "no_call")


def _stub_rng_choice(seed: int, task_id: str, gen_idx: int) -> str:
    h = hashlib.sha256(f"{seed}|{task_id}|{gen_idx}".encode()).hexdigest()
    return _STUB_BEHAVIORS[int(h[:8], 16) % len(_STUB_BEHAVIORS)]


def make_stub_generate_fn(task: Dict[str, Any], behavior: str):
    """Deterministic fake completions exercising distinct reward bands."""
    state = {"turn": 0}
    gold = task.get("gold_calls") or []

    def _fmt(call: Dict[str, Any]) -> str:
        return ("<tool_call_answer>[" + json.dumps(
            {"name": call.get("name"), "arguments": call.get("arguments") or {}},
            ensure_ascii=False) + "]</tool_call_answer>")

    def gen(messages, max_new_tokens):  # noqa: ARG001 - signature fixed by run_episode
        i = state["turn"]
        state["turn"] += 1
        if behavior == "no_call":
            text = "I think the answer is 42."  # unparsable -> no tool call
        elif behavior == "too_few":
            text = _fmt(gold[0]) if i == 0 and gold else "<tool_call_answer>[]</tool_call_answer>"
        elif behavior == "wrong_tool":
            if i < len(gold):
                c = dict(gold[i]); c = {**c, "name": "definitely_not_a_tool"}
                text = _fmt(c)
            else:
                text = "<tool_call_answer>[]</tool_call_answer>"
        elif behavior == "wrong_args":
            if i < len(gold):
                c = gold[i]
                bad_args = {k: "wrong_value" for k in (c.get("arguments") or {})}
                text = _fmt({"name": c.get("name"), "arguments": bad_args})
            else:
                text = "<tool_call_answer>[]</tool_call_answer>"
        else:  # perfect
            text = _fmt(gold[i]) if i < len(gold) else "<tool_call_answer>[]</tool_call_answer>"
        return {"text": text, "prompt_tokens": 10, "completion_tokens": 10,
                "clipped": False, "prompt_overflow": False}

    return gen


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------

def shannon_entropy_bits(values: List[float]) -> float:
    if not values:
        return 0.0
    counts: Dict[float, int] = defaultdict(int)
    for v in values:
        counts[round(v, 3)] += 1
    n = len(values)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def reward_histogram(values: List[float], n_bins: int = 10) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        label = f"[{lo:.1f},{hi:.1f})" if i < n_bins - 1 else f"[{lo:.1f},1.0]"
        hist[label] = 0
    for v in values:
        idx = min(int(v * n_bins), n_bins - 1)
        hist[list(hist.keys())[idx]] += 1
    return hist


def _rate(flags: List[bool]) -> Optional[float]:
    return round(sum(1 for f in flags if f) / len(flags), 4) if flags else None


# ---------------------------------------------------------------------------
# probe core
# ---------------------------------------------------------------------------

def run_probe(args) -> int:
    # --- dataset resolution + guardrails ------------------------------------
    if args.dataset:
        dataset_path = os.path.abspath(args.dataset)
    elif args.stage:
        dataset_path = CANONICAL_STAGE_FILES[int(args.stage)]
    else:
        print("[probe] ERROR: pass --stage N or --dataset PATH", file=sys.stderr)
        return 1
    if not os.path.isfile(dataset_path):
        print(f"[probe] ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        return 1
    if is_legacy_dataset_path(dataset_path) and not args.allow_legacy_dataset:
        print(f"[probe] ERROR: {dataset_path} is in the LEGACY dataset-B tree. "
              "Pass --allow-legacy-dataset to override.", file=sys.stderr)
        return 3

    if args.stage:
        os.environ["TRAIN_STAGE"] = str(args.stage)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(
        args.output_dir
        or os.path.join(V3_ROOT, "outputs", "probes",
                        f"probe_{args.stage or 'ds'}_{args.reward_policy}_{ts}"))

    print(f"[probe] dataset        : {dataset_path}")
    print(f"[probe] reward policy  : {args.reward_policy}")
    print(f"[probe] checkpoint     : {args.checkpoint or '(base model)'}")
    print(f"[probe] tasks x gens   : {args.num_tasks} x {args.num_generations}")
    print(f"[probe] decoding       : T={args.temperature} top_p={args.top_p} seed={args.seed}")
    print(f"[probe] backend        : {args.backend}")
    print(f"[probe] output dir     : {out_dir}")
    if args.backend == "stub":
        print("[probe] WARNING: stub backend — pipeline self-test only, numbers are FAKE.")

    if args.dry_run:
        print("[probe] DRY RUN — nothing executed.")
        return 0

    # --- heavy imports (after dry-run exit) ---------------------------------
    import importlib.util

    def _import_by_path(path: str, name: str):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    base_run = _import_by_path(os.path.join(MINIMAL_ROOT, "run.py"), "mtgrpo_base_run")
    from data import load_tasks  # noqa: E402
    from reward import compute_gold_observations  # noqa: E402
    from rollout import run_episode  # noqa: E402
    from vllm_dp_pool import resolve_reward_info  # noqa: E402

    config = base_run.load_config(os.path.abspath(args.config))
    overrides = [
        f"reward.train_policy={args.reward_policy}",
        f"generation.temperature={args.temperature}",
        f"generation.top_p={args.top_p}",
        f"experiment.seed={args.seed}",
        f"experiment.output_dir={out_dir}",
        f"hardware.use_vllm={'true' if args.backend == 'vllm' else 'false'}",
    ]
    base_run._apply_overrides(config, overrides)
    base_run._normalize_config_paths(config)
    # the probe's own output dir must not be re-rooted by config normalization
    config.setdefault("experiment", {})["output_dir"] = out_dir

    # SAME reward dispatch as training — raises on unknown policy.
    reward_fn, reward_info = resolve_reward_info(config)
    print(f"[probe] reward dispatch: configured={reward_info['configured_policy']} "
          f"resolved={reward_info['resolved_policy']} "
          f"({reward_info['reward_fn_module']}.{reward_info['reward_fn_name']})")
    if reward_info.get("fallback_used"):
        print("[probe] ERROR: reward dispatch fell back to strict — aborting.", file=sys.stderr)
        return 4

    tasks = load_tasks(dataset_path, max_tasks=args.num_tasks, seed=args.seed)
    print(f"[probe] loaded {len(tasks)} tasks")

    # raw rows keyed by id (for the filter output files)
    raw_by_id: Dict[str, str] = {}
    with open(dataset_path, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rid = str(json.loads(line).get("sample_id") or f"task_{idx}")
            except json.JSONDecodeError:
                continue
            raw_by_id[rid] = line

    registry = base_run.build_registry(config)

    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    model = tokenizer = generate_fn = None
    if args.backend != "stub":
        model, tokenizer, generate_fn = base_run._load_inference_backend(
            config, args.checkpoint, mode="eval")

    # --- rollout loop (NO optimizer, NO adapter writes) ----------------------
    groups: List[Dict[str, Any]] = []
    all_rewards: List[float] = []
    for t_i, task in enumerate(tasks):
        gold_obs = compute_gold_observations(task, registry)
        rewards: List[float] = []
        diags: List[Dict[str, Any]] = []
        rseqs: List[List[float]] = []
        for g_i in range(args.num_generations):
            if args.backend == "stub":
                behavior = _stub_rng_choice(args.seed, task["task_id"], g_i)
                gen_fn = make_stub_generate_fn(task, behavior)
            else:
                gen_fn = generate_fn
            traj = run_episode(model, tokenizer, task, config,
                               registry=registry, mode="train", generate_fn=gen_fn)
            rr = reward_fn(traj, task, gold_obs)
            rewards.append(float(rr["episode_reward"]))
            rseqs.append([float(x) for x in (rr.get("r_seq") or [])])
            diags.append(rr.get("diagnostics") or {})
        uniq = sorted({round(r, 6) for r in rewards})
        dead = len(uniq) <= 1
        # position-artifact analog: episode rewards identical but per-turn
        # scores differ across completions -> variance is positional noise
        rseq_variety = len({tuple(round(x, 6) for x in s) for s in rseqs}) > 1
        groups.append({
            "task_id": task["task_id"],
            "motif_type": task.get("motif_type"),
            "stage": task.get("stage"),
            "gold_tool_0": (task.get("gold_calls") or [{}])[0].get("name"),
            "num_gold_calls": task.get("num_calls"),
            "rewards": [round(r, 6) for r in rewards],
            "unique_rewards": len(uniq),
            "dead": dead,
            "mean_reward": round(sum(rewards) / len(rewards), 6),
            "position_artifact": bool(dead and rseq_variety),
            "too_few_rate": _rate([bool(d.get("too_few_calls")) for d in diags]),
            "wrong_tool_rate": _rate([bool(d.get("wrong_tool")) for d in diags]),
            "wrong_args_rate": _rate([bool(d.get("wrong_args")) for d in diags]),
            "parse_error_rate": _rate([bool(d.get("parse_error")) for d in diags]),
            "no_tool_call_rate": _rate([bool(d.get("no_tool_call")) for d in diags]),
            "invalid_ref_rate": _rate([bool(d.get("invalid_reference")) for d in diags]),
            "avg_pred_calls": round(sum(int(d.get("n_pred_calls") or 0) for d in diags)
                                    / len(diags), 4),
            "cap_reasons": sorted({str(d.get("reward_cap_reason")) for d in diags}),
        })
        all_rewards.extend(rewards)
        if (t_i + 1) % 10 == 0:
            print(f"[probe] {t_i + 1}/{len(tasks)} tasks probed", flush=True)

    # --- aggregate -----------------------------------------------------------
    n_groups = len(groups)
    dead_groups = [g for g in groups if g["dead"]]
    dead_low = [g for g in dead_groups if g["mean_reward"] <= args.dead_low_threshold]
    dead_high = [g for g in dead_groups if g["mean_reward"] >= 0.9]
    signal_positive = [g for g in groups if not g["dead"]]

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    dead_group_rate = round(len(dead_groups) / n_groups, 4) if n_groups else None
    mean_unique = _mean([g["unique_rewards"] for g in groups])
    proceed = bool(dead_group_rate is not None and dead_group_rate < 0.5
                   and (mean_unique or 0) >= 2.0)

    per_motif: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for g in groups:
        per_motif[str(g["motif_type"])].append(g)
    motif_rows = [{
        "motif_type": m,
        "n_groups": len(gs),
        "dead_rate": _mean([1.0 if g["dead"] else 0.0 for g in gs]),
        "mean_reward": _mean([g["mean_reward"] for g in gs]),
        "mean_unique_rewards": _mean([g["unique_rewards"] for g in gs]),
        "too_few_rate": _mean([g["too_few_rate"] for g in gs]),
        "wrong_tool_rate": _mean([g["wrong_tool_rate"] for g in gs]),
    } for m, gs in sorted(per_motif.items())]

    per_tool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for g in groups:
        per_tool[str(g["gold_tool_0"])].append(g)
    tool_rows = [{
        "gold_tool_0": t,
        "n_groups": len(gs),
        "dead_rate": _mean([1.0 if g["dead"] else 0.0 for g in gs]),
        "wrong_tool_rate": _mean([g["wrong_tool_rate"] for g in gs]),
        "mean_reward": _mean([g["mean_reward"] for g in gs]),
    } for t, gs in sorted(per_tool.items())]

    report = {
        "probe_version": 1,
        "backend": args.backend,
        "stub_warning": (args.backend == "stub"),
        "dataset": dataset_info(dataset_path),
        "reward": reward_info,
        "checkpoint": args.checkpoint,
        "decoding": {"temperature": args.temperature, "top_p": args.top_p,
                     "seed": args.seed},
        "num_tasks": n_groups,
        "num_generations": args.num_generations,
        "dead_group_rate": dead_group_rate,
        "mixed_group_rate": round(len(signal_positive) / n_groups, 4) if n_groups else None,
        "dead_low_rate": round(len(dead_low) / n_groups, 4) if n_groups else None,
        "dead_high_rate": round(len(dead_high) / n_groups, 4) if n_groups else None,
        "position_artifact_rate": _mean([1.0 if g["position_artifact"] else 0.0 for g in groups]),
        "mean_unique_rewards_per_group": mean_unique,
        "reward_entropy_bits": round(shannon_entropy_bits(all_rewards), 4),
        "reward_histogram": reward_histogram(all_rewards),
        "unique_reward_values": len({round(r, 3) for r in all_rewards}),
        "too_few_calls_rate": _mean([g["too_few_rate"] for g in groups]),
        "avg_predicted_calls": _mean([g["avg_pred_calls"] for g in groups]),
        "wrong_tool_rate": _mean([g["wrong_tool_rate"] for g in groups]),
        "wrong_arg_rate": _mean([g["wrong_args_rate"] for g in groups]),
        "parse_error_rate": _mean([g["parse_error_rate"] for g in groups]),
        "no_tool_call_rate": _mean([g["no_tool_call_rate"] for g in groups]),
        "invalid_reference_rate": _mean([g["invalid_ref_rate"] for g in groups]),
        "proceed_recommendation": proceed,
        "proceed_gate": "dead_group_rate < 0.5 AND mean_unique_rewards_per_group >= 2",
        "groups": groups,
    }

    # --- outputs -------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "PROBE_REPORT.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    def _dump_tasks(fname: str, group_list: List[Dict[str, Any]]) -> int:
        n = 0
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as fh:
            for g in group_list:
                raw = raw_by_id.get(str(g["task_id"]))
                if raw:
                    fh.write(raw + "\n")
                    n += 1
        return n

    n_sig = _dump_tasks("signal_positive_tasks.jsonl", signal_positive)
    n_dl = _dump_tasks("dead_low_tasks.jsonl", dead_low)

    with open(os.path.join(out_dir, "motif_signal_table.csv"), "w",
              encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(motif_rows[0].keys()) if motif_rows else
                                ["motif_type"])
        writer.writeheader()
        writer.writerows(motif_rows)

    md = [
        f"# PROBE REPORT — {os.path.basename(out_dir)}",
        "",
        f"Backend: **{args.backend}**"
        + (" — **STUB (fake numbers, pipeline self-test only)**" if args.backend == "stub" else ""),
        f"Dataset: `{report['dataset']['path']}` (n_probed={n_groups}, "
        f"sha256={report['dataset']['sha256'][:12]}…)",
        f"Reward: `{reward_info['resolved_policy']}` | checkpoint: "
        f"`{args.checkpoint or 'base model'}`",
        f"Decoding: T={args.temperature} top_p={args.top_p} seed={args.seed} | "
        f"{args.num_generations} generations/group",
        "",
        "## GRPO signal",
        "",
        "| metric | value |",
        "|---|---|",
        f"| dead_group_rate | **{report['dead_group_rate']}** |",
        f"| mixed_group_rate | {report['mixed_group_rate']} |",
        f"| dead_low_rate (mean<= {args.dead_low_threshold}) | {report['dead_low_rate']} |",
        f"| dead_high_rate (mean>=0.9, saturated) | {report['dead_high_rate']} |",
        f"| position_artifact_rate | {report['position_artifact_rate']} |",
        f"| mean unique rewards / group | {report['mean_unique_rewards_per_group']} |",
        f"| reward entropy (bits) | {report['reward_entropy_bits']} |",
        f"| distinct reward values | {report['unique_reward_values']} |",
        "",
        "## Behavior",
        "",
        "| metric | value |",
        "|---|---|",
        f"| too_few_calls_rate | {report['too_few_calls_rate']} |",
        f"| avg_predicted_calls | {report['avg_predicted_calls']} |",
        f"| wrong_tool_rate | {report['wrong_tool_rate']} |",
        f"| wrong_arg_rate | {report['wrong_arg_rate']} |",
        f"| parse_error_rate | {report['parse_error_rate']} |",
        f"| no_tool_call_rate | {report['no_tool_call_rate']} |",
        f"| invalid_reference_rate | {report['invalid_reference_rate']} |",
        "",
        "## Reward histogram",
        "",
        "| bin | count |",
        "|---|---|",
    ]
    md += [f"| {b} | {c} |" for b, c in report["reward_histogram"].items()]
    md += [
        "",
        "## Verdict",
        "",
        f"**proceed_recommendation: {proceed}** (gate: {report['proceed_gate']})",
        "" if proceed else
        "GRPO training on this (stage, reward, checkpoint) combination is expected to be "
        "signal-starved. Fix the reward (densify), the init (SFT warmup), or the task mix "
        "(filtering) before spending GPU time.",
        "",
        f"Files: signal_positive_tasks.jsonl ({n_sig} rows), dead_low_tasks.jsonl ({n_dl} rows), "
        "motif_signal_table.csv",
    ]
    with open(os.path.join(out_dir, "PROBE_REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    manifest = build_manifest(
        kind="stage_probe",
        config_path=os.path.abspath(args.config),
        overrides=overrides,
        datasets=[dataset_path],
        seed=args.seed,
        decoding={"temperature": args.temperature, "top_p": args.top_p},
        extra={"reward": reward_info, "checkpoint": args.checkpoint,
               "backend": args.backend, "num_tasks": n_groups,
               "num_generations": args.num_generations,
               "dead_group_rate": dead_group_rate,
               "proceed_recommendation": proceed},
    )
    write_manifest(manifest, os.path.join(out_dir, "manifest.json"))

    print(f"[probe] dead_group_rate={dead_group_rate} mean_unique={mean_unique} "
          f"proceed={proceed}")
    print(f"[probe] report: {os.path.join(out_dir, 'PROBE_REPORT.md')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Forward-only stage probe (no training, no adapter writes).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stage", type=int, choices=[1, 2, 3, 4],
                    help="canonical curriculum stage (mutually exclusive with --dataset)")
    ap.add_argument("--dataset", help="explicit JSONL path")
    ap.add_argument("--checkpoint", default=None, help="LoRA adapter dir (default: base model)")
    ap.add_argument("--reward-policy", default="execution_aware_v3_1_stepwise")
    ap.add_argument("--num-tasks", type=int, default=50)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="rollout temperature (1.0 = training default)")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--backend", choices=["vllm", "hf", "stub"], default="hf")
    ap.add_argument("--dead-low-threshold", type=float, default=0.35)
    ap.add_argument("--allow-legacy-dataset", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return run_probe(args)


if __name__ == "__main__":
    sys.exit(main())
