#!/usr/bin/env python3
"""One-GPU inference worker of the Pilot2 signal probe.

Rolls out a shard of tasks with the SAME rollout code (``rollout.run_episode``,
``mode="train"``), the SAME executor (``executor.mode=synthetic`` on the factory
trainer adapter) and the SAME reward dispatch
(``vllm_dp_pool.resolve_reward_info``) as the planned D1 GRPO run — and nothing
else. There is no optimizer, no backward pass, no LoRA adapter and no checkpoint
write anywhere in this file.

Each rollout is appended to ``--out`` as one JSON line, keyed by a content hash
of the task row plus every knob that could change the model's output, so
``--resume`` re-uses finished work and only regenerates what is missing.

Usage (one process per GPU; the orchestrator sets CUDA_VISIBLE_DEVICES):
    CUDA_VISIBLE_DEVICES=0 python signal_probe_worker.py \
        --data data/train_grpo_pilot2.jsonl --phase P2 --rollouts 4 \
        --shard-index 0 --shard-count 4 --out .../shard_p2_0.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

BUNDLE = Path(__file__).resolve().parent
FACTORY = BUNDLE.parent
EXPERIMENTS = FACTORY.parent
V3 = EXPERIMENTS / "nestful_synthetic_curriculum_v3"
MINIMAL = EXPERIMENTS / "nestful_mtgrpo_minimal"
PARTIAL = EXPERIMENTS / "nestful_mtgrpo_partial"
ADAPTER = FACTORY / "trainer_adapter"

WORKER_VERSION = "signal-probe-worker-1.0.0"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REWARD_ARM = "A4_GATED_VERIFIABLE"
REWARD_POLICY = f"reward_ablation_{REWARD_ARM}"

sys.path.insert(0, str(BUNDLE))
from signal_probe_lib import (  # noqa: E402
    SCHEMA_VERSION, content_hash, derive_rollout_metrics, extract_task_meta,
    rollout_cache_key, values_equal,
)


# ───────────────────────────────────────────────────────────────── helpers ──

def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_cached_rollouts(path: Path) -> List[Dict[str, Any]]:
    """Read a shard tolerantly for ``--resume``.

    A worker killed mid-write leaves a truncated final line. Dropping it (and
    rewriting the file from the intact records) is what makes resume safe:
    appending after a partial line would corrupt every later record too.
    """
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[probe] resume: dropping incomplete record at "
                      f"{path.name}:{line_no}", flush=True)
    return rows


def turn_returns(r_seq: Sequence[float], episode_reward: float,
                 gamma: float, lambda_episode: float) -> List[float]:
    """G_t = sum_{k>=t} gamma^(k-t) r_k + lambda * gamma^(T-t+1) * R_episode.

    Mirrors ``grpo_train._turn_returns``; parity is pinned by
    ``test_signal_probe.py::test_turn_returns_matches_trainer``.
    """
    n = len(r_seq)
    T = n - 1
    out: List[float] = []
    for t in range(n):
        disc = 0.0
        for k in range(t, n):
            disc += (gamma ** (k - t)) * float(r_seq[k])
        disc += lambda_episode * (gamma ** (T - t + 1)) * float(episode_reward)
        out.append(disc)
    return out


def _jsonable(value: Any) -> Any:
    """Make executor observations JSON-safe without hiding their content."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


# ─────────────────────────────────────────────────────────── probe plumbing ──

def build_config(args) -> Dict[str, Any]:
    """Trainer config with D1's rollout settings, forced to BF16 inference."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("mtgrpo_base_run", str(MINIMAL / "run.py"))
    base_run = importlib.util.module_from_spec(spec)
    sys.modules["mtgrpo_base_run"] = base_run
    spec.loader.exec_module(base_run)

    config = base_run.load_config(str(args.config))
    overrides = [
        # ── identical to the planned D1 rollouts ──────────────────────────
        f"reward.train_policy={REWARD_POLICY}",
        "executor.mode=synthetic",
        f"generation.temperature={args.temperature}",
        f"generation.top_p={args.top_p}",
        "data.train_stage=null",
        "data.mixed_replay=false",
        # ── probe-specific: exact BF16 base checkpoint, no quantisation,
        #    no adapter. D1 trains a QLoRA adapter; the probe must measure the
        #    UNMODIFIED starting policy, so 4-bit and LoRA are both off.
        f"model.base_model={args.model}",
        "model.lora_adapter=null",
        "hardware.bf16=true",
        "hardware.load_in_4bit=false",
        "finetuning.load_in_4bit=false",
        "finetuning.method=lora",
        f"experiment.seed={args.seed}",
        f"experiment.output_dir={args.out.parent}",
        f"hardware.use_vllm={'true' if args.backend == 'vllm' else 'false'}",
        "hardware.rollout_data_parallel_gpus=null",
    ]
    overrides.extend(args.override or [])
    base_run._apply_overrides(config, overrides)
    base_run._normalize_config_paths(config)
    config.setdefault("experiment", {})["output_dir"] = str(args.out.parent)
    return base_run, config, overrides


def gold_replay(task: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the gold trace through the real executor.

    Gold arguments contain ``$varN.field$`` references; resolving them makes the
    per-call comparison label-style agnostic, so a correct call is scored as
    correct whether the model wrote ``$var1`` or ``$var_1``.
    """
    from executor import ToolExecutor

    ex = ToolExecutor(task, mode="synthetic")
    calls: List[Dict[str, Any]] = []
    observations: List[Any] = []
    error: Optional[str] = None
    for i, call in enumerate(task.get("gold_calls") or []):
        res = ex.execute(call)
        calls.append({"name": res.name, "arguments": _jsonable(res.arguments_resolved),
                      "error": res.error})
        observations.append(_jsonable(res.observation))
        if res.error is not None:
            error = f"call {i + 1} ({call.get('name')}): {res.error}"
            break
    return {"calls": calls, "observations": observations, "error": error}


def replay_predicted(task: Dict[str, Any],
                     parsed_calls: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Recover RESOLVED arguments for the model's calls.

    ``run_episode`` keeps only the raw parsed call and the observation, but the
    probe has to report what was actually executed. The synthetic executor is
    pure and deterministic, so replaying the same calls in the same order
    reproduces the episode exactly; ``build_record`` cross-checks the replayed
    observations against the episode's own.
    """
    from executor import ToolExecutor

    ex = ToolExecutor(task, mode="synthetic")
    out: List[Dict[str, Any]] = []
    for call in parsed_calls:
        res = ex.execute(call)
        out.append({
            "name": res.name,
            "label": res.label,
            "arguments_raw": _jsonable(call.get("arguments")),
            "arguments_resolved": _jsonable(res.arguments_resolved),
            "observation": _jsonable(res.observation),
            "error": res.error,
        })
    return out


def build_record(*, task: Dict[str, Any], meta: Dict[str, Any], traj,
                 reward: Dict[str, Any], rollout_idx: int, phase: str,
                 cache_key: str, gold: Dict[str, Any],
                 gamma: float, lambda_episode: float) -> Dict[str, Any]:
    """One fully-populated rollout record (spec §2)."""
    turns = list(getattr(traj, "turns", []) or [])
    turn_texts = [str(getattr(t, "model_text", "") or "") for t in turns]
    raw_completion = "\n".join(turn_texts)
    parsed_calls = [t.parsed_call for t in turns if getattr(t, "parsed_call", None)]
    replayed = replay_predicted(task, parsed_calls)

    observations = [_jsonable(getattr(t, "observation", None)) for t in turns
                    if getattr(t, "parsed_call", None) is not None
                    and getattr(t, "fail_reason", None) is None]
    replay_obs = [c["observation"] for c in replayed if c.get("error") is None]
    replay_ok = len(observations) == len(replay_obs) and all(
        values_equal(a, b) for a, b in zip(observations, replay_obs))

    diagnostics = reward.get("diagnostics") or {}
    r_seq = [float(x) for x in (reward.get("r_seq") or [])]
    episode_reward = float(reward.get("episode_reward") or 0.0)
    returns = turn_returns(r_seq, episode_reward, gamma, lambda_episode)

    rec: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cache_key": cache_key,
        "phase": phase,
        "rollout_idx": rollout_idx,
        # ── task identity / structure ─────────────────────────────────────
        "task_id": meta["task_id"],
        "track": meta["track"],
        "call_count": meta["call_count"],
        "motif": meta["motif"],
        "answer_type": meta["answer_type"],
        "generation_cell": meta["generation_cell"],
        "target_skill": meta["target_skill"],
        "target_failure_mode": meta["target_failure_mode"],
        # ── what the model produced ───────────────────────────────────────
        "raw_completion": raw_completion,
        "raw_completion_turns": turn_texts,
        "completion_hash": content_hash(turn_texts),
        "parsed_calls": [_jsonable(c) for c in parsed_calls],
        "resolved_calls": replayed,
        "observations": observations,
        "n_pred_calls": len(parsed_calls),
        "n_successful_calls": sum(1 for t in turns
                                  if getattr(t, "parsed_call", None) is not None
                                  and getattr(t, "fail_reason", None) is None),
        "executable_frac": round(
            (sum(1 for c in replayed if c.get("error") is None) / len(replayed))
            if replayed else 0.0, 6),
        # ── how it ended ──────────────────────────────────────────────────
        "stop_reason": getattr(traj, "stop_reason", None),
        "n_turns": len(turns),
        "parse_error": any(str(getattr(t, "fail_reason", "") or "").startswith("parse:")
                           for t in turns),
        "clipped": bool(getattr(traj, "clipped_any", False)),
        "prompt_overflow": bool(getattr(traj, "prompt_overflow", False)),
        "turn_fail_reasons": [getattr(t, "fail_reason", None) for t in turns],
        "terminal_outcome": diagnostics.get("terminal_class"),
        "success": diagnostics.get("terminal_class") == "official_success",
        # ── reward ────────────────────────────────────────────────────────
        "reward_policy": diagnostics.get("reward_policy"),
        "process_reward": diagnostics.get("process_score"),
        "reward_terminal_score": diagnostics.get("terminal_score"),
        "reward_epsilon": diagnostics.get("epsilon"),
        "episode_reward": episode_reward,
        "r_seq": r_seq,
        "returns": [round(v, 6) for v in returns],
        "return_t0": round(returns[0], 6) if returns else round(episode_reward, 6),
        "reward_components": _jsonable(diagnostics.get("components")),
        # ── integrity ─────────────────────────────────────────────────────
        "gold_replay_observations_match": replay_ok,
        "executor_mode": getattr(traj, "executor_mode", None),
    }
    return derive_rollout_metrics(rec, gold["calls"])


# ──────────────────────────────────────────────────────────────────── main ──

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phase", choices=["P2", "P3"], required=True)
    ap.add_argument("--rollouts", type=int, required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--task-ids", type=Path, default=None,
                    help="restrict to these task ids (one per line); P3 uses this")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--config", type=Path, default=PARTIAL / "config.yaml")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--override", action="append", metavar="KEY=VALUE")
    ap.add_argument("--resume", action="store_true",
                    help="skip rollouts already present in --out (content-hash cache)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve everything, print the plan, generate nothing")
    args = ap.parse_args()

    # The trainer resolves its executable tool registry from this variable; the
    # probe MUST run on the factory adapter, never the legacy Stage-3 registry.
    os.environ.setdefault("SYNTHETIC_TOOLS_DIR", str(ADAPTER))
    os.environ.setdefault("TRAIN_STAGE", "3")
    for p in (str(MINIMAL), str(V3), str(FACTORY / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)

    rows = read_rows(args.data)
    dataset_sha = sha256_file(args.data)

    keep: Optional[set] = None
    if args.task_ids:
        keep = {ln.strip() for ln in args.task_ids.read_text(encoding="utf-8").splitlines()
                if ln.strip()}
        rows = [r for r in rows if str(r.get("sample_id") or r.get("task_id")) in keep]

    # Deterministic, contiguous sharding over the FROZEN file order so every
    # worker covers a disjoint slice and a re-run reproduces the same split.
    shard_rows = [r for i, r in enumerate(rows)
                  if i % max(1, args.shard_count) == args.shard_index]

    from synthetic_tool_registry import load_synthetic_tools_module
    registry_mod = load_synthetic_tools_module(os.environ["SYNTHETIC_TOOLS_DIR"])
    registry_hash = registry_mod.registry_hash()

    probe_signature = {
        "worker_version": WORKER_VERSION,
        "model": args.model,
        "dtype": "bfloat16",
        "lora_adapter": None,
        "reward_arm": REWARD_ARM,
        "executor_mode": "synthetic",
        "registry_hash": registry_hash,
        "registry_version": getattr(registry_mod, "REGISTRY_VERSION", None),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "rollouts": args.rollouts,
        "seed": args.seed,
        "dataset_sha256": dataset_sha,
    }

    print(f"[probe:{args.phase}:{args.shard_index}] model={args.model} dtype=bfloat16 "
          f"backend={args.backend} rollouts={args.rollouts}", flush=True)
    print(f"[probe:{args.phase}:{args.shard_index}] registry={os.environ['SYNTHETIC_TOOLS_DIR']} "
          f"hash={registry_hash[:16]}… n_tools={len(registry_mod.TOOLS)}", flush=True)
    print(f"[probe:{args.phase}:{args.shard_index}] tasks={len(shard_rows)} "
          f"of {len(rows)} (shard {args.shard_index}/{args.shard_count})", flush=True)
    print(f"[probe:{args.phase}:{args.shard_index}] out={args.out}", flush=True)

    done: set = set()
    cached: List[Dict[str, Any]] = []
    if args.resume and args.out.is_file():
        for rec in read_cached_rollouts(args.out):
            key = str(rec.get("cache_key") or "")
            if key and key not in done:
                done.add(key)
                cached.append(rec)
        print(f"[probe:{args.phase}:{args.shard_index}] resume: "
              f"{len(done)} cached rollouts", flush=True)

    planned = len(shard_rows) * args.rollouts
    if args.dry_run:
        print(f"[probe:{args.phase}:{args.shard_index}] DRY RUN — would generate "
              f"{planned - len(done)} of {planned} rollouts; nothing executed.")
        return 0

    base_run, config, overrides = build_config(args)

    from data import normalize_task  # noqa: E402
    from vllm_dp_pool import resolve_reward_info  # noqa: E402
    from rollout import run_episode  # noqa: E402
    from reward import compute_gold_observations  # noqa: E402

    reward_fn, reward_info = resolve_reward_info(config)
    if reward_info.get("fallback_used"):
        print("[probe] ABORT: reward dispatch fell back to strict", file=sys.stderr)
        return 4
    print(f"[probe] reward: configured={reward_info['configured_policy']} "
          f"resolved={reward_info['resolved_policy']}", flush=True)

    mt = config.get("mt_grpo", {}) or {}
    gamma = float(mt.get("gamma", 1.0))
    lambda_episode = float(mt.get("lambda_episode", 1.0))
    executor_mode = str((config.get("executor") or {}).get("mode") or "synthetic")
    if executor_mode != "synthetic":
        print(f"[probe] ABORT: executor.mode={executor_mode!r}, expected "
              "'synthetic' (pilot2 tasks run on the factory adapter)",
              file=sys.stderr)
        return 6

    registry = base_run.build_registry(config)
    try:
        import torch
        torch.manual_seed(args.seed + args.shard_index)
    except ImportError:
        pass

    model, tokenizer, generate_fn = base_run._load_inference_backend(
        config, None, mode="eval")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "worker_version": WORKER_VERSION,
        "phase": args.phase,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "dataset": str(args.data),
        "dataset_sha256": dataset_sha,
        "n_tasks": len(shard_rows),
        "rollouts_per_task": args.rollouts,
        "probe_signature": probe_signature,
        "reward": reward_info,
        "executor_mode": (config.get("executor") or {}).get("mode"),
        "overrides": overrides,
        "mt_grpo": {"gamma": gamma, "lambda_episode": lambda_episode},
        "backend": args.backend,
        "training_performed": False,
        "optimizer_steps": 0,
        "adapter_written": None,
    }
    (args.out.parent / f"manifest_{args.phase.lower()}_{args.shard_index}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    n_written = 0
    n_skipped = 0
    # Always rewrite from scratch, replaying the intact cached records first, so
    # the shard can never end up with a truncated line in the middle.
    with args.out.open("w", encoding="utf-8") as sink:
        for rec in cached:
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
        sink.flush()

        for t_i, row in enumerate(shard_rows):
            meta = extract_task_meta(row)
            task = normalize_task(row, t_i)
            row_hash = content_hash(row)
            gold = gold_replay(task)
            if gold["error"]:
                print(f"[probe] ABORT: gold trace does not replay for "
                      f"{meta['task_id']}: {gold['error']}", file=sys.stderr)
                return 5
            # Gold observations must come from the SAME executor backend the
            # rollouts use; the default mode="auto" would resolve differently.
            gold_obs = compute_gold_observations(task, registry, mode=executor_mode)

            for g_i in range(args.rollouts):
                key = rollout_cache_key(row_hash=row_hash, task_id=meta["task_id"],
                                        rollout_idx=g_i, phase=args.phase,
                                        probe_signature=probe_signature)
                if key in done:
                    n_skipped += 1
                    continue
                traj = run_episode(model, tokenizer, task, config,
                                   registry=registry, mode="train",
                                   generate_fn=generate_fn)
                reward = reward_fn(traj, task, gold_obs)
                rec = build_record(task=task, meta=meta, traj=traj, reward=reward,
                                   rollout_idx=g_i, phase=args.phase, cache_key=key,
                                   gold=gold, gamma=gamma,
                                   lambda_episode=lambda_episode)
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sink.flush()
                done.add(key)
                n_written += 1

            if (t_i + 1) % 5 == 0:
                print(f"[probe:{args.phase}:{args.shard_index}] "
                      f"{t_i + 1}/{len(shard_rows)} tasks "
                      f"({n_written} rollouts written, {n_skipped} cached)", flush=True)

    print(f"[probe:{args.phase}:{args.shard_index}] DONE {n_written} written, "
          f"{n_skipped} cached -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
