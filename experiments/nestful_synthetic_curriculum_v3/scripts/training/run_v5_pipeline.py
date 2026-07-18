#!/usr/bin/env python3
"""Curriculum v5 training pipeline — ONE explicit path, no silent overrides.

Per run:
  * exactly one dataset (path + sha256 recorded in the run manifest; verified
    before every epoch);
  * exactly one executor mode (default: synthetic — REAL execution of the
    versioned tool registry; gold_replay must be requested explicitly and is
    marked LEGACY);
  * exactly one reward policy;
  * registry-hash consistency: the dataset's recorded registry_hash must match
    the registry the trainer will execute (abort on mismatch);
  * after every epoch: checkpoint + train summary + DETERMINISTIC dev
    evaluation (temperature=0.0, top_p=1.0, 1 rollout, fixed NESTFUL dev set,
    full executor, official scorer, ReAct);
  * best checkpoint = max (official ReAct win rate, F1 param, full-sequence
    accuracy) — lexicographic;
  * optional early stopping with a minimum epoch count and configurable
    patience, driven ONLY by the deterministic dev win rate;
  * resume prints and records the exact source checkpoint.

State lives in <run_dir>/pipeline_state.json; the manifest in
<run_dir>/run_manifest.json. Training/eval subprocesses are the existing
run.py entry points — this orchestrator only composes explicit --override
lists (all printed) and never mutates config files.

GPU/vLLM knobs come from environment variables (printed at startup):
  USE_VLLM=1                enable vLLM (default 0: HF generate)
  ROLLOUT_DP_GPUS=1,2,3     data-parallel rollout workers (training)
  EVAL_TP=4                 vLLM tensor-parallel size for eval
  VLLM_GPU_UTIL=0.85        gpu_memory_utilization
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3 = os.path.normpath(os.path.join(_HERE, "..", ".."))
_MINIMAL = os.path.normpath(os.path.join(_V3, "..", "nestful_mtgrpo_minimal"))
_PARTIAL = os.path.normpath(os.path.join(_V3, "..", "nestful_mtgrpo_partial"))
_RUN_PY = os.path.join(_V3, "run.py")
_DEFAULT_CONFIG = os.path.join(_PARTIAL, "config.yaml")
_DEFAULT_DEV = os.path.join(_MINIMAL, "data", "splits", "nestful_dev.jsonl")

sys.path.insert(0, _V3)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state() -> dict:
    def g(*a):
        try:
            r = subprocess.run(["git", *a], capture_output=True, text=True,
                               timeout=20, cwd=_V3)
            return r.stdout.strip() if r.returncode == 0 else None
        except OSError:
            return None
    return {"commit": g("rev-parse", "HEAD"),
            "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(g("status", "--porcelain"))}


def _dataset_registry_hash(path: str):
    """registry_hash recorded in the dataset rows (None for non-v5 data)."""
    hashes = set()
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rh = row.get("registry_hash")
            if rh is None:
                prov = row.get("provenance") or {}
                rh = prov.get("registry_hash")
            hashes.add(rh)
            if i >= 200:  # sampling 200 rows is enough to detect a mix
                break
    hashes.discard(None)
    if len(hashes) > 1:
        raise SystemExit(f"[v5] ABORT: dataset {path} mixes registry hashes: "
                         f"{sorted(h[:16] for h in hashes)}")
    return next(iter(hashes), None)


def _current_registry_hash():
    from lib.synthetic_tools import registry_hash, REGISTRY_VERSION
    return registry_hash(), REGISTRY_VERSION


def _run_logged(cmd, log_path, env=None) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(f"[v5] exec: {' '.join(cmd)}")
    print(f"[v5] log:  {log_path}")
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {datetime.now(timezone.utc).isoformat()} =====\n")
        lf.write(" ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def _vllm_train_overrides() -> list:
    out = []
    if os.environ.get("USE_VLLM", "0") == "1":
        out += ["--override", "hardware.use_vllm=true"]
        dp = os.environ.get("ROLLOUT_DP_GPUS", "").strip()
        if dp:
            out += ["--override", f"hardware.rollout_data_parallel_gpus={dp}"]
        util = os.environ.get("VLLM_GPU_UTIL", "").strip()
        if util:
            out += ["--override", f"hardware.vllm_gpu_memory_utilization={util}"]
    return out


def _vllm_eval_overrides() -> list:
    out = []
    if os.environ.get("USE_VLLM", "0") == "1":
        out += ["--override", "hardware.use_vllm=true"]
        tp = os.environ.get("EVAL_TP", "").strip()
        if tp:
            out += ["--override", f"hardware.vllm_tensor_parallel_size={tp}"]
        util = os.environ.get("VLLM_GPU_UTIL", "").strip()
        if util:
            out += ["--override", f"hardware.vllm_gpu_memory_utilization={util}"]
    return out


def _exec_rate_from_trajectories(out_dir: str):
    path = os.path.join(out_dir, "final_eval_trajectories.jsonl")
    if not os.path.isfile(path):
        return None
    n = ok = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ex = (row.get("_traj") or {}).get("executable")
            if ex is None:
                continue
            n += 1
            ok += bool(ex)
    return (ok / n) if n else None


def deterministic_dev_eval(config: str, checkpoint, dev_set: str, out_dir: str,
                           max_tasks, log_path: str) -> dict:
    """Deterministic official eval (react, temp 0). Returns metrics dict."""
    cmd = [sys.executable, _RUN_PY, "--mode", "final_eval",
           "--config", config,
           "--override", f"experiment.output_dir={out_dir}",
           "--override", f"paths.full_nestful_jsonl={dev_set}",
           "--override", "data.eval_paradigm=react",
           "--override", "data.num_eval_rollouts=1",
           "--override", "generation.temperature=0.0",
           "--override", "generation.top_p=1.0",
           ] + _vllm_eval_overrides()
    if max_tasks:
        cmd += ["--override", f"data.max_eval_tasks={int(max_tasks)}"]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    else:
        cmd += ["--override", "model.lora_adapter=null"]
    rc = _run_logged(cmd, log_path)
    if rc != 0:
        raise SystemExit(f"[v5] ABORT: dev eval failed (rc={rc}); see {log_path}")
    mpath = os.path.join(out_dir, "metrics_official.json")
    if not os.path.isfile(mpath):
        raise SystemExit(f"[v5] ABORT: dev eval produced no metrics_official.json "
                         f"in {out_dir}")
    with open(mpath, encoding="utf-8") as fh:
        m = json.load(fh)
    m["executable_rate"] = _exec_rate_from_trajectories(out_dir)
    return m


def _selection_key(m: dict):
    """Lexicographic: ReAct win -> F1 param -> full-sequence accuracy."""
    return (float(m.get("win_rate") or 0.0),
            float(m.get("f1_param") or 0.0),
            float(m.get("full_sequence_accuracy") or 0.0))


def _load_state(run_dir: str) -> dict:
    p = os.path.join(run_dir, "pipeline_state.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {"epochs": []}


def _save_state(run_dir: str, state: dict) -> None:
    p = os.path.join(run_dir, "pipeline_state.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="training JSONL (ONE file)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--min-epochs", type=int, default=2,
                    help="early stopping never fires before this many epochs")
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop patience in epochs; 0 disables early stopping")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--reward-policy", default="execution_aware_v3_2_dense")
    ap.add_argument("--executor-mode", default="synthetic",
                    choices=["synthetic", "gold_replay", "full"],
                    help="gold_replay is LEGACY (cannot falsify wrong values)")
    ap.add_argument("--dev-set", default=_DEFAULT_DEV)
    ap.add_argument("--dev-max-tasks", type=int, default=0,
                    help="cap dev tasks (0 = full dev set)")
    ap.add_argument("--config", default=_DEFAULT_CONFIG)
    ap.add_argument("--checkpoint-in", default=None,
                    help="adapter to initialise epoch 1 from (fresh run only)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from pipeline_state.json in --run-dir")
    ap.add_argument("--max-train-tasks", type=int, default=0,
                    help="cap train tasks per epoch (0 = all; use for smoke)")
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--kl-beta", type=float, default=None)
    ap.add_argument("--allow-registry-mismatch", action="store_true")
    ap.add_argument("--skip-baseline-dev", action="store_true",
                    help="skip the epoch-0 baseline dev eval")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved configuration and exit")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    dataset = os.path.abspath(args.dataset)
    for p, what in [(dataset, "dataset"), (args.config, "config"),
                    (args.dev_set, "dev set")]:
        if not os.path.isfile(p):
            print(f"[v5] ABORT: {what} not found: {p}", file=sys.stderr)
            return 2
    if args.resume and args.checkpoint_in:
        print("[v5] ABORT: --resume and --checkpoint-in are mutually exclusive "
              "(resume reads the source checkpoint from pipeline_state.json)",
              file=sys.stderr)
        return 2

    dataset_sha = _sha256_file(dataset)
    ds_reg_hash = _dataset_registry_hash(dataset)
    cur_reg_hash, cur_reg_version = _current_registry_hash()
    if args.executor_mode == "synthetic":
        if ds_reg_hash is None:
            print("[v5] ABORT: executor-mode=synthetic but the dataset rows carry "
                  "no registry_hash — this is not a v5 registry dataset. "
                  "Regenerate with build_v5_dataset.py or pick a different "
                  "executor mode explicitly.", file=sys.stderr)
            return 2
        if ds_reg_hash != cur_reg_hash and not args.allow_registry_mismatch:
            print(f"[v5] ABORT: dataset registry_hash {ds_reg_hash[:16]}… != "
                  f"current registry {cur_reg_hash[:16]}… (v{cur_reg_version}). "
                  "The trainer would execute DIFFERENT tool implementations than "
                  "the generator. Regenerate the dataset, or pass "
                  "--allow-registry-mismatch to proceed anyway.", file=sys.stderr)
            return 2
    if args.executor_mode == "gold_replay":
        print("[v5] WARNING: executor-mode=gold_replay is LEGACY — it cannot "
              "falsify wrong argument values. Use only as an ablation baseline.")

    resolved = {
        "run_dir": run_dir,
        "dataset": dataset,
        "dataset_sha256": dataset_sha,
        "dataset_registry_hash": ds_reg_hash,
        "registry_hash_current": cur_reg_hash,
        "registry_version_current": cur_reg_version,
        "executor_mode": args.executor_mode,
        "reward_policy": args.reward_policy,
        "epochs": args.epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "num_generations": args.num_generations,
        "dev_set": os.path.abspath(args.dev_set),
        "dev_max_tasks": args.dev_max_tasks or None,
        "dev_decoding": {"temperature": 0.0, "top_p": 1.0, "num_rollouts": 1,
                         "paradigm": "react"},
        "selection": ["win_rate", "f1_param", "full_sequence_accuracy"],
        "config": os.path.abspath(args.config),
        "checkpoint_in": args.checkpoint_in,
        "resume": args.resume,
        "max_train_tasks": args.max_train_tasks or None,
        "learning_rate": args.learning_rate,
        "kl_beta": args.kl_beta,
        "env": {k: os.environ.get(k) for k in
                ("USE_VLLM", "ROLLOUT_DP_GPUS", "EVAL_TP", "VLLM_GPU_UTIL",
                 "CUDA_VISIBLE_DEVICES", "WANDB_PROJECT")},
        "git": _git_state(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    print("[v5] --- resolved configuration ---")
    print(json.dumps(resolved, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("[v5] dry run — nothing executed")
        return 0

    os.makedirs(run_dir, exist_ok=True)
    state = _load_state(run_dir)
    manifest_path = os.path.join(run_dir, "run_manifest.json")

    if args.resume:
        if not state["epochs"]:
            print("[v5] ABORT: --resume but no completed epochs in "
                  f"{run_dir}/pipeline_state.json", file=sys.stderr)
            return 2
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as fh:
                prev = json.load(fh)
            for key in ("dataset_sha256", "executor_mode", "reward_policy"):
                if prev.get(key) != resolved[key]:
                    print(f"[v5] ABORT: resume changes {key!r}: manifest has "
                          f"{prev.get(key)!r}, this invocation has "
                          f"{resolved[key]!r}. One run = one configuration.",
                          file=sys.stderr)
                    return 2
        start_epoch = state["epochs"][-1]["epoch"] + 1
        checkpoint_in = state["epochs"][-1]["adapter"]
        print(f"[v5] RESUME: epoch {start_epoch}, source checkpoint = "
              f"{checkpoint_in}")
    else:
        if state["epochs"]:
            print(f"[v5] ABORT: {run_dir} already has completed epochs; pass "
                  "--resume or choose a fresh --run-dir", file=sys.stderr)
            return 2
        start_epoch = 1
        checkpoint_in = args.checkpoint_in
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(resolved, fh, indent=2, ensure_ascii=False)
        print(f"[v5] manifest -> {manifest_path}")

    # ── Baseline dev eval (epoch 0) — the selection reference ────────────────
    if not args.skip_baseline_dev and not any(
            e["epoch"] == 0 for e in state["epochs"]):
        print("[v5] baseline (no-adapter) deterministic dev eval …")
        m = deterministic_dev_eval(
            args.config, None, args.dev_set,
            os.path.join(run_dir, "dev_eval", "epoch_0"),
            args.dev_max_tasks or None,
            os.path.join(run_dir, "logs", "dev_eval_epoch_0.log"))
        state["epochs"].append({"epoch": 0, "adapter": None, "dev_metrics": m})
        state["epochs"].sort(key=lambda e: e["epoch"])
        _save_state(run_dir, state)
        print(f"[v5] baseline dev: win={m.get('win_rate')} "
              f"f1_param={m.get('f1_param')}")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    no_improve = 0
    trained_epochs = [e for e in state["epochs"] if e["epoch"] > 0]
    best_key = max((_selection_key(e["dev_metrics"]) for e in trained_epochs),
                   default=None)

    for epoch in range(start_epoch, args.epochs + 1):
        if _sha256_file(dataset) != dataset_sha:
            print(f"[v5] ABORT: dataset changed on disk mid-run: {dataset}",
                  file=sys.stderr)
            return 2

        train_out = os.path.join(run_dir, "train", f"epoch_{epoch}")
        ckpt_dir = os.path.join(train_out, "checkpoints")
        os.makedirs(train_out, exist_ok=True)
        print(f"[v5] -- epoch {epoch}/{args.epochs} -- init: "
              f"{checkpoint_in or 'BASE MODEL (fresh LoRA)'}")

        cmd = [sys.executable, _RUN_PY, "--mode", "train",
               "--config", args.config,
               "--override", f"experiment.output_dir={train_out}",
               "--override", f"model.output_adapter_dir={ckpt_dir}",
               "--override", f"paths.train_jsonl={dataset}",
               "--override", "data.train_stage=null",
               "--override", "data.mixed_replay=false",
               "--override", f"executor.mode={args.executor_mode}",
               "--override", f"reward.train_policy={args.reward_policy}",
               "--override", "training.epochs=1",
               "--override", f"generation.num_generations={args.num_generations}",
               ] + _vllm_train_overrides()
        if args.max_train_tasks:
            cmd += ["--override", f"data.max_train_tasks={args.max_train_tasks}"]
        if args.learning_rate is not None:
            cmd += ["--override", f"training.learning_rate={args.learning_rate}"]
        if args.kl_beta is not None:
            cmd += ["--override", f"training.kl_beta={args.kl_beta}"]
        if checkpoint_in:
            cmd += ["--checkpoint", checkpoint_in]
        env = dict(os.environ)
        env["REWARD_POLICY"] = args.reward_policy

        rc = _run_logged(cmd, os.path.join(run_dir, "logs",
                                           f"train_epoch_{epoch}.log"), env=env)
        if rc != 0:
            print(f"[v5] ABORT: training failed at epoch {epoch} (rc={rc})",
                  file=sys.stderr)
            return rc

        adapter = os.path.join(ckpt_dir, "adapter_epoch_1")
        if not os.path.isfile(os.path.join(adapter, "adapter_config.json")):
            print(f"[v5] ABORT: no adapter saved at {adapter}", file=sys.stderr)
            return 1

        m = deterministic_dev_eval(
            args.config, adapter, args.dev_set,
            os.path.join(run_dir, "dev_eval", f"epoch_{epoch}"),
            args.dev_max_tasks or None,
            os.path.join(run_dir, "logs", f"dev_eval_epoch_{epoch}.log"))
        print(f"[v5] epoch {epoch} dev: win={m.get('win_rate')} "
              f"f1_param={m.get('f1_param')} "
              f"full_seq={m.get('full_sequence_accuracy')}")

        state["epochs"].append({
            "epoch": epoch, "adapter": adapter, "init_checkpoint": checkpoint_in,
            "dev_metrics": m,
            "train_summary": os.path.join(train_out, "train_summary.json"),
        })
        _save_state(run_dir, state)

        key = _selection_key(m)
        if best_key is None or key > best_key:
            best_key = key
            no_improve = 0
        else:
            no_improve += 1
        checkpoint_in = adapter

        if (args.patience > 0 and epoch >= args.min_epochs
                and no_improve >= args.patience):
            print(f"[v5] early stop: no dev-win improvement for "
                  f"{no_improve} epoch(s) (patience={args.patience})")
            break

    # ── Checkpoint selection ─────────────────────────────────────────────────
    trained = [e for e in state["epochs"] if e["epoch"] > 0]
    if not trained:
        print("[v5] no trained epochs — nothing to select", file=sys.stderr)
        return 1
    ranked = sorted(trained, key=lambda e: _selection_key(e["dev_metrics"]),
                    reverse=True)
    best = ranked[0]
    selection = {
        "criteria": ["win_rate", "f1_param", "full_sequence_accuracy"],
        "decoding": {"temperature": 0.0, "top_p": 1.0, "num_rollouts": 1},
        "dev_set": os.path.abspath(args.dev_set),
        "best_epoch": best["epoch"],
        "best_adapter": best["adapter"],
        "final_epoch": trained[-1]["epoch"],
        "final_adapter": trained[-1]["adapter"],
        "ranking": [{"epoch": e["epoch"],
                     "win_rate": e["dev_metrics"].get("win_rate"),
                     "f1_param": e["dev_metrics"].get("f1_param"),
                     "full_sequence_accuracy":
                         e["dev_metrics"].get("full_sequence_accuracy"),
                     "executable_rate": e["dev_metrics"].get("executable_rate")}
                    for e in ranked],
    }
    best_dir = os.path.join(run_dir, "best_adapter")
    if os.path.isdir(best_dir):
        shutil.rmtree(best_dir)
    shutil.copytree(best["adapter"], best_dir)
    selection["best_adapter_copy"] = best_dir
    with open(os.path.join(run_dir, "checkpoint_selection.json"), "w",
              encoding="utf-8") as fh:
        json.dump(selection, fh, indent=2, ensure_ascii=False)
    print("[v5] --- checkpoint selection ---")
    print(json.dumps(selection, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
