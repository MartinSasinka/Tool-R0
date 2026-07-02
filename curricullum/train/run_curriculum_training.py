#!/usr/bin/env python3
"""Curriculum GRPO Training Orchestrator.

Runs the full 6-stage curriculum loop:
  1. One-time baseline evaluation on all NESTFUL n+1 groups.
  2. For each stage S in [1..6]:
     a. Train 1 epoch via train_grpo_stage.py (subprocess, frees GPU on exit).
     b. Evaluate on NESTFUL tasks with (S+1) calls via evaluate_nestful_stage.py.
     c. Gate: advance if threshold met, plateau detected, or max_epochs reached.
  3. Print summary table and write curriculum_summary.csv.

Usage:
    python curricullum/train/run_curriculum_training.py \\
        --config curricullum/train/configs/qwen3_4b_curriculum_v2.yaml \\
        --wandb_project nestful-curriculum-toolr0 \\
        --run_group qwen3-4b-curriculum-v2

    # Resume from stage 3 (stages 1-2 adapters must exist in checkpoints_root):
    python ... --resume_stage 3

    # Skip baseline eval, use cached file:
    python ... --baseline_cache curricullum/training/results/baseline_nestful.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Make CUDA device ids match `nvidia-smi` ordering. Without this, CUDA defaults to
# FASTEST_FIRST ordering, which ranks the slow DGX Display GPU last — so e.g.
# CUDA_VISIBLE_DEVICES=2,4 (nvidia-smi ids) can silently map index 4 onto the 4GB
# display GPU and OOM a training rank. Set before any CUDA init or subprocess launch
# so the orchestrator, accelerate training, and eval shards all agree on ids.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import wandb as _wandb
except ImportError:
    _wandb = None  # type: ignore

STAGE_KEYS = ["stage_1", "stage_2", "stage_3", "stage_4", "stage_5", "stage_6"]

# ─── Logging ──────────────────────────────────────────────────────────────────

_log_file: Optional[Path] = None


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with _log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ─── Config helpers ───────────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def stage_num(stage_key: str) -> int:
    return int(stage_key.split("_")[1])


# ─── Checkpoint manifest ──────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def best_checkpoint(manifest: Dict[str, Any], stage_key: str) -> Optional[str]:
    entry = manifest.get(stage_key)
    if entry:
        return entry.get("best_path")
    return None


# ─── Subprocess helpers ───────────────────────────────────────────────────────

def _run_subprocess(cmd: List[str], label: str, env: Optional[Dict[str, str]] = None) -> int:
    """Run a subprocess, stream its output, return exit code."""
    _log(f"  $ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=sys.stdout, stderr=sys.stderr, env=env)
    proc.wait()
    if proc.returncode != 0:
        _log(f"  WARNING: {label} exited with code {proc.returncode}")
    return proc.returncode


def _eval_gpu_ids(args: argparse.Namespace) -> List[str]:
    """Physical GPU ids available for eval (data-parallel sharding)."""
    raw = (
        getattr(args, "eval_gpus", "")
        or os.environ.get("CUDA_VISIBLE_DEVICES", "")
        or getattr(args, "train_gpus", "")
        or os.environ.get("TRAIN_GPUS", "")
    )
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    return ids or ["0"]


def _train_gpu_ids(args: argparse.Namespace) -> List[str]:
    """Physical GPU ids to pin training onto via `accelerate launch --gpu_ids`.

    accelerate's DeepSpeed launcher does NOT reliably honor an inherited
    CUDA_VISIBLE_DEVICES (DeepSpeed ignores it once --num_gpus is passed and grabs
    GPUs 0,1 by absolute index). Passing --gpu_ids forces accelerate to set
    CUDA_VISIBLE_DEVICES itself, so the ranks land on the intended physical GPUs.
    Resolution order: --train_gpus arg, then $TRAIN_GPUS, then $CUDA_VISIBLE_DEVICES.
    With CUDA_DEVICE_ORDER=PCI_BUS_ID (set at import) these ids match nvidia-smi.
    """
    raw = (
        getattr(args, "train_gpus", "")
        or os.environ.get("TRAIN_GPUS", "")
        or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    )
    return [x.strip() for x in raw.split(",") if x.strip()]


def _merge_eval_shards(shard_jsons: List[Path], out_json: Path, n_calls: int) -> Dict[str, Any]:
    """Aggregate per-shard metric JSONs into a single combined metrics file.

    Rates are recomputed from summed raw counts so the merge is exact.
    """
    agg = {
        "total_tasks": 0, "exec_pass_count": 0, "parse_fail_count": 0,
        "tool_acc_sum": 0.0, "partial_sum": 0.0, "turns_sum": 0.0,
        "clipped_gens": 0, "total_gens": 0,
    }
    for sj in shard_jsons:
        if not sj.is_file():
            _log(f"  WARNING: missing shard output {sj}")
            continue
        d = json.loads(sj.read_text(encoding="utf-8"))
        for k in agg:
            agg[k] += d.get(k, 0)

    n = agg["total_tasks"]
    tg = agg["total_gens"]
    merged = {
        "n_calls": n_calls,
        "total_tasks": n,
        "exec_pass_rate": agg["exec_pass_count"] / n if n else 0.0,
        "tool_call_acc": agg["tool_acc_sum"] / n if n else 0.0,
        "partial_score": agg["partial_sum"] / n if n else 0.0,
        "parse_fail_rate": agg["parse_fail_count"] / n if n else 0.0,
        "avg_turns_completed": agg["turns_sum"] / n if n else 0.0,
        "clipped_frac": round(agg["clipped_gens"] / tg, 4) if tg else 0.0,
        **agg,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    # Concatenate per-shard failure files
    merged_failures: List[str] = []
    for sj in shard_jsons:
        fp = sj.with_suffix(".failures.jsonl")
        if fp.is_file():
            merged_failures.extend(l for l in fp.read_text(encoding="utf-8").splitlines() if l.strip())
    if merged_failures:
        out_json.with_suffix(".failures.jsonl").write_text(
            "\n".join(merged_failures) + "\n", encoding="utf-8"
        )
    return merged


def _run_eval(base_cmd: List[str], out_json: Path, label: str, eval_gpus: List[str], n_calls: int) -> int:
    """Run an eval command, sharded data-parallel across eval_gpus if >1 GPU.

    base_cmd MUST NOT include --output_json / --shard_id / --num_shards;
    those are added per shard here.
    """
    if len(eval_gpus) <= 1:
        return _run_subprocess(base_cmd + ["--output_json", str(out_json)], label)

    n_shards = len(eval_gpus)
    shard_jsons: List[Path] = []
    procs = []
    for shard_id, gpu in enumerate(eval_gpus):
        shard_json = out_json.with_suffix(f".shard{shard_id}.json")
        shard_jsons.append(shard_json)
        cmd = base_cmd + [
            "--output_json", str(shard_json),
            "--shard_id", str(shard_id),
            "--num_shards", str(n_shards),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        _log(f"  $ [gpu {gpu}] {' '.join(cmd)}")
        procs.append(subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=sys.stdout, stderr=sys.stderr))

    rcs = [p.wait() for p in procs]
    for shard_id, rc in enumerate(rcs):
        if rc != 0:
            _log(f"  WARNING: {label} shard {shard_id} exited with code {rc}")

    _merge_eval_shards(shard_jsons, out_json, n_calls)
    return 0 if all(rc == 0 for rc in rcs) else 1


# ─── Baseline evaluation ──────────────────────────────────────────────────────

def run_baseline_eval(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    results_dir: Path,
) -> Dict[str, Any]:
    """Run baseline (no adapter) on NESTFUL groups 2..7. Returns dict keyed by str(n_calls)."""
    baseline: Dict[str, Any] = {}
    max_stage = max(stage_num(k) for k in STAGE_KEYS)
    smoke = getattr(args, "smoke_test", False)
    eval_gpus = _eval_gpu_ids(args)

    for n_calls in range(2, max_stage + 2):  # groups 2..7 for stages 1..6
        out_json = results_dir / f"baseline_n{n_calls}.json"
        _log(f"baseline eval  n_calls={n_calls} ...")
        cmd = [
            sys.executable, "-u",
            str(Path(SCRIPT_DIR) / "evaluate_nestful_stage.py"),
            "--base_model", cfg["model"]["name"],
            "--no_adapter",
            "--nestful_path", args.nestful_path,
            "--call_dist_path", args.call_dist_path,
            "--n_calls", str(n_calls),
            "--max_new_tokens", str(cfg.get("eval", {}).get("max_new_tokens", 1024)),
            "--batch_size", str(cfg.get("eval", {}).get("batch_size", 8)),
            "--save_failures", str(cfg.get("eval", {}).get("save_failure_examples", 20)),
        ]
        if smoke:
            cmd += ["--max_tasks", "5"]
        _run_eval(cmd, out_json, f"baseline n_calls={n_calls}", eval_gpus, n_calls)
        if out_json.is_file():
            metrics = json.loads(out_json.read_text(encoding="utf-8"))
            baseline[str(n_calls)] = metrics
            _log(
                f"  baseline n_calls={n_calls}:"
                f"  exec_pass={metrics.get('exec_pass_rate', 0):.3f}"
                f"  tasks={metrics.get('total_tasks', 0)}"
            )
        else:
            _log(f"  WARNING: no output for baseline n_calls={n_calls}")
            baseline[str(n_calls)] = {"exec_pass_rate": 0.0, "total_tasks": 0}

    return baseline


# ─── Train one epoch ──────────────────────────────────────────────────────────

def train_epoch(
    stage_key: str,
    epoch: int,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    prev_adapter: Optional[str],
    checkpoint_dir: Path,
    replay_jsonl: Optional[str],
    wandb_run_group: Optional[str],
) -> str:
    """Call train_grpo_stage.py for one epoch. Returns output adapter path."""
    stage_cfg = cfg["stages"][stage_key]
    output_dir = checkpoint_dir / f"{stage_key}_epoch{epoch}"
    run_name = f"{wandb_run_group or 'curriculum'}-{stage_key}-e{epoch}"
    smoke = getattr(args, "smoke_test", False)

    cmd = [
        sys.executable, "-u",
        str(Path(SCRIPT_DIR) / "train_grpo_stage.py"),
        "--config", args.config,
        "--stage", stage_key,
        "--model_name", cfg["model"]["name"],
        "--data_path", stage_cfg["data_path"],
        "--output_dir", str(output_dir),
        "--previous_adapter", prev_adapter or "none",
        "--wandb_project", args.wandb_project,
        "--wandb_run_name", run_name,
        "--num_train_epochs", "1.0",
        "--training_format", "tool_r0",
        "--overwrite",
    ]
    if smoke:
        cmd += ["--max_steps", "3"]
    if wandb_run_group:
        cmd += ["--wandb_run_group", wandb_run_group]
    if replay_jsonl:
        cmd += ["--replay_jsonl", replay_jsonl]
    train_gpus = _train_gpu_ids(args)
    # num_processes follows the explicit GPU list when given, so the two never disagree.
    num_processes = len(train_gpus) if train_gpus else int(args.num_processes or 1)
    train_env: Optional[Dict[str, str]] = None
    if num_processes > 1:
        # accelerate launch manages the python interpreter itself —
        # strip [sys.executable, "-u"] from the front and pass only the script + args.
        script_and_args = cmd[2:]  # drop sys.executable and "-u"
        accel = [
            "accelerate", "launch",
            "--config_file", args.deepspeed_config,
            "--num_processes", str(num_processes),
        ]
        if train_gpus:
            # Force device selection; inherited CUDA_VISIBLE_DEVICES is ignored by the
            # DeepSpeed launcher once --num_gpus is set, which silently lands ranks on
            # GPUs 0,1. --gpu_ids makes accelerate set CUDA_VISIBLE_DEVICES itself.
            accel += ["--gpu_ids", ",".join(train_gpus)]
            # Clear any inherited CUDA_VISIBLE_DEVICES so accelerate sees all physical
            # GPUs and --gpu_ids selects them by absolute id (with PCI_BUS_ID order).
            train_env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        cmd = accel + script_and_args

    rc = _run_subprocess(cmd, f"train {stage_key} epoch {epoch}", env=train_env)
    if rc != 0:
        _log(f"ERROR: training failed for {stage_key} epoch {epoch} (exit {rc})")
        raise RuntimeError(f"train_grpo_stage exited with {rc}")

    return str(output_dir)


# ─── Evaluate checkpoint ──────────────────────────────────────────────────────

def eval_checkpoint(
    stage_key: str,
    epoch: int,
    n_calls: int,
    adapter_path: str,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    results_dir: Path,
) -> Dict[str, Any]:
    """Run evaluate_nestful_stage.py on NESTFUL n_calls group. Returns metrics dict."""
    out_json = results_dir / f"{stage_key}_epoch{epoch}_val.json"
    smoke = getattr(args, "smoke_test", False)
    eval_gpus = _eval_gpu_ids(args)
    cmd = [
        sys.executable, "-u",
        str(Path(SCRIPT_DIR) / "evaluate_nestful_stage.py"),
        "--base_model", cfg["model"]["name"],
        "--adapter_path", adapter_path,
        "--nestful_path", args.nestful_path,
        "--call_dist_path", args.call_dist_path,
        "--n_calls", str(n_calls),
        "--max_new_tokens", str(cfg.get("eval", {}).get("max_new_tokens", 1024)),
        "--batch_size", str(cfg.get("eval", {}).get("batch_size", 8)),
        "--save_failures", str(cfg.get("eval", {}).get("save_failure_examples", 20)),
    ]
    if smoke:
        cmd += ["--max_tasks", "5"]
    else:
        per_epoch_limit = cfg.get("eval", {}).get("max_tasks_per_epoch")
        if per_epoch_limit:
            cmd += ["--max_tasks", str(per_epoch_limit)]
    rc = _run_eval(cmd, out_json, f"eval {stage_key} epoch {epoch}", eval_gpus, n_calls)
    if rc != 0:
        _log(f"WARNING: eval failed for {stage_key} epoch {epoch}")
        return {"exec_pass_rate": 0.0, "total_tasks": 0, "error": f"exit_{rc}"}

    if out_json.is_file():
        return json.loads(out_json.read_text(encoding="utf-8"))
    return {"exec_pass_rate": 0.0, "total_tasks": 0, "error": "no_output"}


# ─── Gating decision ──────────────────────────────────────────────────────────

def gating_decision(
    metrics_history: List[Dict[str, Any]],
    threshold: float,
    plateau_patience: int,
    max_epochs: int,
) -> Tuple[bool, str]:
    """Return (should_advance, reason).

    Advance if:
      - exec_pass_rate >= threshold   → "threshold"
      - no improvement for patience epochs  → "plateau"
      - epoch count >= max_epochs           → "max_epochs"
    """
    epoch = len(metrics_history)
    current = metrics_history[-1].get("exec_pass_rate", 0.0)

    if current >= threshold:
        return True, "threshold"

    if epoch >= max_epochs:
        return True, "max_epochs"

    if len(metrics_history) >= plateau_patience + 1:
        recent = [m.get("exec_pass_rate", 0.0) for m in metrics_history[-(plateau_patience + 1):]]
        best_prev = max(recent[:-1])
        if current <= best_prev:
            return True, "plateau"

    return False, "continue"


# ─── W&B curriculum-level logging ─────────────────────────────────────────────

def _wandb_log_epoch(
    wandb_run,
    stage_key: str,
    epoch: int,
    metrics: Dict[str, Any],
    baseline_val: float,
    threshold: float,
) -> None:
    if wandb_run is None:
        return
    s = stage_num(stage_key)
    _wandb.log(
        {
            f"curriculum/stage{s}_exec_pass": metrics.get("exec_pass_rate", 0.0),
            f"curriculum/stage{s}_tool_acc": metrics.get("tool_call_acc", 0.0),
            f"curriculum/stage{s}_partial": metrics.get("partial_score", 0.0),
            f"curriculum/stage{s}_parse_fail": metrics.get("parse_fail_rate", 0.0),
            f"curriculum/stage{s}_delta_vs_baseline": metrics.get("exec_pass_rate", 0.0) - baseline_val,
            f"curriculum/stage{s}_threshold": threshold,
            "curriculum/current_stage": s,
            "curriculum/current_epoch": epoch,
        }
    )


# ─── Summary table ────────────────────────────────────────────────────────────

_COL_W = [8, 10, 10, 10, 8, 8, 12]
_HEADERS = ["Stage", "Val calls", "Baseline", "Final", "Delta", "Epochs", "Reason"]


def _row_str(vals: List[str]) -> str:
    return "  ".join(f"{v:<{w}}" for v, w in zip(vals, _COL_W))


def print_summary(summary_rows: List[Dict[str, Any]]) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print("  CURRICULUM TRAINING SUMMARY")
    print(sep)
    print(_row_str(_HEADERS))
    print(sep)
    for r in summary_rows:
        warn = " ⚠" if r["reason"] in ("max_epochs", "plateau") else ""
        print(_row_str([
            r["stage"],
            str(r["val_n_calls"]),
            f"{r['baseline']:.1%}",
            f"{r['final']:.1%}",
            f"{r['final'] - r['baseline']:+.1%}",
            f"{r['epochs_used']}/{r['max_epochs']}",
            r["reason"] + warn,
        ]))
    print(sep + "\n")


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "val_n_calls", "baseline", "final", "delta", "epochs_used", "max_epochs", "reason"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "stage": r["stage"],
                "val_n_calls": r["val_n_calls"],
                "baseline": f"{r['baseline']:.4f}",
                "final": f"{r['final']:.4f}",
                "delta": f"{r['final'] - r['baseline']:+.4f}",
                "epochs_used": r["epochs_used"],
                "max_epochs": r["max_epochs"],
                "reason": r["reason"],
            })


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Curriculum GRPO training orchestrator")
    ap.add_argument("--config", default="curricullum/train/configs/qwen3_4b_curriculum_v2.yaml")
    ap.add_argument("--wandb_project", default="nestful-curriculum-toolr0")
    ap.add_argument("--run_group", default=None)
    ap.add_argument("--nestful_path", default="eval/data/NESTFUL-main/data_v2/nestful_data.jsonl")
    ap.add_argument("--call_dist_path", default="helper_calculations/output/nestful_call_distribution.json")
    ap.add_argument("--baseline_cache", default=None, help="Path to cached baseline JSON; skips baseline eval")
    ap.add_argument("--resume_stage", type=int, default=1, help="Start from this stage (1-indexed); prior stage adapters must exist")
    ap.add_argument("--num_processes", default=os.environ.get("NUM_PROCESSES", "2"), help="GPU processes for accelerate launch (overridden by the count of --train_gpus when set)")
    ap.add_argument("--train_gpus", default="", help="Comma list of physical GPU ids to pin training onto via accelerate --gpu_ids, e.g. 2,4 (default: $TRAIN_GPUS or $CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--eval_gpus", default="", help="Comma list of physical GPU ids for data-parallel eval sharding (default: CUDA_VISIBLE_DEVICES)")
    # On 40GB A100 with GRPO (policy + generation KV cache for num_generations
    # rollouts), CPU offload of optimizer state is needed for headroom — without
    # it, generation OOMs on the KV cache. Non-offload (deepseed_zero2.yaml) is
    # faster but only fits if you also cut memory (beta=0, num_generations=2, or
    # shorter prompts).
    ap.add_argument("--deepspeed_config", default="configs/deepseed_zero2_offload.yaml")
    ap.add_argument("--stages", default="", help="Comma list to run subset, e.g. 1,2,3")
    ap.add_argument(
        "--smoke_test", action="store_true",
        help="Dry-run mode: 3 gradient steps/epoch, 5 eval tasks, 1 epoch/stage, no baseline",
    )
    return ap.parse_args()


def main() -> None:
    global _log_file
    args = parse_args()
    cfg = load_config(args.config)

    # Resolve paths relative to repo root
    results_dir = Path(REPO_ROOT) / "curricullum" / "training" / "results"
    logs_dir = Path(REPO_ROOT) / "curricullum" / "training" / "logs"
    checkpoint_root = Path(REPO_ROOT) / cfg.get("checkpoints_root", "curricullum/checkpoints/qwen3_4b_curriculum_v2")
    manifest_path = checkpoint_root / "checkpoints.json"

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file = logs_dir / f"curriculum_run_{run_ts}.log"

    # Stage filter
    if args.stages.strip():
        stage_nums_filter = {int(x.strip()) for x in args.stages.split(",") if x.strip()}
        active_stage_keys = [k for k in STAGE_KEYS if stage_num(k) in stage_nums_filter]
    else:
        active_stage_keys = [k for k in STAGE_KEYS if stage_num(k) >= args.resume_stage]
        # Filter to only stages defined in config
        active_stage_keys = [k for k in active_stage_keys if k in cfg.get("stages", {})]

    curriculum_cfg = cfg.get("curriculum", {})
    threshold_delta = float(curriculum_cfg.get("advance_threshold_delta", 0.12))
    plateau_patience = int(curriculum_cfg.get("plateau_patience", 2))
    replay_fraction = float(curriculum_cfg.get("replay_fraction", 0.15))
    max_epochs_map: Dict[str, int] = {
        k: int(v) for k, v in (curriculum_cfg.get("max_epochs_per_stage") or {}).items()
    }

    smoke = args.smoke_test
    if smoke:
        _log("*** SMOKE TEST MODE: max_steps=3, max_eval_tasks=5, max_epochs=1, no baseline ***")
        # Override epoch limits to 1 per stage in smoke mode
        for k in max_epochs_map:
            max_epochs_map[k] = 1
        for k in STAGE_KEYS:
            if k not in max_epochs_map:
                max_epochs_map[k] = 1

    _log("=" * 60)
    _log(f"Curriculum GRPO Training  stages={[stage_num(k) for k in active_stage_keys]}")
    _log(f"threshold_delta={threshold_delta}  plateau_patience={plateau_patience}  replay={replay_fraction}")
    _log("=" * 60)

    # ── Baseline evaluation ────────────────────────────────────────────────
    if smoke:
        _log("Smoke test: skipping baseline eval, using zeros")
        baseline: Dict[str, Any] = {str(n): {"exec_pass_rate": 0.0, "total_tasks": 0} for n in range(2, 9)}
    elif args.baseline_cache and Path(args.baseline_cache).is_file():
        _log(f"Loading baseline from cache: {args.baseline_cache}")
        baseline = json.loads(Path(args.baseline_cache).read_text(encoding="utf-8"))
    else:
        _log("Running one-time baseline evaluation ...")
        baseline = run_baseline_eval(cfg, args, results_dir)
        baseline_path = results_dir / "baseline_nestful.json"
        baseline_path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
        _log(f"Baseline saved to {baseline_path}")

    # ── W&B curriculum run ─────────────────────────────────────────────────
    wandb_run = None
    if _wandb is not None:
        wandb_run = _wandb.init(
            project=args.wandb_project,
            name=f"{args.run_group or 'curriculum'}-{run_ts}",
            group=args.run_group,
            config={
                "config_path": args.config,
                "model": cfg["model"]["name"],
                "stages": [stage_num(k) for k in active_stage_keys],
                "threshold_delta": threshold_delta,
                "plateau_patience": plateau_patience,
                "replay_fraction": replay_fraction,
                "lora": cfg.get("lora", {}),
                "grpo": cfg.get("grpo", {}),
            },
        )
        # Log baseline metrics
        for n_calls_str, bm in baseline.items():
            _wandb.log({f"baseline/exec_pass_n{n_calls_str}": bm.get("exec_pass_rate", 0.0)})

    # ── Load checkpoint manifest ───────────────────────────────────────────
    manifest = load_manifest(manifest_path)

    # Resolve previous adapter for resume
    prev_adapter: Optional[str] = None
    if args.resume_stage > 1:
        prev_stage_key = f"stage_{args.resume_stage - 1}"
        prev_adapter = best_checkpoint(manifest, prev_stage_key)
        if prev_adapter:
            _log(f"Resuming from stage {args.resume_stage}; previous adapter: {prev_adapter}")
        else:
            _log(f"WARNING: no manifest entry for {prev_stage_key}; starting without adapter")

    summary_rows: List[Dict[str, Any]] = []
    run_aborted = False

    data_dir = Path(REPO_ROOT) / "curricullum" / "data" / "filtered_toolr0_synthetic"

    # ── Stage loop ─────────────────────────────────────────────────────────
    for stage_key in active_stage_keys:
        s = stage_num(stage_key)
        stage_cfg = cfg["stages"][stage_key]
        val_n_calls = s + 1
        max_epochs = max_epochs_map.get(stage_key, 4 + s)

        baseline_val = baseline.get(str(val_n_calls), {}).get("exec_pass_rate", 0.0)
        threshold = baseline_val + threshold_delta

        _log(f"\n{'═'*60}")
        _log(f"STAGE {s}  ({stage_key})  val_n_calls={val_n_calls}  max_epochs={max_epochs}")
        _log(f"  baseline={baseline_val:.3f}  threshold={threshold:.3f}  data={stage_cfg['data_path']}")
        _log(f"{'═'*60}")

        metrics_history: List[Dict[str, Any]] = []
        best_val = 0.0
        best_epoch_adapter: str = ""
        advance_reason = "max_epochs"
        stage_failed = False

        for epoch in range(1, max_epochs + 1):
            epoch_label = f"Stage {s} | Epoch {epoch}/{max_epochs}"
            _log(f"\n[{epoch_label}] training ...")
            t_train = time.time()

            # Sample replay data
            replay_path = None
            if replay_fraction > 0 and s > 1:
                from curricullum.train.replay_buffer import sample_replay
                # Estimate expanded dataset size (each multi-call sample → N turn records)
                dataset_size = sum(1 for _ in open(stage_cfg["data_path"], encoding="utf-8") if _.strip())
                avg_calls = stage_cfg.get("num_calls", s)
                expanded_size = dataset_size * avg_calls
                replay_path = sample_replay(
                    stage=s,
                    dataset_size=expanded_size,
                    replay_fraction=replay_fraction,
                    seed=42 + s * 100 + epoch,
                    data_dir=data_dir,
                )

            # Train
            stage_failed = False
            try:
                adapter_path = train_epoch(
                    stage_key=stage_key,
                    epoch=epoch,
                    cfg=cfg,
                    args=args,
                    prev_adapter=prev_adapter,
                    checkpoint_dir=checkpoint_root,
                    replay_jsonl=str(replay_path) if replay_path else None,
                    wandb_run_group=args.run_group,
                )
            except RuntimeError as e:
                _log(f"  FATAL: {e} — stopping stage {s} and all subsequent stages")
                stage_failed = True
                break
            finally:
                if replay_path and replay_path.is_file():
                    replay_path.unlink(missing_ok=True)

            train_elapsed = time.time() - t_train
            _log(f"[{epoch_label}] train done  elapsed={train_elapsed:.0f}s")

            # Read training summary for reward_mean
            summary_json = Path(adapter_path) / "training_summary.json"
            reward_mean = None
            if summary_json.is_file():
                ts_data = json.loads(summary_json.read_text(encoding="utf-8"))
                reward_mean = ts_data.get("final_train_reward")
            if reward_mean is not None:
                _log(f"[{epoch_label}] reward_mean={reward_mean:.4f}")

            # Evaluate
            _log(f"[{epoch_label}] evaluating n_calls={val_n_calls} ...")
            t_eval = time.time()
            metrics = eval_checkpoint(
                stage_key=stage_key,
                epoch=epoch,
                n_calls=val_n_calls,
                adapter_path=adapter_path,
                cfg=cfg,
                args=args,
                results_dir=results_dir,
            )
            eval_elapsed = time.time() - t_eval

            exec_pass = metrics.get("exec_pass_rate", 0.0)
            _log(
                f"[{epoch_label}] eval"
                f"  n_calls={val_n_calls}"
                f"  tasks={metrics.get('total_tasks', '?')}"
                f"  exec_pass={exec_pass:.3f}"
                f"  tool_acc={metrics.get('tool_call_acc', 0.0):.3f}"
                f"  partial={metrics.get('partial_score', 0.0):.3f}"
                f"  elapsed={eval_elapsed:.0f}s"
            )
            _log(
                f"             baseline={baseline_val:.3f}"
                f"  threshold={threshold:.3f}"
                f"  delta={exec_pass - baseline_val:+.3f}"
            )

            # Update best checkpoint
            if exec_pass >= best_val:
                best_val = exec_pass
                best_epoch_adapter = adapter_path
            metrics_history.append(metrics)

            # W&B
            _wandb_log_epoch(wandb_run, stage_key, epoch, metrics, baseline_val, threshold)

            # Gating
            advance, advance_reason = gating_decision(
                metrics_history, threshold, plateau_patience, max_epochs
            )
            _log(f"             gating → {advance_reason.upper()}")
            if advance:
                break

        # Stop entire run if this stage failed (training subprocess crashed)
        if stage_failed:
            run_aborted = True
            _log(f"  Aborting remaining stages due to fatal failure in stage {s}.")
            break

        # Best checkpoint carries forward
        prev_adapter = best_epoch_adapter if best_epoch_adapter else prev_adapter

        # Update manifest
        final_metrics = metrics_history[-1] if metrics_history else {}
        manifest[stage_key] = {
            "best_path": best_epoch_adapter,
            "best_exec_pass": best_val,
            "epochs_used": len(metrics_history),
            "advance_reason": advance_reason,
            "val_n_calls": val_n_calls,
            "threshold": threshold,
            "baseline_exec_pass": baseline_val,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        save_manifest(manifest_path, manifest)

        summary_rows.append({
            "stage": stage_key,
            "val_n_calls": val_n_calls,
            "baseline": baseline_val,
            "final": best_val,
            "epochs_used": len(metrics_history),
            "max_epochs": max_epochs,
            "reason": advance_reason,
        })

        _log(
            f"\nSTAGE {s} DONE"
            f"  best_exec_pass={best_val:.3f}"
            f"  reason={advance_reason}"
            f"  best_ckpt={best_epoch_adapter}"
        )

    # ── Final eval on all NESTFUL groups ──────────────────────────────────
    # Skip final eval after a fatal stage failure: the previous run still
    # left a crashed (OOM'd) trainer behind, and launching the multi-GPU eval
    # here spawns shard children that survive as orphans if the user Ctrl+C's
    # the orchestrator — squatting ~3 GiB each and OOM-ing the next attempt.
    if run_aborted:
        _log(
            "\nSkipping final evaluation because the run aborted on a fatal stage failure."
            "\n  Before re-running, kill leftover processes on the training GPUs:"
            "\n    ps -u $USER -o pid,etime,cmd | grep -E 'run_curriculum|train_grpo_stage|evaluate_nestful|accelerate' | grep -v grep"
            "\n    kill -9 <pid>   # only your own PIDs"
        )
    elif prev_adapter and active_stage_keys:
        _log("\nRunning final evaluation on all NESTFUL groups ...")
        final_all: Dict[str, Any] = {}
        final_adapter = prev_adapter
        max_s = max(stage_num(k) for k in active_stage_keys)
        smoke = args.smoke_test
        final_limit = cfg.get("eval", {}).get("max_tasks_final")
        eval_gpus = _eval_gpu_ids(args)
        for n_calls in range(2, max_s + 3):
            out_json = results_dir / f"final_n{n_calls}.json"
            cmd = [
                sys.executable, "-u",
                str(Path(SCRIPT_DIR) / "evaluate_nestful_stage.py"),
                "--base_model", cfg["model"]["name"],
                "--adapter_path", final_adapter,
                "--nestful_path", args.nestful_path,
                "--call_dist_path", args.call_dist_path,
                "--n_calls", str(n_calls),
                "--max_new_tokens", str(
                    cfg.get("eval", {}).get(
                        "max_new_tokens_final",
                        cfg.get("eval", {}).get("max_new_tokens", 1024),
                    )
                ),
                "--batch_size", str(cfg.get("eval", {}).get("batch_size", 8)),
            ]
            if smoke:
                cmd += ["--max_tasks", "20"]
            elif final_limit:
                cmd += ["--max_tasks", str(final_limit)]
            _run_eval(cmd, out_json, f"final eval n_calls={n_calls}", eval_gpus, n_calls)
            if out_json.is_file():
                final_all[str(n_calls)] = json.loads(out_json.read_text(encoding="utf-8"))

        final_path = results_dir / "final_nestful.json"
        final_path.write_text(json.dumps(final_all, indent=2, ensure_ascii=False), encoding="utf-8")
        _log(f"Final eval saved to {final_path}")

        if wandb_run is not None:
            for n_calls_str, fm in final_all.items():
                _wandb.log({f"final/exec_pass_n{n_calls_str}": fm.get("exec_pass_rate", 0.0)})

    # ── Summary ───────────────────────────────────────────────────────────
    if summary_rows:
        print_summary(summary_rows)
        csv_path = results_dir / "curriculum_summary.csv"
        write_summary_csv(csv_path, summary_rows)
        _log(f"Summary CSV -> {csv_path}")

    if wandb_run is not None:
        _wandb.finish()

    _log("All done.")


if __name__ == "__main__":
    main()
