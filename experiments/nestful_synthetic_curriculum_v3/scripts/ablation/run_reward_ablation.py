#!/usr/bin/env python3
"""Reward ablation — main training CLI (reports/reward_ablation/ABLATION_PLAN.md §7).

Reuses the EXISTING training/rollout/reward/executor/credit-assignment/
checkpoint infrastructure verbatim:
  - scripts/training/two_phase_train_session.py::TwoPhaseTrainSession
    (same GRPO trainer, same `train_phase` = exactly one epoch over the
    given dataset, same vLLM DP rollout pool, same synthetic executor).
  - scripts/training/two_phase_utils.py (atomic checkpoint publish, repro
    manifest, GPU-eval prep, dataset audits) — same helpers the production
    pure_stage3_2ep run uses.
  - scripts/training/preflight_training_datasets.py (same synthetic-executor
    replay gate).
  - scripts/eval/final_eval_v5.py (same eval CLI: same prompt/parser/
    executor/scorer, forces temperature=0/top_p=1/react).
  - lib/reward_ablation_registry.py (the ONLY new "reward" surface),
    dispatched through vllm_dp_pool.resolve_reward_info exactly like every
    other reward policy in this repo.

Does NOT re-implement the GRPO trainer, rollout loop, or credit assignment.

CLI:
  --round {1,2,3}
  --reward-arm {A0_R0_CURRENT,A1_OUTCOME_ONLY,A2_R3_OUTCOME_FIRST,
                A3_VERIFIABLE_PROCESS,A4_GATED_VERIFIABLE}
  --seed INT
  --train-subset PATH        (default: reports/reward_ablation/data/train_subset_160.jsonl)
  --eval-subset PATH         (default: reports/reward_ablation/data/nestful_diagnostic_500_ids.json)
  --resume
  --smoke                    (8 train tasks, 8 rollouts, ~2 optimizer steps, 20 eval tasks)
  --run-id STR               (default: auto-generated unique experiment ID)
  --wandb-project STR        (default: nestful-reward-ablation)
  --wandb-group STR          (default: reward_ablation_round{round}_<timestamp>)
  --output-root PATH         (default: outputs/runs)
  --dry-run                  (resolve config + write manifest, execute nothing — no GPU needed)
  --force-fresh              (delete an incomplete run dir — no SUCCESS marker — and restart)
  --skip-c0-eval             (reuse an existing shared C0-on-500-subset eval; C0 is evaluated
                              ONCE per (eval-subset, seed-for-decoding) across all arms, never
                              per-arm, per spec §11)

GPU topology (RunPod, 1 pod, 4 GPUs): GPU0=learner, GPU1-3=rollout workers.
Eval only runs AFTER `session.close()` releases the learner/optimizer (see
`_run_training` below) — no concurrent TP4 eval while the learner is
resident, matching spec §7.

Usage (RunPod):
  python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation.py \\
      --round 1 --reward-arm A2_R3_OUTCOME_FIRST --seed 20260724

Usage (dry-run, no GPU, works anywhere):
  python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation.py \\
      --round 1 --reward-arm A2_R3_OUTCOME_FIRST --seed 20260724 --dry-run
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
_REPO = _V3.parents[1]
_MINIMAL = _V3.parents[0] / "nestful_mtgrpo_minimal"
_PARTIAL = _V3.parents[0] / "nestful_mtgrpo_partial"
# Force _V3 to sys.path[0] (not just "insert if absent") so the real
# `lib.*` package always wins over `scripts/lib/__init__.py`, regardless of
# what other test/CLI modules already touched sys.path first (see
# tests/test_reward_ablation.py for the original shadowing bug this guards).
if str(_V3) in sys.path:
    sys.path.remove(str(_V3))
sys.path.insert(0, str(_V3))

CONFIGS_DIR = _V3 / "configs" / "reward_ablation"
DATA_DIR = _V3 / "reports" / "reward_ablation" / "data"
DEFAULT_TRAIN_SUBSET = DATA_DIR / "train_subset_160.jsonl"
DEFAULT_EVAL_SUBSET_IDS = DATA_DIR / "nestful_diagnostic_500_ids.json"
DEFAULT_OUTPUT_ROOT = _V3 / "outputs" / "runs"
NESTFUL_TEST = _MINIMAL / "data" / "splits" / "nestful_test.jsonl"
DEFAULT_TRAIN_CONFIG = _PARTIAL / "config.yaml"

ARM_IDS = (
    "A0_R0_CURRENT",
    "A1_OUTCOME_ONLY",
    "A2_R3_OUTCOME_FIRST",
    "A3_VERIFIABLE_PROCESS",
    "A4_GATED_VERIFIABLE",
)
TRAIN_POLICY = {
    "A0_R0_CURRENT": "execution_aware_v3_2_dense",
    "A1_OUTCOME_ONLY": "reward_ablation_A1_OUTCOME_ONLY",
    "A2_R3_OUTCOME_FIRST": "reward_ablation_A2_R3_OUTCOME_FIRST",
    "A3_VERIFIABLE_PROCESS": "reward_ablation_A3_VERIFIABLE_PROCESS",
    "A4_GATED_VERIFIABLE": "reward_ablation_A4_GATED_VERIFIABLE",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_REPO),
                              capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNKNOWN ({exc})"


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_effective_config(reward_arm: str) -> Dict[str, Any]:
    base = yaml.safe_load((CONFIGS_DIR / "round1_base.yaml").read_text(encoding="utf-8"))
    arm = yaml.safe_load((CONFIGS_DIR / "arms" / f"{reward_arm}.yaml").read_text(encoding="utf-8"))
    return _deep_merge(base, arm)


def standard_run_id(round_: int, reward_arm: str, seed: int) -> str:
    """Canonical run directory name (no timestamp). Must match run_reward_ablation_round1.sh."""
    return f"reward_ablation_r{round_}_{reward_arm}_seed{seed}"


def build_experiment_id(reward_arm: str, round_: int, seed: int, run_id: Optional[str]) -> str:
    if run_id:
        return run_id
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{standard_run_id(round_, reward_arm, seed)}_{ts}"


def prepare_run_dir(
    run_dir: Path,
    *,
    resume: bool,
    force_fresh: bool,
    dry_run: bool,
    canonical_run_id: Optional[str] = None,
) -> None:
    """Refuse to clobber an existing run unless --resume or --force-fresh."""
    if not run_dir.is_dir() or dry_run:
        return
    if resume:
        return
    if (run_dir / "SUCCESS").is_file():
        raise SystemExit(
            f"[run_reward_ablation] ABORT: {run_dir} already finished (SUCCESS). "
            "Use a new --run-id for a parallel experiment."
        )
    if force_fresh:
        shutil.rmtree(run_dir)
        print(f"[run_reward_ablation] --force-fresh: removed incomplete run dir {run_dir}")
        return
    hint = ""
    if canonical_run_id and run_dir.name != canonical_run_id:
        hint = (
            f"\n  expected canonical --run-id: {canonical_run_id}\n"
            "  (avoid patterns like reward_ablation_r1_1_<ARM> — use r{round}_<ARM> only once)"
        )
    raise SystemExit(
        f"[run_reward_ablation] ABORT: {run_dir} exists (incomplete or failed prior attempt).\n"
        "  --resume       continue where it left off\n"
        "  --force-fresh  delete incomplete run and restart\n"
        "  --run-id NEW   start a parallel experiment with a different id"
        f"{hint}"
    )


def materialize_eval_subset(eval_subset_arg: Path, out_dir: Path) -> Path:
    """`final_eval_v5.py` takes a JSONL --eval-set. If `eval_subset_arg` is
    the frozen *_ids.json (task IDs only), materialize the matching rows
    from nestful_test.jsonl into a JSONL the eval CLI can consume directly
    — no changes to final_eval_v5.py needed. If it's already a .jsonl, use
    it as-is."""
    if eval_subset_arg.suffix == ".jsonl":
        return eval_subset_arg
    ids = set(json.loads(eval_subset_arg.read_text(encoding="utf-8"))["task_ids"])
    out_path = out_dir / "eval_subset_materialized.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    found = 0
    with open(NESTFUL_TEST, encoding="utf-8") as src, open(out_path, "w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id") or row.get("task_id") or "")
            if sid in ids:
                dst.write(line if line.endswith("\n") else line + "\n")
                found += 1
    if found != len(ids):
        raise SystemExit(f"[run_reward_ablation] ABORT: materialized {found}/{len(ids)} eval subset rows")
    return out_path


def build_run_config(args) -> Dict[str, Any]:
    effective = load_effective_config(args.reward_arm)
    effective["reward"]["train_policy"] = TRAIN_POLICY[args.reward_arm]
    effective.setdefault("seeds", {}).update({
        "SEED": args.seed, "DATA_SEED": args.seed, "ROLLOUT_SEED": args.seed,
    })
    effective.setdefault("data", {})["train_dataset"] = str(args.train_subset)
    effective["data"]["eval_dataset_ids"] = str(args.eval_subset)
    effective["round"] = args.round
    if args.smoke:
        effective.setdefault("smoke", {}).update({
            "enabled": True,
            "max_train_tasks": 8,
            "num_generations": 8,
            "eval_max_tasks": 20,
        })
        effective["training"]["num_generations"] = 8
    return effective


def compute_hashes(args, effective_config: Dict[str, Any]) -> Dict[str, str]:
    from scripts.training.run_v5_pipeline import _current_registry_hash
    registry_hash, registry_version = _current_registry_hash()
    frozen_specs_path = _V3 / "reports" / "reward_ablation" / "FROZEN_REWARD_SPECS.json"
    reward_spec_hash = (_sha256_file(frozen_specs_path) if frozen_specs_path.is_file()
                        else "UNFROZEN (run scripts/ablation/freeze_reward_specs.py first)")
    return {
        "dataset_hash": _sha256_file(args.train_subset) if args.train_subset.is_file() else "MISSING",
        "eval_subset_hash": _sha256_file(args.eval_subset) if args.eval_subset.is_file() else "MISSING",
        "config_hash": _sha256_json(effective_config),
        "reward_spec_hash": reward_spec_hash,
        "executor_hash": registry_hash,
        "registry_version": registry_version,
        "prompt_hash": _sha256_file(_MINIMAL / "prompts.py") if (_MINIMAL / "prompts.py").is_file() else "N/A",
    }


# ─────────────────────────────────────────────────────────────────────────
# Resume-safe run state (same pattern as run_pure_stage3_two_epoch.py's
# overnight_state.json, generalized to a single-epoch ablation run).
# ─────────────────────────────────────────────────────────────────────────

def _state_path(run_dir: Path) -> Path:
    return run_dir / "ablation_run_state.json"


def load_state(run_dir: Path) -> Dict[str, Any]:
    p = _state_path(run_dir)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"version": "reward_ablation_v1", "steps": {}}


def save_state(run_dir: Path, state: Dict[str, Any]) -> None:
    _state_path(run_dir).write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def step_done(state: Dict[str, Any], name: str) -> bool:
    return bool((state.get("steps") or {}).get(name, {}).get("done"))


def mark_step(state: Dict[str, Any], name: str, **payload) -> None:
    state.setdefault("steps", {})[name] = {"done": True, **payload, "at": _now()}


def assert_resume_compatible(run_dir: Path, args, experiment_id: str) -> None:
    """Never resume into a different reward arm / seed / optimizer than the
    run that owns this directory (spec §14)."""
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return
    prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    if prior.get("reward_arm") != args.reward_arm:
        raise SystemExit(
            f"[run_reward_ablation] ABORT: --resume into {run_dir} would switch reward arm "
            f"{prior.get('reward_arm')} -> {args.reward_arm}. Never overwrite another arm's checkpoint."
        )
    if int(prior.get("seed", -1)) != int(args.seed):
        raise SystemExit(
            f"[run_reward_ablation] ABORT: --resume into {run_dir} would switch seed "
            f"{prior.get('seed')} -> {args.seed}."
        )


# ─────────────────────────────────────────────────────────────────────────
# Training + eval (GPU-executing; requires the same RunPod stack as the
# production pure_stage3 run — TwoPhaseTrainSession, vLLM DP pool, executor)
# ─────────────────────────────────────────────────────────────────────────

def _build_session_overrides(effective_config: Dict[str, Any], smoke: bool) -> List[str]:
    ovs = [
        "executor.mode=synthetic",
        f"reward.train_policy={effective_config['reward']['train_policy']}",
        "training.epochs=1",
        f"generation.num_generations={effective_config['training']['num_generations']}",
        f"training.learning_rate={effective_config['training']['learning_rate']}",
        f"training.kl_beta={effective_config['training']['kl_beta']}",
        "training.max_grad_norm=1.0",
        f"generation.temperature={effective_config['generation']['temperature']}",
        f"generation.top_p={effective_config['generation']['top_p']}",
        "data.train_stage=null",
        "data.mixed_replay=false",
    ]
    if os.environ.get("USE_VLLM", "0") == "1":
        ovs.append("hardware.use_vllm=true")
        dp = os.environ.get("ROLLOUT_DP_GPUS", "").strip()
        if dp:
            ovs.append(f"hardware.rollout_data_parallel_gpus={dp}")
    return ovs


def run_training(args, effective_config: Dict[str, Any], run_dir: Path, state: Dict[str, Any]) -> str:
    """Runs exactly ONE epoch via TwoPhaseTrainSession.train_phase (the SAME
    trainer/rollout/credit-assignment code the production run uses).
    Returns the path to the published FINAL checkpoint."""
    from scripts.training.two_phase_train_session import TwoPhaseTrainSession  # noqa: E402
    from scripts.training.two_phase_utils import (  # noqa: E402
        atomic_publish_checkpoint,
        discard_incomplete_checkpoint,
    )

    dest = str(run_dir / "checkpoints" / "FINAL")
    if step_done(state, "train"):
        print(f"[run_reward_ablation] train already done -> {dest}")
        return dest

    overrides = _build_session_overrides(effective_config, args.smoke)
    seed = args.seed
    session = TwoPhaseTrainSession(
        str(DEFAULT_TRAIN_CONFIG), overrides,
        seed=seed, data_seed=seed, rollout_seed=seed,
    )
    try:
        session.load_learner(checkpoint=None)  # always start from C0 / base model
        session.start_rollout_workers(adapter_path=None)
        max_train_tasks = 8 if args.smoke else 0
        expected_rows = None if args.smoke else 160
        adapter, summary = session.train_phase(
            dataset_path=str(args.train_subset),
            train_out=str(run_dir / "train"),
            phase_name=f"reward_ablation_{args.reward_arm}_r{args.round}",
            max_train_tasks=max_train_tasks,
            expected_rows=expected_rows,
            wandb_run_name=args.wandb_run_name,
        )
        discard_incomplete_checkpoint(dest)
        man = atomic_publish_checkpoint(adapter, dest, label="FINAL")
        mark_step(state, "train", checkpoint=dest, checkpoint_manifest=man,
                  global_step=session.global_step, summary_path=str(run_dir / "train" / "train_summary.json"))
        (run_dir / "train").mkdir(parents=True, exist_ok=True)
        (run_dir / "train" / "train_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        save_state(run_dir, state)
    finally:
        session.close()  # release learner + rollout workers BEFORE any eval (spec §7)
        print("[run_reward_ablation] training session closed; GPUs free for eval", flush=True)
    return dest


def run_eval(args, checkpoint: Optional[str], run_dir: Path, state: Dict[str, Any], step_name: str,
             label: str) -> Dict[str, Any]:
    from scripts.training.two_phase_utils import prep_gpus_for_eval  # noqa: E402

    if step_done(state, step_name):
        return (state["steps"][step_name] or {}).get("metrics") or {}

    eval_jsonl = materialize_eval_subset(args.eval_subset, run_dir / "eval_data")
    out_dir = run_dir / "eval" / args.reward_arm / str(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_tasks = 20 if args.smoke else 0

    prep_gpus_for_eval()
    final_eval = _V3 / "scripts" / "eval" / "final_eval_v5.py"
    cmd = [sys.executable, str(final_eval), "run",
           "--label", label, "--out-dir", str(out_dir),
           "--eval-set", str(eval_jsonl), "--config", str(DEFAULT_TRAIN_CONFIG),
           "--max-tasks", str(max_tasks)]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    env = dict(os.environ)
    env["WANDB_RUN_NAME"] = f"{args.wandb_group}-{label}-eval"
    log_path = run_dir / "logs" / f"eval_{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"[run_reward_ablation] ABORT: eval {label} failed; see {log_path}")
    metrics_path = out_dir / "metrics_official.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    mark_step(state, step_name, metrics=metrics, out_dir=str(out_dir))
    save_state(run_dir, state)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--reward-arm", required=True, choices=ARM_IDS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-subset", type=Path, default=DEFAULT_TRAIN_SUBSET)
    ap.add_argument("--eval-subset", type=Path, default=DEFAULT_EVAL_SUBSET_IDS)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-fresh", action="store_true",
                    help="Delete incomplete run dir (no SUCCESS) and restart from scratch")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--wandb-project", default="nestful-reward-ablation")
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-c0-eval", action="store_true")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.resume and args.force_fresh:
        raise SystemExit("[run_reward_ablation] ABORT: --resume and --force-fresh are mutually exclusive")

    canonical_run_id = standard_run_id(args.round, args.reward_arm, args.seed)
    experiment_id = build_experiment_id(args.reward_arm, args.round, args.seed, args.run_id)
    if args.run_id and args.run_id != canonical_run_id:
        print(f"[run_reward_ablation] NOTE: --run-id {args.run_id!r} differs from canonical "
              f"{canonical_run_id!r} (launcher uses canonical id)")
    args.wandb_group = args.wandb_group or f"reward_ablation_round{args.round}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    args.wandb_run_name = f"{args.reward_arm}_seed{args.seed}"

    run_dir = args.output_root / experiment_id
    prepare_run_dir(
        run_dir,
        resume=args.resume,
        force_fresh=args.force_fresh,
        dry_run=args.dry_run,
        canonical_run_id=canonical_run_id,
    )
    if args.resume:
        assert_resume_compatible(run_dir, args, experiment_id)

    effective_config = build_run_config(args)
    hashes = compute_hashes(args, effective_config)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.json").write_text(
        json.dumps(effective_config, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "experiment_id": experiment_id,
        "round": args.round,
        "reward_arm": args.reward_arm,
        "reward_train_policy": TRAIN_POLICY[args.reward_arm],
        "seed": args.seed,
        "train_subset": str(args.train_subset),
        "eval_subset": str(args.eval_subset),
        "smoke": args.smoke,
        "wandb": {"project": args.wandb_project, "group": args.wandb_group, "run_name": args.wandb_run_name},
        "hashes": hashes,
        "git_commit": _git_commit(),
        "created_at": _now(),
        "output_dir": str(run_dir),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[run_reward_ablation] experiment_id={experiment_id}")
    print(f"[run_reward_ablation] config -> {run_dir / 'config_used.json'}")
    print(f"[run_reward_ablation] manifest -> {run_dir / 'run_manifest.json'}")
    for k, v in hashes.items():
        print(f"[run_reward_ablation]   {k}: {v}")

    if args.dry_run:
        print("[run_reward_ablation] DRY RUN — no training/eval executed.")
        return 0

    if "UNFROZEN" in hashes["reward_spec_hash"] and args.reward_arm != "A0_R0_CURRENT":
        raise SystemExit("[run_reward_ablation] ABORT: reward specs not frozen — run "
                         "scripts/ablation/freeze_reward_specs.py before Round 1.")

    state = load_state(run_dir)
    manifest["experiment_start_at"] = manifest.get("experiment_start_at") or _now()

    from scripts.training.preflight_training_datasets import validate_file  # noqa: E402
    if not step_done(state, "preflight"):
        report = validate_file(str(args.train_subset))
        mark_step(state, "preflight", report=report)
        save_state(run_dir, state)

    checkpoint = run_training(args, effective_config, run_dir, state)

    if not args.skip_c0_eval:
        run_eval(args, checkpoint=None, run_dir=run_dir, state=state, step_name="eval_C0", label="C0")
    run_eval(args, checkpoint=checkpoint, run_dir=run_dir, state=state, step_name="eval_arm",
             label=args.reward_arm)

    manifest["experiment_end_at"] = _now()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "SUCCESS").write_text(_now() + "\n", encoding="utf-8")
    print(f"[run_reward_ablation] SUCCESS -> {run_dir / 'SUCCESS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
