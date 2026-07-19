#!/usr/bin/env python3
"""Pure Stage 3 — two continuous GRPO epochs from base model (no C1/Stage2).

Workflow:
  materialize/audit (caller) → preflight → eval C0
  → open session → epoch1 → atomic S3_E1 → weight sync
  → epoch2 (same optimizer) → atomic S3_E2 → teardown
  → eval E1/E2 (dev) → eval C0/E2 (test) → compare → reports

Usage:
  python scripts/training/run_pure_stage3_two_epoch.py --run-dir outputs/runs/...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3 = os.path.normpath(os.path.join(_HERE, "..", ".."))
_MINIMAL = os.path.normpath(os.path.join(_V3, "..", "nestful_mtgrpo_minimal"))
_PARTIAL = os.path.normpath(os.path.join(_V3, "..", "nestful_mtgrpo_partial"))
_PREFLIGHT = os.path.join(_HERE, "preflight_training_datasets.py")
_FINAL_EVAL = os.path.join(_V3, "scripts", "eval", "final_eval_v5.py")
_CREDIT_PROBE = os.path.join(_V3, "scripts", "analysis", "pure_stage3_credit_probe.py")
_DEFAULT_CONFIG = os.path.join(_PARTIAL, "config.yaml")
_DEFAULT_DEV = os.path.join(_MINIMAL, "data", "splits", "nestful_dev.jsonl")
_DEFAULT_TEST = os.path.join(_MINIMAL, "data", "splits", "nestful_test.jsonl")
_DEFAULT_STAGE3 = os.path.join(
    _V3, "data", "training_ready_v5", "filtered", "stage3_train_ready.jsonl")

if _V3 not in sys.path:
    sys.path.insert(0, _V3)

from scripts.training.run_v5_pipeline import (  # noqa: E402
    _current_registry_hash,
    _git_state,
    _run_logged,
    _sha256_file,
    _vllm_train_overrides,
)
from scripts.training.two_phase_train_session import TwoPhaseTrainSession  # noqa: E402
from scripts.training.two_phase_utils import (  # noqa: E402
    assert_canonical_training,
    atomic_publish_checkpoint,
    audit_dataset_ids,
    collect_repro_manifest,
    count_jsonl_rows,
    discard_incomplete_checkpoint,
    verify_dev_test_disjoint,
)

EXPECTED_ROWS = 326


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def _state_path(run_dir: str) -> str:
    return os.path.join(run_dir, "overnight_state.json")


def _load_state(run_dir: str) -> dict:
    p = _state_path(run_dir)
    if os.path.isfile(p):
        return _load_json(p)
    return {"version": "pure_stage3_two_epoch_v1", "steps": {},
            "resume_kind": "phase_level"}


def _save_state(run_dir: str, state: dict) -> None:
    _save_json(_state_path(run_dir), state)


def _step_done(state: dict, name: str) -> bool:
    return bool((state.get("steps") or {}).get(name, {}).get("done"))


def _mark_step(state: dict, name: str, **payload) -> None:
    state.setdefault("steps", {})[name] = {"done": True, **payload, "at": _now()}


def _build_overrides(args) -> List[str]:
    ovs = [
        "executor.mode=synthetic",
        f"reward.train_policy={args.reward_policy}",
        "training.epochs=1",
        f"generation.num_generations={args.num_generations}",
        f"training.learning_rate={args.learning_rate}",
        f"training.kl_beta={args.kl_beta}",
        "training.max_grad_norm=1.0",
        f"generation.temperature={args.temperature}",
        f"generation.top_p={args.top_p}",
        "data.train_stage=null",
        "data.mixed_replay=false",
    ]
    ovs += [x.replace("--override ", "") for x in _vllm_train_overrides()]
    return ovs


def _run_preflight(dataset: str, report_path: str) -> dict:
    cmd = [sys.executable, _PREFLIGHT, dataset, "--report", report_path]
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit("[pure-s3] preflight failed")
    return _load_json(report_path)


def _assert_pure_stage3(dataset: str) -> None:
    n = count_jsonl_rows(dataset)
    if n != EXPECTED_ROWS and int(os.environ.get("MAX_TRAIN_TASKS", "0") or 0) == 0:
        raise SystemExit(f"[pure-s3] expected {EXPECTED_ROWS} rows, got {n}")
    stage2 = 0
    with open(dataset, encoding="utf-8-sig") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            st = str(row.get("stage") or "")
            if "stage2" in st:
                stage2 += 1
            if row.get("num_calls") != 3:
                raise SystemExit(
                    f"[pure-s3] {row.get('sample_id')} num_calls!="
                    f"3 ({row.get('num_calls')})")
    if stage2:
        raise SystemExit(f"[pure-s3] ABORT: {stage2} stage2 rows in dataset")


def _eval(
    *,
    config: str,
    checkpoint: Optional[str],
    eval_set: str,
    out_dir: str,
    max_tasks: int,
    log_path: str,
    label: str,
    wandb_run_name: str,
) -> dict:
    cmd = [sys.executable, _FINAL_EVAL, "run",
           "--label", label,
           "--out-dir", out_dir,
           "--eval-set", eval_set,
           "--config", config,
           "--max-tasks", str(max_tasks or 0)]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    env = dict(os.environ)
    env["WANDB_RUN_NAME"] = wandb_run_name
    env.setdefault("USE_VLLM", "1")
    rc = _run_logged(cmd, log_path, env=env)
    if rc != 0:
        raise SystemExit(f"[pure-s3] eval {label} failed; see {log_path}")
    return _load_json(os.path.join(out_dir, "metrics_official.json"))


def _compare(c0: str, e1: str, e2: str, out_dir: str) -> dict:
    cmd = [sys.executable, _FINAL_EVAL, "compare",
           "--baseline", c0, "--best", e1, "--final", e2, "--out", out_dir]
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit("[pure-s3] compare failed")
    return _load_json(os.path.join(out_dir, "final_compare_report.json"))


def _epoch_diag_from_summary(summary: dict, train_log: str) -> dict:
    """Aggregate training diagnostics for one epoch from summary + log."""
    dead = all_ok = all_fail = same_partial = mixed = 0
    n = 0
    rewards: List[float] = []
    with open(train_log, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line) if line.strip() else None
            if not r or "episode_rewards" not in r:
                continue
            n += 1
            ep = r["episode_rewards"]
            rewards.extend(ep)
            if r.get("dead_group"):
                dead += 1
            if r.get("group_all_one"):
                all_ok += 1
            if r.get("group_all_zero"):
                all_fail += 1
            if r.get("group_mixed"):
                mixed += 1
            uniq = set(round(float(x), 6) for x in ep)
            if len(uniq) == 1 and 0.0 < next(iter(uniq)) < 0.99:
                same_partial += 1
    rewards_sorted = sorted(rewards)
    med = (rewards_sorted[len(rewards_sorted) // 2]
           if rewards_sorted else None)
    return {
        "from_summary": {
            k: summary.get(k) for k in (
                "dead_group_rate", "avg_predicted_calls", "no_tool_call_rate",
                "too_few_calls_rate", "global_step_start", "global_step_end",
                "contributing_turns_total", "n_unique_reward_values",
                "fractional_rewards_present", "continuous_training",
                "phase_name", "trained",
            )
        },
        "groups": n,
        "dead_group_rate": dead / n if n else None,
        "all_success_dead_groups": all_ok,
        "all_failure_dead_groups": all_fail,
        "same_partial_reward_dead_groups": same_partial,
        "mixed_groups": mixed,
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
        "median_reward": med,
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "terminal_synthetic_success_rate": summary.get("dead_group_rate_last_epoch"),
        # win_rate in summary is over groups; also pull mean from log if present
        "mean_group_win_rate": summary.get("eligible_for_best"),
    }


def _write_final_reports(run_dir: str, state: dict) -> None:
    """Lightweight markdown/json summary from overnight_state + eval dirs."""
    reports = os.path.join(_V3, "reports")
    os.makedirs(reports, exist_ok=True)
    payload = {
        "generated_at": _now(),
        "run_dir": run_dir,
        "state": state,
        "kind": "pure_stage3_two_epoch",
    }
    # Pull official metrics if present
    for key, rel in (
        ("C0_dev", "eval/C0_dev"),
        ("E1_dev", "eval/S3_E1_dev"),
        ("E2_dev", "eval/S3_E2_dev"),
        ("C0_test", "eval/C0_test"),
        ("E2_test", "eval/S3_E2_test"),
    ):
        p = os.path.join(run_dir, rel, "metrics_official.json")
        if os.path.isfile(p):
            payload.setdefault("official_metrics", {})[key] = _load_json(p)

    jp = os.path.join(reports, "PURE_STAGE3_TWO_EPOCH_RESULT.json")
    _save_json(jp, payload)
    # Also under run_dir
    _save_json(os.path.join(run_dir, "PURE_STAGE3_TWO_EPOCH_RESULT.json"), payload)

    lines = [
        "# Pure Stage 3 — Two-Epoch Result",
        "",
        f"Generated: {payload['generated_at']}",
        f"Run dir: `{run_dir}`",
        "",
        "## Official metrics",
        "",
        "| Arm | split | win_rate | f1_func | f1_param | full_seq |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, m in (payload.get("official_metrics") or {}).items():
        lines.append(
            f"| {key} | — | {m.get('win_rate')} | {m.get('f1_func')} | "
            f"{m.get('f1_param')} | {m.get('full_sequence_accuracy')} |"
        )
    lines += [
        "",
        "## Training",
        "",
        f"- E1 global_step end: "
        f"{(state.get('steps') or {}).get('epoch1_train', {}).get('global_step')}",
        f"- E2 global_step end: "
        f"{(state.get('steps') or {}).get('epoch2_train', {}).get('global_step')}",
        f"- Optimizer unchanged: "
        f"{(state.get('steps') or {}).get('epoch2_train', {}).get('optimizer_unchanged')}",
        "",
        "See `overnight_state.json`, epoch summaries, credit probe, and "
        "`eval/compare_*` for paired statistics.",
        "",
    ]
    mp = os.path.join(reports, "PURE_STAGE3_TWO_EPOCH_RESULT.md")
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(run_dir, "PURE_STAGE3_TWO_EPOCH_RESULT.md"),
              "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[pure-s3] reports -> {mp}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default=_DEFAULT_STAGE3)
    ap.add_argument("--dev-set", default=_DEFAULT_DEV)
    ap.add_argument("--test-set", default=_DEFAULT_TEST)
    ap.add_argument("--config", default=_DEFAULT_CONFIG)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=3e-7)
    ap.add_argument("--kl-beta", type=float, default=0.15)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--reward-policy", default="execution_aware_v3_2_dense")
    ap.add_argument("--max-train-tasks", type=int, default=0)
    ap.add_argument("--dev-max-tasks", type=int, default=0)
    ap.add_argument("--test-max-tasks", type=int, default=0)
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--skip-baseline-eval", action="store_true")
    ap.add_argument("--skip-test-eval", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--syntax-audit-verdict", default="NO_MISMATCH")
    ap.add_argument("--syntax-audit-path", default=None)
    args = ap.parse_args()

    # Env caps for smoke
    if os.environ.get("MAX_TRAIN_TASKS"):
        args.max_train_tasks = int(os.environ["MAX_TRAIN_TASKS"])
    if os.environ.get("DEV_MAX_TASKS"):
        args.dev_max_tasks = int(os.environ["DEV_MAX_TASKS"])

    seed = int(os.environ.get("SEED", "42"))
    data_seed = int(os.environ.get("DATA_SEED", "42"))
    rollout_seed = int(os.environ.get("ROLLOUT_SEED", "42"))

    run_dir = os.path.abspath(args.run_dir)
    dataset = os.path.abspath(args.dataset)
    dev_set = os.path.abspath(args.dev_set)
    test_set = os.path.abspath(args.test_set)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    success_marker = os.path.join(run_dir, "SUCCESS")
    failed_marker = os.path.join(run_dir, "FAILED")
    if os.path.isfile(success_marker) and not args.resume:
        raise SystemExit(f"[pure-s3] SUCCESS marker exists; new RUN_DIR or RESUME=1")

    cur_hash, cur_ver = _current_registry_hash()
    assert_canonical_training(
        executor_mode="synthetic",
        reward_policy=args.reward_policy,
        registry_version=cur_ver,
    )
    _assert_pure_stage3(dataset)

    state = _load_state(run_dir)
    if (not args.resume) and state.get("steps") and any(
            (state["steps"].get(k) or {}).get("done")
            for k in ("epoch1_train", "epoch2_train")):
        raise SystemExit("[pure-s3] run_dir has training state; use --resume")

    ds_sha = _sha256_file(dataset)
    manifest = collect_repro_manifest(
        git=_git_state(),
        registry_version=cur_ver,
        registry_hash=cur_hash,
        datasets=[(dataset, ds_sha)],
    )
    manifest.update({
        "kind": "pure_stage3_two_epoch",
        "created_at": _now(),
        "run_dir": run_dir,
        "dataset": dataset,
        "dev_set": dev_set,
        "test_set": test_set,
        "dev_test_audit": verify_dev_test_disjoint(dev_set),
        "dataset_id_audit": audit_dataset_ids(dataset),
        "syntax_audit_verdict": args.syntax_audit_verdict,
        "syntax_audit_path": args.syntax_audit_path,
        "hyperparameters": {
            "num_generations": args.num_generations,
            "learning_rate": args.learning_rate,
            "kl_beta": args.kl_beta,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "epochs": 2,
            "executor_mode": "synthetic",
            "reward_policy": args.reward_policy,
            "mt_grpo_mode": "turn_level_minimal",
            "gamma": 1.0,
            "lambda_episode": 1.0,
            "max_grad_norm": 1.0,
            "start_from": "base_model_C0",
            "no_stage2": True,
            "no_gold_replay": True,
        },
        "seeds": {"SEED": seed, "DATA_SEED": data_seed, "ROLLOUT_SEED": rollout_seed},
        "wandb_project": os.environ.get("WANDB_PROJECT", "nestful-v5-pure-stage3"),
    })
    _save_json(os.path.join(run_dir, "run_manifest.json"), manifest)

    if args.dry_run:
        print("[pure-s3] dry-run OK — manifest written")
        return 0

    wandb_group = os.environ.get("WANDB_GROUP") or os.environ.get(
        "WANDB_RUN_GROUP") or os.path.basename(run_dir)
    os.environ.setdefault("WANDB_PROJECT", "nestful-v5-pure-stage3")
    os.environ["WANDB_RUN_GROUP"] = wandb_group

    try:
        # ── preflight ──────────────────────────────────────────────
        if not args.skip_preflight and not _step_done(state, "preflight"):
            report = _run_preflight(
                dataset, os.path.join(run_dir, "preflight_report.json"))
            if report.get("total_rows") not in (EXPECTED_ROWS, args.max_train_tasks or EXPECTED_ROWS):
                if args.max_train_tasks == 0 and report.get("total_rows") != EXPECTED_ROWS:
                    raise SystemExit("[pure-s3] preflight row count mismatch")
            _mark_step(state, "preflight", report=report)
            _save_state(run_dir, state)

        # ── C0 dev eval ────────────────────────────────────────────
        c0_dir = os.path.join(run_dir, "eval", "C0_dev")
        if not args.skip_baseline_eval and not _step_done(state, "eval_C0_dev"):
            metrics = _eval(
                config=args.config, checkpoint=None, eval_set=dev_set,
                out_dir=c0_dir, max_tasks=args.dev_max_tasks,
                log_path=os.path.join(run_dir, "logs", "eval_C0_dev.log"),
                label="C0_baseline",
                wandb_run_name=f"{wandb_group}-C0-dev-eval",
            )
            _mark_step(state, "eval_C0_dev", metrics=metrics, out_dir=c0_dir)
            _save_state(run_dir, state)

        # ── training session ───────────────────────────────────────
        session: Optional[TwoPhaseTrainSession] = None
        overrides = _build_overrides(args)
        expected = None if args.max_train_tasks else EXPECTED_ROWS

        need_train = (not _step_done(state, "epoch1_train")
                      or not _step_done(state, "epoch2_train"))
        if need_train:
            session = TwoPhaseTrainSession(
                args.config, overrides,
                seed=seed, data_seed=data_seed, rollout_seed=rollout_seed,
            )
            # Resume mid-run: if E1 done but E2 not, load E1 adapter with FRESH
            # optimizer (Adam not on disk) — marked explicitly.
            e1_ckpt = os.path.join(run_dir, "checkpoints", "S3_E1")
            if _step_done(state, "epoch1_train") and not _step_done(state, "epoch2_train"):
                print("[pure-s3] RESUME: loading S3_E1 with FRESH optimizer "
                      "(Adam state not persisted)", flush=True)
                session.load_learner(checkpoint=e1_ckpt)
                session.global_step = int(
                    state["steps"]["epoch1_train"].get("global_step") or 0)
                session.start_rollout_workers(adapter_path=e1_ckpt)
                session.sync_rollout_policy(e1_ckpt, label="S3_E1")
                state.setdefault("resume_events", []).append({
                    "kind": "fresh_optimizer_from_E1",
                    "at": _now(),
                    "note": "exact Adam resume unsupported",
                })
                _save_state(run_dir, state)
            else:
                session.load_learner(checkpoint=None)
                session.start_rollout_workers(adapter_path=None)

            # Persist rollout PIDs for shell trap
            _save_json(os.path.join(run_dir, "logs", "rollout_worker_pids.json"),
                       session._last_pool_pids)

            # ── Epoch 1 ────────────────────────────────────────────
            if not _step_done(state, "epoch1_train"):
                e1_out = os.path.join(run_dir, "epoch_1", "train")
                adapter, summary = session.train_phase(
                    dataset_path=dataset,
                    train_out=e1_out,
                    phase_name="pure_stage3_epoch1",
                    max_train_tasks=args.max_train_tasks,
                    expected_rows=expected,
                    wandb_run_name=f"{wandb_group}-pure-stage3-train",
                )
                dest = os.path.join(run_dir, "checkpoints", "S3_E1")
                discard_incomplete_checkpoint(dest)
                man = atomic_publish_checkpoint(adapter, dest, label="S3_E1")
                # Weight sync barrier before epoch 2
                sync_hash = session.sync_rollout_policy(dest, label="S3_E1")
                diag = _epoch_diag_from_summary(
                    summary, os.path.join(e1_out, "train_log.jsonl"))
                _save_json(os.path.join(run_dir, "epoch_1", "train_summary.json"),
                           summary)
                _save_json(os.path.join(run_dir, "epoch_1", "epoch_coverage.json"),
                           summary.get("epoch_coverage") or {})
                _save_json(os.path.join(run_dir, "epoch_1", "diagnostics.json"), diag)
                _mark_step(
                    state, "epoch1_train",
                    checkpoint=dest,
                    checkpoint_manifest=man,
                    summary=summary,
                    diagnostics=diag,
                    global_step=session.global_step,
                    optimizer_id=id(session.optimizer),
                    adapter_sync_hash=sync_hash,
                    rollout_pids=list(session._last_pool_pids),
                )
                _save_state(run_dir, state)

            # ── Epoch 2 ────────────────────────────────────────────
            if not _step_done(state, "epoch2_train"):
                e1_ckpt = os.path.join(run_dir, "checkpoints", "S3_E1")
                # Ensure workers on E1 weights
                session.sync_rollout_policy(e1_ckpt, label="S3_E1")
                opt_id_before = id(session.optimizer)
                e2_out = os.path.join(run_dir, "epoch_2", "train")
                adapter, summary = session.train_phase(
                    dataset_path=dataset,
                    train_out=e2_out,
                    phase_name="pure_stage3_epoch2",
                    max_train_tasks=args.max_train_tasks,
                    expected_rows=expected,
                    wandb_run_name=f"{wandb_group}-pure-stage3-train",
                )
                dest = os.path.join(run_dir, "checkpoints", "S3_E2")
                discard_incomplete_checkpoint(dest)
                man = atomic_publish_checkpoint(adapter, dest, label="S3_E2")
                diag = _epoch_diag_from_summary(
                    summary, os.path.join(e2_out, "train_log.jsonl"))
                _save_json(os.path.join(run_dir, "epoch_2", "train_summary.json"),
                           summary)
                _save_json(os.path.join(run_dir, "epoch_2", "epoch_coverage.json"),
                           summary.get("epoch_coverage") or {})
                _save_json(os.path.join(run_dir, "epoch_2", "diagnostics.json"), diag)
                _mark_step(
                    state, "epoch2_train",
                    checkpoint=dest,
                    checkpoint_manifest=man,
                    summary=summary,
                    diagnostics=diag,
                    global_step=session.global_step,
                    optimizer_id=id(session.optimizer),
                    optimizer_unchanged=(id(session.optimizer) == opt_id_before),
                    continuous_from_epoch1=True,
                    rollout_pids=list(session._last_pool_pids),
                )
                _save_state(run_dir, state)

            # Training comparison
            cmp = {
                "epoch1": (state["steps"].get("epoch1_train") or {}).get("diagnostics"),
                "epoch2": (state["steps"].get("epoch2_train") or {}).get("diagnostics"),
                "optimizer_unchanged": (state["steps"].get("epoch2_train") or {}).get(
                    "optimizer_unchanged"),
                "global_step_e1": (state["steps"].get("epoch1_train") or {}).get(
                    "global_step"),
                "global_step_e2": (state["steps"].get("epoch2_train") or {}).get(
                    "global_step"),
                "adapter_hash_e1": ((state["steps"].get("epoch1_train") or {})
                                    .get("checkpoint_manifest") or {}).get("adapter_hash"),
                "adapter_hash_e2": ((state["steps"].get("epoch2_train") or {})
                                    .get("checkpoint_manifest") or {}).get("adapter_hash"),
            }
            _save_json(os.path.join(run_dir, "pure_stage3_training_comparison.json"), cmp)

            # Credit probe (post-hoc; no training change)
            if not _step_done(state, "credit_probe"):
                e1_log = os.path.join(run_dir, "epoch_1", "train", "train_log.jsonl")
                e2_log = os.path.join(run_dir, "epoch_2", "train", "train_log.jsonl")
                out_j = os.path.join(_V3, "reports", "pure_stage3_credit_probe.jsonl")
                out_m = os.path.join(_V3, "reports", "pure_stage3_credit_probe_summary.md")
                os.makedirs(os.path.join(_V3, "reports"), exist_ok=True)
                cmd = [sys.executable, _CREDIT_PROBE,
                       "--epoch1-log", e1_log, "--epoch2-log", e2_log,
                       "--out-jsonl", out_j, "--out-md", out_m,
                       "--n-groups", "100", "--seed", str(seed)]
                if os.path.isfile(e1_log) and os.path.isfile(e2_log):
                    subprocess.run(cmd, check=False)
                    _mark_step(state, "credit_probe", jsonl=out_j, md=out_m)
                    _save_state(run_dir, state)

            print("[pure-s3] training session closed; all GPUs free for eval",
                  flush=True)
            session.close()
            session = None
            _save_json(os.path.join(run_dir, "logs", "rollout_worker_pids.json"), [])

        # ── Dev evals E1 / E2 ──────────────────────────────────────
        e1_ckpt = os.path.join(run_dir, "checkpoints", "S3_E1")
        e2_ckpt = os.path.join(run_dir, "checkpoints", "S3_E2")
        e1_dev = os.path.join(run_dir, "eval", "S3_E1_dev")
        e2_dev = os.path.join(run_dir, "eval", "S3_E2_dev")

        if not _step_done(state, "eval_E1_dev"):
            m = _eval(config=args.config, checkpoint=e1_ckpt, eval_set=dev_set,
                      out_dir=e1_dev, max_tasks=args.dev_max_tasks,
                      log_path=os.path.join(run_dir, "logs", "eval_E1_dev.log"),
                      label="S3_E1",
                      wandb_run_name=f"{wandb_group}-E1-dev-eval")
            _mark_step(state, "eval_E1_dev", metrics=m, out_dir=e1_dev)
            _save_state(run_dir, state)

        if not _step_done(state, "eval_E2_dev"):
            m = _eval(config=args.config, checkpoint=e2_ckpt, eval_set=dev_set,
                      out_dir=e2_dev, max_tasks=args.dev_max_tasks,
                      log_path=os.path.join(run_dir, "logs", "eval_E2_dev.log"),
                      label="S3_E2",
                      wandb_run_name=f"{wandb_group}-E2-dev-eval")
            _mark_step(state, "eval_E2_dev", metrics=m, out_dir=e2_dev)
            _save_state(run_dir, state)

        if not _step_done(state, "compare_dev"):
            out = os.path.join(run_dir, "eval", "compare_C0_E1_E2_dev")
            rep = _compare(c0_dir, e1_dev, e2_dev, out)
            _mark_step(state, "compare_dev", report=rep, out_dir=out)
            _save_state(run_dir, state)

        # ── Final test: C0 + E2 only ───────────────────────────────
        if not args.skip_test_eval:
            c0_test = os.path.join(run_dir, "eval", "C0_test")
            e2_test = os.path.join(run_dir, "eval", "S3_E2_test")
            if not _step_done(state, "eval_C0_test"):
                m = _eval(config=args.config, checkpoint=None, eval_set=test_set,
                          out_dir=c0_test, max_tasks=args.test_max_tasks,
                          log_path=os.path.join(run_dir, "logs", "eval_C0_test.log"),
                          label="C0_test",
                          wandb_run_name=f"{wandb_group}-C0-final-test")
                _mark_step(state, "eval_C0_test", metrics=m, out_dir=c0_test)
                _save_state(run_dir, state)
            if not _step_done(state, "eval_E2_test"):
                m = _eval(config=args.config, checkpoint=e2_ckpt, eval_set=test_set,
                          out_dir=e2_test, max_tasks=args.test_max_tasks,
                          log_path=os.path.join(run_dir, "logs", "eval_E2_test.log"),
                          label="S3_E2_test",
                          wandb_run_name=f"{wandb_group}-E2-final-test")
                _mark_step(state, "eval_E2_test", metrics=m, out_dir=e2_test)
                _save_state(run_dir, state)
            if not _step_done(state, "compare_test"):
                # compare needs 3 arms; use E1_dev as placeholder "best" exploratory
                # Primary headline is C0_test vs E2_test — also write pairwise note
                out = os.path.join(run_dir, "eval", "compare_C0_E2_test")
                # final_eval compare requires best; pass E2 as both best and final
                # if E1 test not run. Use E1_dev as exploratory middle arm.
                rep = _compare(c0_test, e1_dev, e2_test, out)
                _mark_step(state, "compare_test", report=rep, out_dir=out,
                           note="best arm is E1_dev (exploratory); final=E2_test")
                _save_state(run_dir, state)

        _write_final_reports(run_dir, state)
        with open(success_marker, "w", encoding="utf-8") as fh:
            fh.write(_now() + "\n")
        if os.path.isfile(failed_marker):
            os.remove(failed_marker)
        print(f"[pure-s3] SUCCESS -> {success_marker}")
        return 0

    except BaseException as exc:
        with open(failed_marker, "w", encoding="utf-8") as fh:
            fh.write(f"{_now()}\n{type(exc).__name__}: {exc}\n")
        print(f"[pure-s3] FAILED: {exc}", file=sys.stderr)
        raise
    finally:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
