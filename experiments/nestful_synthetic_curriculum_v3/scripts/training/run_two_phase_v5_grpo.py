#!/usr/bin/env python3
"""Two-phase v5 GRPO — single-process continuous training session.

Workflow (strict order):
  preflight → eval C0 → open training session
           → Phase 1 (429× S2, 1 epoch) → atomic C1 save
           → Phase 2 (466× S3+replay, 1 epoch, same optimizer/global_step) → atomic C2 save
           → teardown session (free all GPUs)
           → eval C1 → eval C2 → compare C0/C1/C2

C1 dev eval is deferred until after Phase 2 so EVAL_TP=4 can use all four GPUs
without evicting the learner + AdamW optimizer from GPU 0.

Training runs in ONE Python process via :class:`TwoPhaseTrainSession` so AdamW
and ``global_step`` continue across phases. Rollout workers stay up between
phases; they are shut down only after training completes.

Resume (``--resume``) skips *completed* steps in ``two_phase_state.json`` — this
is **phase-level** resume, NOT exact optimizer-step resume (no Adam state is
persisted to disk). A crash during Phase 2 keeps C1, discards incomplete C2,
and restarts Phase 2 from C1 with a fresh optimizer.

Usage:
  python scripts/training/run_two_phase_v5_grpo.py --run-dir outputs/runs/two_phase_...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3 = os.path.normpath(os.path.join(_HERE, "..", ".."))
_MINIMAL = os.path.normpath(os.path.join(_V3, "..", "nestful_mtgrpo_minimal"))
_PARTIAL = os.path.normpath(os.path.join(_V3, "..", "nestful_mtgrpo_partial"))
_PREFLIGHT = os.path.join(_HERE, "preflight_training_datasets.py")
_FINAL_EVAL = os.path.join(_V3, "scripts", "eval", "final_eval_v5.py")
_DEFAULT_CONFIG = os.path.join(_PARTIAL, "config.yaml")
_DEFAULT_DEV = os.path.join(_MINIMAL, "data", "splits", "nestful_dev.jsonl")
_DEFAULT_PHASE1 = os.path.join(
    _V3, "data", "training_ready_v5", "filtered", "phase1_stage2_train.jsonl")
_DEFAULT_PHASE2 = os.path.join(
    _V3, "data", "training_ready_v5", "filtered",
    "phase2_stage3_plus_stage2_replay.jsonl")

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

PHASE1_ROWS = 429
PHASE2_ROWS = 466


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
    return os.path.join(run_dir, "two_phase_state.json")


def _load_state(run_dir: str) -> dict:
    p = _state_path(run_dir)
    if os.path.isfile(p):
        return _load_json(p)
    return {"version": "two_phase_v3", "steps": {}, "resume_kind": "phase_level"}


def _save_state(run_dir: str, state: dict) -> None:
    _save_json(_state_path(run_dir), state)


def _step_done(state: dict, name: str) -> bool:
    return bool((state.get("steps") or {}).get(name, {}).get("done"))


def _mark_step(state: dict, name: str, **payload) -> None:
    state.setdefault("steps", {})[name] = {"done": True, **payload, "at": _now()}


def _record_resume_event(state: dict, kind: str, **payload) -> None:
    state.setdefault("resume_events", []).append(
        {"kind": kind, "at": _now(), **payload})


def _build_train_overrides(args) -> List[str]:
    ovs = [
        f"executor.mode=synthetic",
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


def _run_preflight(phase1: str, phase2: str, report_path: str) -> dict:
    cmd = [sys.executable, _PREFLIGHT, phase1, phase2, "--report", report_path]
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit("[two-phase] preflight failed")
    return _load_json(report_path)


def _dev_eval_subprocess(
    *,
    config: str,
    checkpoint: Optional[str],
    dev_set: str,
    out_dir: str,
    max_tasks: int,
    log_path: str,
    label: str,
    wandb_run_name: str,
) -> dict:
    cmd = [sys.executable, _FINAL_EVAL, "run",
           "--label", label,
           "--out-dir", out_dir,
           "--eval-set", dev_set,
           "--config", config,
           "--max-tasks", str(max_tasks or 0)]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    env = dict(os.environ)
    env["WANDB_RUN_NAME"] = wandb_run_name
    env.setdefault("USE_VLLM", "1")
    rc = _run_logged(cmd, log_path, env=env)
    if rc != 0:
        raise SystemExit(f"[two-phase] eval {label} failed; see {log_path}")
    with open(os.path.join(out_dir, "metrics_official.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _stage_metrics_from_log(train_log: str, dataset_path: str) -> Dict[str, Any]:
    stage_by_id: Dict[str, str] = {}
    with open(dataset_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id") or row.get("task_id"))
            st = str(row.get("stage") or "")
            stage_by_id[sid] = "stage2" if "stage2" in st else (
                "stage3" if "stage3" in st else "other")

    buckets: Dict[str, List[dict]] = defaultdict(list)
    if not os.path.isfile(train_log):
        return {"error": f"missing {train_log}"}
    with open(train_log, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "mean_reward" not in rec:
                continue
            tid = str(rec.get("task_id", ""))
            buckets[stage_by_id.get(tid, "other")].append(rec)

    def _agg(recs: List[dict]) -> dict:
        if not recs:
            return {"n_tasks": 0}
        n = len(recs)
        def mean(k):
            vals = [float(r[k]) for r in recs if r.get(k) is not None]
            return sum(vals) / len(vals) if vals else None
        return {
            "n_tasks": n,
            "mean_reward": mean("mean_reward"),
            "mean_kl": mean("kl"),
            "mean_grad_norm": None,
            "dead_group_rate": mean("dead_group"),
            "all_correct_rate": mean("win_rate"),
            "unique_rewards_mean": mean("n_unique_episode_rewards"),
            "too_few_calls_rate": mean("too_few_calls_rate"),
            "executable_trajectory_rate": mean("executable_trajectory_rate"),
            "invalid_reference_rate": mean("invalid_reference_rate"),
            "executor_error_rate": mean("executor_error_rate"),
        }

    return {k: _agg(v) for k, v in sorted(buckets.items())}


def _log_stage_metrics_wandb(group: str, metrics: Dict[str, Any]) -> None:
    if not os.environ.get("WANDB_PROJECT"):
        return
    try:
        import wandb
        run = wandb.init(
            project=os.environ["WANDB_PROJECT"],
            name=f"{group}-phase2-stage-metrics",
            group=group,
            entity=os.environ.get("WANDB_ENTITY") or None,
            reinit=True,
        )
        for stage, m in metrics.items():
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    wandb.log({f"{stage}/{k}": v})
        wandb.finish()
    except Exception as exc:
        print(f"[two-phase] wandb stage metrics skipped: {exc}")


def _compare_three(c0: str, c1: str, c2: str, out_dir: str) -> dict:
    cmd = [sys.executable, _FINAL_EVAL, "compare",
           "--baseline", c0, "--best", c1, "--final", c2, "--out", out_dir]
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit("[two-phase] compare failed")
    return _load_json(os.path.join(out_dir, "final_compare_report.json"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--phase1-dataset", default=_DEFAULT_PHASE1)
    ap.add_argument("--phase2-dataset", default=_DEFAULT_PHASE2)
    ap.add_argument("--dev-set", default=_DEFAULT_DEV)
    ap.add_argument("--config", default=_DEFAULT_CONFIG)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=3e-7)
    ap.add_argument("--kl-beta", type=float, default=0.15)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--reward-policy", default="execution_aware_v3_2_dense")
    ap.add_argument("--dev-max-tasks", type=int, default=0)
    ap.add_argument("--max-train-tasks", type=int, default=0)
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--skip-baseline-eval", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="phase-level resume: skip completed steps (NOT exact "
                         "optimizer-step resume)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seed = int(os.environ.get("SEED", "42"))
    data_seed = int(os.environ.get("DATA_SEED", "42"))
    rollout_seed = int(os.environ.get("ROLLOUT_SEED", "42"))

    run_dir = os.path.abspath(args.run_dir)
    phase1 = os.path.abspath(args.phase1_dataset)
    phase2 = os.path.abspath(args.phase2_dataset)
    dev_set = os.path.abspath(args.dev_set)
    cur_hash, cur_ver = _current_registry_hash()

    assert_canonical_training(
        executor_mode="synthetic",
        reward_policy=args.reward_policy,
        registry_version=cur_ver,
    )

    p1_sha, p2_sha = _sha256_file(phase1), _sha256_file(phase2)
    manifest = collect_repro_manifest(
        git=_git_state(),
        registry_version=cur_ver,
        registry_hash=cur_hash,
        datasets=[(phase1, p1_sha), (phase2, p2_sha)],
    )
    manifest.update({
        "kind": "two_phase_v5_grpo",
        "created_at": _now(),
        "run_dir": run_dir,
        "dev_set": dev_set,
        "dev_test_audit": verify_dev_test_disjoint(dev_set),
        "dataset_id_audit": {
            "phase1": audit_dataset_ids(phase1),
            "phase2": audit_dataset_ids(phase2),
        },
        "hyperparameters": manifest.get("hyperparameters") or {
            "num_generations": args.num_generations,
            "learning_rate": args.learning_rate,
            "kl_beta": args.kl_beta,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "epochs_per_phase": 1,
            "executor_mode": "synthetic",
            "reward_policy": args.reward_policy,
        },
        "training_mode": "continuous_in_process_two_phase",
        "eval_order": "C0_before_training_C1_C2_after_teardown",
        "resume_kind": "phase_level_only",
        "eval_decoding": {"temperature": 0.0, "top_p": 1.0, "num_rollouts": 1,
                          "paradigm": "react"},
        "wandb_project": os.environ.get("WANDB_PROJECT", "nestful-v5-curriculum"),
        "wandb_entity": os.environ.get("WANDB_ENTITY"),
    })
    if not manifest["dev_test_audit"]["ok"]:
        raise SystemExit(f"[two-phase] dev/test hygiene failed: "
                         f"{manifest['dev_test_audit']}")
    for label, aud in manifest["dataset_id_audit"].items():
        if not aud["ok"]:
            raise SystemExit(f"[two-phase] duplicate sample_id in {label}: {aud}")

    print("[two-phase] --- resolved configuration ---")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if args.dry_run:
        return 0

    os.makedirs(run_dir, exist_ok=True)
    logs = os.path.join(run_dir, "logs")
    os.makedirs(logs, exist_ok=True)

    state = _load_state(run_dir) if args.resume else {
        "version": "two_phase_v3", "steps": {}, "resume_kind": "phase_level"}
    if not args.resume and state.get("steps"):
        raise SystemExit("[two-phase] run_dir has state; use --resume or fresh dir")

    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if not args.resume:
        _save_json(manifest_path, manifest)
    wandb_group = os.environ.get("WANDB_GROUP") or os.environ.get(
        "WANDB_RUN_GROUP") or os.path.basename(run_dir)
    os.environ.setdefault("WANDB_PROJECT", "nestful-v5-curriculum")
    os.environ["WANDB_RUN_GROUP"] = wandb_group

    train_overrides = _build_train_overrides(args)
    session: Optional[TwoPhaseTrainSession] = None
    managed_pids: List[int] = []
    c1_link = os.path.join(run_dir, "checkpoints", "C1")
    c2_link = os.path.join(run_dir, "checkpoints", "C2")
    c0_dir = os.path.join(run_dir, "eval", "C0_baseline")

    try:
        # ── 1 Preflight ───────────────────────────────────────────────────
        if not args.skip_preflight and not _step_done(state, "preflight"):
            report = _run_preflight(phase1, phase2,
                                    os.path.join(run_dir, "preflight_report.json"))
            _mark_step(state, "preflight", report=report)
            _save_state(run_dir, state)

        # ── 2 C0 eval (no training session; all GPUs free) ─────────────────
        if not args.skip_baseline_eval and not _step_done(state, "eval_C0"):
            m = _dev_eval_subprocess(
                config=args.config, checkpoint=None, dev_set=dev_set,
                out_dir=c0_dir, max_tasks=args.dev_max_tasks,
                log_path=os.path.join(logs, "eval_C0.log"),
                label="C0_baseline", wandb_run_name=f"{wandb_group}-eval-C0",
            )
            _mark_step(state, "eval_C0", metrics=m, out_dir=c0_dir)
            _save_state(run_dir, state)

        # ── 3–7 Continuous training session (Phase 1 → C1 → Phase 2 → C2) ──
        need_training = (
            not _step_done(state, "phase1_train")
            or not _step_done(state, "phase2_train")
        )
        resume_phase2_only = (
            args.resume
            and _step_done(state, "phase1_train")
            and not _step_done(state, "phase2_train")
        )

        if need_training:
            if resume_phase2_only:
                discard_incomplete_checkpoint(c2_link)
                _record_resume_event(
                    state,
                    "phase2_restart",
                    from_checkpoint=c1_link if os.path.isdir(c1_link) else
                    state["steps"]["phase1_train"].get("checkpoint"),
                    reason="phase_level_resume_fresh_optimizer",
                )
                _save_state(run_dir, state)
                print(
                    "[two-phase] RESUME: Phase 1 complete — loading C1, "
                    "discarded incomplete C2, fresh optimizer for Phase 2 "
                    "(not exact step resume)",
                    flush=True,
                )

            session = TwoPhaseTrainSession(
                args.config, train_overrides,
                seed=seed, data_seed=data_seed, rollout_seed=rollout_seed,
            )

            init_ckpt = None
            if resume_phase2_only:
                init_ckpt = state["steps"]["phase1_train"]["checkpoint"]
                session.global_step = int(
                    state["steps"]["phase1_train"].get("global_step") or 0)
            session.load_learner(checkpoint=init_ckpt)

            p1_out = os.path.join(run_dir, "phase1", "train", "epoch_1")
            phase1_optimizer_id: Optional[int] = None

            # ── Phase 1 ───────────────────────────────────────────────────
            if not _step_done(state, "phase1_train"):
                managed_pids = session.start_rollout_workers(adapter_path=None)
                _save_json(os.path.join(logs, "rollout_worker_pids.json"), managed_pids)
                adapter, summary = session.train_phase(
                    dataset_path=phase1,
                    train_out=p1_out,
                    phase_name="phase1_stage2",
                    max_train_tasks=args.max_train_tasks,
                    expected_rows=PHASE1_ROWS,
                    wandb_run_name=f"{wandb_group}-phase1-train",
                )
                phase1_optimizer_id = id(session.optimizer)
                c1_manifest = atomic_publish_checkpoint(adapter, c1_link, label="C1")
                session.sync_rollout_policy(c1_link, label="C1")
                _mark_step(
                    state, "phase1_train",
                    checkpoint=c1_link,
                    checkpoint_manifest=c1_manifest,
                    summary=summary,
                    rollout_pids=managed_pids,
                    global_step=session.global_step,
                    optimizer_id=phase1_optimizer_id,
                )
                _save_state(run_dir, state)
                print(
                    f"[two-phase] C1 -> {c1_link} "
                    f"(global_step={session.global_step}, "
                    f"optimizer_id={phase1_optimizer_id})",
                    flush=True,
                )

            c1 = state["steps"]["phase1_train"]["checkpoint"]

            # ── Phase 2 (same session; workers stay up when continuous) ───
            p2_out = os.path.join(run_dir, "phase2", "train", "epoch_1")
            if not _step_done(state, "phase2_train"):
                if resume_phase2_only:
                    managed_pids = session.start_rollout_workers(adapter_path=c1)
                    _save_json(os.path.join(logs, "rollout_worker_pids.json"), managed_pids)
                elif session.rollout_pool is None and session.vllm_gen is None:
                    managed_pids = session.start_rollout_workers(adapter_path=c1)
                    _save_json(os.path.join(logs, "rollout_worker_pids.json"), managed_pids)

                session.sync_rollout_policy(c1, label="C1")

                p1_opt = state["steps"]["phase1_train"].get("optimizer_id")
                opt_at_p2_start = id(session.optimizer)
                if p1_opt is not None and not resume_phase2_only:
                    if opt_at_p2_start != p1_opt:
                        raise SystemExit(
                            "[two-phase] optimizer object changed between phases "
                            f"({p1_opt} -> {opt_at_p2_start})")
                    print(
                        f"[two-phase] continuous optimizer confirmed "
                        f"(optimizer_id={opt_at_p2_start})",
                        flush=True,
                    )

                adapter, summary = session.train_phase(
                    dataset_path=phase2,
                    train_out=p2_out,
                    phase_name="phase2_stage3_replay",
                    max_train_tasks=args.max_train_tasks,
                    expected_rows=PHASE2_ROWS,
                    wandb_run_name=f"{wandb_group}-phase2-train",
                )
                opt_at_p2_end = id(session.optimizer)
                c2_manifest = atomic_publish_checkpoint(adapter, c2_link, label="C2")
                stage_metrics = _stage_metrics_from_log(
                    os.path.join(p2_out, "train_log.jsonl"), phase2)
                _save_json(os.path.join(run_dir, "phase2", "stage_split_metrics.json"),
                           stage_metrics)
                _log_stage_metrics_wandb(wandb_group, stage_metrics)

                shutdown_pids = session.shutdown_rollout_workers()
                managed_pids = []
                _save_json(os.path.join(logs, "rollout_worker_pids.json"), [])

                _mark_step(
                    state, "phase2_train",
                    checkpoint=c2_link,
                    checkpoint_manifest=c2_manifest,
                    summary=summary,
                    stage_split_metrics=stage_metrics,
                    global_step=session.global_step,
                    continuous_from_phase1=not resume_phase2_only,
                    optimizer_id=opt_at_p2_end,
                    optimizer_unchanged=(
                        p1_opt is not None and opt_at_p2_end == p1_opt
                        and not resume_phase2_only
                    ),
                )
                _save_state(run_dir, state)
                print(
                    f"[two-phase] C2 -> {c2_link} (global_step={session.global_step})",
                    flush=True,
                )

        c1 = state["steps"]["phase1_train"]["checkpoint"]
        c2 = state["steps"]["phase2_train"]["checkpoint"]

        # ── 8 Teardown training session (free all GPUs for eval) ────────────
        if session is not None:
            session.close()
            session = None
            print("[two-phase] training session closed; learner unloaded; "
                  "all GPUs free for eval", flush=True)

        # ── 9 Eval C1 (deferred until after Phase 2) ──────────────────────
        c1_dir = state.get("steps", {}).get("eval_C1", {}).get(
            "out_dir") or os.path.join(run_dir, "eval", "C1_phase1")
        if not _step_done(state, "eval_C1"):
            m = _dev_eval_subprocess(
                config=args.config, checkpoint=c1, dev_set=dev_set,
                out_dir=c1_dir, max_tasks=args.dev_max_tasks,
                log_path=os.path.join(logs, "eval_C1.log"),
                label="C1_phase1", wandb_run_name=f"{wandb_group}-eval-C1",
            )
            _mark_step(state, "eval_C1", metrics=m, out_dir=c1_dir)
            _save_state(run_dir, state)

        # ── 10 Eval C2 ────────────────────────────────────────────────────
        c2_dir = state.get("steps", {}).get("eval_C2", {}).get(
            "out_dir") or os.path.join(run_dir, "eval", "C2_phase2")
        if not _step_done(state, "eval_C2"):
            m = _dev_eval_subprocess(
                config=args.config, checkpoint=c2, dev_set=dev_set,
                out_dir=c2_dir, max_tasks=args.dev_max_tasks,
                log_path=os.path.join(logs, "eval_C2.log"),
                label="C2_phase2", wandb_run_name=f"{wandb_group}-eval-C2",
            )
            _mark_step(state, "eval_C2", metrics=m, out_dir=c2_dir)
            _save_state(run_dir, state)

        # ── 11 Compare C0 / C1 / C2 ───────────────────────────────────────
        if not _step_done(state, "compare"):
            if not _step_done(state, "eval_C0") and args.skip_baseline_eval:
                print("[two-phase] skipping compare (no C0)")
            else:
                c0 = state.get("steps", {}).get("eval_C0", {}).get(
                    "out_dir") or c0_dir
                out_cmp = os.path.join(run_dir, "eval", "compare_C0_C1_C2")
                report = _compare_three(c0, c1_dir, c2_dir, out_cmp)
                _mark_step(state, "compare", report_path=os.path.join(
                    out_cmp, "final_compare_report.json"))
                if os.environ.get("WANDB_PROJECT"):
                    try:
                        import wandb
                        wr = wandb.init(project=os.environ["WANDB_PROJECT"],
                                        name=f"{wandb_group}-compare",
                                        group=wandb_group,
                                        entity=os.environ.get("WANDB_ENTITY") or None,
                                        reinit=True)
                        for arm in ("baseline", "best", "final"):
                            off = report.get("aggregate_official", {}).get(arm, {})
                            for k, v in off.items():
                                if isinstance(v, (int, float)):
                                    wandb.log({f"compare/{arm}/{k}": v})
                        wandb.finish()
                    except Exception:
                        pass
                _save_state(run_dir, state)

    finally:
        if session is not None:
            session.close()

    print("[two-phase] COMPLETE")
    print(f"  manifest : {manifest_path}")
    print(f"  state    : {_state_path(run_dir)}")
    print(f"  mode     : continuous_in_process (C1/C2 eval after teardown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
