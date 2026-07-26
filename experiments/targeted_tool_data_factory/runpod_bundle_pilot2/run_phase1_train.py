#!/usr/bin/env python3
"""Launch the Phase-1 GRPO canary with the offline-selected reward variant.

Applies ``phase1_reward_patch`` in-process, then runs the stock
``run_reward_ablation.py`` entry point against ``recommended_phase1_train.jsonl``.

Fixed small optimizer budget (1 epoch, grad_accum=4 → 20 steps on 80 tasks).
Never starts full NESTFUL-1661 eval (``--skip-eval``); C0/C1 eval is done by
``run_eval_all.py`` afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
FACTORY = BUNDLE.parent
EXPERIMENTS = FACTORY.parent
V3 = EXPERIMENTS / "nestful_synthetic_curriculum_v3"
ABLATION = V3 / "scripts" / "ablation" / "run_reward_ablation.py"

# 80 tasks / gradient_accumulation_steps 4 = 20 optimizer steps per epoch.
PHASE1_N_TASKS = 80
PHASE1_GRAD_ACCUM = 4
PHASE1_OPTIMIZER_STEPS = PHASE1_N_TASKS // PHASE1_GRAD_ACCUM  # 20
PHASE1_NUM_GENERATIONS = 8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-subset", type=Path, required=True)
    ap.add_argument("--variant-file", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--wandb-project", default="ttdf-pilot2")
    ap.add_argument("--wandb-group", default="pilot2_phase1_canary")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sel = json.loads(args.variant_file.read_text(encoding="utf-8"))
    if not sel.get("selected") or sel.get("hard_gate") == "FAIL":
        print(f"[phase1_train] ABORT: unsafe variant selection: {sel}",
              file=sys.stderr)
        return 2

    train_policy = sel.get("train_policy") or "reward_ablation_A4_GATED_VERIFIABLE"
    # Map train_policy -> arm id expected by run_reward_ablation.
    if train_policy.endswith("A1_OUTCOME_ONLY"):
        reward_arm = "A1_OUTCOME_ONLY"
    else:
        reward_arm = "A4_GATED_VERIFIABLE"

    run_id = args.run_id or f"pilot2_C1_phase1_seed{args.seed}"
    n_rows = sum(1 for line in args.train_subset.read_text(encoding="utf-8").splitlines()
                 if line.strip())
    if n_rows != PHASE1_N_TASKS:
        print(f"[phase1_train] ABORT: expected {PHASE1_N_TASKS} tasks, got {n_rows}",
              file=sys.stderr)
        return 2

    print(f"[phase1_train] variant={sel['selected']} arm={reward_arm} "
          f"policy={train_policy}")
    print(f"[phase1_train] tasks={n_rows} gens={PHASE1_NUM_GENERATIONS} "
          f"grad_accum={PHASE1_GRAD_ACCUM} "
          f"optimizer_steps~={PHASE1_OPTIMIZER_STEPS}")
    print(f"[phase1_train] run_id={run_id}")

    if args.dry_run:
        print("[phase1_train] DRY RUN - training not started")
        return 0

    os.environ["PHASE1_REWARD_VARIANT_FILE"] = str(args.variant_file.resolve())
    os.environ["PHASE1_REWARD_VARIANT"] = str(sel["selected"])
    os.environ["SYNTHETIC_TOOLS_DIR"] = str(FACTORY / "trainer_adapter")
    os.environ["CANARY_TRAJ_LOG"] = "1"
    os.environ["USE_VLLM"] = os.environ.get("USE_VLLM", "1")
    os.environ.setdefault("ROLLOUT_DP_GPUS", "1,2,3")

    # Patch BEFORE the trainer resolves the reward fn.
    sys.path.insert(0, str(BUNDLE))
    from phase1_reward_patch import apply_phase1_reward_variant  # noqa: WPS433
    apply_phase1_reward_variant(sel)

    argv = [
        str(ABLATION),
        "--round", "3",
        "--reward-arm", reward_arm,
        "--seed", str(args.seed),
        "--train-subset", str(args.train_subset),
        "--expected-rows", str(PHASE1_N_TASKS),
        "--skip-eval", "--skip-c0-eval",
        "--output-root", str(args.output_root),
        "--run-id", run_id,
        "--wandb-project", args.wandb_project,
        "--wandb-group", args.wandb_group,
    ]
    if args.resume:
        argv.append("--resume")

    # Persist the canary budget contract next to the run.
    args.output_root.mkdir(parents=True, exist_ok=True)
    budget = {
        "n_tasks": n_rows,
        "num_generations": PHASE1_NUM_GENERATIONS,
        "gradient_accumulation_steps": PHASE1_GRAD_ACCUM,
        "epochs": 1,
        "expected_optimizer_steps": PHASE1_OPTIMIZER_STEPS,
        "variant": sel,
        "run_id": run_id,
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "full_nestful_1661": False,
    }
    (args.output_root / f"{run_id}_budget.json").write_text(
        json.dumps(budget, indent=2), encoding="utf-8")

    sys.argv = argv
    print(f"[phase1_train] + python {' '.join(argv)}", flush=True)
    runpy.run_path(str(ABLATION), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
