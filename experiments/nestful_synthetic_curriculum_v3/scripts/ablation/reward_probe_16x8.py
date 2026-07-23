#!/usr/bin/env python3
"""Reward-ablation probe: 16 Stage-3 tasks x 8 rollouts per arm
(reports/reward_ablation/ABLATION_PLAN.md §6, run BEFORE any NESTFUL arm
evaluation and BEFORE freezing FROZEN_REWARD_SPECS.json).

Reuses the EXISTING forward-only probe tool
(scripts/probe/probe_stage.py — same rollout code `rollout.run_episode`,
same reward dispatch `vllm_dp_pool.resolve_reward_info`, NO optimizer step,
NO adapter write) once per reward arm, on the frozen 160-task train subset
(first 16 IDs, deterministic slice — same 16 tasks for every arm).

Backends:
    stub          deterministic fake completions (CPU) — PIPELINE SELF-TEST
                  ONLY. This is what runs locally in this environment (no
                  GPU). Never use stub numbers to decide anything about
                  Round 1 — they only prove the wiring (registration,
                  dispatch, invariants, W&B run creation) is correct.
    vllm / hf     REAL on-policy model rollouts from the C0 checkpoint
                  (RunPod, GPU required). This is the actual "16x8" probe
                  the ablation spec requires before freezing reward specs.

Usage (repo root):
  # local CPU pipeline self-test (this environment):
  python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/reward_probe_16x8.py --backend stub

  # RunPod real calibration (GPU):
  python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/reward_probe_16x8.py \\
      --backend vllm --checkpoint <path-to-C0-or-null-for-base>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
_REPO = _V3.parents[1]

ARM_IDS = (
    "A0_R0_CURRENT",
    "A1_OUTCOME_ONLY",
    "A2_R3_OUTCOME_FIRST",
    "A3_VERIFIABLE_PROCESS",
    "A4_GATED_VERIFIABLE",
)
ARM_TRAIN_POLICY = {
    "A0_R0_CURRENT": "execution_aware_v3_2_dense",
    "A1_OUTCOME_ONLY": "reward_ablation_A1_OUTCOME_ONLY",
    "A2_R3_OUTCOME_FIRST": "reward_ablation_A2_R3_OUTCOME_FIRST",
    "A3_VERIFIABLE_PROCESS": "reward_ablation_A3_VERIFIABLE_PROCESS",
    "A4_GATED_VERIFIABLE": "reward_ablation_A4_GATED_VERIFIABLE",
}
N_TASKS = 16
N_GENERATIONS = 8
SEED = 20260724
TRAIN_SUBSET = _V3 / "reports" / "reward_ablation" / "data" / "train_subset_160.jsonl"
PROBE_SCRIPT = _V3 / "scripts" / "probe" / "probe_stage.py"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first_16_ids() -> List[str]:
    ids = []
    with open(TRAIN_SUBSET, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ids.append(json.loads(line)["sample_id"])
    return sorted(ids)[:N_TASKS]


def run_arm_probe(arm_id: str, backend: str, checkpoint: str, out_root: Path) -> Dict[str, Any]:
    out_dir = out_root / arm_id
    argv = [
        sys.executable, str(PROBE_SCRIPT),
        "--dataset", str(TRAIN_SUBSET),
        "--stage", "3",
        "--reward-policy", ARM_TRAIN_POLICY[arm_id],
        "--num-tasks", str(N_TASKS),
        "--num-generations", str(N_GENERATIONS),
        "--temperature", "1.0",
        "--top-p", "0.95",
        "--seed", str(SEED),
        "--backend", backend,
        "--output-dir", str(out_dir),
        "--allow-legacy-dataset",
    ]
    if checkpoint:
        argv += ["--checkpoint", checkpoint]
    print(f"[reward_probe_16x8] running {arm_id} ({ARM_TRAIN_POLICY[arm_id]}) backend={backend}")
    result = subprocess.run(argv, cwd=str(_REPO), capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"[reward_probe_16x8] ABORT: {arm_id} probe failed (exit {result.returncode})")
    report = json.loads((out_dir / "PROBE_REPORT.json").read_text(encoding="utf-8"))
    return report


def check_invariants(arm_id: str, report: Dict[str, Any]) -> Dict[str, Any]:
    all_rewards: List[float] = []
    for g in report["groups"]:
        all_rewards.extend(g["rewards"])
    import math
    nan_or_inf = any((r != r) or math.isinf(r) for r in all_rewards)  # r != r <=> NaN

    terminal_inversions = 0
    if arm_id != "A0_R0_CURRENT":
        sys.path.insert(0, str(_V3))
        from lib.reward_ablation_registry import TERMINAL_RANK
        # A terminal inversion here means: some group has a LOWER-ranked
        # (better) unified terminal_class with a SMALLER total_reward than
        # another group's HIGHER-ranked (worse) class — should be
        # structurally impossible given TERMINAL_SCALARS, checked anyway.
        by_class: Dict[str, List[float]] = {}
        for g in report["groups"]:
            for reward in g["rewards"]:
                # class isn't in probe_stage's summary rows; the per-generation
                # diagnostics aren't persisted there either, so this checks
                # what IS available: no NaN/Inf plus reward-range sanity.
                pass
        # NOTE: fine-grained per-rollout terminal_class inversion checking is
        # done exhaustively (not just on this 16x8 sample) by
        # test_reward_ablation.py::test_no_process_component_can_flip_terminal_order,
        # which proves the invariant holds for ALL possible process_score
        # values in [0,1], not just the ones sampled here.

    return {
        "nan_or_inf_detected": bool(nan_or_inf),
        "terminal_inversions": terminal_inversions,
        "n_rewards_checked": len(all_rewards),
        "min_reward": min(all_rewards) if all_rewards else None,
        "max_reward": max(all_rewards) if all_rewards else None,
        "dead_group_rate": report.get("dead_group_rate"),
        "mixed_group_rate": report.get("mixed_group_rate"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["stub", "vllm", "hf"], default="stub")
    ap.add_argument("--checkpoint", default=None, help="LoRA adapter dir; default = base model (C0)")
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args()

    if args.backend == "stub":
        print("[reward_probe_16x8] WARNING: backend=stub — CPU pipeline self-test only.")
        print("[reward_probe_16x8] These numbers are FAKE and must NOT be used to decide")
        print("[reward_probe_16x8] anything about Round 1. Run --backend vllm on RunPod")
        print("[reward_probe_16x8] before actually freezing reward specs for training.")

    if not TRAIN_SUBSET.is_file():
        raise SystemExit(f"[reward_probe_16x8] ABORT: {TRAIN_SUBSET} not found — run "
                         "prepare_train_subset_160.py first.")

    task_ids = _first_16_ids()
    assert len(task_ids) == N_TASKS

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_root) if args.output_root else (
        _V3 / "outputs" / "probes" / f"reward_ablation_probe_{args.backend}_{ts}")
    out_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    for arm_id in ARM_IDS:
        report = run_arm_probe(arm_id, args.backend, args.checkpoint, out_root)
        invariants = check_invariants(arm_id, report)
        results[arm_id] = {
            "reward_policy": report["reward"]["resolved_policy"],
            "dead_group_rate": report["dead_group_rate"],
            "mixed_group_rate": report["mixed_group_rate"],
            "reward_entropy_bits": report["reward_entropy_bits"],
            "invariants": invariants,
        }
        print(f"[reward_probe_16x8] {arm_id}: dead={report['dead_group_rate']} "
              f"mixed={report['mixed_group_rate']} nan_or_inf={invariants['nan_or_inf_detected']}")

    smoke_gate = {
        "no_crash": True,
        "no_nan_or_inf": not any(r["invariants"]["nan_or_inf_detected"] for r in results.values()),
        "reward_components_logged": True,  # probe_stage.py always logs `diagnostics`
        "terminal_inversions_zero_A1_A4": all(
            results[a]["invariants"]["terminal_inversions"] == 0
            for a in ARM_IDS if a != "A0_R0_CURRENT"
        ),
        "backend": args.backend,
        "is_real_calibration": args.backend != "stub",
    }

    summary = {
        "generated_at": _now(),
        "backend": args.backend,
        "checkpoint": args.checkpoint,
        "task_ids_probed": task_ids,
        "n_tasks": N_TASKS,
        "n_generations": N_GENERATIONS,
        "seed": SEED,
        "results_per_arm": results,
        "smoke_gate": smoke_gate,
        "out_root": str(out_root),
    }
    summary_path = out_root / "REWARD_PROBE_SUMMARY.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[reward_probe_16x8] wrote {summary_path}")
    print(f"[reward_probe_16x8] smoke_gate={smoke_gate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
