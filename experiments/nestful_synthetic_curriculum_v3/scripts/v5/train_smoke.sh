#!/usr/bin/env bash
# 4/9 — Short Stage-2 SMOKE training (tiny task cap, 1 epoch, small dev cap).
#
# Verifies the whole path end-to-end (real synthetic executor, manifest,
# dev eval, selection) in minutes, NOT a real training run.
#
# Env:  PYTHON=python3
#       DATASET=      (required) curriculum-v5 stage-2 JSONL
#       RUN_DIR=      (default <v3>/outputs/runs/v5_smoke_<ts>)
#       SMOKE_TASKS=8       train tasks for the smoke epoch
#       DEV_MAX_TASKS=20    dev tasks for the deterministic eval
#       NUM_GENERATIONS=4
#       USE_VLLM=0  ROLLOUT_DP_GPUS=  EVAL_TP=  VLLM_GPU_UTIL=
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATASET="${DATASET:?set DATASET to a curriculum_v5 stage JSONL}"
RUN_DIR="${RUN_DIR:-$V3/outputs/runs/v5_smoke_$(date +%Y%m%d_%H%M%S)}"
require_file "$DATASET" "DATASET"

banner "stage-2 smoke training"
print_env PYTHON DATASET RUN_DIR SMOKE_TASKS DEV_MAX_TASKS NUM_GENERATIONS \
          USE_VLLM ROLLOUT_DP_GPUS EVAL_TP VLLM_GPU_UTIL

cd "$V3"
"$PY" scripts/training/run_v5_pipeline.py \
  --dataset "$DATASET" \
  --run-dir "$RUN_DIR" \
  --epochs 1 \
  --min-epochs 1 \
  --max-train-tasks "${SMOKE_TASKS:-8}" \
  --dev-max-tasks "${DEV_MAX_TASKS:-20}" \
  --num-generations "${NUM_GENERATIONS:-4}" \
  --executor-mode synthetic

banner "smoke run finished: $RUN_DIR"
