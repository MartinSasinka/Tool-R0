#!/usr/bin/env bash
# 5/9 — FULL Stage-2 training on curriculum v5 (real synthetic executor).
#
# One dataset, one executor mode, one reward policy per run; a run manifest
# and per-epoch deterministic dev evals land in $RUN_DIR. This launches a
# REAL multi-hour training run — it must be invoked deliberately.
#
# Env:  PYTHON=python3
#       DATASET=      (required) curriculum-v5 stage-2 JSONL
#       RUN_DIR=      (required, must be fresh)
#       EPOCHS=3  MIN_EPOCHS=2  PATIENCE=0 (0 = no early stopping)
#       NUM_GENERATIONS=8
#       REWARD_POLICY=execution_aware_v3_2_dense
#       CHECKPOINT_IN=        optional adapter to initialise from
#       DEV_MAX_TASKS=0       0 = full dev set
#       LEARNING_RATE=  KL_BETA=      optional overrides
#       USE_VLLM=1  ROLLOUT_DP_GPUS=1,2,3  EVAL_TP=4  VLLM_GPU_UTIL=0.85
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATASET="${DATASET:?set DATASET to the stage-2 curriculum_v5 JSONL}"
RUN_DIR="${RUN_DIR:?set RUN_DIR to a fresh run directory}"
require_file "$DATASET" "DATASET"
if [ -f "$RUN_DIR/pipeline_state.json" ]; then
  echo "[v5] ERROR: $RUN_DIR already contains a run; use resume.sh" >&2
  exit 1
fi

banner "stage-2 FULL training"
print_env PYTHON DATASET RUN_DIR EPOCHS MIN_EPOCHS PATIENCE NUM_GENERATIONS \
          REWARD_POLICY CHECKPOINT_IN DEV_MAX_TASKS LEARNING_RATE KL_BETA \
          USE_VLLM ROLLOUT_DP_GPUS EVAL_TP VLLM_GPU_UTIL

ARGS=(scripts/training/run_v5_pipeline.py
  --dataset "$DATASET"
  --run-dir "$RUN_DIR"
  --epochs "${EPOCHS:-3}"
  --min-epochs "${MIN_EPOCHS:-2}"
  --patience "${PATIENCE:-0}"
  --num-generations "${NUM_GENERATIONS:-8}"
  --reward-policy "${REWARD_POLICY:-execution_aware_v3_2_dense}"
  --dev-max-tasks "${DEV_MAX_TASKS:-0}"
  --executor-mode synthetic)
if [ -n "${CHECKPOINT_IN:-}" ]; then
  require_adapter "$CHECKPOINT_IN" "CHECKPOINT_IN"
  ARGS+=(--checkpoint-in "$CHECKPOINT_IN")
fi
if [ -n "${LEARNING_RATE:-}" ]; then ARGS+=(--learning-rate "$LEARNING_RATE"); fi
if [ -n "${KL_BETA:-}" ]; then ARGS+=(--kl-beta "$KL_BETA"); fi

cd "$V3"
"$PY" "${ARGS[@]}"

banner "training finished: $RUN_DIR (see checkpoint_selection.json)"
