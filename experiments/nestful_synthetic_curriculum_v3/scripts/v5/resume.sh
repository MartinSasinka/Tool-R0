#!/usr/bin/env bash
# 6/9 — Resume an interrupted v5 run.
#
# The source checkpoint is read from $RUN_DIR/pipeline_state.json and printed;
# dataset / executor mode / reward policy are cross-checked against the run
# manifest and the invocation aborts on any mismatch.
#
# Env:  PYTHON=python3
#       DATASET=   (required) same dataset the run was started with
#       RUN_DIR=   (required) existing run directory
#       EPOCHS=3  MIN_EPOCHS=2  PATIENCE=0  NUM_GENERATIONS=8
#       REWARD_POLICY=execution_aware_v3_2_dense
#       DEV_MAX_TASKS=0
#       USE_VLLM=1  ROLLOUT_DP_GPUS=  EVAL_TP=  VLLM_GPU_UTIL=
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DATASET="${DATASET:?set DATASET (same file the run was started with)}"
RUN_DIR="${RUN_DIR:?set RUN_DIR to the existing run directory}"
require_file "$DATASET" "DATASET"
require_file "$RUN_DIR/pipeline_state.json" "pipeline_state.json (nothing to resume)"

banner "resume run"
print_env PYTHON DATASET RUN_DIR EPOCHS MIN_EPOCHS PATIENCE NUM_GENERATIONS \
          REWARD_POLICY DEV_MAX_TASKS USE_VLLM ROLLOUT_DP_GPUS EVAL_TP VLLM_GPU_UTIL

cd "$V3"
"$PY" scripts/training/run_v5_pipeline.py \
  --dataset "$DATASET" \
  --run-dir "$RUN_DIR" \
  --resume \
  --epochs "${EPOCHS:-3}" \
  --min-epochs "${MIN_EPOCHS:-2}" \
  --patience "${PATIENCE:-0}" \
  --num-generations "${NUM_GENERATIONS:-8}" \
  --reward-policy "${REWARD_POLICY:-execution_aware_v3_2_dense}" \
  --dev-max-tasks "${DEV_MAX_TASKS:-0}" \
  --executor-mode synthetic

banner "resume finished: $RUN_DIR"
