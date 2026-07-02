#!/bin/bash
# Smoke test for the curriculum GRPO training pipeline.
#
# Runs all 6 stages with:
#   - 3 gradient steps per training epoch  (instead of full epoch)
#   - 5 NESTFUL tasks per evaluation       (instead of 400-600)
#   - 1 epoch per stage, no gating         (just check the pipeline runs)
#   - no baseline eval (uses zeros)
#
# Expected runtime on 2x A100: ~15-25 minutes total.
# If all stages complete and checkpoints.json is written, the pipeline is healthy.
#
# Usage:
#   bash curricullum/run_smoke_test.sh
#   bash curricullum/run_smoke_test.sh --stages 1,2   # test only stages 1-2
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

CONFIG="${CONFIG:-curricullum/train/configs/qwen3_4b_curriculum_v2.yaml}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
WANDB_PROJECT="${WANDB_PROJECT:-nestful-curriculum-smoketest}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-smoke-$(date +%Y%m%d-%H%M)}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
DEEPSPEED_CONFIG="${TOOL_R0_DEEPSPEED_CONFIG:-./configs/deepseed_zero2_offload.yaml}"
NESTFUL_PATH="${NESTFUL_PATH:-eval/data/NESTFUL-main/data_v2/nestful_data.jsonl}"
CALL_DIST_PATH="${CALL_DIST_PATH:-helper_calculations/output/nestful_call_distribution.json}"
STAGES_ARG="${1:-}"   # optional: --stages 1,2

# Validate required data files
if [[ ! -f "$CONFIG" ]]; then
  echo "[smoke] ERROR: config not found: $CONFIG"
  exit 1
fi
if [[ ! -f "$NESTFUL_PATH" ]]; then
  echo "[smoke] ERROR: NESTFUL data not found: $NESTFUL_PATH"
  exit 1
fi
if [[ ! -f "$CALL_DIST_PATH" ]]; then
  echo "[smoke] ERROR: call distribution not found: $CALL_DIST_PATH"
  exit 1
fi

# Check synthetic data
for e in 1 2 3 4 5 6; do
  DPATH="curricullum/data/filtered_toolr0_synthetic/epoch_${e}_${e}call.jsonl"
  if [[ ! -f "$DPATH" ]]; then
    echo "[smoke] WARNING: missing synthetic data for stage ${e}: $DPATH"
  fi
done

echo ""
echo "=========================================================="
echo "  SMOKE TEST: Curriculum GRPO Pipeline"
echo "  config:   $CONFIG"
echo "  model:    $MODEL_NAME"
echo "  wandb:    $WANDB_PROJECT / $WANDB_RUN_GROUP"
echo "  stages:   ${STAGES_ARG:-(all)}"
echo "  mode:     3 grad steps, 5 eval tasks, 1 epoch/stage"
echo "=========================================================="
echo ""

STAGES_FLAG=()
if [[ -n "$STAGES_ARG" ]]; then
  STAGES_FLAG=(--stages "$STAGES_ARG")
fi

python -u curricullum/train/run_curriculum_training.py \
  --config "$CONFIG" \
  --wandb_project "$WANDB_PROJECT" \
  --run_group "$WANDB_RUN_GROUP" \
  --nestful_path "$NESTFUL_PATH" \
  --call_dist_path "$CALL_DIST_PATH" \
  --num_processes "$NUM_PROCESSES" \
  --deepspeed_config "$DEEPSPEED_CONFIG" \
  --smoke_test \
  "${STAGES_FLAG[@]}"

echo ""
echo "=========================================================="
echo "  SMOKE TEST COMPLETE"
echo ""
echo "  Check outputs:"
echo "    curricullum/checkpoints/qwen3_4b_curriculum_v2/checkpoints.json"
echo "    curricullum/training/results/"
echo "    curricullum/training/logs/"
echo "=========================================================="
