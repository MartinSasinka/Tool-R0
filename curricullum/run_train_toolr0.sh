#!/bin/bash
# Train 2-stage Tool-R0 + IBM-exec aligned curriculum (eval format).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

CONFIG="${CONFIG:-curricullum/train/configs/qwen3_4b_lora_grpo_toolr0.yaml}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
WANDB_PROJECT="${WANDB_PROJECT:-nestful-curriculum-toolr0}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-qwen3-4b-toolr0-overnight}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
OVERWRITE="${OVERWRITE:-1}"
MAX_STAGES="${MAX_STAGES:-3}"
TRAINING_FORMAT="${TRAINING_FORMAT:-tool_r0}"
CKPT_ROOT="${CKPT_ROOT:-curricullum/checkpoints/qwen3_4b_lora_grpo_toolr0}"
DEEPSPEED_CONFIG="${TOOL_R0_DEEPSPEED_CONFIG:-./configs/deepseed_zero2_offload.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAX_STEPS="${MAX_STEPS:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
DATA_DIR="${DATA_DIR:-curricullum/data/filtered_toolr0_synthetic}"

export TRAINING_FORMAT

count_jsonl_rows() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo 0
    return
  fi
  python - <<'PY' "$path"
import sys
path = sys.argv[1]
n = 0
with open(path, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            n += 1
print(n)
PY
}

validate_stage_data() {
  local epoch="$1"
  local path="$DATA_DIR/epoch_${epoch}_${epoch}call.jsonl"
  if [[ ! -f "$path" ]]; then
    echo "[err] missing training data: $path"
    exit 1
  fi
  local rows
  rows="$(count_jsonl_rows "$path")"
  if [[ "$rows" -lt 1 ]]; then
    echo "[err] empty training data: $path"
    exit 1
  fi
  echo "[train] epoch ${epoch}: ${rows} task rows (uses all available, no padding)"
}

if [[ ! -f "$DATA_DIR/epoch_1_1call.jsonl" ]]; then
  echo "[err] Run dataset generation first (filtered_toolr0_synthetic/)"
  exit 1
fi

for e in 1 2 3; do
  if [[ "$e" -gt "$MAX_STAGES" ]]; then
    break
  fi
  validate_stage_data "$e"
done

OVERWRITE_FLAG=()
if [[ "$OVERWRITE" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

MAX_STEPS_FLAG=()
if [[ -n "$MAX_STEPS" ]]; then
  MAX_STEPS_FLAG=(--max_steps "$MAX_STEPS")
fi

GRAD_ACCUM_FLAG=()
if [[ -n "$GRADIENT_ACCUMULATION_STEPS" ]]; then
  GRAD_ACCUM_FLAG=(--gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS")
fi

PREV="none"

run_stage() {
  local STAGE_KEY="$1"
  local STAGE_NAME="$2"
  local DATA_PATH="$3"
  local OUTPUT_DIR="$4"
  local RUN_NAME="$5"

  echo ""
  echo "========== $STAGE_KEY ($STAGE_NAME) training_format=$TRAINING_FORMAT =========="

  accelerate launch \
    --config_file "$DEEPSPEED_CONFIG" \
    --num_processes "$NUM_PROCESSES" \
    curricullum/train/train_grpo_stage.py \
    --config "$CONFIG" \
    --stage "$STAGE_KEY" \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --previous_adapter "$PREV" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$RUN_NAME" \
    --wandb_run_group "$WANDB_RUN_GROUP" \
    --num_generations "$NUM_GENERATIONS" \
    --training_format "$TRAINING_FORMAT" \
    "${OVERWRITE_FLAG[@]}" \
    "${MAX_STEPS_FLAG[@]}" \
    "${GRAD_ACCUM_FLAG[@]}"

  PREV="$OUTPUT_DIR"
}

mkdir -p "$CKPT_ROOT"

run_stage stage_1 stage1_1call_toolr0 \
  "$DATA_DIR/epoch_1_1call.jsonl" \
  "$CKPT_ROOT/stage1_1call" \
  "${WANDB_RUN_GROUP}-stage1"

if [[ "$MAX_STAGES" -ge 2 ]]; then
  run_stage stage_2 stage2_2call_toolr0 \
    "$DATA_DIR/epoch_2_2call.jsonl" \
    "$CKPT_ROOT/stage2_2call" \
    "${WANDB_RUN_GROUP}-stage2"
fi

if [[ "$MAX_STAGES" -ge 3 ]]; then
  run_stage stage_3 stage3_3call_toolr0 \
    "$DATA_DIR/epoch_3_3call.jsonl" \
    "$CKPT_ROOT/stage3_3call" \
    "${WANDB_RUN_GROUP}-stage3"
fi

echo ""
if [[ "$MAX_STAGES" -ge 3 ]]; then
  echo "Done. Final adapter: $CKPT_ROOT/stage3_3call"
elif [[ "$MAX_STAGES" -ge 2 ]]; then
  echo "Done. Final adapter: $CKPT_ROOT/stage2_2call"
else
  echo "Done. Final adapter: $CKPT_ROOT/stage1_1call"
fi
echo "Eval: bash curricullum/evaluation/run_toolr0_eval.sh"
