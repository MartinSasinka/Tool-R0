#!/bin/bash
set -euo pipefail

MODEL_PATH=$1
DATA_PATH=$2
OUTPUT_DIR=$3
RUN_NAME=$4
STEPS=$5
STEP_NAME="step3_solver_train"
STEP_DIR="${TOOL_R0_RUN_DIR:-$OUTPUT_DIR}/iter${TOOL_R0_ITERATION:-unknown}/${STEP_NAME}"
mkdir -p "$STEP_DIR"
export TOOL_R0_STEP_NAME="$STEP_NAME"
export TOOL_R0_STEP_DIR="$STEP_DIR"
STEP_LOG_FILE="${STEP_DIR}/step.log"
exec > >(tee -a "$STEP_LOG_FILE") 2>&1
echo "Step dir: $STEP_DIR"
echo "Model path: $MODEL_PATH"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Run name: $RUN_NAME"
echo "Max steps: $STEPS"

STEP13_GPUS="${STEP13_GPUS:-0,1,2}"
STEP13_NUM_PROCESSES="${STEP13_NUM_PROCESSES:-3}"
DEEPSPEED_CONFIG="${TOOL_R0_DEEPSPEED_CONFIG:-./configs/deepseed_zero2_offload.yaml}"
STEP3_PER_DEVICE_TRAIN_BATCH_SIZE="${STEP3_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
STEP3_GRADIENT_ACCUMULATION_STEPS="${STEP3_GRADIENT_ACCUMULATION_STEPS:-8}"
STEP3_NUM_GENERATIONS="${STEP3_NUM_GENERATIONS:-2}"
STEP3_MAX_COMPLETION_LENGTH="${STEP3_MAX_COMPLETION_LENGTH:-3072}"

# LoRA (TRL ModelConfig flags). Defaults match Qwen3 LLaMA-style 7 projections.
# Set TOOL_R0_USE_PEFT=false to fall back to full fine-tuning.
USE_PEFT="${TOOL_R0_USE_PEFT:-true}"
LORA_R="${TOOL_R0_LORA_R:-32}"
LORA_ALPHA="${TOOL_R0_LORA_ALPHA:-64}"
LORA_DROPOUT="${TOOL_R0_LORA_DROPOUT:-0.05}"
LORA_TARGETS="${TOOL_R0_LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj}"

PEFT_ARGS=()
if [ "$USE_PEFT" = "true" ]; then
    PEFT_ARGS=(
        --use_peft true
        --lora_r "$LORA_R"
        --lora_alpha "$LORA_ALPHA"
        --lora_dropout "$LORA_DROPOUT"
        --lora_target_modules $LORA_TARGETS
    )
    echo "Step3 LoRA: r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT targets=[$LORA_TARGETS]"
else
    echo "Step3 LoRA: DISABLED (full fine-tuning)"
fi

export CUDA_VISIBLE_DEVICES="$STEP13_GPUS"
echo "Step3 GPUs: $CUDA_VISIBLE_DEVICES"
echo "Step3 num_processes: $STEP13_NUM_PROCESSES"
echo "DeepSpeed config: $DEEPSPEED_CONFIG"
echo "Step3 GRPO: per_device_batch=$STEP3_PER_DEVICE_TRAIN_BATCH_SIZE grad_accum=$STEP3_GRADIENT_ACCUMULATION_STEPS num_generations=$STEP3_NUM_GENERATIONS max_completion_length=$STEP3_MAX_COMPLETION_LENGTH"

accelerate launch \
    --config_file "$DEEPSPEED_CONFIG" --num_processes "$STEP13_NUM_PROCESSES" \
    step3_solver.py \
    --model_name_or_path "$MODEL_PATH" \
    --generated_data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps $STEPS \
    --save_steps 50 \
    --save_total_limit 1 \
    --save_only_model true \
    --per_device_train_batch_size "$STEP3_PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$STEP3_GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate 1e-6 \
    --dtype bfloat16 \
    --max_completion_length "$STEP3_MAX_COMPLETION_LENGTH" \
    --gradient_checkpointing true \
    --log_completions \
    --num_generations "$STEP3_NUM_GENERATIONS" \
    --remove_unused_columns False \
    --loss_type grpo \
    --logging_steps 1 \
    --run_name "$RUN_NAME" \
    --report_to wandb \
    "${PEFT_ARGS[@]}"