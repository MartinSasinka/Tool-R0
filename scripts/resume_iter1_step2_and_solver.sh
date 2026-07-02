#!/bin/bash
# Resume iter1 from step2 (data pipeline) then step3 (solver GRPO).
# Prerequisite: vLLM >= 0.10 and iter1 generator checkpoint already exists.
set -euo pipefail
cd "$(dirname "$0")/.."

ABBREVIATION="qwen3-4b-tool-r0"
BASE_DIR="./${ABBREVIATION}"
RUN_ID=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${BASE_DIR}/runs/${RUN_ID}"
mkdir -p "$RUN_DIR"

# ─── W&B + logging ──────────────────────────────────────────────────
export WANDB_DISABLED=false
export WANDB_PROJECT="self-play-${ABBREVIATION}"
export TOOL_R0_RUN_ID="$RUN_ID"
export TOOL_R0_RUN_DIR="$RUN_DIR"
export TOOL_R0_TRACE_SAMPLES_PER_STEP="${TOOL_R0_TRACE_SAMPLES_PER_STEP:-2}"
export TOOL_R0_TRACE_TEXT_LIMIT="${TOOL_R0_TRACE_TEXT_LIMIT:-4000}"
export TOOL_R0_ITERATION="${TOOL_R0_ITERATION:-1}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# ─── GPU layout (3 compute GPUs: 0,1,2) ─────────────────────────────
export STEP2_GPUS="${STEP2_GPUS:-0,1,2}"
export STEP2_TP="${STEP2_TP:-1}"
export STEP2_VERIFY_BATCH_SIZE="${STEP2_VERIFY_BATCH_SIZE:-16}"
export STEP2_JUDGE_BATCH_SIZE="${STEP2_JUDGE_BATCH_SIZE:-16}"
export STEP2_GPU_MEM_UTIL="${STEP2_GPU_MEM_UTIL:-0.75}"
export STEP13_GPUS="${STEP13_GPUS:-0,1,2}"
export STEP13_NUM_PROCESSES="${STEP13_NUM_PROCESSES:-3}"

# ─── DeepSpeed + GRPO hyper-params ──────────────────────────────────
export TOOL_R0_DEEPSPEED_CONFIG="${TOOL_R0_DEEPSPEED_CONFIG:-./configs/deepseed_zero2_offload.yaml}"
export STEP3_PER_DEVICE_TRAIN_BATCH_SIZE="${STEP3_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export STEP3_GRADIENT_ACCUMULATION_STEPS="${STEP3_GRADIENT_ACCUMULATION_STEPS:-8}"
export STEP3_NUM_GENERATIONS="${STEP3_NUM_GENERATIONS:-2}"
export STEP3_MAX_COMPLETION_LENGTH="${STEP3_MAX_COMPLETION_LENGTH:-3072}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ─── NCCL / Torch distributed ───────────────────────────────────────
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TRITON_CACHE_DIR="./triton_autotune"
mkdir -p "$TRITON_CACHE_DIR"

# ─── Paths ───────────────────────────────────────────────────────────
GEN_CKPT="${GEN_CKPT:-./qwen3-4b-tool-r0/iter1_generator/checkpoint-50}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
OUT_JSON="${OUT_JSON:-./qwen3-4b-tool-r0/iter1_data.json}"
SOLVER_DIR="${SOLVER_DIR:-./qwen3-4b-tool-r0/iter1_solver}"
STEPS="${STEPS:-50}"
WANDB_NAME="${WANDB_NAME:-${ABBREVIATION}-iter1-solver}"

RUN_LOG="${RUN_DIR}/resume_iter1.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "================================================================"
echo "Resume iter1: step2 + step3"
echo "Generator checkpoint: $GEN_CKPT"
echo "Base model:           $BASE_MODEL"
echo "Output data:          $OUT_JSON"
echo "Solver output:        $SOLVER_DIR"
echo "Run dir:              $RUN_DIR"
echo "================================================================"

echo "=== Step 2: Data Pipeline ==="
bash run_step2.sh "$GEN_CKPT" "$BASE_MODEL" "$OUT_JSON"

echo "=== Step 3: Solver GRPO Training ==="
bash run_step3.sh \
  "$BASE_MODEL" \
  "$OUT_JSON" \
  "$SOLVER_DIR" \
  "$WANDB_NAME" \
  "$STEPS"

echo "================================================================"
echo "Done. Solver checkpoint: ${SOLVER_DIR}/checkpoint-${STEPS}"
echo "================================================================"
