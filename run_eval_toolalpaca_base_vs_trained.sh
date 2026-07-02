#!/bin/bash

set -euo pipefail

: "${BASE_MODEL:?BASE_MODEL is required}"
: "${TRAINED_MODEL:?TRAINED_MODEL is required}"

SIM_DATASET_PATH="${SIM_DATASET_PATH:-./data/eval_simulated.json}"
REAL_DATASET_PATH="${REAL_DATASET_PATH:-./data/eval_real.json}"
RUN_REAL_EVAL="${RUN_REAL_EVAL:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/toolalpaca_eval}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
VLLM_USE_V1="${VLLM_USE_V1:-0}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

mkdir -p "$OUTPUT_DIR"

export VLLM_USE_V1
export VLLM_WORKER_MULTIPROC_METHOD

echo "============================================================"
echo "ToolAlpaca base-vs-trained compare run"
echo "Primary split: $SIM_DATASET_PATH (headline)"
echo "Secondary split (optional): $REAL_DATASET_PATH (supplementary)"
echo "Base model:    $BASE_MODEL"
echo "Trained model: $TRAINED_MODEL"
echo "Output dir:    $OUTPUT_DIR"
echo "VLLM_USE_V1: $VLLM_USE_V1"
echo "VLLM_WORKER_MULTIPROC_METHOD: $VLLM_WORKER_MULTIPROC_METHOD"
echo "============================================================"

python ./scripts/eval_toolalpaca.py \
  --model_path "$BASE_MODEL" \
  --dataset_path "$SIM_DATASET_PATH" \
  --output_path "$OUTPUT_DIR/base_simulated_toolalpaca_eval.json" \
  --table_path "$OUTPUT_DIR/base_simulated_toolalpaca_eval.md" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --max_model_len "$MAX_MODEL_LEN"

python ./scripts/eval_toolalpaca.py \
  --model_path "$TRAINED_MODEL" \
  --dataset_path "$SIM_DATASET_PATH" \
  --output_path "$OUTPUT_DIR/trained_simulated_toolalpaca_eval.json" \
  --table_path "$OUTPUT_DIR/trained_simulated_toolalpaca_eval.md" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --max_model_len "$MAX_MODEL_LEN"

python ./scripts/compare_eval_results.py \
  --base_result "$OUTPUT_DIR/base_simulated_toolalpaca_eval.json" \
  --trained_result "$OUTPUT_DIR/trained_simulated_toolalpaca_eval.json" \
  --output_path "$OUTPUT_DIR/toolalpaca_comparison_simulated.json" \
  --table_path "$OUTPUT_DIR/toolalpaca_comparison_simulated.md"

if [[ "$RUN_REAL_EVAL" == "1" ]]; then
  python ./scripts/eval_toolalpaca.py \
    --model_path "$BASE_MODEL" \
    --dataset_path "$REAL_DATASET_PATH" \
    --output_path "$OUTPUT_DIR/base_real_toolalpaca_eval.json" \
    --table_path "$OUTPUT_DIR/base_real_toolalpaca_eval.md" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --max_model_len "$MAX_MODEL_LEN"

  python ./scripts/eval_toolalpaca.py \
    --model_path "$TRAINED_MODEL" \
    --dataset_path "$REAL_DATASET_PATH" \
    --output_path "$OUTPUT_DIR/trained_real_toolalpaca_eval.json" \
    --table_path "$OUTPUT_DIR/trained_real_toolalpaca_eval.md" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --max_model_len "$MAX_MODEL_LEN"

  python ./scripts/compare_eval_results.py \
    --base_result "$OUTPUT_DIR/base_real_toolalpaca_eval.json" \
    --trained_result "$OUTPUT_DIR/trained_real_toolalpaca_eval.json" \
    --output_path "$OUTPUT_DIR/toolalpaca_comparison_real.json" \
    --table_path "$OUTPUT_DIR/toolalpaca_comparison_real.md"
fi

echo
echo "Done. Main compare table (headline, simulated split):"
cat "$OUTPUT_DIR/toolalpaca_comparison_simulated.md"

