#!/bin/bash

set -euo pipefail

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

run_eval () {
    local label="$1"
    local model_path="$2"
    local dataset_path="$3"
    local split_tag="$4"
    local output_path="$OUTPUT_DIR/${label}_${split_tag}_toolalpaca_eval.json"
    local table_path="$OUTPUT_DIR/${label}_${split_tag}_toolalpaca_eval.md"

    echo "============================================================"
    echo "Running ToolAlpaca eval: $label ($split_tag)"
    echo "Model:   $model_path"
    echo "Dataset: $dataset_path"
    echo "Output:  $output_path"
    echo "VLLM_USE_V1: $VLLM_USE_V1"
    echo "VLLM_WORKER_MULTIPROC_METHOD: $VLLM_WORKER_MULTIPROC_METHOD"
    echo "============================================================"

    python ./scripts/eval_toolalpaca.py \
        --model_path "$model_path" \
        --dataset_path "$dataset_path" \
        --output_path "$output_path" \
        --table_path "$table_path" \
        --batch_size "$BATCH_SIZE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        --max_model_len "$MAX_MODEL_LEN"
}

if [[ -n "${MODEL_PATH:-}" ]]; then
    run_eval "single" "$MODEL_PATH" "$SIM_DATASET_PATH" "simulated"
    if [[ "$RUN_REAL_EVAL" == "1" ]]; then
        run_eval "single" "$MODEL_PATH" "$REAL_DATASET_PATH" "real"
    fi
    exit 0
fi

if [[ -n "${BASE_MODEL:-}" ]]; then
    run_eval "base" "$BASE_MODEL" "$SIM_DATASET_PATH" "simulated"
    if [[ "$RUN_REAL_EVAL" == "1" ]]; then
        run_eval "base" "$BASE_MODEL" "$REAL_DATASET_PATH" "real"
    fi
fi

if [[ -n "${TRAINED_MODEL:-}" ]]; then
    run_eval "trained" "$TRAINED_MODEL" "$SIM_DATASET_PATH" "simulated"
    if [[ "$RUN_REAL_EVAL" == "1" ]]; then
        run_eval "trained" "$TRAINED_MODEL" "$REAL_DATASET_PATH" "real"
    fi
fi

if [[ -n "${BASE_MODEL:-}" && -n "${TRAINED_MODEL:-}" ]]; then
    python ./scripts/compare_eval_results.py \
        --base_result "$OUTPUT_DIR/base_simulated_toolalpaca_eval.json" \
        --trained_result "$OUTPUT_DIR/trained_simulated_toolalpaca_eval.json" \
        --output_path "$OUTPUT_DIR/toolalpaca_comparison_simulated.json" \
        --table_path "$OUTPUT_DIR/toolalpaca_comparison_simulated.md"

    if [[ "$RUN_REAL_EVAL" == "1" ]]; then
        python ./scripts/compare_eval_results.py \
            --base_result "$OUTPUT_DIR/base_real_toolalpaca_eval.json" \
            --trained_result "$OUTPUT_DIR/trained_real_toolalpaca_eval.json" \
            --output_path "$OUTPUT_DIR/toolalpaca_comparison_real.json" \
            --table_path "$OUTPUT_DIR/toolalpaca_comparison_real.md"
    fi
fi

