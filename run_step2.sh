#!/bin/bash
set -euo pipefail

GEN_CKPT=$1
SOLVER_NAME=$2
OUT_JSON=$3
STEP_NAME="step2_data_pipeline"
STEP_DIR="${TOOL_R0_RUN_DIR:-.}/iter${TOOL_R0_ITERATION:-unknown}/${STEP_NAME}"
mkdir -p "$STEP_DIR"
export TOOL_R0_STEP_NAME="$STEP_NAME"
export TOOL_R0_STEP_DIR="$STEP_DIR"
STEP_LOG_FILE="${STEP_DIR}/step.log"
exec > >(tee -a "$STEP_LOG_FILE") 2>&1
echo "Step dir: $STEP_DIR"
echo "Generator checkpoint: $GEN_CKPT"
echo "Solver model: $SOLVER_NAME"
echo "Output json: $OUT_JSON"

# Defaults match run_main.sh (3× compute GPU); override if needed.
STEP2_GPUS="${STEP2_GPUS:-0,1,2}"
STEP2_TP="${STEP2_TP:-1}"
STEP2_N_GENERATE="${STEP2_N_GENERATE:-4000}"
STEP2_VERIFY_BATCH_SIZE="${STEP2_VERIFY_BATCH_SIZE:-16}"
STEP2_JUDGE_BATCH_SIZE="${STEP2_JUDGE_BATCH_SIZE:-16}"
STEP2_GPU_MEM_UTIL="${STEP2_GPU_MEM_UTIL:-0.75}"

export CUDA_VISIBLE_DEVICES="$STEP2_GPUS"
echo "Step2 GPUs: $CUDA_VISIBLE_DEVICES"
echo "Step2 TP: $STEP2_TP"
echo "Step2 n_generate: $STEP2_N_GENERATE"
echo "Step2 verify_batch_size: $STEP2_VERIFY_BATCH_SIZE"
echo "Step2 judge_batch_size: $STEP2_JUDGE_BATCH_SIZE"
echo "Step2 gpu_memory_utilization: $STEP2_GPU_MEM_UTIL"

INTERMEDIATE_JSON="${OUT_JSON}.intermediate.json"
VERIFIED_JSON="${OUT_JSON}.intermediate.verified.json"
VERIFY_REPORT="${OUT_JSON}.verify_report.txt"

# Auto-fix checkpoints for vLLM before loading (handles weight key remap,
# missing processor configs, tokenizer_class issues).
BASE_HUB="${TOOL_R0_BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
echo "Pre-flight: fixing checkpoints for vLLM compatibility..."
python -c "
from grpo_processing import fix_checkpoint_for_vllm
import os, sys
for ckpt in sys.argv[1:]:
    if os.path.isdir(ckpt):
        print(f'[preflight] Fixing {ckpt}')
        fix_checkpoint_for_vllm(ckpt, '$BASE_HUB')
    else:
        print(f'[preflight] Not a dir, skipping: {ckpt}')
" "$GEN_CKPT" "$SOLVER_NAME"

python step2_gen.py \
    --generator_model "$GEN_CKPT" \
    --tokenizer "$SOLVER_NAME" \
    --out_intermediate_json "$INTERMEDIATE_JSON" \
    --n_generate "$STEP2_N_GENERATE" \
    --max_tokens_gen 4096 \
    --tensor_parallel_size "$STEP2_TP" \
    --gpu_memory_utilization "$STEP2_GPU_MEM_UTIL"


python step2_genverify.py \
    --solver_model "$SOLVER_NAME" \
    --in_intermediate_json "$INTERMEDIATE_JSON" \
    --out_intermediate_json "$VERIFIED_JSON" \
    --report_txt "$VERIFY_REPORT" \
    --k_verify 10 \
    --tau_verify 0.20 \
    --temp_verify 0.001 \
    --max_tokens_verify 1024 \
    --verify_batch_size "$STEP2_VERIFY_BATCH_SIZE" \
    --tensor_parallel_size "$STEP2_TP" \
    --gpu_memory_utilization "$STEP2_GPU_MEM_UTIL"

python step2_select_curriculum.py \
    --judge_model "$SOLVER_NAME" \
    --in_json "$VERIFIED_JSON" \
    --out_json "$OUT_JSON" \
    --n_final 2000 \
    --tensor_parallel_size "$STEP2_TP" \
    --gpu_memory_utilization "$STEP2_GPU_MEM_UTIL" \
    --max_model_len 4096 \
    --batch_size "$STEP2_JUDGE_BATCH_SIZE" \
    --temp_judge 0.0 \
    --max_tokens_judge 10 \
    --mix_easy 0.20 \
    --mix_medium 0.50 \
    --mix_hard 0.30 \
    --seed 13 \
    --default_diff medium