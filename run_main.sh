#!/bin/bash
set -euo pipefail

BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
ABBREVIATION="qwen3-4b-tool-r0"
ITERATIONS=3
TRAIN_STEPS_GEN=50
TRAIN_STEPS_SOLVER=50
BASE_DIR="./${ABBREVIATION}"
RUN_ID=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${BASE_DIR}/runs/${RUN_ID}"

export WANDB_DISABLED=false
export WANDB_PROJECT="self-play-$ABBREVIATION"
export TOOL_R0_BASE_MODEL="$BASE_MODEL"
export TOOL_R0_RUN_ID="$RUN_ID"
export TOOL_R0_RUN_DIR="$RUN_DIR"
export TOOL_R0_TRACE_SAMPLES_PER_STEP="${TOOL_R0_TRACE_SAMPLES_PER_STEP:-2}"
export TOOL_R0_TRACE_TEXT_LIMIT="${TOOL_R0_TRACE_TEXT_LIMIT:-4000}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# 3-GPU DGX profile: step0(vLLM solver)=2, step1/3(train)=0,1,2, step2(data)=0,1,2.
# GPU 3 is DGX Display (4GB) — excluded from all workloads.
# With LoRA r=32 on a 4B base the GRPO trainer fits ~14 GB/GPU on bf16 + grad ckpt at
# max_completion_length=3072, so all three compute GPUs participate in DDP.
export STEP0_GPU="${STEP0_GPU:-2}"
export STEP0_TP="${STEP0_TP:-1}"
export STEP0_GPU_MEM_UTIL="${STEP0_GPU_MEM_UTIL:-0.80}"
export STEP13_GPUS="${STEP13_GPUS:-0,1,2}"
export STEP13_NUM_PROCESSES="${STEP13_NUM_PROCESSES:-3}"
export STEP2_GPUS="${STEP2_GPUS:-0,1,2}"
export STEP2_TP="${STEP2_TP:-1}"
export STEP2_VERIFY_BATCH_SIZE="${STEP2_VERIFY_BATCH_SIZE:-16}"
export STEP2_JUDGE_BATCH_SIZE="${STEP2_JUDGE_BATCH_SIZE:-16}"
export STEP2_GPU_MEM_UTIL="${STEP2_GPU_MEM_UTIL:-0.75}"

# DeepSpeed: ZeRO-2 (full weight replicas + sharded optimizer/gradients) is the default for Qwen3-4B + LoRA.
# CPU optimizer offload absorbs the ~5–6 GB Adam spike on 40 GB cards; remove it for faster training if VRAM fits.
# ZeRO-3 (param sharding) is opt-in for full FT only — saves ~5 GB/GPU on 4B but risks PEFT #2892
# (empty adapter on save). With LoRA the optimizer state is tiny (~640 MB), so ZeRO-3 buys nothing.
export TOOL_R0_DEEPSPEED_CONFIG="${TOOL_R0_DEEPSPEED_CONFIG:-./configs/deepseed_zero2_offload.yaml}"
# Faster (no offload): export TOOL_R0_DEEPSPEED_CONFIG=./configs/deepseed_zero2.yaml
# Full FT only:        export TOOL_R0_DEEPSPEED_CONFIG=./configs/deepseed_zero3.yaml  TOOL_R0_USE_PEFT=false

# GRPO: per_device_batch / num_generations still affect activation memory; max_completion_length raised
# from 2048 -> 3072 to stop truncating multi-call <tool_call_answer> generations on NESTFUL/BFCL.
export STEP1_PER_DEVICE_TRAIN_BATCH_SIZE="${STEP1_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export STEP1_GRADIENT_ACCUMULATION_STEPS="${STEP1_GRADIENT_ACCUMULATION_STEPS:-8}"
export STEP1_NUM_GENERATIONS="${STEP1_NUM_GENERATIONS:-2}"
export STEP1_MAX_COMPLETION_LENGTH="${STEP1_MAX_COMPLETION_LENGTH:-3072}"
export STEP3_PER_DEVICE_TRAIN_BATCH_SIZE="${STEP3_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export STEP3_GRADIENT_ACCUMULATION_STEPS="${STEP3_GRADIENT_ACCUMULATION_STEPS:-8}"
export STEP3_NUM_GENERATIONS="${STEP3_NUM_GENERATIONS:-2}"
export STEP3_MAX_COMPLETION_LENGTH="${STEP3_MAX_COMPLETION_LENGTH:-3072}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# LoRA (TRL ModelConfig CLI flags). Defaults target the standard 7 projections of
# LLaMA-style Qwen3 attention+MLP. Qwen3-4B has tied word embeddings, so we DO NOT
# add embed_tokens / lm_head to modules_to_save (transformers #45127). Set
# TOOL_R0_USE_PEFT=false to fall back to full fine-tuning.
export TOOL_R0_USE_PEFT="${TOOL_R0_USE_PEFT:-true}"
export TOOL_R0_LORA_R="${TOOL_R0_LORA_R:-32}"
export TOOL_R0_LORA_ALPHA="${TOOL_R0_LORA_ALPHA:-64}"
export TOOL_R0_LORA_DROPOUT="${TOOL_R0_LORA_DROPOUT:-0.05}"
export TOOL_R0_LORA_TARGET_MODULES="${TOOL_R0_LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj}"

mkdir -p "$RUN_DIR"
RUN_MAIN_LOG="${RUN_DIR}/run_main.log"
exec > >(tee -a "$RUN_MAIN_LOG") 2>&1
cat > "${RUN_DIR}/run_manifest.txt" <<EOF
run_id=$RUN_ID
base_model=$BASE_MODEL
abbreviation=$ABBREVIATION
iterations=$ITERATIONS
train_steps_gen=$TRAIN_STEPS_GEN
train_steps_solver=$TRAIN_STEPS_SOLVER
wandb_project=$WANDB_PROJECT
deepspeed_config=$TOOL_R0_DEEPSPEED_CONFIG
step13_gpus=$STEP13_GPUS
step13_num_processes=$STEP13_NUM_PROCESSES
step1_max_completion_length=$STEP1_MAX_COMPLETION_LENGTH
step3_max_completion_length=$STEP3_MAX_COMPLETION_LENGTH
use_peft=$TOOL_R0_USE_PEFT
lora_r=$TOOL_R0_LORA_R
lora_alpha=$TOOL_R0_LORA_ALPHA
lora_dropout=$TOOL_R0_LORA_DROPOUT
lora_target_modules=$TOOL_R0_LORA_TARGET_MODULES
EOF

stop_solver_server_for_iter() {
    local iter="$1"
    local pid_file="${RUN_DIR}/iter${iter}/step0_host_solver/vllm_server.pid"
    local meta_file="${RUN_DIR}/iter${iter}/step0_host_solver/vllm_server.meta"
    local cmd_marker="vllm.entrypoints.openai.api_server"
    if [ ! -f "$pid_file" ]; then
        return 0
    fi

    local pid
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [ -z "$pid" ]; then
        rm -f "$pid_file" "$meta_file"
        return 0
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$pid_file" "$meta_file"
        return 0
    fi

    local cmdline
    cmdline=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)
    if [[ "$cmdline" != *"$cmd_marker"* ]]; then
        echo "[Iter ${iter}] Skip stop: PID=$pid is not vLLM server (cmdline mismatch)."
        rm -f "$pid_file" "$meta_file"
        return 0
    fi

    local expected_start=""
    if [ -f "$meta_file" ]; then
        expected_start=$(grep '^start_time=' "$meta_file" | head -n 1 | cut -d= -f2-)
    fi

    local current_start=""
    if [ -r "/proc/${pid}/stat" ]; then
        current_start=$(awk '{print $22}' "/proc/${pid}/stat" 2>/dev/null || true)
    fi

    if [ -n "$expected_start" ] && [ -n "$current_start" ] && [ "$expected_start" != "$current_start" ]; then
        echo "[Iter ${iter}] Skip stop: PID=$pid start_time mismatch (expected=$expected_start current=$current_start)."
        rm -f "$pid_file" "$meta_file"
        return 0
    fi

    echo "[Iter ${iter}] Stopping owned vLLM server PID=$pid"
    kill "$pid" 2>/dev/null || true
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file" "$meta_file"
}

cleanup_on_exit() {
    if [ -n "${TOOL_R0_ITERATION:-}" ]; then
        stop_solver_server_for_iter "$TOOL_R0_ITERATION"
    fi
}

trap cleanup_on_exit EXIT INT TERM

echo "----------------------------------------------------------------"
echo "Starting Self-Play (Iter 1 explicit + Loop for 2..$ITERATIONS)"
echo "Base Model: $BASE_MODEL"
echo "Run Logs: $RUN_DIR"
echo "----------------------------------------------------------------"

export TRITON_CACHE_DIR="./triton_autotune"
mkdir -p "$TRITON_CACHE_DIR"

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export VLLM_DISABLE_COMPILE_CACHE=1


# Resume support:
#   RESUME_FROM_ITER=2 RESUME_FROM_STEP=2 bash run_main.sh
#     → skip iter 1 entirely, in iter 2 skip step0+step1, start at step2.
#   RESUME_FROM_STEP only (no ITER): applies to iter 1 as before.
RESUME_FROM_ITER="${RESUME_FROM_ITER:-1}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:-0}"
echo "Resume: ITER>=${RESUME_FROM_ITER}, STEP>=${RESUME_FROM_STEP} (within first resumed iter)"

# ---- helper: run one iteration (works for iter 1 AND iter 2+) ----
run_iteration() {
    local i="$1"
    local model_for_gen="$2"       # model to train generator on
    local model_for_solver="$3"    # model to train solver on
    local solver_for_step0="$4"    # model to host as vLLM solver (step0)
    local solver_for_step2="$5"    # solver model for step2 data generation
    local gen_dir="$6"
    local solver_dir="$7"
    local data_json="$8"
    local step_skip="${9:-0}"      # skip steps < this value

    export TOOL_R0_ITERATION="$i"
    stop_solver_server_for_iter "$i"

    local gen_ckpt="${gen_dir}/checkpoint-${TRAIN_STEPS_GEN}"
    local tokenizer_for_step0="$BASE_MODEL"

    # Step 0 + Step 1 (train generator)
    if [ "$step_skip" -lt 2 ]; then
        if [ -n "$solver_for_step0" ]; then
            echo "[Iter $i] Step 0: Hosting Solver..."
            bash run_step0.sh "$solver_for_step0" 5000 "$tokenizer_for_step0"
        fi

        echo "[Iter $i] Step 1: Training Generator..."
        bash run_step1.sh \
            "$model_for_gen" \
            "$gen_dir" \
            "${ABBREVIATION}-iter${i}-gen" \
            "$TRAIN_STEPS_GEN"

        sleep 5
        stop_solver_server_for_iter "$i"
    else
        echo "[Iter $i] SKIPPING step0+step1 (step_skip=$step_skip)"
        echo "[Iter $i] Using existing generator: $gen_ckpt"
    fi

    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    # Step 2 (generate data)
    if [ "$step_skip" -lt 3 ]; then
        echo "[Iter $i] Step 2: Generating Data..."
        bash run_step2.sh \
            "$gen_ckpt" \
            "$solver_for_step2" \
            "$data_json"

        sleep 5
        stop_solver_server_for_iter "$i"
    else
        echo "[Iter $i] SKIPPING step2 (step_skip=$step_skip)"
        echo "[Iter $i] Using existing data: $data_json"
    fi

    # Step 3 (train solver) — always runs
    if [ "$step_skip" -lt 4 ]; then
        echo "[Iter $i] Step 3: Training Solver..."
        bash run_step3.sh \
            "$model_for_solver" \
            "$data_json" \
            "$solver_dir" \
            "${ABBREVIATION}-iter${i}-solver" \
            "$TRAIN_STEPS_SOLVER"
    else
        echo "[Iter $i] SKIPPING step3 (step_skip=$step_skip)"
    fi

    echo ">>> FINISHED ITERATION $i"
}

# ---- Iteration 1 ----
if [ "$RESUME_FROM_ITER" -le 1 ]; then
    echo ">>> STARTING ITERATION 1 (Initialization)"
    run_iteration 1 \
        "$BASE_MODEL" \
        "$BASE_MODEL" \
        "$BASE_MODEL" \
        "$BASE_MODEL" \
        "${BASE_DIR}/iter1_generator" \
        "${BASE_DIR}/iter1_solver" \
        "${BASE_DIR}/iter1_data.json" \
        "$RESUME_FROM_STEP"
    RESUME_FROM_STEP=0
else
    echo ">>> SKIPPING ITERATION 1 (RESUME_FROM_ITER=$RESUME_FROM_ITER)"
fi

# ---- Iterations 2+ ----
for (( i=2; i<=ITERATIONS; i++ ))
do
    prev=$((i-1))
    PREV_SOLVER_CKPT="${BASE_DIR}/iter${prev}_solver/checkpoint-${TRAIN_STEPS_SOLVER}"
    PREV_GEN_CKPT="${BASE_DIR}/iter${prev}_generator/checkpoint-${TRAIN_STEPS_GEN}"

    if [ "$RESUME_FROM_ITER" -gt "$i" ]; then
        echo ">>> SKIPPING ITERATION $i (RESUME_FROM_ITER=$RESUME_FROM_ITER)"
        continue
    fi

    # Determine step skip for this iteration
    local_step_skip=0
    if [ "$RESUME_FROM_ITER" -eq "$i" ]; then
        local_step_skip="$RESUME_FROM_STEP"
        RESUME_FROM_STEP=0
    fi

    if [ ! -d "$PREV_SOLVER_CKPT" ]; then
        echo "ERROR: Previous solver checkpoint not found at $PREV_SOLVER_CKPT"
        exit 1
    fi
    if [ ! -d "$PREV_GEN_CKPT" ]; then
        echo "ERROR: Previous generator checkpoint not found at $PREV_GEN_CKPT"
        exit 1
    fi

    echo ">>> STARTING ITERATION $i (Derived from v$prev)"
    run_iteration "$i" \
        "$PREV_GEN_CKPT" \
        "$PREV_SOLVER_CKPT" \
        "$PREV_SOLVER_CKPT" \
        "$PREV_SOLVER_CKPT" \
        "${BASE_DIR}/iter${i}_generator" \
        "${BASE_DIR}/iter${i}_solver" \
        "${BASE_DIR}/iter${i}_data.json" \
        "$local_step_skip"
done

echo "================================================================"
echo "Self-Play Complete."
echo "Model: ${BASE_DIR}/iter${ITERATIONS}_solver/checkpoint-${TRAIN_STEPS_SOLVER}"
echo "================================================================"