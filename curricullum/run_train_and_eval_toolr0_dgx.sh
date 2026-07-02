#!/bin/bash
# DGX: train Tool-R0 synthetic curriculum (3 stages) then NESTFUL IBM eval.
#
# Usage:
#   bash curricullum/run_train_and_eval_toolr0_dgx.sh
#
# Pilot (train + small eval):
#   MAX_TASKS=50 NUM_ROLLOUTS=2 bash curricullum/run_train_and_eval_toolr0_dgx.sh
#
# Eval only (training already done):
#   SKIP_TRAIN=1 bash curricullum/run_train_and_eval_toolr0_dgx.sh
#
# Train only:
#   SKIP_EVAL=1 bash curricullum/run_train_and_eval_toolr0_dgx.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# --- GPU / distributed (DGX: exclude display GPU if needed → 0,1,2,4) ---
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

# --- Training ---
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
TRAINING_FORMAT="${TRAINING_FORMAT:-tool_r0}"
MAX_STAGES="${MAX_STAGES:-3}"
OVERWRITE="${OVERWRITE:-1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
CONFIG="${CONFIG:-curricullum/train/configs/qwen3_4b_lora_grpo_toolr0.yaml}"
CKPT_ROOT="${CKPT_ROOT:-curricullum/checkpoints/qwen3_4b_lora_grpo_toolr0}"
DATA_DIR="${DATA_DIR:-curricullum/data/filtered_toolr0_synthetic}"
NESTFUL_REPO_DIR="${NESTFUL_REPO_DIR:-nestful_repo}"
TOOL_R0_DEEPSPEED_CONFIG="${TOOL_R0_DEEPSPEED_CONFIG:-./configs/deepseed_zero2.yaml}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
# Keep ~same optimizer steps as 1-GPU default (batch=1, accum=8): 1*2*4=8
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
MAX_STEPS="${MAX_STEPS:-}"
WANDB_PROJECT="${WANDB_PROJECT:-nestful-curriculum-toolr0}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-qwen3-4b-toolr0-dgx}"

# --- Eval (real NESTFUL via nestful_evaluation/run.py) ---
OUTPUT_DIR="${OUTPUT_DIR:-curricullum/evaluation/results_toolr0}"
PREPARED_ROOT="${PREPARED_ROOT:-curricullum/evaluation/prepared_toolr0}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
MAX_TASKS="${MAX_TASKS:-}"
MAX_STEPS_EVAL="${MAX_STEPS_EVAL:-10}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
EVAL_BASELINE="${EVAL_BASELINE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

log() { echo "[toolr0_dgx] $*" >&2; }

resolve_final_stage_key() {
  if [[ "$MAX_STAGES" -ge 3 ]] && [[ -d "$CKPT_ROOT/stage3_3call" ]]; then
    echo "stage3_3call"
  elif [[ "$MAX_STAGES" -ge 2 ]] && [[ -d "$CKPT_ROOT/stage2_2call" ]]; then
    echo "stage2_2call"
  else
    echo "stage1_1call"
  fi
}

preflight() {
  if [[ ! -d "$NESTFUL_REPO_DIR/data_v2/executable_functions" ]]; then
    log "nestful_repo missing — attempting clone (training continues with fallbacks if this fails)..."
    if ! git clone --depth 1 https://github.com/IBM/NESTFUL.git "$NESTFUL_REPO_DIR"; then
      log "WARNING: could not clone nestful_repo; train uses gold_answer exec fallbacks"
    fi
  fi
  export NESTFUL_REPO_DIR

  if [[ "$SKIP_TRAIN" != "1" ]]; then
    for e in 1 2 3; do
      if [[ "$e" -gt "$MAX_STAGES" ]]; then
        break
      fi
      f="$DATA_DIR/epoch_${e}_${e}call.jsonl"
      if [[ ! -f "$f" ]]; then
        log "ERROR: missing training data: $f"
        log "Run first: python curricullum/run_generate_toolr0_curriculum.py"
        exit 1
      fi
      rows="$(grep -cve '^[[:space:]]*$' "$f" || true)"
      if [[ "$rows" -lt 1 ]]; then
        log "ERROR: empty training data: $f"
        exit 1
      fi
      log "epoch ${e}: ${rows} task rows (uses all available)"
    done
  fi
}

run_train() {
  log "=== TRAIN (GPUs=$CUDA_DEVICES processes=$NUM_PROCESSES stages=$MAX_STAGES) ==="
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
  export NUM_PROCESSES
  export TRAINING_FORMAT
  export MAX_STAGES
  export OVERWRITE
  export MODEL_NAME
  export CONFIG
  export CKPT_ROOT
  export NESTFUL_REPO_DIR
  export TOOL_R0_DEEPSPEED_CONFIG
  export NUM_GENERATIONS
  export GRADIENT_ACCUMULATION_STEPS
  export WANDB_PROJECT
  export WANDB_RUN_GROUP
  if [[ -n "$MAX_STEPS" ]]; then
    export MAX_STEPS
  fi

  bash curricullum/run_train_toolr0.sh
}

run_eval() {
  local final_key
  final_key="$(resolve_final_stage_key)"
  local final_ckpt="$CKPT_ROOT/$final_key"

  if [[ ! -f "$final_ckpt/adapter_config.json" ]]; then
    log "ERROR: final checkpoint missing: $final_ckpt"
    exit 1
  fi

  local only_profiles="$final_key"
  if [[ "$EVAL_BASELINE" == "1" ]]; then
    only_profiles="baseline $final_key"
  fi

  log "=== EVAL (NESTFUL IBM, nestful_evaluation/run.py) ==="
  log "Final checkpoint: $final_ckpt"
  log "Profiles: $only_profiles"
  log "Results -> $OUTPUT_DIR"

  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
  export CKPT_ROOT
  export OUTPUT_DIR
  export PREPARED_ROOT
  export BASE_MODEL="$MODEL_NAME"
  export NESTFUL_REPO_DIR
  export NUM_ROLLOUTS
  export MAX_TASKS
  export MAX_STEPS="$MAX_STEPS_EVAL"
  export TENSOR_PARALLEL_SIZE
  export GPU_MEMORY_UTILIZATION
  export SKIP_EXISTING
  export ONLY="$only_profiles"

  bash curricullum/evaluation/run_toolr0_eval.sh
}

main() {
  log "repo=$ROOT"
  preflight

  if [[ "$SKIP_TRAIN" != "1" ]]; then
    run_train
  else
    log "SKIP_TRAIN=1 — skipping training"
  fi

  if [[ "$SKIP_EVAL" != "1" ]]; then
    run_eval
  else
    log "SKIP_EVAL=1 — skipping eval"
  fi

  log "Done."
  log "Checkpoints: $CKPT_ROOT"
  log "Eval JSONL:  $OUTPUT_DIR"
  log "Viewer:      open eval_viewer.html and load *_multiturn_predictions.jsonl"
}

main "$@"
