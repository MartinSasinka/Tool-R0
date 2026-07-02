#!/bin/bash
# NESTFUL multi-turn eval for Tool-R0 synthetic curriculum checkpoints.
# Driver: nestful_evaluation/run.py (TOOL_R0_SYSTEM_PROMPT, vLLM, IBM helpers).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
CKPT_ROOT="${CKPT_ROOT:-curricullum/checkpoints/qwen3_4b_lora_grpo_toolr0}"
OUTPUT_DIR="${OUTPUT_DIR:-curricullum/evaluation/results_toolr0}"
PREPARED_ROOT="${PREPARED_ROOT:-curricullum/evaluation/prepared_toolr0}"

NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
MAX_TASKS="${MAX_TASKS:-}"
MAX_STEPS="${MAX_STEPS:-10}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SEED="${SEED:-0}"
NESTFUL_REPO_DIR="${NESTFUL_REPO_DIR:-nestful_repo}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
# Default: final curriculum stage + baseline for comparison
ONLY="${ONLY:-baseline stage3_3call}"

echo "=== Tool-R0 curriculum NESTFUL eval ==="
echo "Base model:     $BASE_MODEL"
echo "Checkpoints:    $CKPT_ROOT"
echo "Output:         $OUTPUT_DIR"
echo "Prepared merge: $PREPARED_ROOT"
echo "NESTFUL repo:   $NESTFUL_REPO_DIR"
echo "Rollouts/task:  $NUM_ROLLOUTS  max_steps: $MAX_STEPS"
if [[ -n "$MAX_TASKS" ]]; then
  echo "Pilot max_tasks: $MAX_TASKS"
fi
echo "Profiles:       $ONLY"

ARGS=(
  --base-model "$BASE_MODEL"
  --ckpt-root "$CKPT_ROOT"
  --output-dir "$OUTPUT_DIR"
  --prepared-root "$PREPARED_ROOT"
  --num-rollouts "$NUM_ROLLOUTS"
  --max-steps "$MAX_STEPS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --max-model-len "$MAX_MODEL_LEN"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --seed "$SEED"
  --nestful-repo-dir "$NESTFUL_REPO_DIR"
)

if [[ -n "$MAX_TASKS" ]]; then
  ARGS+=(--max-tasks "$MAX_TASKS")
fi
if [[ "$SKIP_EXISTING" == "1" ]]; then
  ARGS+=(--skip-existing)
fi
if [[ -n "$ONLY" ]]; then
  # shellcheck disable=SC2206
  ONLY_ARR=($ONLY)
  ARGS+=(--only "${ONLY_ARR[@]}")
fi

python curricullum/evaluation/run_eval.py "${ARGS[@]}" "$@"
