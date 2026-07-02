#!/bin/bash
# Generate synthetic Tool-R0 curriculum (OpenRouter) — 3 stages, IBM exec verify, no benchmark leakage.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -n "${NESTFUL_PATH:-}" ]]; then
  NESTFUL="$NESTFUL_PATH"
elif [[ -f "eval/data/NESTFUL-main/data_v2/nestful_data.jsonl" ]]; then
  NESTFUL="eval/data/NESTFUL-main/data_v2/nestful_data.jsonl"
else
  echo "[err] NESTFUL data not found (needed for tool schema + contamination filter only)"
  exit 1
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "[err] OPENROUTER_API_KEY is not set"
  exit 1
fi

# Defaults tuned for overnight: 500 samples/stage, parallel epochs
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
N_GENERATE="${N_GENERATE:-1500}"
MAX_GENERATE="${MAX_GENERATE:-2500}"
N_FINAL="${N_FINAL:-500}"
PARALLEL_EPOCHS="${PARALLEL_EPOCHS:-1}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-16}"
MAX_STAGES="${MAX_STAGES:-3}"
USE_EXECUTOR="${USE_EXECUTOR:-1}"
SEED="${SEED:-42}"
SEED_MODE="${SEED_MODE:-schema_only}"
DEPENDENCY_MODE="${DEPENDENCY_MODE:-strict_chain}"

DATA_DIR="${DATA_DIR:-curricullum/data/filtered_toolr0_synthetic}"
RAW_DIR="curricullum/data/raw_toolr0"
VERIFIED_DIR="curricullum/data/verified_toolr0"
REJECTED_DIR="curricullum/data/rejected_toolr0"

mkdir -p "$RAW_DIR" "$VERIFIED_DIR" "$REJECTED_DIR" "$DATA_DIR" curricullum/data/reports

echo "=== Tool-R0 synthetic curriculum generation ==="
echo "Model: $MODEL"
echo "NESTFUL (schema/contamination only): $NESTFUL"
echo "Stages: 1..$MAX_STAGES | N_FINAL/stage: $N_FINAL | N_GENERATE/stage: $N_GENERATE"
echo "Parallel: epochs=$PARALLEL_EPOCHS workers=$PARALLEL_WORKERS"
echo "IBM exec verify: $USE_EXECUTOR"
echo "Output: $DATA_DIR"

batch_for_epoch() {
  case "$1" in
    1) echo "${BATCH_SIZE_E1:-14}" ;;
    2) echo "${BATCH_SIZE_E2:-12}" ;;
    3) echo "${BATCH_SIZE_E3:-10}" ;;
    *) echo 10 ;;
  esac
}

gen_max_tokens_for_epoch() {
  case "$1" in
    1) echo 1024 ;;
    2) echo 1536 ;;
    3) echo 2048 ;;
    *) echo 2048 ;;
  esac
}

run_step1() {
  local EPOCH="$1"
  local BATCH_SIZE
  BATCH_SIZE="$(batch_for_epoch "$EPOCH")"
  local MAX_TOKENS
  MAX_TOKENS="$(gen_max_tokens_for_epoch "$EPOCH")"

  echo ""
  echo "========== Epoch $EPOCH STEP1 (batch=$BATCH_SIZE) =========="

  python -u curricullum/data/step1_gen_candidates.py \
    --nestful_path "$NESTFUL" \
    --out_json "$RAW_DIR/epoch_${EPOCH}_candidates.json" \
    --epoch "$EPOCH" \
    --n_generate "$N_GENERATE" \
    --max_generate "$MAX_GENERATE" \
    --batch_size "$BATCH_SIZE" \
    --parallel_workers "$PARALLEL_WORKERS" \
    --max_tokens "$MAX_TOKENS" \
    --model "$MODEL" \
    --seed "$SEED" \
    --seed_mode "$SEED_MODE" \
    --dependency_mode "$DEPENDENCY_MODE" \
    --n_seed_examples 0
}

run_step2_step3() {
  local EPOCH="$1"
  local CALL_LABEL="${EPOCH}call"
  local EXEC_FLAG=()
  if [[ "$USE_EXECUTOR" == "1" ]]; then
    EXEC_FLAG=(--use_executor)
  else
    EXEC_FLAG=(--no_executor)
  fi

  python -u curricullum/data/step2_verify_candidates.py \
    --in_json "$RAW_DIR/epoch_${EPOCH}_candidates.json" \
    --out_json "$VERIFIED_DIR/epoch_${EPOCH}_verified.json" \
    --rejected_json "$REJECTED_DIR/epoch_${EPOCH}_rejected.json" \
    --nestful_path "$NESTFUL" \
    --epoch "$EPOCH" \
    --dependency_mode "$DEPENDENCY_MODE" \
    "${EXEC_FLAG[@]}"

  python -u curricullum/data/step3_select_curriculum.py \
    --in_json "$VERIFIED_DIR/epoch_${EPOCH}_verified.json" \
    --out_jsonl "$DATA_DIR/epoch_${EPOCH}_${CALL_LABEL}.jsonl" \
    --n_final "$N_FINAL" \
    --epoch "$EPOCH" \
    --seed "$SEED"
}

# --- STEP1 (parallel optional) ---
if [[ "$PARALLEL_EPOCHS" == "1" ]]; then
  echo ""
  echo "========== PARALLEL STEP1 (epochs 1-$MAX_STAGES) =========="
  pids=()
  for EPOCH in $(seq 1 "$MAX_STAGES"); do
    run_step1 "$EPOCH" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
else
  for EPOCH in $(seq 1 "$MAX_STAGES"); do
    run_step1 "$EPOCH"
  done
fi

# --- STEP2+3 (parallel per epoch) ---
echo ""
echo "========== PARALLEL STEP2+STEP3 (epochs 1-$MAX_STAGES) =========="
pids=()
for EPOCH in $(seq 1 "$MAX_STAGES"); do
  run_step2_step3 "$EPOCH" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo ""
echo "Concatenating -> curriculum_toolr0_all.jsonl"
cat "$DATA_DIR"/epoch_*_*call.jsonl > "$DATA_DIR/curriculum_toolr0_all.jsonl" 2>/dev/null || true

python -u curricullum/data/inspect_dataset.py --path "$DATA_DIR/curriculum_toolr0_all.jsonl" || true

echo ""
echo "Done. Train with:"
echo "  export TRAINING_FORMAT=tool_r0"
echo "  export CONFIG=curricullum/train/configs/qwen3_4b_lora_grpo_toolr0.yaml"
echo "  bash curricullum/run_train_toolr0.sh"
