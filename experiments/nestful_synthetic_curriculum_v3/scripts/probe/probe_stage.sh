#!/usr/bin/env bash
# Stage probe wrapper — forward-only GRPO signal check (no training).
#
# Usage (repo root, Linux pod):
#   STAGE=2 bash experiments/nestful_synthetic_curriculum_v3/scripts/probe/probe_stage.sh
#   STAGE=2 REWARD_POLICY=execution_aware_v3_2_dense NUM_TASKS=50 NUM_GENERATIONS=8 \
#     bash experiments/nestful_synthetic_curriculum_v3/scripts/probe/probe_stage.sh
#   DRY_RUN=1 STAGE=1 bash .../probe_stage.sh
#
# Env knobs:
#   STAGE (1-4) | DATASET (explicit jsonl, overrides STAGE) | CHECKPOINT
#   REWARD_POLICY (default execution_aware_v3_1_stepwise)
#   NUM_TASKS (50) | NUM_GENERATIONS (8) | TEMPERATURE (1.0) | TOP_P (0.95)
#   SEED (42) | OUTPUT_DIR | BACKEND (vllm|hf|stub; default vllm)
#   DRY_RUN (0/1) | CUDA_DEVICE (sets CUDA_VISIBLE_DEVICES)
if grep -q $'\r' "$0" 2>/dev/null; then
  exec /bin/bash <(sed 's/\r$//' "$0") "$@"
fi
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
V3="$REPO/experiments/nestful_synthetic_curriculum_v3"
PYTHON="${PYTHON:-python}"

STAGE="${STAGE:-}"
DATASET="${DATASET:-}"
if [ -z "$STAGE" ] && [ -z "$DATASET" ]; then
  echo "[probe] ERROR: set STAGE=1..4 or DATASET=path.jsonl" >&2
  exit 1
fi

ARGS=(
  --reward-policy "${REWARD_POLICY:-execution_aware_v3_1_stepwise}"
  --num-tasks "${NUM_TASKS:-50}"
  --num-generations "${NUM_GENERATIONS:-8}"
  --temperature "${TEMPERATURE:-1.0}"
  --top-p "${TOP_P:-0.95}"
  --seed "${SEED:-42}"
  --backend "${BACKEND:-vllm}"
)
[ -n "$STAGE" ] && ARGS+=(--stage "$STAGE")
[ -n "$DATASET" ] && ARGS+=(--dataset "$DATASET")
[ -n "${CHECKPOINT:-}" ] && ARGS+=(--checkpoint "$CHECKPOINT")
[ -n "${OUTPUT_DIR:-}" ] && ARGS+=(--output-dir "$OUTPUT_DIR")
[ "${DRY_RUN:-0}" = "1" ] && ARGS+=(--dry-run)

if [ -n "${CUDA_DEVICE:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

echo "[probe] repo=$REPO"
echo "[probe] cmd : $PYTHON $V3/scripts/probe/probe_stage.py ${ARGS[*]}"
exec "$PYTHON" "$V3/scripts/probe/probe_stage.py" "${ARGS[@]}"
