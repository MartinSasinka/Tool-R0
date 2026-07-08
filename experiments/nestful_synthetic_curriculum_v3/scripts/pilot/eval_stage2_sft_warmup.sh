#!/usr/bin/env bash
# Stage2 continuation SFT — evaluation wrapper.
#
# Evaluates BASE model vs the SFT checkpoint produced by
# run_stage2_continuation_sft_warmup.sh, in two modes:
#   1. continuation-conditioned eval (Stage2 val set; given call1+obs1, model
#      must generate call2 + terminal finish)
#   2. free ReAct eval (Stage2 val set AND full NESTFUL dev; model must find
#      the whole trace unaided) — reuses the EXISTING GRPO eval machinery
#      (nestful_mtgrpo_minimal/run.py --mode rollout_eval / val_eval) so
#      metric definitions are identical to the rest of the pipeline.
#
# Usage:
#   CHECKPOINT=experiments/nestful_synthetic_curriculum_v3/outputs/sft/stage2_continuation/run_<ts>/adapter/epoch_1 \
#     bash experiments/nestful_synthetic_curriculum_v3/scripts/pilot/eval_stage2_sft_warmup.sh
#
# Configurable env vars:
#   CHECKPOINT       (required) SFT LoRA adapter dir to evaluate.
#   STAGE2_VAL       (default: outputs/sft/stage2_continuation/val.jsonl)
#   NESTFUL_DEV      (default: nestful_mtgrpo_minimal/data/splits/nestful_dev.jsonl)
#   EVAL_TEMPERATURE (default: 0.0)
#   OUT_DIR          (default: <checkpoint's run dir>/eval_<timestamp>)
#   USE_VLLM=1       (opt-in; default off — plain HF generation)
#   SKIP_FREE_REACT=1 / SKIP_CONTINUATION=1  (partial runs)

# Re-exec without CRLF when checked out on Windows.
if grep -q $'\r' "$0" 2>/dev/null; then
  exec /bin/bash <(sed 's/\r$//' "$0") "$@"
fi
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."

V3="experiments/nestful_synthetic_curriculum_v3"
SFT_DIR="$V3/scripts/sft"
DEFAULT_DATA_DIR="$V3/outputs/sft/stage2_continuation"

PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-}"
STAGE2_VAL="${STAGE2_VAL:-$DEFAULT_DATA_DIR/val.jsonl}"
NESTFUL_DEV="${NESTFUL_DEV:-experiments/nestful_mtgrpo_minimal/data/splits/nestful_dev.jsonl}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
USE_VLLM="${USE_VLLM:-0}"
SKIP_FREE_REACT="${SKIP_FREE_REACT:-0}"
SKIP_CONTINUATION="${SKIP_CONTINUATION:-0}"

if [ -z "$CHECKPOINT" ]; then
  echo "[eval-stage2-sft] ERROR: CHECKPOINT is required, e.g." >&2
  echo "  CHECKPOINT=$DEFAULT_DATA_DIR/run_<ts>/adapter/epoch_1 bash $0" >&2
  exit 1
fi
if [ ! -f "$CHECKPOINT/adapter_config.json" ]; then
  echo "[eval-stage2-sft] ERROR: CHECKPOINT is not a valid LoRA adapter dir (no adapter_config.json): $CHECKPOINT" >&2
  exit 1
fi
if [ ! -f "$STAGE2_VAL" ]; then
  echo "[eval-stage2-sft] ERROR: Stage2 val set not found: $STAGE2_VAL" >&2
  echo "  Build it first: python $SFT_DIR/build_stage2_sft_dataset.py" >&2
  exit 1
fi

if [ -z "${OUT_DIR:-}" ]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  RUN_ROOT="$(dirname "$(dirname "$CHECKPOINT")")"   # .../run_<ts>/adapter/epoch_N -> .../run_<ts>
  OUT_DIR="$RUN_ROOT/eval_${TS}"
fi
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Stage2 continuation SFT — evaluation"
echo "checkpoint       = $CHECKPOINT"
echo "stage2_val       = $STAGE2_VAL"
echo "nestful_dev      = $NESTFUL_DEV"
echo "eval_temperature = $EVAL_TEMPERATURE"
echo "out_dir          = $OUT_DIR"
echo "use_vllm=$USE_VLLM skip_free_react=$SKIP_FREE_REACT skip_continuation=$SKIP_CONTINUATION"
echo "============================================================"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --out-dir "$OUT_DIR"
  --stage2-val "$STAGE2_VAL"
  --nestful-dev "$NESTFUL_DEV"
  --eval-temperature "$EVAL_TEMPERATURE"
)
if [ "$USE_VLLM" = "1" ]; then ARGS+=(--use-vllm); fi
if [ "$SKIP_FREE_REACT" = "1" ]; then ARGS+=(--skip-free-react); fi
if [ "$SKIP_CONTINUATION" = "1" ]; then ARGS+=(--skip-continuation); fi

"$PYTHON" "$SFT_DIR/eval_stage2_sft.py" "${ARGS[@]}"
RC=$?

echo "============================================================"
echo "[eval-stage2-sft] done (rc=$RC)."
echo "  summary: $OUT_DIR/SFT_STAGE2_EVAL.md"
echo "  json:    $OUT_DIR/SFT_STAGE2_EVAL.json"
echo "============================================================"
exit "$RC"
