#!/usr/bin/env bash
# Stage2 continuation SFT warmup — pure supervised fine-tuning, NO GRPO.
#
# Trains on the Stage2 continuation SFT view (a derived serialization of the
# EXISTING GRPO Stage2 curriculum file, NOT a new dataset — see
# scripts/sft/build_stage2_sft_dataset.py). This is a separate, standalone
# experiment: it does not touch the GRPO pipeline, does not train stage 3/4,
# and does not mix SFT with GRPO in the same run.
#
# Usage:
#   python experiments/nestful_synthetic_curriculum_v3/scripts/sft/build_stage2_sft_dataset.py
#   SFT_EPOCHS=1 SFT_LR=1e-5 SFT_BATCH_SIZE=1 SFT_GRAD_ACCUM=16 \
#     bash experiments/nestful_synthetic_curriculum_v3/scripts/pilot/run_stage2_continuation_sft_warmup.sh
#
# Configurable env vars (all optional; defaults shown):
#   SFT_EPOCHS=1  SFT_LR=1e-5  SFT_BATCH_SIZE=1  SFT_GRAD_ACCUM=16
#   SFT_MAX_SEQ_LEN=3072  SFT_SEED=42
#   SFT_LORA_R=16  SFT_LORA_ALPHA=32  SFT_LORA_DROPOUT=0.05  SFT_LOAD_IN_4BIT=1
#   TRAIN_PATH / VAL_PATH  (default: outputs/sft/stage2_continuation/{train,val}.jsonl)
#   OUTPUT_DIR             (default: outputs/sft/stage2_continuation/run_<timestamp>)
#   RESUME_CHECKPOINT      (optional existing LoRA adapter dir to continue from)
#   BASE_MODEL             (default: Qwen/Qwen3-4B-Instruct-2507)
#   CUDA_VISIBLE_DEVICES   (default: unset = all GPUs visible; e.g. "0" for one GPU)
#   SFT_HF_DEVICE_MAP      (default: auto; use '{"": 0}' to pin whole model on GPU 0)
#   DRY_RUN=1              (tokenize+mask sanity check only; no GPU/model needed)
#
# Requires (real training): torch+CUDA, transformers, peft, bitsandbytes.
# Quick install on a fresh pod:
#   pip install 'peft>=0.12' 'bitsandbytes>=0.43' 'accelerate>=0.33'
# or full GRPO stack:
#   bash experiments/nestful_mtgrpo_minimal/install_deps.sh
# This script does NOT require vLLM, does NOT start rollout workers, and does
# NOT read/write anything under outputs/runs/ (the GRPO run directories).

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
TRAIN_PATH="${TRAIN_PATH:-$DEFAULT_DATA_DIR/train.jsonl}"
VAL_PATH="${VAL_PATH:-$DEFAULT_DATA_DIR/val.jsonl}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

SFT_EPOCHS="${SFT_EPOCHS:-1}"
SFT_LR="${SFT_LR:-1e-5}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-16}"
SFT_MAX_SEQ_LEN="${SFT_MAX_SEQ_LEN:-3072}"
SFT_SEED="${SFT_SEED:-42}"
SFT_LORA_R="${SFT_LORA_R:-16}"
SFT_LORA_ALPHA="${SFT_LORA_ALPHA:-32}"
SFT_LORA_DROPOUT="${SFT_LORA_DROPOUT:-0.05}"
SFT_LOAD_IN_4BIT="${SFT_LOAD_IN_4BIT:-1}"
SFT_HF_DEVICE_MAP="${SFT_HF_DEVICE_MAP:-auto}"
DRY_RUN="${DRY_RUN:-0}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES
fi

if [ "$DRY_RUN" != "1" ]; then
  if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[stage2-sft] ERROR: CUDA not available. Need a GPU pod with torch+cuda." >&2
    exit 1
  fi
  if [ "$SFT_LOAD_IN_4BIT" = "1" ]; then
    if ! "$PYTHON" -c "import importlib.metadata as m; m.version('bitsandbytes')" 2>/dev/null; then
      echo "[stage2-sft] ERROR: bitsandbytes missing (required for QLoRA 4-bit)." >&2
      echo "  pip install 'peft>=0.12' 'bitsandbytes>=0.43' 'accelerate>=0.33'" >&2
      echo "  or: bash experiments/nestful_mtgrpo_minimal/install_deps.sh" >&2
      exit 1
    fi
  fi
  if ! "$PYTHON" -c "import peft" 2>/dev/null; then
    echo "[stage2-sft] ERROR: peft missing." >&2
    echo "  pip install 'peft>=0.12' 'bitsandbytes>=0.43' 'accelerate>=0.33'" >&2
    exit 1
  fi
fi

if [ ! -f "$TRAIN_PATH" ] || [ ! -f "$VAL_PATH" ]; then
  echo "[stage2-sft] ERROR: train/val jsonl not found. Build them first:" >&2
  echo "  python $SFT_DIR/build_stage2_sft_dataset.py" >&2
  exit 1
fi

if [ -z "${OUTPUT_DIR:-}" ]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_DIR="$DEFAULT_DATA_DIR/run_${TS}"
fi
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Stage2 continuation SFT warmup (pure SFT — no GRPO, no reward, no rollouts)"
echo "train_path   = $TRAIN_PATH"
echo "val_path     = $VAL_PATH"
echo "output_dir   = $OUTPUT_DIR"
echo "base_model   = $BASE_MODEL"
echo "epochs=$SFT_EPOCHS lr=$SFT_LR batch_size=$SFT_BATCH_SIZE grad_accum=$SFT_GRAD_ACCUM"
echo "max_seq_len=$SFT_MAX_SEQ_LEN seed=$SFT_SEED"
echo "lora: r=$SFT_LORA_R alpha=$SFT_LORA_ALPHA dropout=$SFT_LORA_DROPOUT 4bit=$SFT_LOAD_IN_4BIT"
echo "cuda_visible_devices = ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "hf_device_map        = $SFT_HF_DEVICE_MAP"
echo "resume_checkpoint = ${RESUME_CHECKPOINT:-<none — fresh LoRA from base model>}"
echo "dry_run = $DRY_RUN"
echo "============================================================"

ARGS=(
  --train-path "$TRAIN_PATH"
  --val-path "$VAL_PATH"
  --output-dir "$OUTPUT_DIR"
  --base-model "$BASE_MODEL"
  --epochs "$SFT_EPOCHS"
  --lr "$SFT_LR"
  --batch-size "$SFT_BATCH_SIZE"
  --grad-accum "$SFT_GRAD_ACCUM"
  --max-seq-len "$SFT_MAX_SEQ_LEN"
  --seed "$SFT_SEED"
  --lora-r "$SFT_LORA_R"
  --lora-alpha "$SFT_LORA_ALPHA"
  --lora-dropout "$SFT_LORA_DROPOUT"
  --hf-device-map "$SFT_HF_DEVICE_MAP"
)
if [ "$SFT_LOAD_IN_4BIT" != "1" ]; then
  ARGS+=(--no-4bit)
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
  ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi
if [ "$DRY_RUN" = "1" ]; then
  ARGS+=(--dry-run)
fi

"$PYTHON" "$SFT_DIR/train_stage2_continuation_sft.py" "${ARGS[@]}"
RC=$?

echo "============================================================"
if [ "$DRY_RUN" = "1" ]; then
  echo "[stage2-sft] DRY_RUN complete (rc=$RC) — see $OUTPUT_DIR/DRY_RUN_TOKENIZE_REPORT.json"
else
  echo "[stage2-sft] training complete (rc=$RC)."
  echo "  adapter:  $OUTPUT_DIR/adapter/epoch_${SFT_EPOCHS}"
  echo "  summary:  $OUTPUT_DIR/SFT_STAGE2_TRAINING_SUMMARY.md"
  echo "  train log: $OUTPUT_DIR/train_log.jsonl"
fi
echo "============================================================"
exit "$RC"
