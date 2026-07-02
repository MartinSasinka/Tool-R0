#!/usr/bin/env bash
# NESTFUL Synthetic Curriculum v3 — final eval launcher (POD ONLY skeleton).
#
# Usage (pod):
#   DATASET=experiments/nestful_mtgrpo_minimal/data/splits/nestful_test.jsonl \
#     CKPT_ROOT=experiments/nestful_synthetic_curriculum_v3/outputs/runs/<timestamp> \
#     bash experiments/nestful_synthetic_curriculum_v3/scripts/run_eval_v3.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
V3="$REPO/experiments/nestful_synthetic_curriculum_v3"
MINIMAL="$REPO/experiments/nestful_mtgrpo_minimal"
PARTIAL="$REPO/experiments/nestful_mtgrpo_partial"
PYTHON="${PYTHON:-python}"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

DATASET="${DATASET:-$MINIMAL/data/splits/nestful_test.jsonl}"
CKPT_ROOT="${CKPT_ROOT:-$V3/outputs/runs/latest}"
OUT="$V3/outputs/final_eval_v3"
mkdir -p "$OUT"

BEST="${BEST_ADAPTER:-$CKPT_ROOT/best_react_win_adapter}"
TS="$(date +%Y%m%d_%H%M%S)"
REPORT="$OUT/FINAL_RESULTS_V3.md"

if [ ! -f "$DATASET" ]; then
  echo "[eval_v3] ERROR: test split not found: $DATASET" >&2
  exit 1
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  cat > "$REPORT" <<EOF
# Final Results v3 (DRY RUN skeleton)

Test split: $DATASET
Best adapter: $BEST

## TODO (pod + GPU)
1. Recompute baseline on same test split
2. Eval best checkpoint via partial final_eval
3. Write motif_level_eval.csv, win_loss_overlap_v3.csv, failure_taxonomy_v3.csv

## Commands
\`\`\`bash
DATASET=$DATASET CKPT_ROOT=$CKPT_ROOT \\
  bash $PARTIAL/run_final_eval_v2_parallel.sh
\`\`\`
EOF
  echo "[eval_v3] DRY_RUN skeleton -> $REPORT"
  exit 0
fi

echo "[eval_v3] running parallel final eval (baseline + best) ..."
CELLS="baseline= best_v3=$BEST" \
  DATASET="$DATASET" \
  OUT_ROOT="$OUT/run_$TS" \
  CKPT_ROOT="$CKPT_ROOT" \
  bash "$PARTIAL/run_final_eval_v2_parallel.sh"

"$PYTHON" "$V3/scripts/motif_level_eval.py" \
  --split "$DATASET" \
  --out_dir "$OUT" || true

echo "[eval_v3] see $OUT for results"
