#!/usr/bin/env bash
# run_all.sh (PARTIAL) — same unified entry point as the sibling, wired to THIS
# experiment (partial reward for training; strict + official metrics for eval).
#
# Delegates to ../nestful_mtgrpo_minimal/run_all.sh with RUN_PY/CONFIG/OUTPUT_ROOT
# pointing at this folder, so all stabilisation fixes and the 4×GPU split are
# inherited automatically.
#
# Usage mirrors the sibling run_all.sh:
#   MODE=final CHECKPOINT_IN=outputs/curriculum/stage_4/checkpoints/adapter_epoch_4 \
#     USE_VLLM=1 bash run_all.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIBLING="$(cd "$HERE/../nestful_mtgrpo_minimal" && pwd)"
SIBLING_RUN_ALL="$SIBLING/run_all.sh"

if [ ! -f "$SIBLING_RUN_ALL" ]; then
    echo "[partial] ERROR: sibling runner not found: $SIBLING_RUN_ALL" >&2
    echo "          Keep nestful_mtgrpo_partial next to nestful_mtgrpo_minimal." >&2
    exit 1
fi

# Wire the unified runner to THIS experiment.
export RUN_PY="${RUN_PY:-$HERE/run.py}"
export CONFIG="${CONFIG:-$HERE/config.yaml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$HERE/outputs/curriculum}"
# Use the sibling's curriculum loop (it reads the env above).
export CURRICULUM_RUNNER="$SIBLING/run_curriculum.sh"

if [ -n "${WANDB_API_KEY:-}" ] && [ -z "${WANDB_PROJECT:-}" ]; then
    export WANDB_PROJECT="nestful-mtgrpo-partial"
    echo "[partial] WANDB_PROJECT defaulted to $WANDB_PROJECT"
fi

echo "=============================================================="
echo " NESTFUL MT-GRPO PARTIAL run_all"
echo "   RUN_PY      : $RUN_PY"
echo "   CONFIG      : $CONFIG"
echo "   OUTPUT_ROOT : $OUTPUT_ROOT"
echo "=============================================================="

exec bash "$SIBLING_RUN_ALL" "$@"
