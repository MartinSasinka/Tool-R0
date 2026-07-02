#!/usr/bin/env bash
# run_curriculum.sh (PARTIAL) — staged MT-GRPO curriculum with the PARTIAL reward.
#
# Reuses the sibling ../nestful_mtgrpo_minimal/run_curriculum.sh (gate logic,
# plateau detection, manifests, W&B grouping) but points it at THIS experiment's
# entry point + config so training uses the graded partial reward. Evaluation
# stays strict + official (comparable to the strict experiment).
#
# Everything is driven via env vars the sibling script already supports:
#   RUN_PY       -> this folder's run.py (partial reward injected for train)
#   CONFIG       -> this folder's config.yaml (reward.train_policy=partial_gold_trace)
#   OUTPUT_ROOT  -> this folder's outputs/curriculum
#   DATA_BASE    -> sibling data (default of the sibling script; dataset lives there)
#
# Usage (identical knobs to the strict runner):
#   # pilot smoke
#   CUDA_VISIBLE_DEVICES=0 PROFILE=pilot STAGES="3" bash run_curriculum.sh
#
#   # full overnight curriculum with vLLM + W&B
#   export WANDB_API_KEY=...; export WANDB_PROJECT=nestful-mtgrpo-partial
#   CUDA_VISIBLE_DEVICES=0 USE_VLLM=1 PROFILE=curriculum STAGES="1 2 3 4" \
#     bash run_curriculum.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIBLING="$(cd "$HERE/../nestful_mtgrpo_minimal" && pwd)"
SIBLING_RUNNER="$SIBLING/run_curriculum.sh"

if [ ! -f "$SIBLING_RUNNER" ]; then
    echo "[partial] ERROR: sibling runner not found: $SIBLING_RUNNER" >&2
    echo "          Keep nestful_mtgrpo_partial next to nestful_mtgrpo_minimal." >&2
    exit 1
fi

# Wire the sibling curriculum loop to THIS experiment.
export RUN_PY="${RUN_PY:-$HERE/run.py}"
export CONFIG="${CONFIG:-$HERE/config.yaml}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$HERE/outputs/curriculum}"
# DATA_BASE intentionally left unset -> sibling default ($SIBLING/data/...).

# Default W&B project for this experiment (only used if WANDB_PROJECT is set by user;
# we just provide a sensible default name when they enable W&B without naming it).
if [ -n "${WANDB_API_KEY:-}" ] && [ -z "${WANDB_PROJECT:-}" ]; then
    export WANDB_PROJECT="nestful-mtgrpo-partial"
    echo "[partial] WANDB_PROJECT defaulted to $WANDB_PROJECT"
fi

echo "=============================================================="
echo " NESTFUL MT-GRPO PARTIAL curriculum"
echo "   RUN_PY      : $RUN_PY"
echo "   CONFIG      : $CONFIG"
echo "   OUTPUT_ROOT : $OUTPUT_ROOT"
echo "   sibling     : $SIBLING"
echo "=============================================================="

# Hand off to the sibling curriculum loop (passes through all PROFILE/STAGES/... env).
exec bash "$SIBLING_RUNNER" "$@"
