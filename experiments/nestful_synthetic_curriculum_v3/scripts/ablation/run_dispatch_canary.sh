#!/usr/bin/env bash
# Dispatch canary — RunPod launcher (A1 + A4, 24 tasks × 8 rollouts, no NESTFUL eval).
#
# Purpose: prove the reward-dispatch fix on the real GPU stack and dump
# per-rollout trajectories for credit audit. Does NOT measure NESTFUL Win Rate.
#
# Usage (from repo root on RunPod):
#   export WANDB_API_KEY=...
#   export HF_TOKEN=...
#   bash experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_dispatch_canary.sh
#
# One arm only:
#   bash .../run_dispatch_canary.sh --arm A1_OUTCOME_ONLY
#
# Resume / force-fresh:
#   bash .../run_dispatch_canary.sh --resume
#   bash .../run_dispatch_canary.sh --force-fresh
#
# After both arms finish:
#   python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/validate_dispatch_canary.py
set -Eeuo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V3="$(cd "$_HERE/../.." && pwd)"
REPO="$(cd "$V3/../.." && pwd)"
PY="${PYTHON:-python3}"

ARM=""
RESUME=0
FORCE_FRESH=0
VALIDATE_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --force-fresh) FORCE_FRESH=1; shift ;;
    --validate-only) VALIDATE_ONLY=1; shift ;;
    *) echo "[dispatch-canary] ERROR: unknown arg $1" >&2; exit 1 ;;
  esac
done

SEED="${SEED:-20260724}"
WANDB_PROJECT="${WANDB_PROJECT:-nestful-reward-ablation}"
WANDB_GROUP="${WANDB_GROUP:-dispatch_canary_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$V3/outputs/runs}"
CANARY_SUBSET="${CANARY_SUBSET:-$V3/reports/reward_ablation/data/canary_subset_24.jsonl}"

banner() {
  echo "──────────────────────────────────────────────────────────────"
  echo "[dispatch-canary] $1"
  echo "──────────────────────────────────────────────────────────────"
}

cd "$REPO"

if [ "$VALIDATE_ONLY" = "1" ]; then
  banner "validate only"
  exec "$PY" "$V3/scripts/ablation/validate_dispatch_canary.py" \
    --output-root "$OUTPUT_ROOT" --seed "$SEED"
fi

banner "GPU / env check"
command -v "$PY" >/dev/null || { echo "python missing" >&2; exit 1; }
"$PY" -c "import torch; assert torch.cuda.is_available(); print('GPUs', torch.cuda.device_count())"
export USE_VLLM="${USE_VLLM:-1}"
export ROLLOUT_DP_GPUS="${ROLLOUT_DP_GPUS:-1,2,3}"
export CANARY_TRAJ_LOG=1

banner "prepare canary subset (24 tasks)"
"$PY" "$V3/scripts/ablation/prepare_canary_subset_24.py"
[ -f "$CANARY_SUBSET" ] || { echo "missing $CANARY_SUBSET" >&2; exit 1; }

ARMS=(A1_OUTCOME_ONLY A4_GATED_VERIFIABLE)
if [ -n "$ARM" ]; then
  ARMS=("$ARM")
fi

for arm in "${ARMS[@]}"; do
  banner "train arm=$arm seed=$SEED (canary, no NESTFUL eval)"
  ARGS=(
    --round 2
    --reward-arm "$arm"
    --seed "$SEED"
    --canary
    --train-subset "$CANARY_SUBSET"
    --wandb-project "$WANDB_PROJECT"
    --wandb-group "$WANDB_GROUP"
    --output-root "$OUTPUT_ROOT"
    --run-id "dispatch_canary_${arm}_seed${SEED}"
  )
  if [ "$RESUME" = "1" ]; then ARGS+=(--resume); fi
  if [ "$FORCE_FRESH" = "1" ]; then ARGS+=(--force-fresh); fi
  "$PY" "$V3/scripts/ablation/run_reward_ablation.py" "${ARGS[@]}"
done

banner "validate canary gates"
"$PY" "$V3/scripts/ablation/validate_dispatch_canary.py" \
  --output-root "$OUTPUT_ROOT" --seed "$SEED"

banner "done"
echo "[dispatch-canary] If VERDICT=PASS → use A4_GATED_VERIFIABLE as working reward."
echo "[dispatch-canary] Round 1 remains reward_ablation_round1_INVALID_DISPATCH."
