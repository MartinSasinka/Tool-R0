#!/usr/bin/env bash
# SFT warmup + GRPO chain (Phase 1f / RESEARCH_FIX_PLAN E3-E4).
#
# Runs (or reuses) a Stage2 continuation SFT adapter, then starts GRPO FROM that
# adapter. Reuses the existing SFT script and the existing GRPO stack unchanged;
# this wrapper only chains them and records provenance.
#
# Cells this produces (evaluate ONLY via scripts/eval/eval_batch_temp0.sh):
#   base       -> no training                (baseline cell of the eval batch)
#   sft_only   -> <out>/sft/adapter/epoch_N
#   sft_grpo   -> <out>/grpo/... best adapter (GRPO initialized from SFT)
#   (pure GRPO comes from scripts/training/run_grpo.sh without CHECKPOINT_IN)
#
# Usage (repo root, Linux pod):
#   bash experiments/nestful_synthetic_curriculum_v3/scripts/training/run_sft_plus_grpo.sh
#   SFT_ADAPTER=/abs/path/adapter bash .../run_sft_plus_grpo.sh   # skip SFT step
#   DRY_RUN=1 bash .../run_sft_plus_grpo.sh                        # print-only
#   SMOKE=1 bash .../run_sft_plus_grpo.sh                          # tiny SFT + tiny GRPO
#
# Env knobs:
#   SFT_ADAPTER      existing adapter dir (skips the SFT step)
#   SFT_EPOCHS=1 SFT_LR=1e-5 SFT_BATCH_SIZE=1 SFT_GRAD_ACCUM=16   (SFT step)
#   GRPO_STAGES="2"  REWARD_POLICY=execution_aware_v3_1_stepwise  (GRPO step)
#   ROLLOUT_DP_GPUS / DP_LEARNER_GPU                              (topology)
#   OUTPUT_ROOT      default outputs/runs/<ts>_sft_plus_grpo
#   DRY_RUN=1 | SMOKE=1
if grep -q $'\r' "$0" 2>/dev/null; then
  exec /bin/bash <(sed 's/\r$//' "$0") "$@"
fi
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
V3="$REPO/experiments/nestful_synthetic_curriculum_v3"
PYTHON="${PYTHON:-python}"

DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
GRPO_STAGES="${GRPO_STAGES:-2}"

if [ -z "${OUTPUT_ROOT:-}" ]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_ROOT="$V3/outputs/runs/${TS}_sft_plus_grpo"
fi
SFT_OUT="$OUTPUT_ROOT/sft"
GRPO_OUT="$OUTPUT_ROOT/grpo"

echo "[sft+grpo] output root : $OUTPUT_ROOT"
echo "[sft+grpo] cells       : sft_only -> $SFT_OUT | sft_grpo -> $GRPO_OUT"
echo "[sft+grpo] grpo stages : $GRPO_STAGES | reward: ${REWARD_POLICY:-execution_aware_v3_1_stepwise}"

# ── step 1: SFT adapter (reuse existing script, or take a prebuilt adapter) ──
SFT_ADAPTER="${SFT_ADAPTER:-}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
if [ -n "$SFT_ADAPTER" ]; then
  if [ ! -f "$SFT_ADAPTER/adapter_config.json" ]; then
    echo "[sft+grpo] ERROR: SFT_ADAPTER=$SFT_ADAPTER has no adapter_config.json" >&2
    exit 1
  fi
  echo "[sft+grpo] step 1: SKIPPED (using existing SFT adapter: $SFT_ADAPTER)"
else
  SFT_CMD_ENV=(
    "OUTPUT_DIR=$SFT_OUT"
    "SFT_EPOCHS=$SFT_EPOCHS"
    "SFT_LR=${SFT_LR:-1e-5}"
    "SFT_BATCH_SIZE=${SFT_BATCH_SIZE:-1}"
    "SFT_GRAD_ACCUM=${SFT_GRAD_ACCUM:-16}"
  )
  if [ "$SMOKE" = "1" ]; then
    # tiny smoke: 1 epoch, short sequences; still real training on few steps
    SFT_CMD_ENV+=("SFT_EPOCHS=1" "SFT_MAX_SEQ_LEN=1024")
  fi
  echo "[sft+grpo] step 1: SFT warmup"
  echo "[sft+grpo]   cmd: ${SFT_CMD_ENV[*]} DRY_RUN=$DRY_RUN bash $V3/scripts/pilot/run_stage2_continuation_sft_warmup.sh"
  env "${SFT_CMD_ENV[@]}" DRY_RUN="$DRY_RUN" \
    bash "$V3/scripts/pilot/run_stage2_continuation_sft_warmup.sh"
  SFT_ADAPTER="$SFT_OUT/adapter/epoch_${SFT_EPOCHS}"
  if [ "$DRY_RUN" != "1" ] && [ ! -f "$SFT_ADAPTER/adapter_config.json" ]; then
    echo "[sft+grpo] ERROR: SFT step did not produce $SFT_ADAPTER" >&2
    exit 1
  fi
fi

# ── step 2: GRPO from the SFT adapter (via the validated wrapper) ───────────
echo "[sft+grpo] step 2: GRPO from SFT adapter"
GRPO_ENV=(
  "STAGES=$GRPO_STAGES"
  "OUTPUT_ROOT=$GRPO_OUT"
  "CHECKPOINT_IN=$SFT_ADAPTER"
  "INIT_FROM=checkpoint"
  "DRY_RUN=$DRY_RUN"
  "SMOKE=$SMOKE"
)
[ -n "${REWARD_POLICY:-}" ] && GRPO_ENV+=("REWARD_POLICY=$REWARD_POLICY")
[ -n "${ROLLOUT_DP_GPUS:-}" ] && GRPO_ENV+=("ROLLOUT_DP_GPUS=$ROLLOUT_DP_GPUS")
[ -n "${DP_LEARNER_GPU:-}" ] && GRPO_ENV+=("DP_LEARNER_GPU=$DP_LEARNER_GPU")
echo "[sft+grpo]   cmd: ${GRPO_ENV[*]} bash $V3/scripts/training/run_grpo.sh"
env "${GRPO_ENV[@]}" bash "$V3/scripts/training/run_grpo.sh"

# ── chain manifest (records the init adapter — Phase 1f acceptance) ─────────
if [ "$DRY_RUN" != "1" ]; then
  EXTRA_JSON="$("$PYTHON" -c "
import json
print(json.dumps({
    'chain': 'sft_plus_grpo',
    'init_adapter': '$SFT_ADAPTER',
    'sft_output': '$SFT_OUT',
    'grpo_output': '$GRPO_OUT',
    'grpo_stages': '$GRPO_STAGES',
    'smoke': '$SMOKE' == '1',
}))")"
  "$PYTHON" "$V3/scripts/lib/run_manifest.py" \
    --out "$OUTPUT_ROOT/manifest.json" --kind sft_plus_grpo --extra "$EXTRA_JSON"
fi

echo "[sft+grpo] done."
echo "[sft+grpo] evaluate with the eval batch runner, e.g.:"
echo "  CELLS=\"baseline sft_only=$SFT_ADAPTER sft_grpo=$GRPO_OUT/best_react_win_adapter\" \\"
echo "    DATASET=nestful_dev bash $V3/scripts/eval/eval_batch_temp0.sh"
