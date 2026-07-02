#!/usr/bin/env bash
# NESTFUL Synthetic Curriculum v3 — training launcher (POD ONLY skeleton).
#
# Preflight: validation + distribution audit + gold replay + preflight gates must pass.
# Does NOT start training automatically during local dev — run explicitly on pod.
#
# Usage (pod):
#   USE_VLLM=1 ROLLOUT_DP_GPUS="1,2,3" DP_LEARNER_GPU=0 \
#     bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
V3="$REPO/experiments/nestful_synthetic_curriculum_v3"
MINIMAL="$REPO/experiments/nestful_mtgrpo_minimal"
PARTIAL="$REPO/experiments/nestful_mtgrpo_partial"
PYTHON="${PYTHON:-python}"

echo "[curriculum_v3] preflight checks ..."

CURR="$V3/outputs/curriculum_v3"
if [ ! -f "$CURR/curriculum_manifest.json" ]; then
  echo "[curriculum_v3] ERROR: run build_curriculum_v3.py first" >&2
  exit 1
fi

DEV="$MINIMAL/data/splits/nestful_dev.jsonl"
TEST="$MINIMAL/data/splits/nestful_test.jsonl"
for f in "$DEV" "$TEST"; do
  if [ ! -f "$f" ]; then
    echo "[curriculum_v3] ERROR: missing split $f — run make_nestful_dev_split.py" >&2
    exit 1
  fi
done

"$PYTHON" "$V3/scripts/validate_synthetic_tasks.py" --input "$CURR"
"$PYTHON" "$V3/scripts/run_distribution_audit.py" || {
  echo "[curriculum_v3] WARNING: distribution audit warnings (continuing if not FAIL)"
}
"$PYTHON" "$V3/scripts/replay_synthetic_gold_traces.py" --input "$CURR"
"$PYTHON" "$V3/scripts/run_tool_family_realism.py"

PREFLIGHT_ARGS=()
if [ "${ALLOW_PROTOTYPE_TRAINING:-0}" = "1" ]; then
  PREFLIGHT_ARGS+=(--prototype-only)
fi
"$PYTHON" "$V3/scripts/run_preflight_gates.py" "${PREFLIGHT_ARGS[@]}"
PREFLIGHT_STATUS="$( "$PYTHON" -c "import json; print(json.load(open('$V3/outputs/preflight_gates_summary.json'))['status'])" )"

if [ "$PREFLIGHT_STATUS" = "FAIL" ]; then
  echo "[curriculum_v3] ERROR: preflight gates FAIL — training blocked" >&2
  exit 1
fi
if [ "$PREFLIGHT_STATUS" = "PASS_PROTOTYPE_ONLY" ] && [ "${ALLOW_PROTOTYPE_TRAINING:-0}" != "1" ]; then
  echo "[curriculum_v3] ERROR: prototype-only dataset — set ALLOW_PROTOTYPE_TRAINING=1 to train" >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
export OUTPUT_ROOT="$V3/outputs/runs/$TS"
export RUN_PY="$V3/run.py"
export CONFIG="${CONFIG:-$PARTIAL/config.yaml}"
export VAL_JSONL="$DEV"
export PROFILE=stabilized_curriculum
export STAGES="${STAGES:-1 2}"
export MAX_EPOCHS_PER_STAGE="${MAX_EPOCHS_PER_STAGE:-2}"
export STABILIZED_LR="${STABILIZED_LR:-5e-7}"
export STABILIZED_KL="${STABILIZED_KL:-0.15}"
export REGRESSION_GUARD="${REGRESSION_GUARD:-1}"
export REGRESSION_EARLY_ABORT="${REGRESSION_EARLY_ABORT:-1}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export EXTRA_TRAIN_OVERRIDES_STR="${EXTRA_TRAIN_OVERRIDES_STR:---override reward.train_policy=execution_aware_v2_1_motif --override training.kl_beta=0.15 --override training.max_epochs_per_stage=2}"

if echo "$STAGES" | grep -qE '(^| )3( |$)|(^| )4( |$)'; then
  echo "[curriculum_v3] ERROR: stage 3/4 require explicit advance_gates on pod (dev Win vs baseline)" >&2
  exit 1
fi

# TODO: per-stage TRAIN_JSONL from curriculum_manifest (not epoch_N_Ncall.jsonl).
export DATA_BASE="$OUTPUT_ROOT/data_base"
mkdir -p "$DATA_BASE"
ln -sf "$CURR/stage1_linear_simple.jsonl" "$DATA_BASE/epoch_1_1call.jsonl"
ln -sf "$CURR/stage2_reference_reuse.jsonl" "$DATA_BASE/epoch_2_2call.jsonl"
ln -sf "$CURR/stage3_structural_motifs.jsonl" "$DATA_BASE/epoch_3_3call.jsonl"
ln -sf "$CURR/stage4_nestful_like_mixed.jsonl" "$DATA_BASE/epoch_4_4call.jsonl"

echo "[curriculum_v3] CONFIG=$CONFIG"
echo "[curriculum_v3] RUN_PY=$RUN_PY"
echo "[curriculum_v3] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[curriculum_v3] STAGES=$STAGES (default 1 2; stage 3/4 gated)"
echo "[curriculum_v3] preflight=$PREFLIGHT_STATUS"
echo "[curriculum_v3] reward override: execution_aware_v2_1_motif via v3/run.py"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[curriculum_v3] DRY_RUN=1 — skipping training invocation"
  exit 0
fi

USE_VLLM="${USE_VLLM:-1}" \
  ROLLOUT_DP_GPUS="${ROLLOUT_DP_GPUS:-1,2,3}" \
  DP_LEARNER_GPU="${DP_LEARNER_GPU:-0}" \
  bash "$MINIMAL/run_curriculum.sh"

echo "[curriculum_v3] done. Best checkpoint: $OUTPUT_ROOT/best_react_win_adapter"
