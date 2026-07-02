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

if [ -n "${OUTPUT_ROOT:-}" ]; then
  echo "[curriculum_v3] resuming existing OUTPUT_ROOT=$OUTPUT_ROOT"
else
  TS="$(date +%Y%m%d_%H%M%S)"
  export OUTPUT_ROOT="$V3/outputs/runs/$TS"
fi
# Always use absolute paths so symlinks and DATA_BASE survive cwd changes.
if [ -d "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
else
  OUTPUT_ROOT="$(cd "$(dirname "$OUTPUT_ROOT")" 2>/dev/null && pwd)/$(basename "$OUTPUT_ROOT")"
fi
export OUTPUT_ROOT
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

# Resume: CHECKPOINT_IN requires INIT_FROM=checkpoint (baseline ignores the adapter).
if [ -n "${CHECKPOINT_IN:-}" ]; then
  export INIT_FROM="${INIT_FROM:-checkpoint}"
fi

if echo "$STAGES" | grep -qE '(^| )3( |$)|(^| )4( |$)'; then
  echo "[curriculum_v3] ERROR: stage 3/4 require explicit advance_gates on pod (dev Win vs baseline)" >&2
  exit 1
fi

# TODO: per-stage TRAIN_JSONL from curriculum_manifest (not epoch_N_Ncall.jsonl).
export DATA_BASE="$OUTPUT_ROOT/data_base"
mkdir -p "$DATA_BASE"
DATA_BASE="$(cd "$DATA_BASE" && pwd)"
export DATA_BASE

_link_stage_file() {
  local stage_file="$1" link_name="$2"
  local target="$CURR/$stage_file"
  local link="$DATA_BASE/$link_name"
  if [ ! -f "$target" ]; then
    echo "[curriculum_v3] ERROR: missing curriculum file: $target" >&2
    exit 1
  fi
  ln -sf "$target" "$link"
  if [ ! -f "$link" ]; then
    echo "[curriculum_v3] ERROR: failed to link $link -> $target" >&2
    exit 1
  fi
}

_link_stage_file "stage1_linear_simple.jsonl" "epoch_1_1call.jsonl"
_link_stage_file "stage2_reference_reuse.jsonl" "epoch_2_2call.jsonl"
_link_stage_file "stage3_structural_motifs.jsonl" "epoch_3_3call.jsonl"
_link_stage_file "stage4_nestful_like_mixed.jsonl" "epoch_4_4call.jsonl"
echo "[curriculum_v3] DATA_BASE=$DATA_BASE (symlinks verified)"

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

if [ -n "${CHECKPOINT_IN:-}" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  if [ ! -f "$CHECKPOINT_IN/adapter_config.json" ]; then
    echo "[curriculum_v3] ERROR: CHECKPOINT_IN is not a valid adapter: $CHECKPOINT_IN" >&2
    exit 1
  fi
  echo "[curriculum_v3] CHECKPOINT_IN=$CHECKPOINT_IN INIT_FROM=$INIT_FROM"
fi

USE_VLLM="${USE_VLLM:-1}" \
  ROLLOUT_DP_GPUS="${ROLLOUT_DP_GPUS:-1,2,3}" \
  DP_LEARNER_GPU="${DP_LEARNER_GPU:-0}" \
  bash "$MINIMAL/run_curriculum.sh"

echo "[curriculum_v3] done. Best checkpoint: $OUTPUT_ROOT/best_react_win_adapter"
