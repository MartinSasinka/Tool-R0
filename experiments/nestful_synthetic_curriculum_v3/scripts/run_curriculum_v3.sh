#!/usr/bin/env bash
# NESTFUL Synthetic Curriculum v3 / v3.1 — training launcher.
#
# Usage (v3.1 pod dry-run):
#   DRY_RUN=1 ALLOW_PROTOTYPE_TRAINING=1 CURRICULUM_VERSION=v3_1 STAGES="1 2" \
#     bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
V3="$REPO/experiments/nestful_synthetic_curriculum_v3"
MINIMAL="$REPO/experiments/nestful_mtgrpo_minimal"
PARTIAL="$REPO/experiments/nestful_mtgrpo_partial"
PYTHON="${PYTHON:-python}"

CURRICULUM_VERSION="${CURRICULUM_VERSION:-v3}"
CURRICULUM_VERSION="${CURRICULUM_VERSION,,}"
CURRICULUM_VERSION="${CURRICULUM_VERSION//-/_}"

if [ "$CURRICULUM_VERSION" = "v3_1" ] || [ "$CURRICULUM_VERSION" = "v31" ]; then
  CURR="$V3/outputs/curriculum_v3_1/filtered"
  CURR_RAW="$V3/outputs/curriculum_v3_1"
  MANIFEST="$CURR_RAW/curriculum_v3_1_manifest.json"
  PREFLIGHT_VERSION="v3_1"
  REWARD_POLICY="execution_aware_v3_1_stepwise"
  STAGE1_FILE="stage1_1call_atomic.jsonl"
  STAGE2_FILE="stage2_2call_dependency.jsonl"
  STAGE3_FILE="stage3_3call_composition.jsonl"
  STAGE4_FILE="stage4_4to6call_persistence.jsonl"
  REPLAY_S2="${CURRICULUM_REPLAY_WEIGHTS_S2:-0.20}"
  REPLAY_S3="${CURRICULUM_REPLAY_WEIGHTS_S3:-0.25}"
  REPLAY_S4="${CURRICULUM_REPLAY_WEIGHTS_S4:-0.30}"
else
  CURR="$V3/outputs/curriculum_v3"
  CURR_RAW="$CURR"
  MANIFEST="$CURR/curriculum_manifest.json"
  PREFLIGHT_VERSION="v3"
  REWARD_POLICY="execution_aware_v2_1_motif"
  STAGE1_FILE="stage1_linear_simple.jsonl"
  STAGE2_FILE="stage2_reference_reuse.jsonl"
  STAGE3_FILE="stage3_structural_motifs.jsonl"
  STAGE4_FILE="stage4_nestful_like_mixed.jsonl"
  REPLAY_S2="${CURRICULUM_REPLAY_WEIGHTS_S2:-0.35}"
  REPLAY_S3="${CURRICULUM_REPLAY_WEIGHTS_S3:-0.50}"
  REPLAY_S4="${CURRICULUM_REPLAY_WEIGHTS_S4:-0.65}"
fi

echo "[curriculum] version=$CURRICULUM_VERSION preflight=$PREFLIGHT_VERSION reward=$REWARD_POLICY"

if [ ! -f "$MANIFEST" ]; then
  echo "[curriculum] ERROR: missing manifest $MANIFEST — run build pipeline first" >&2
  exit 1
fi

# Prefer filtered dir for v3_1; fall back to raw stage files
if [ ! -d "$CURR" ] || [ -z "$(ls -A "$CURR"/stage*.jsonl 2>/dev/null || true)" ]; then
  CURR="$CURR_RAW"
fi

DEV="$MINIMAL/data/splits/nestful_dev.jsonl"
TEST="$MINIMAL/data/splits/nestful_test.jsonl"
for f in "$DEV" "$TEST"; do
  if [ ! -f "$f" ]; then
    echo "[curriculum] ERROR: missing split $f" >&2
    exit 1
  fi
done

if [ "$CURRICULUM_VERSION" = "v3_1" ] || [ "$CURRICULUM_VERSION" = "v31" ]; then
  PREFLIGHT_ARGS=(--curriculum-version v3_1 --out_dir "$V3/outputs")
  if [ "${ALLOW_PROTOTYPE_TRAINING:-0}" = "1" ]; then
    PREFLIGHT_ARGS+=(--prototype-only)
  fi
  "$PYTHON" "$V3/scripts/run_preflight_gates.py" "${PREFLIGHT_ARGS[@]}"
  PREFLIGHT_STATUS="$( "$PYTHON" -c "import json; print(json.load(open('$CURR_RAW/preflight_gates_summary.json'))['status'])" )"
else
  "$PYTHON" "$V3/scripts/validate_synthetic_tasks.py" --input "$CURR"
  "$PYTHON" "$V3/scripts/run_distribution_audit.py" || true
  "$PYTHON" "$V3/scripts/replay_synthetic_gold_traces.py" --input "$CURR"
  "$PYTHON" "$V3/scripts/run_tool_family_realism.py"
  PREFLIGHT_ARGS=()
  if [ "${ALLOW_PROTOTYPE_TRAINING:-0}" = "1" ]; then
    PREFLIGHT_ARGS+=(--prototype-only)
  fi
  "$PYTHON" "$V3/scripts/run_preflight_gates.py" "${PREFLIGHT_ARGS[@]}"
  PREFLIGHT_STATUS="$( "$PYTHON" -c "import json; print(json.load(open('$V3/outputs/preflight_gates_summary.json'))['status'])" )"
fi

if [ "$PREFLIGHT_STATUS" = "FAIL" ]; then
  echo "[curriculum] ERROR: preflight FAIL" >&2
  exit 1
fi
if [ "$PREFLIGHT_STATUS" = "PASS_PROTOTYPE_ONLY" ] && [ "${ALLOW_PROTOTYPE_TRAINING:-0}" != "1" ]; then
  echo "[curriculum] ERROR: prototype-only — set ALLOW_PROTOTYPE_TRAINING=1" >&2
  exit 1
fi

if [ -n "${OUTPUT_ROOT:-}" ]; then
  echo "[curriculum] resuming OUTPUT_ROOT=$OUTPUT_ROOT"
else
  TS="$(date +%Y%m%d_%H%M%S)"
  export OUTPUT_ROOT="$V3/outputs/runs/${TS}_v3_1"
  if [ "$CURRICULUM_VERSION" != "v3_1" ] && [ "$CURRICULUM_VERSION" != "v31" ]; then
    export OUTPUT_ROOT="$V3/outputs/runs/$TS"
  fi
fi

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
export CURRICULUM_VERSION
export CURRICULUM_REPLAY_WEIGHTS_S2="$REPLAY_S2"
export CURRICULUM_REPLAY_WEIGHTS_S3="$REPLAY_S3"
export CURRICULUM_REPLAY_WEIGHTS_S4="$REPLAY_S4"

EXTRA_REWARD="--override reward.train_policy=$REWARD_POLICY"
export EXTRA_TRAIN_OVERRIDES_STR="${EXTRA_TRAIN_OVERRIDES_STR:-$EXTRA_REWARD --override training.kl_beta=0.15 --override training.max_epochs_per_stage=2}"

if [ -n "${CHECKPOINT_IN:-}" ]; then
  export INIT_FROM="${INIT_FROM:-checkpoint}"
fi

if echo "$STAGES" | grep -qE '(^| )3( |$)|(^| )4( |$)'; then
  echo "[curriculum] ERROR: stage 3/4 require advance_gates on pod" >&2
  exit 1
fi

export DATA_BASE="$OUTPUT_ROOT/data_base"
mkdir -p "$DATA_BASE"
DATA_BASE="$(cd "$DATA_BASE" && pwd)"
export DATA_BASE

_link_stage_file() {
  local stage_file="$1" link_name="$2"
  local target="$CURR/$stage_file"
  local link="$DATA_BASE/$link_name"
  if [ ! -f "$target" ]; then
    target="$CURR_RAW/$stage_file"
  fi
  if [ ! -f "$target" ]; then
    echo "[curriculum] ERROR: missing $stage_file in $CURR or $CURR_RAW" >&2
    exit 1
  fi
  ln -sf "$target" "$link"
  if [ ! -f "$link" ]; then
    echo "[curriculum] ERROR: symlink failed $link -> $target" >&2
    exit 1
  fi
}

_link_stage_file "$STAGE1_FILE" "epoch_1_1call.jsonl"
_link_stage_file "$STAGE2_FILE" "epoch_2_2call.jsonl"
_link_stage_file "$STAGE3_FILE" "epoch_3_3call.jsonl"
_link_stage_file "$STAGE4_FILE" "epoch_4_4call.jsonl"

echo "[curriculum] DATA_BASE=$DATA_BASE (symlinks verified)"
echo "[curriculum] mapping: $STAGE1_FILE -> epoch_1_1call.jsonl"
echo "[curriculum] mapping: $STAGE2_FILE -> epoch_2_2call.jsonl"
echo "[curriculum] CONFIG=$CONFIG RUN_PY=$RUN_PY OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[curriculum] STAGES=$STAGES preflight=$PREFLIGHT_STATUS reward=$REWARD_POLICY"
echo "[curriculum] replay_weights s2=$REPLAY_S2 s3=$REPLAY_S3 s4=$REPLAY_S4"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[curriculum] DRY_RUN=1 — skipping training"
  exit 0
fi

if [ -n "${CHECKPOINT_IN:-}" ]; then
  if [ ! -f "$CHECKPOINT_IN/adapter_config.json" ]; then
    echo "[curriculum] ERROR: invalid CHECKPOINT_IN" >&2
    exit 1
  fi
fi

USE_VLLM="${USE_VLLM:-1}" \
  ROLLOUT_DP_GPUS="${ROLLOUT_DP_GPUS:-1,2,3}" \
  DP_LEARNER_GPU="${DP_LEARNER_GPU:-0}" \
  bash "$MINIMAL/run_curriculum.sh"

echo "[curriculum] done. Best: $OUTPUT_ROOT/best_react_win_adapter"
