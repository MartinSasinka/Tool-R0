#!/usr/bin/env bash
# NESTFUL Synthetic Curriculum v3 / v3.1 — training launcher.
#
# Usage (v3.1 pod dry-run):
#   DRY_RUN=1 ALLOW_PROTOTYPE_TRAINING=1 CURRICULUM_VERSION=v3_1 STAGES="1 2" \
#     bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
# Re-exec without CRLF when checked out on Windows (RunPod bash rejects pipefail\r).
if grep -q $'\r' "$0" 2>/dev/null; then
  exec /bin/bash <(sed 's/\r$//' "$0") "$@"
fi
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
V3="$REPO/experiments/nestful_synthetic_curriculum_v3"
MINIMAL="$REPO/experiments/nestful_mtgrpo_minimal"
PARTIAL="$REPO/experiments/nestful_mtgrpo_partial"
PYTHON="${PYTHON:-python}"

# Default is the CANONICAL v3.1 corpus (cleanup Phase K). The pre-v3.1 corpus
# was archived to archive/curriculum_v3/ and now requires an explicit opt-in.
CURRICULUM_VERSION="${CURRICULUM_VERSION:-v3_1}"
CURRICULUM_VERSION="${CURRICULUM_VERSION,,}"
CURRICULUM_VERSION="${CURRICULUM_VERSION//-/_}"

if [ "$CURRICULUM_VERSION" = "v3_1" ] || [ "$CURRICULUM_VERSION" = "v31" ]; then
  CURR="$V3/outputs/curriculum_v3_1/filtered"
  CURR_RAW="$V3/outputs/curriculum_v3_1"
  MANIFEST="$CURR_RAW/curriculum_v3_1_manifest.json"
  PREFLIGHT_VERSION="v3_1"
  # REWARD_POLICY is overridable so the SAME pipeline can run the v2.1 motif
  # ablation on the v3.1 dataset: REWARD_POLICY=execution_aware_v2_1_motif.
  REWARD_POLICY="${REWARD_POLICY:-execution_aware_v3_1_stepwise}"
  STAGE1_FILE="stage1_1call_atomic.jsonl"
  STAGE2_FILE="stage2_2call_dependency.jsonl"
  STAGE3_FILE="stage3_3call_composition.jsonl"
  STAGE4_FILE="stage4_4to6call_persistence.jsonl"
  # Scalar per-stage values are REPLAY RATIOS (audit Bug 6): 0.20 => previous
  # stages total 20%, current stage 80%. run_curriculum.sh forwards a scalar
  # as data.replay_ratio (a CSV list keeps explicit per-stage weights).
  REPLAY_S2="${CURRICULUM_REPLAY_WEIGHTS_S2:-0.20}"
  REPLAY_S3="${CURRICULUM_REPLAY_WEIGHTS_S3:-0.25}"
  REPLAY_S4="${CURRICULUM_REPLAY_WEIGHTS_S4:-0.30}"
else
  # LEGACY corpus (pre-v3.1), archived by cleanup Phase K. Kept runnable only
  # for reproducing the July-2 pilot; requires an explicit opt-in.
  if [ "${ALLOW_LEGACY_CURRICULUM_V3:-0}" != "1" ]; then
    echo "[curriculum] ABORT: CURRICULUM_VERSION=v3 selects the ARCHIVED pre-v3.1 corpus" >&2
    echo "  (archive/curriculum_v3). Use CURRICULUM_VERSION=v3_1 (canonical), or set" >&2
    echo "  ALLOW_LEGACY_CURRICULUM_V3=1 to run the legacy corpus explicitly." >&2
    exit 1
  fi
  CURR="$V3/archive/curriculum_v3"
  CURR_RAW="$CURR"
  MANIFEST="$CURR/curriculum_manifest.json"
  PREFLIGHT_VERSION="v3"
  REWARD_POLICY="${REWARD_POLICY:-execution_aware_v2_1_motif}"
  STAGE1_FILE="stage1_linear_simple.jsonl"
  STAGE2_FILE="stage2_reference_reuse.jsonl"
  STAGE3_FILE="stage3_structural_motifs.jsonl"
  STAGE4_FILE="stage4_nestful_like_mixed.jsonl"
  REPLAY_S2="${CURRICULUM_REPLAY_WEIGHTS_S2:-0.35}"
  REPLAY_S3="${CURRICULUM_REPLAY_WEIGHTS_S3:-0.50}"
  REPLAY_S4="${CURRICULUM_REPLAY_WEIGHTS_S4:-0.65}"
fi
export REWARD_POLICY

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
  # stdin redirect (not open(path)) so MSYS-style /c/... paths work on Windows
  PREFLIGHT_STATUS="$( "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['status'])" < "$CURR_RAW/preflight_gates_summary.json" )"
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

# Ensure parent exists before path normalization — otherwise a missing
# outputs/runs/ dir resolves to /${TS}_v3_1 at filesystem root.
mkdir -p "$(dirname "$OUTPUT_ROOT")"

if [ -d "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
else
  OUTPUT_ROOT="$(cd "$(dirname "$OUTPUT_ROOT")" && pwd)/$(basename "$OUTPUT_ROOT")"
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
export ALLOW_NO_REGRESSION_GUARD="${ALLOW_NO_REGRESSION_GUARD:-0}"
export ALLOW_STRICT_REWARD_FALLBACK="${ALLOW_STRICT_REWARD_FALLBACK:-0}"
# Hard per-stage advancement gates (check_stage_gates.py) — default ON for
# v3/v3.1 runs; a failing stage stops the run instead of advancing.
export STAGE_GATES="${STAGE_GATES:-1}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
# Teacher-forced continuation training (off by default). Intended for a
# dedicated Stage2b-style sub-run, NOT the main curriculum — see
# scripts/pilot/run_stage2b_teacher_forced.sh.
export TEACHER_FORCED_PREFIX_CALLS="${TEACHER_FORCED_PREFIX_CALLS:-0}"
export CURRICULUM_VERSION
export CURRICULUM_REPLAY_WEIGHTS_S2="$REPLAY_S2"
export CURRICULUM_REPLAY_WEIGHTS_S3="$REPLAY_S3"
export CURRICULUM_REPLAY_WEIGHTS_S4="$REPLAY_S4"
# Optional deterministic eval decoding (pilot: EVAL_TEMPERATURE=0.0).
if [ -n "${EVAL_TEMPERATURE:-}" ]; then export EVAL_TEMPERATURE; fi

# Always inject the reward policy; append optional trainer knobs. A wrapper
# may pre-set EXTRA_TRAIN_OVERRIDES_STR — the reward override is appended so
# it can never be silently dropped.
EXTRA_REWARD="--override reward.train_policy=$REWARD_POLICY"
if [ -z "${EXTRA_TRAIN_OVERRIDES_STR:-}" ]; then
  EXTRA_TRAIN_OVERRIDES_STR="$EXTRA_REWARD --override training.kl_beta=${TRAIN_KL_BETA:-0.15}"
elif ! echo "$EXTRA_TRAIN_OVERRIDES_STR" | grep -q "reward.train_policy="; then
  EXTRA_TRAIN_OVERRIDES_STR="$EXTRA_TRAIN_OVERRIDES_STR $EXTRA_REWARD"
fi
if [ -n "${TRAIN_TEMPERATURE:-}" ] && \
   ! echo "$EXTRA_TRAIN_OVERRIDES_STR" | grep -q "generation.temperature="; then
  EXTRA_TRAIN_OVERRIDES_STR="$EXTRA_TRAIN_OVERRIDES_STR --override generation.temperature=$TRAIN_TEMPERATURE"
fi
export EXTRA_TRAIN_OVERRIDES_STR

if [ -n "${CHECKPOINT_IN:-}" ]; then
  export INIT_FROM="${INIT_FROM:-checkpoint}"
fi

# Stage 4 is NEVER allowed from this launcher; stage 3 requires the hard
# stage gates (STAGE_GATES=1) so it only runs when stage 2 passed.
if echo "$STAGES" | grep -qE '(^| )4( |$)'; then
  echo "[curriculum] ERROR: stage 4 is not allowed in this pilot launcher" >&2
  exit 1
fi
if echo "$STAGES" | grep -qE '(^| )3( |$)' && [ "$STAGE_GATES" != "1" ]; then
  echo "[curriculum] ERROR: stage 3 requires STAGE_GATES=1 (hard advancement gates)" >&2
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
  # Optional per-stage dataset override (absolute JSONL path), e.g. a probe-
  # filtered signal-positive file: STAGE2_FILE_OVERRIDE=/path/to/filtered.jsonl.
  local stage_n="${link_name#epoch_}"; stage_n="${stage_n%%_*}"
  local override_var="STAGE${stage_n}_FILE_OVERRIDE"
  local override_val="${!override_var:-}"
  if [ -n "$override_val" ]; then
    if [ ! -f "$override_val" ]; then
      echo "[curriculum] ERROR: $override_var=$override_val does not exist" >&2
      exit 1
    fi
    case "$override_val" in
      *filtered_toolr0_synthetic*)
        echo "[curriculum] ERROR: $override_var points into LEGACY dataset B" >&2
        exit 1;;
    esac
    echo "[curriculum] stage $stage_n dataset OVERRIDE: $override_val"
    target="$override_val"
  fi
  if [ ! -f "$target" ]; then
    target="$CURR_RAW/$stage_file"
  fi
  if [ ! -f "$target" ]; then
    echo "[curriculum] ERROR: missing $stage_file in $CURR or $CURR_RAW" >&2
    exit 1
  fi
  # Symlink on the pod; plain copy where symlinks are unavailable (Windows
  # Git Bash dry-runs) so DRY_RUN=1 works on any platform.
  ln -sf "$target" "$link" 2>/dev/null || cp "$target" "$link"
  if [ ! -f "$link" ]; then
    echo "[curriculum] ERROR: could not stage $link -> $target" >&2
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
