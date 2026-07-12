#!/usr/bin/env bash
# Multi-GPU MT-GRPO wrapper (Phase 1h) — validates topology / dataset / reward,
# writes a run manifest, prints the exact command, then forwards to the
# EXISTING trainer stack (scripts/run_curriculum_v3.sh -> nestful_mtgrpo_minimal/
# run_curriculum.sh). It does NOT reimplement or modify the trainer.
#
# Usage (repo root, Linux pod):
#   STAGES="2" bash experiments/nestful_synthetic_curriculum_v3/scripts/training/run_grpo.sh
#   DRY_RUN=1 STAGES="1 2" bash .../run_grpo.sh                     # print-only
#   SMOKE=1 STAGES="2" bash .../run_grpo.sh                         # tiny run
#   REWARD_POLICY=execution_aware_v3_2_dense STAGES="2" bash .../run_grpo.sh
#   STAGE2_FILE_OVERRIDE=/abs/path/filtered.jsonl STAGES="2" bash .../run_grpo.sh
#
# Env knobs:
#   STAGES               required, e.g. "2" or "1 2" (stage 4 always refused)
#   REWARD_POLICY        default execution_aware_v3_1_stepwise
#   ROLLOUT_DP_GPUS      default "1,2,3" (comma-separated GPU ids for rollout workers)
#   DP_LEARNER_GPU       default "0" (single GPU id for the learner)
#   NUM_GENERATIONS      default 4 | MAX_EPOCHS_PER_STAGE default 2
#   CHECKPOINT_IN        optional LoRA adapter to init from (INIT_FROM=checkpoint)
#   OUTPUT_ROOT          default outputs/runs/<ts>_v3_1
#   STAGE<N>_FILE_OVERRIDE  optional per-stage dataset override (e.g. probe-filtered)
#   WANDB_MODE / WANDB_PROJECT / WANDB_ENTITY / WANDB_GROUP / WANDB_TAGS  optional
#   DRY_RUN=1            validate + print + manifest preview, execute nothing
#   SMOKE=1              tiny run: 8 tasks, 2 generations, 1 epoch
if grep -q $'\r' "$0" 2>/dev/null; then
  exec /bin/bash <(sed 's/\r$//' "$0") "$@"
fi
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
V3="$REPO/experiments/nestful_synthetic_curriculum_v3"
PYTHON="${PYTHON:-python}"

STAGES="${STAGES:-}"
if [ -z "$STAGES" ]; then
  echo "[grpo] ERROR: STAGES is required (e.g. STAGES=\"2\")" >&2
  exit 1
fi
if echo "$STAGES" | grep -qE '(^| )4( |$)'; then
  echo "[grpo] ERROR: stage 4 is not allowed (pilot launcher policy)" >&2
  exit 1
fi

REWARD_POLICY="${REWARD_POLICY:-execution_aware_v3_1_stepwise}"
case "$REWARD_POLICY" in
  execution_aware_v3_1_stepwise|execution_aware_v3_2_dense|execution_aware_v2_1_motif|partial_gold_trace|strict) ;;
  *)
    echo "[grpo] ERROR: unknown REWARD_POLICY '$REWARD_POLICY'." >&2
    echo "  known: execution_aware_v3_1_stepwise execution_aware_v3_2_dense" >&2
    echo "         execution_aware_v2_1_motif partial_gold_trace strict" >&2
    exit 1;;
esac

ROLLOUT_DP_GPUS="${ROLLOUT_DP_GPUS:-1,2,3}"
DP_LEARNER_GPU="${DP_LEARNER_GPU:-0}"
export REWARD_POLICY STAGES ROLLOUT_DP_GPUS DP_LEARNER_GPU

# ── GPU topology validation ──────────────────────────────────────────────────
GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
fi
if ! echo "$DP_LEARNER_GPU" | grep -qE '^[0-9]+$'; then
  echo "[grpo] ERROR: DP_LEARNER_GPU must be a single GPU id (got '$DP_LEARNER_GPU')" >&2
  exit 1
fi
_ROLLOUT_LIST="$(echo "$ROLLOUT_DP_GPUS" | tr ',' ' ')"
for g in $_ROLLOUT_LIST; do
  if ! echo "$g" | grep -qE '^[0-9]+$'; then
    echo "[grpo] ERROR: ROLLOUT_DP_GPUS must be comma-separated ids (got '$ROLLOUT_DP_GPUS')" >&2
    exit 1
  fi
  if [ "$g" = "$DP_LEARNER_GPU" ]; then
    echo "[grpo] ERROR: learner GPU $DP_LEARNER_GPU overlaps ROLLOUT_DP_GPUS ($ROLLOUT_DP_GPUS)." >&2
    echo "       Rollout workers and the learner must use disjoint GPUs." >&2
    exit 1
  fi
done
_MAX_ID="$DP_LEARNER_GPU"
for g in $_ROLLOUT_LIST; do [ "$g" -gt "$_MAX_ID" ] && _MAX_ID="$g"; done
if [ "$GPU_COUNT" -gt 0 ]; then
  if [ "$_MAX_ID" -ge "$GPU_COUNT" ]; then
    if [ "${DRY_RUN:-0}" != "1" ]; then
      echo "[grpo] ERROR: topology references GPU $_MAX_ID but only $GPU_COUNT GPUs are visible." >&2
      exit 1
    fi
    echo "[grpo] WARNING: topology references GPU $_MAX_ID with $GPU_COUNT visible — OK only because DRY_RUN=1."
  fi
else
  if [ "${DRY_RUN:-0}" != "1" ]; then
    echo "[grpo] ERROR: no GPUs detected (nvidia-smi missing/empty) and DRY_RUN != 1." >&2
    exit 1
  fi
  echo "[grpo] WARNING: no GPUs detected — OK only because DRY_RUN=1."
fi

# ── dataset validation ───────────────────────────────────────────────────────
CURR="$V3/outputs/curriculum_v3_1/filtered"
AGENTIC_DATASET_DETECTED=0
for n in $STAGES; do
  ov_var="STAGE${n}_FILE_OVERRIDE"
  ov="${!ov_var:-}"
  if [ -n "$ov" ]; then
    if [ ! -f "$ov" ]; then
      echo "[grpo] ERROR: $ov_var=$ov not found" >&2; exit 1
    fi
    case "$ov" in *filtered_toolr0_synthetic*)
      echo "[grpo] ERROR: $ov_var points into LEGACY dataset B" >&2; exit 1;;
    esac
    # Agentic synthetic-tool datasets (curriculum_v4_*_agentic_*) are NOT real
    # IBM NESTFUL functions. executor.mode=auto resolves to `full` (the real
    # registry IS present in this repo) and either hard-fails every episode
    # on unknown_function, or worse, silently executes a colliding real IBM
    # function instead of the synthetic gold trace — corrupting the reward
    # either way. See docs/AGENTIC_DATA_GENERATION.md.
    case "$ov" in *agentic_openrouter*|*agentic_hybrid*|*agentic_workers*|*nestful_like_agentic*|*curriculum_v4*)
      AGENTIC_DATASET_DETECTED=1;;
    esac
    echo "[grpo] stage $n dataset: OVERRIDE $ov"
  else
    f="$(ls "$CURR"/stage${n}_*.jsonl 2>/dev/null | head -n1 || true)"
    if [ -z "$f" ]; then
      echo "[grpo] ERROR: no canonical stage-$n file under $CURR" >&2; exit 1
    fi
    echo "[grpo] stage $n dataset: $f"
  fi
done
if [ "$AGENTIC_DATASET_DETECTED" = "1" ]; then
  if ! echo "${EXTRA_TRAIN_OVERRIDES_STR:-}" | grep -q "executor.mode="; then
    EXTRA_TRAIN_OVERRIDES_STR="${EXTRA_TRAIN_OVERRIDES_STR:-} --override executor.mode=gold_replay"
    echo "[grpo] AGENTIC dataset detected — forcing executor.mode=gold_replay"
    echo "       (pass EXTRA_TRAIN_OVERRIDES_STR with executor.mode=... yourself to override)"
  fi
  export EXTRA_TRAIN_OVERRIDES_STR
fi

# ── init checkpoint validation ───────────────────────────────────────────────
if [ -n "${CHECKPOINT_IN:-}" ]; then
  if [ ! -f "$CHECKPOINT_IN/adapter_config.json" ]; then
    if [ "${DRY_RUN:-0}" != "1" ]; then
      echo "[grpo] ERROR: CHECKPOINT_IN=$CHECKPOINT_IN has no adapter_config.json" >&2
      exit 1
    fi
    # dry-run chains reference adapters a preceding (skipped) step would create
    echo "[grpo] WARNING: CHECKPOINT_IN does not exist yet — OK only because DRY_RUN=1."
  fi
  export INIT_FROM="${INIT_FROM:-checkpoint}"
  echo "[grpo] init adapter: $CHECKPOINT_IN"
fi

# ── output dir ───────────────────────────────────────────────────────────────
if [ -z "${OUTPUT_ROOT:-}" ]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  SUFFIX="_v3_1"
  [ "${SMOKE:-0}" = "1" ] && SUFFIX="_v3_1_smoke"
  export OUTPUT_ROOT="$V3/outputs/runs/${TS}${SUFFIX}"
fi

# ── smoke-mode knobs ─────────────────────────────────────────────────────────
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export MAX_EPOCHS_PER_STAGE="${MAX_EPOCHS_PER_STAGE:-2}"
EXTRA_TRAIN_OVERRIDES_STR="${EXTRA_TRAIN_OVERRIDES_STR:-}"
if [ "${SMOKE:-0}" = "1" ]; then
  export NUM_GENERATIONS=2
  export MAX_EPOCHS_PER_STAGE=1
  EXTRA_TRAIN_OVERRIDES_STR="$EXTRA_TRAIN_OVERRIDES_STR --override data.max_train_tasks=8"
  echo "[grpo] SMOKE=1 — 8 tasks, 2 generations, 1 epoch"
fi
export EXTRA_TRAIN_OVERRIDES_STR

echo "[grpo] stages          : $STAGES"
echo "[grpo] reward          : $REWARD_POLICY"
echo "[grpo] topology        : learner=GPU$DP_LEARNER_GPU rollout=[$ROLLOUT_DP_GPUS] (visible=$GPU_COUNT)"
echo "[grpo] output root     : $OUTPUT_ROOT"
echo "[grpo] wandb           : mode=${WANDB_MODE:-<unset>} project=${WANDB_PROJECT:-<disabled>}"

LAUNCH_ENV=(
  "CURRICULUM_VERSION=v3_1"
  "ALLOW_PROTOTYPE_TRAINING=1"
  "STAGES=$STAGES"
  "REWARD_POLICY=$REWARD_POLICY"
  "ROLLOUT_DP_GPUS=$ROLLOUT_DP_GPUS"
  "DP_LEARNER_GPU=$DP_LEARNER_GPU"
  "OUTPUT_ROOT=$OUTPUT_ROOT"
)
CMD="bash $V3/scripts/run_curriculum_v3.sh"
echo "[grpo] exact command   : ${LAUNCH_ENV[*]} $CMD"

# ── manifest (written BEFORE launch so aborted runs keep provenance) ─────────
_manifest_datasets=()
for n in $STAGES; do
  ov_var="STAGE${n}_FILE_OVERRIDE"; ov="${!ov_var:-}"
  if [ -n "$ov" ]; then _manifest_datasets+=(--dataset "$ov")
  else _manifest_datasets+=(--dataset "$(ls "$CURR"/stage${n}_*.jsonl | head -n1)"); fi
done
EXTRA_JSON="$("$PYTHON" -c "
import json, os
print(json.dumps({
    'reward_policy': os.environ.get('REWARD_POLICY'),
    'stages': os.environ.get('STAGES'),
    'topology': {'learner_gpu': os.environ.get('DP_LEARNER_GPU'),
                 'rollout_dp_gpus': os.environ.get('ROLLOUT_DP_GPUS')},
    'init_adapter': os.environ.get('CHECKPOINT_IN') or None,
    'num_generations': os.environ.get('NUM_GENERATIONS'),
    'max_epochs_per_stage': os.environ.get('MAX_EPOCHS_PER_STAGE'),
    'smoke': os.environ.get('SMOKE', '0') == '1',
    'wandb_project': os.environ.get('WANDB_PROJECT') or None,
}))" 2>/dev/null || echo '{}')"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[grpo] DRY_RUN=1 — forwarding dry-run to curriculum launcher (no training)."
  DRY_RUN=1 env "${LAUNCH_ENV[@]}" $CMD
  echo "[grpo] dry run complete; nothing trained."
  exit 0
fi

mkdir -p "$OUTPUT_ROOT"
REWARD_POLICY="$REWARD_POLICY" STAGES="$STAGES" \
  DP_LEARNER_GPU="$DP_LEARNER_GPU" ROLLOUT_DP_GPUS="$ROLLOUT_DP_GPUS" \
  "$PYTHON" "$V3/scripts/lib/run_manifest.py" \
    --out "$OUTPUT_ROOT/manifest.json" --kind grpo_train \
    "${_manifest_datasets[@]}" --extra "$EXTRA_JSON"

env "${LAUNCH_ENV[@]}" $CMD
echo "[grpo] done — outputs in $OUTPUT_ROOT"
