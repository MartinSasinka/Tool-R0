#!/usr/bin/env bash
# Two-phase v5 GRPO — continuous in-process training (Stage2 → C1 → Stage3+replay → C2).
# C1/C2 dev eval run AFTER training teardown (EVAL_TP=4 needs all GPUs).
#
# ALWAYS export RUN_DIR and mkdir BEFORE tee:
#   export RUN_DIR="outputs/runs/two_phase_$(date +%Y%m%d_%H%M%S)"
#   mkdir -p "$RUN_DIR"
#   bash scripts/v5/run_two_phase_grpo.sh 2>&1 | tee "$RUN_DIR/console.log"
#
# Phase-level resume (NOT exact optimizer-step resume):
#   export RUN_DIR="outputs/runs/<existing>"
#   export RESUME=1
#   bash scripts/v5/run_two_phase_grpo.sh 2>&1 | tee -a "$RUN_DIR/console.log"
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

RUN_DIR="${RUN_DIR:?export RUN_DIR first (see header comments)}"
mkdir -p "$RUN_DIR"

PHASE1="${PHASE1_DATASET:-$V3/data/training_ready_v5/filtered/phase1_stage2_train.jsonl}"
PHASE2="${PHASE2_DATASET:-$V3/data/training_ready_v5/filtered/phase2_stage3_plus_stage2_replay.jsonl}"
DEV_SET="${DEV_SET:-$MINIMAL/data/splits/nestful_dev.jsonl}"

require_file "$PHASE1" "PHASE1_DATASET"
require_file "$PHASE2" "PHASE2_DATASET"
require_file "$DEV_SET" "DEV_SET"

if [ "${RESUME:-0}" != "1" ] && [ -f "$RUN_DIR/two_phase_state.json" ]; then
  echo "[two-phase] ERROR: $RUN_DIR has state; export RESUME=1 or pick new RUN_DIR" >&2
  exit 1
fi

export SEED="${SEED:-42}"
export DATA_SEED="${DATA_SEED:-42}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export WANDB_PROJECT="${WANDB_PROJECT:-nestful-v5-curriculum}"
export WANDB_RUN_GROUP="${WANDB_GROUP:-$(basename "$RUN_DIR")}"

# Targeted cleanup: only rollout worker PIDs recorded by the orchestrator.
_cleanup() {
  local pidfile="$RUN_DIR/logs/rollout_worker_pids.json"
  if [ -f "$pidfile" ]; then
    "$PY" - <<'PY' "$pidfile" 2>/dev/null || true
import json, os, signal, sys
path = sys.argv[1]
try:
    pids = json.load(open(path, encoding="utf-8"))
except Exception:
    sys.exit(0)
for pid in pids:
    try:
        os.kill(int(pid), signal.SIGTERM)
    except (ProcessLookupError, ValueError, PermissionError):
        pass
PY
  fi
}
trap _cleanup EXIT INT TERM

banner "two-phase v5 GRPO (continuous session)"
print_env RUN_DIR PHASE1 PHASE2 DEV_SET SEED DATA_SEED ROLLOUT_SEED \
          NUM_GENERATIONS LEARNING_RATE KL_BETA TEMPERATURE TOP_P REWARD_POLICY \
          MAX_TRAIN_TASKS DEV_MAX_TASKS USE_VLLM ROLLOUT_DP_GPUS EVAL_TP VLLM_GPU_UTIL \
          WANDB_PROJECT WANDB_ENTITY WANDB_RUN_GROUP RESUME

ARGS=(scripts/training/run_two_phase_v5_grpo.py --run-dir "$RUN_DIR"
  --phase1-dataset "$PHASE1" --phase2-dataset "$PHASE2" --dev-set "$DEV_SET"
  --num-generations "${NUM_GENERATIONS:-8}"
  --learning-rate "${LEARNING_RATE:-3e-7}"
  --kl-beta "${KL_BETA:-0.15}"
  --temperature "${TEMPERATURE:-1.0}"
  --top-p "${TOP_P:-0.95}"
  --reward-policy "${REWARD_POLICY:-execution_aware_v3_2_dense}"
  --max-train-tasks "${MAX_TRAIN_TASKS:-0}"
  --dev-max-tasks "${DEV_MAX_TASKS:-0}")
[ "${SKIP_PREFLIGHT:-0}" = "1" ] && ARGS+=(--skip-preflight)
[ "${SKIP_BASELINE_EVAL:-0}" = "1" ] && ARGS+=(--skip-baseline-eval)
[ "${RESUME:-0}" = "1" ] && ARGS+=(--resume)

cd "$V3"
"$PY" "${ARGS[@]}"

banner "finished: $RUN_DIR"
