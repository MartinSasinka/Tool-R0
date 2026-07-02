#!/usr/bin/env bash
# Stabilized v2 PILOT run (execution_aware_v2, conservative, short).
#
# This wraps the sibling curriculum loop (../nestful_mtgrpo_minimal/run_curriculum.sh)
# with the partial driver (run.py, which can install execution_aware_v2) and the v2
# settings from the request:
#   reward = execution_aware_v2, init = baseline, 1-2 stages, 1 epoch/stage,
#   eval_every_epoch, num_gen 4, T 0.7, top_p 0.95, lr 5e-7, kl 0.05,
#   mixed_replay (synthetic-only), early stop on validation ReAct Win (patience 1),
#   max_turns_train = gold_n + 1, validation on the held-out SYNTHETIC split.
#
# Runs on the GPU pod. It first runs the OFFLINE preflight gates and aborts if
# they fail. NOTHING here is executed automatically by the v2 build step.
#
# Usage (on pod):
#   USE_VLLM=1 bash experiments/nestful_mtgrpo_partial/run_pilot_v2.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINIMAL="$(cd "$HERE/../nestful_mtgrpo_minimal" && pwd)"
COMPARISON="$(cd "$HERE/../comparison" && pwd)"
PYTHON="${PYTHON:-python}"

# ── Preflight gates (offline) ────────────────────────────────────────────────
echo "[pilot_v2] running offline preflight gates ..."
if ! "$PYTHON" "$COMPARISON/check_gates.py" preflight; then
    echo "[pilot_v2] ABORT: preflight gates failed." >&2
    exit 1
fi

# ── v2 pilot configuration ────────────────────────────────────────────────────
export PROFILE="${PROFILE:-stabilized_curriculum}"
export RUN_PY="$HERE/run.py"                       # partial driver (v2 reward aware)
export CONFIG="${CONFIG:-$HERE/config.yaml}"
export STAGES="${STAGES:-1 2}"                     # short pilot: 1-2 stages
export EPOCHS_PER_STAGE="${EPOCHS_PER_STAGE:-1}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export EVAL_EVERY_EPOCH=1
export EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-react_win_rate}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-1}"
export EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.005}"
export STABILIZED_LR="${STABILIZED_LR:-5e-7}"      # conservative
export STABILIZED_KL="${STABILIZED_KL:-0.10}"
export CURRICULUM_MIXED_REPLAY="${CURRICULUM_MIXED_REPLAY:-1}"
export DATA_BASE="${DATA_BASE:-$MINIMAL/data/clean_curriculum}"
export VAL_JSONL="${VAL_JSONL:-$MINIMAL/data/splits/nestful_dev.jsonl}"
export REGRESSION_GUARD="${REGRESSION_GUARD:-1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$HERE/outputs/execution_v2_pilot}"
export FINAL_EVAL=0                                 # pilot: no full final eval
# v2 reward + turn budget injected into the training invocation.
export EXTRA_TRAIN_OVERRIDES_STR="${EXTRA_TRAIN_OVERRIDES_STR:---override reward.train_policy=execution_aware_v2 --override generation.temperature=0.7 --override generation.top_p=0.95 --override training.max_extra_turns_train=1}"

echo "[pilot_v2] launching curriculum: reward=execution_aware_v2 stages='$STAGES' "
echo "           lr=$STABILIZED_LR kl=$STABILIZED_KL mixed_replay=$CURRICULUM_MIXED_REPLAY"
echo "           val=$VAL_JSONL out=$OUTPUT_ROOT"

bash "$MINIMAL/run_curriculum.sh"

echo "[pilot_v2] done. Build PILOT_REPORT.md from $OUTPUT_ROOT, then run:"
echo "  python $COMPARISON/check_gates.py pilot --metrics <pilot_metrics.json>"
