#!/usr/bin/env bash
# Stabilized v2 FULL run (execution_aware_v2) — only after the pilot passes gates.
#
# Same wiring as run_pilot_v2.sh but full curriculum: stages 1-4, up to 2-3
# epochs/stage, early stopping on validation ReAct Win, best checkpoint selected
# by validation ReAct Win (best_react_win_adapter). Final eval uses that adapter,
# NOT the last checkpoint.
#
# Usage (on pod, after pilot gates pass):
#   USE_VLLM=1 bash experiments/nestful_mtgrpo_partial/run_full_v2.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINIMAL="$(cd "$HERE/../nestful_mtgrpo_minimal" && pwd)"
COMPARISON="$(cd "$HERE/../comparison" && pwd)"
PYTHON="${PYTHON:-python}"

echo "[full_v2] running offline preflight gates ..."
if ! "$PYTHON" "$COMPARISON/check_gates.py" preflight; then
    echo "[full_v2] ABORT: preflight gates failed." >&2
    exit 1
fi

export PROFILE="${PROFILE:-stabilized_curriculum}"
export RUN_PY="$HERE/run.py"
export CONFIG="${CONFIG:-$HERE/config.yaml}"
export STAGES="${STAGES:-1 2 3 4}"
export EPOCHS_PER_STAGE="${EPOCHS_PER_STAGE:-2}"   # early stopping caps this (anti-forgetting)
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export EVAL_EVERY_EPOCH=1
export EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-react_win_rate}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
export EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.005}"
export STABILIZED_LR="${STABILIZED_LR:-5e-7}"
export STABILIZED_KL="${STABILIZED_KL:-0.10}"     # higher KL = less forgetting vs base
export CURRICULUM_MIXED_REPLAY="${CURRICULUM_MIXED_REPLAY:-1}"
export DATA_BASE="${DATA_BASE:-$MINIMAL/data/clean_curriculum}"
# Real NESTFUL dev (disjoint from nestful_test.jsonl) for val_eval / checkpoint selection.
# Build once: python experiments/comparison/make_nestful_dev_split.py
export VAL_JSONL="${VAL_JSONL:-$MINIMAL/data/splits/nestful_dev.jsonl}"
export REGRESSION_GUARD="${REGRESSION_GUARD:-1}"
export REGRESSION_EARLY_ABORT="${REGRESSION_EARLY_ABORT:-1}"
export REGRESSION_ABORT_PATIENCE="${REGRESSION_ABORT_PATIENCE:-3}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$HERE/outputs/execution_v2_mixed_replay_full}"
export FINAL_EVAL="${FINAL_EVAL:-1}"
export FINAL_EVAL_ADAPTER="${FINAL_EVAL_ADAPTER:-$OUTPUT_ROOT/best_react_win_adapter}"
export EXTRA_TRAIN_OVERRIDES_STR="${EXTRA_TRAIN_OVERRIDES_STR:---override reward.train_policy=execution_aware_v2 --override generation.temperature=0.7 --override generation.top_p=0.95 --override training.max_extra_turns_train=1}"

echo "[full_v2] launching FULL curriculum: reward=execution_aware_v2 stages='$STAGES'"
echo "          best checkpoint = best_react_win_adapter (NOT last)"

bash "$MINIMAL/run_curriculum.sh"

echo "[full_v2] done. Final eval should use: $FINAL_EVAL_ADAPTER"
