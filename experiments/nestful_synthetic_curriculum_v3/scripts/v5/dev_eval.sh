#!/usr/bin/env bash
# 7/9 — Deterministic dev evaluation of ONE checkpoint.
#
# temperature=0.0, top_p=1.0, 1 rollout, ReAct, full NESTFUL executor,
# official scorer, fixed dev set. Decoding is forced explicitly — it never
# inherits training settings.
#
# Env:  PYTHON=python3
#       CHECKPOINT=        adapter dir; leave EMPTY for the plain baseline
#       OUT_DIR=           (required) where metrics land
#       DEV_SET=<minimal>/data/splits/nestful_dev.jsonl
#       MAX_TASKS=0        0 = full dev set
#       USE_VLLM=0  EVAL_TP=  VLLM_GPU_UTIL=
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

OUT_DIR="${OUT_DIR:?set OUT_DIR for the eval outputs}"
DEV_SET="${DEV_SET:-$MINIMAL/data/splits/nestful_dev.jsonl}"
require_file "$DEV_SET" "DEV_SET"

banner "deterministic dev eval"
print_env PYTHON CHECKPOINT OUT_DIR DEV_SET MAX_TASKS USE_VLLM EVAL_TP VLLM_GPU_UTIL

ARGS=(scripts/eval/final_eval_v5.py run
  --label dev
  --out-dir "$OUT_DIR"
  --eval-set "$DEV_SET"
  --max-tasks "${MAX_TASKS:-0}")
if [ -n "${CHECKPOINT:-}" ]; then
  require_adapter "$CHECKPOINT" "CHECKPOINT"
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

cd "$V3"
"$PY" "${ARGS[@]}"

banner "dev eval done: $OUT_DIR/metrics_official.json"
