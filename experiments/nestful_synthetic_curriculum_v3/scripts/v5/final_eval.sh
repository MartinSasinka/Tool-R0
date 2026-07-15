#!/usr/bin/env bash
# 8/9 — Final temp-0 NESTFUL evaluation of ONE arm (baseline / best / final).
#
# Full NESTFUL test set, temperature=0.0, top_p=1.0, one rollout, ReAct,
# full executor, official scorer. Run once per arm, then compare_ckpts.sh.
#
# Env:  PYTHON=python3
#       LABEL=       (required) baseline | best | final
#       CHECKPOINT=  adapter dir; leave EMPTY for LABEL=baseline
#       OUT_DIR=     (required)
#       EVAL_SET=    optional NESTFUL jsonl (default: config full_nestful_jsonl)
#       MAX_TASKS=0  0 = full set (only cap for smoke checks)
#       USE_VLLM=1  EVAL_TP=4  VLLM_GPU_UTIL=0.85
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

LABEL="${LABEL:?set LABEL to baseline / best / final}"
OUT_DIR="${OUT_DIR:?set OUT_DIR for the eval outputs}"
if [ "$LABEL" = "baseline" ] && [ -n "${CHECKPOINT:-}" ]; then
  echo "[v5] ERROR: LABEL=baseline conflicts with CHECKPOINT=$CHECKPOINT" >&2
  exit 1
fi
if [ "$LABEL" != "baseline" ] && [ -z "${CHECKPOINT:-}" ]; then
  echo "[v5] ERROR: LABEL=$LABEL requires CHECKPOINT" >&2
  exit 1
fi

banner "final temp-0 eval — arm '$LABEL'"
print_env PYTHON LABEL CHECKPOINT OUT_DIR EVAL_SET MAX_TASKS USE_VLLM EVAL_TP VLLM_GPU_UTIL

ARGS=(scripts/eval/final_eval_v5.py run
  --label "$LABEL"
  --out-dir "$OUT_DIR"
  --max-tasks "${MAX_TASKS:-0}")
if [ -n "${CHECKPOINT:-}" ]; then
  require_adapter "$CHECKPOINT" "CHECKPOINT"
  ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [ -n "${EVAL_SET:-}" ]; then
  require_file "$EVAL_SET" "EVAL_SET"
  ARGS+=(--eval-set "$EVAL_SET")
fi

cd "$V3"
"$PY" "${ARGS[@]}"

banner "final eval done: $OUT_DIR/metrics_official.json"
