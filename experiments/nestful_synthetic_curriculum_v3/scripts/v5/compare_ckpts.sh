#!/usr/bin/env bash
# 9/9 — Paired comparison of finished final-eval arms.
#
# Reads the output directories written by final_eval.sh and produces
# final_compare_report.json: aggregate official metrics, metrics by call
# count (2/3/4+), gained/regressed task lists and a paired bootstrap 95% CI
# on the win-rate delta.
#
# Env:  PYTHON=python3
#       BASELINE_DIR=  (required) baseline arm output dir
#       BEST_DIR=      (required) best-checkpoint arm output dir
#       FINAL_DIR=     optional final-checkpoint arm output dir
#       OUT_DIR=       (required) where the report lands
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

BASELINE_DIR="${BASELINE_DIR:?set BASELINE_DIR}"
BEST_DIR="${BEST_DIR:?set BEST_DIR}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"
require_file "$BASELINE_DIR/metrics_official.json" "baseline metrics"
require_file "$BEST_DIR/metrics_official.json" "best metrics"

banner "checkpoint comparison"
print_env PYTHON BASELINE_DIR BEST_DIR FINAL_DIR OUT_DIR

ARGS=(scripts/eval/final_eval_v5.py compare
  --baseline "$BASELINE_DIR"
  --best "$BEST_DIR"
  --out "$OUT_DIR")
if [ -n "${FINAL_DIR:-}" ]; then
  require_file "$FINAL_DIR/metrics_official.json" "final metrics"
  ARGS+=(--final "$FINAL_DIR")
fi

cd "$V3"
"$PY" "${ARGS[@]}"

banner "report: $OUT_DIR/final_compare_report.json"
