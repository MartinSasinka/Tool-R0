#!/bin/bash
# Run a full baseline-vs-finetuned comparison on a single benchmark.
#
# Usage:
#   BENCHMARK=bfcl       bash eval/scripts/run_benchmark.sh
#   BENCHMARK=toolalpaca bash eval/scripts/run_benchmark.sh
#   BENCHMARK=bfcl MAX_TASKS=50 bash eval/scripts/run_benchmark.sh
#
# Environment variables:
#   BENCHMARK          - bfcl | toolalpaca (required)
#   BASELINE_CONFIG    - path to baseline config YAML (default: eval/configs/baseline.yaml)
#   FINETUNED_CONFIG   - path to finetuned config YAML (default: eval/configs/finetuned.yaml)
#   MAX_TASKS          - optional task limit (per category for BFCL)
#   DRY_RUN            - set to 1 for smoke test
#   CATEGORY           - BFCL only: comma-separated categories (default: all)

set -euo pipefail

BENCHMARK="${BENCHMARK:?Set BENCHMARK=bfcl or BENCHMARK=toolalpaca}"
BASELINE_CONFIG="${BASELINE_CONFIG:-eval/configs/baseline.yaml}"
FINETUNED_CONFIG="${FINETUNED_CONFIG:-eval/configs/finetuned.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/results/${BENCHMARK}}"
MAX_TASKS_FLAG=""
DRY_RUN_FLAG=""
CATEGORY_FLAG=""

if [[ -n "${MAX_TASKS:-}" ]]; then
    MAX_TASKS_FLAG="--max-tasks $MAX_TASKS"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN_FLAG="--dry-run"
fi

if [[ -n "${CATEGORY:-}" ]]; then
    CATEGORY_FLAG="--category $CATEGORY"
fi

echo "============================================================"
echo "  Tool-R0 Eval: ${BENCHMARK}"
echo "  Baseline:  ${BASELINE_CONFIG}"
echo "  Finetuned: ${FINETUNED_CONFIG}"
echo "  Output:    ${OUTPUT_DIR}"
echo "============================================================"

# Baseline
python -m eval.run_eval \
    --benchmark "$BENCHMARK" \
    --config "$BASELINE_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --profile-name baseline \
    $MAX_TASKS_FLAG $DRY_RUN_FLAG $CATEGORY_FLAG

# Finetuned
python -m eval.run_eval \
    --benchmark "$BENCHMARK" \
    --config "$FINETUNED_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --profile-name finetuned \
    $MAX_TASKS_FLAG $DRY_RUN_FLAG $CATEGORY_FLAG

# Compare
python -m eval.scripts.compare \
    --baseline "$OUTPUT_DIR/baseline_summary.json" \
    --finetuned "$OUTPUT_DIR/finetuned_summary.json" \
    --output "$OUTPUT_DIR/comparison.json" \
    --table "$OUTPUT_DIR/comparison.md"

echo ""
echo "Done. Results in: $OUTPUT_DIR/"
