#!/bin/bash
# ==========================================================================
#  Tool-R0: Complete Evaluation Pipeline
#
#  Runs ALL benchmarks for baseline and finetuned models, then compares.
#  Single script — just set the model paths and run on DGX.
#
#  Usage:
#    bash eval/scripts/run_all.sh
#
#  Environment variables (optional overrides):
#    BASELINE_MODEL   - baseline model path (default: Qwen/Qwen2.5-1.5B-Instruct)
#    FINETUNED_MODEL  - finetuned model path (default: qwen2.5-1.5b-instruct-tool-r0)
#    MAX_TASKS        - limit tasks per category (default: no limit = full eval)
#    CUDA_DEVICES     - GPU selection (default: 0,1,2,4)
# ==========================================================================

BASELINE_MODEL="${BASELINE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
FINETUNED_MODEL="${FINETUNED_MODEL:-qwen2.5-1.5b-instruct-tool-r0/iter5_solver/checkpoint-50}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,4}"
MAX_TASKS_FLAG=""
if [[ -n "${MAX_TASKS:-}" ]]; then
    MAX_TASKS_FLAG="--max-tasks $MAX_TASKS"
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PASS=0
FAIL=0

echo "=================================================================="
echo "  Tool-R0 Complete Evaluation"
echo "  Baseline:  $BASELINE_MODEL"
echo "  Finetuned: $FINETUNED_MODEL"
echo "  GPUs:      $CUDA_DEVICES"
echo "  Max tasks: ${MAX_TASKS:-full}"
echo "=================================================================="
echo ""

run_benchmark() {
    local LABEL=$1
    local BENCHMARK=$2
    local CONFIG=$3
    local PROFILE=$4
    local OUTPUT_DIR=$5
    shift 5

    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "  START: $LABEL"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo ""

    python -m eval.run_eval \
        --benchmark "$BENCHMARK" \
        --config "$CONFIG" \
        --profile-name "$PROFILE" \
        --output-dir "$OUTPUT_DIR" \
        $MAX_TASKS_FLAG "$@"
    local RC=$?

    sleep 2

    if [[ $RC -eq 0 ]]; then
        echo ""
        echo "  [OK] $LABEL completed successfully."
        PASS=$((PASS + 1))
    else
        echo ""
        echo "  [FAIL] $LABEL failed (exit code $RC)."
        FAIL=$((FAIL + 1))
    fi

    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo ""
}

# ==================================================================
#  1. BFCL — AST categories (1040 tasks)
# ==================================================================
echo "============================================================"
echo "  PHASE 1/7: BFCL — AST + Exec categories"
echo "============================================================"

run_benchmark "BFCL AST — baseline" \
    bfcl eval/configs/baseline.yaml baseline eval/results/bfcl \
    --model-path "$BASELINE_MODEL" --category simple,multiple,parallel,irrelevance

run_benchmark "BFCL AST — finetuned" \
    bfcl eval/configs/finetuned.yaml finetuned eval/results/bfcl \
    --model-path "$FINETUNED_MODEL" --category simple,multiple,parallel,irrelevance

run_benchmark "BFCL Exec — baseline" \
    bfcl eval/configs/baseline.yaml baseline_exec eval/results/bfcl_exec \
    --model-path "$BASELINE_MODEL" --category exec_simple,exec_multiple,exec_parallel,exec_parallel_multiple

run_benchmark "BFCL Exec — finetuned" \
    bfcl eval/configs/finetuned.yaml finetuned_exec eval/results/bfcl_exec \
    --model-path "$FINETUNED_MODEL" --category exec_simple,exec_multiple,exec_parallel,exec_parallel_multiple

# ==================================================================
#  2. ToolAlpaca (real API schemas, same scorer as training)
# ==================================================================
echo "============================================================"
echo "  PHASE 2/7: ToolAlpaca"
echo "============================================================"

run_benchmark "ToolAlpaca — baseline" \
    toolalpaca eval/configs/baseline.yaml baseline eval/results/toolalpaca \
    --model-path "$BASELINE_MODEL"

run_benchmark "ToolAlpaca — finetuned" \
    toolalpaca eval/configs/finetuned.yaml finetuned eval/results/toolalpaca \
    --model-path "$FINETUNED_MODEL"

# ==================================================================
#  3. API-Bank (73 real APIs, Level 1 — given-desc)
# ==================================================================
echo "============================================================"
echo "  PHASE 3/7: API-Bank (real API benchmark)"
echo "============================================================"

run_benchmark "API-Bank — baseline" \
    apibank eval/configs/baseline.yaml baseline eval/results/apibank \
    --model-path "$BASELINE_MODEL"

run_benchmark "API-Bank — finetuned" \
    apibank eval/configs/finetuned.yaml finetuned eval/results/apibank \
    --model-path "$FINETUNED_MODEL"

# ==================================================================
#  4. ToolTalk (multi-turn real API, 78 conversations)
# ==================================================================
echo "============================================================"
echo "  PHASE 4/7: ToolTalk (multi-turn real API)"
echo "============================================================"

run_benchmark "ToolTalk — baseline" \
    tooltalk eval/configs/baseline.yaml baseline eval/results/tooltalk \
    --model-path "$BASELINE_MODEL"

run_benchmark "ToolTalk — finetuned" \
    tooltalk eval/configs/finetuned.yaml finetuned eval/results/tooltalk \
    --model-path "$FINETUNED_MODEL"

# ==================================================================
#  5. NESTFUL (nested API sequences, 1861 tasks)
# ==================================================================
echo "============================================================"
echo "  PHASE 5/7: NESTFUL (nested API sequences)"
echo "============================================================"

run_benchmark "NESTFUL — baseline" \
    nestful eval/configs/baseline.yaml baseline eval/results/nestful \
    --model-path "$BASELINE_MODEL"

run_benchmark "NESTFUL — finetuned" \
    nestful eval/configs/finetuned.yaml finetuned eval/results/nestful \
    --model-path "$FINETUNED_MODEL"

# ==================================================================
#  6. AppWorld (real API execution, requires appworld package)
# ==================================================================
echo "============================================================"
echo "  PHASE 6/7: AppWorld (real API execution)"
echo "============================================================"

if python -c "import appworld" 2>/dev/null; then
    run_benchmark "AppWorld — baseline" \
        appworld eval/configs/baseline.yaml baseline eval/results/appworld \
        --model-path "$BASELINE_MODEL" --appworld-dataset train --appworld-max-difficulty 1

    run_benchmark "AppWorld — finetuned" \
        appworld eval/configs/finetuned.yaml finetuned eval/results/appworld \
        --model-path "$FINETUNED_MODEL" --appworld-dataset train --appworld-max-difficulty 1
else
    echo "  [SKIP] appworld not installed. Run: pip install appworld && appworld install && appworld download data"
fi

# ==================================================================
#  7. Compare all results
# ==================================================================
echo "============================================================"
echo "  PHASE 7/7: Comparing results"
echo "============================================================"

compare_if_exists() {
    local DIR=$1
    local BASE_PROFILE=$2
    local TUNED_PROFILE=$3

    local BASE_FILE="$DIR/${BASE_PROFILE}_summary.json"
    local TUNED_FILE="$DIR/${TUNED_PROFILE}_summary.json"

    if [[ -f "$BASE_FILE" && -f "$TUNED_FILE" ]]; then
        echo "  Comparing $DIR ..."
        python -m eval.scripts.compare \
            --baseline "$BASE_FILE" \
            --finetuned "$TUNED_FILE" \
            --output "$DIR/comparison.json" \
            --table "$DIR/comparison.md" || echo "  [WARN] comparison failed for $DIR"
    else
        echo "  Skipping $DIR (missing summary files)"
    fi
}

compare_if_exists eval/results/bfcl baseline finetuned
compare_if_exists eval/results/bfcl_exec baseline_exec finetuned_exec
compare_if_exists eval/results/toolalpaca baseline finetuned
compare_if_exists eval/results/apibank baseline finetuned
compare_if_exists eval/results/tooltalk baseline finetuned
compare_if_exists eval/results/nestful baseline finetuned
compare_if_exists eval/results/appworld baseline finetuned

# ==================================================================
#  Final summary
# ==================================================================
echo ""
echo "=================================================================="
echo "  EVALUATION COMPLETE  ($PASS passed, $FAIL failed)"
echo "=================================================================="
echo ""
echo "Results:"
echo "  BFCL AST:        eval/results/bfcl/"
echo "  BFCL Exec:       eval/results/bfcl_exec/"
echo "  ToolAlpaca:      eval/results/toolalpaca/"
echo "  API-Bank:        eval/results/apibank/"
echo "  ToolTalk:        eval/results/tooltalk/"
echo "  NESTFUL:         eval/results/nestful/"
echo "  AppWorld:        eval/results/appworld/"
echo ""

for DIR in eval/results/bfcl eval/results/bfcl_exec eval/results/toolalpaca eval/results/apibank eval/results/tooltalk eval/results/nestful eval/results/appworld; do
    if [[ -f "$DIR/comparison.md" ]]; then
        echo "--- $DIR ---"
        cat "$DIR/comparison.md"
        echo ""
    fi
done

echo "Done. ($PASS passed, $FAIL failed)"
