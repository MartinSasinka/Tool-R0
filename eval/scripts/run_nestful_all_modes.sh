#!/bin/bash
# ==========================================================================
#  Tool-R0: NESTFUL evaluation in all three modes (end-to-end, 2 GPUs).
#
#  What this script does, end-to-end:
#    1. Ensures the IBM/NESTFUL repo is cloned into data/nestful_repo/ so
#       the executor can dispatch unknown tool calls into the dataset's
#       own Python implementations (replaces the LLM judge fallback).
#    2. Runs the BASELINE model in structural + execute + multiturn modes
#       inside a SINGLE Python process (one vLLM load).
#    3. Runs the FINETUNED model the same way (second vLLM load).
#    4. Builds the structural baseline-vs-finetuned compare.md and prints
#       headline numbers from every summary JSON.
#
#  Why it's fast:
#    Loading a 4B model into vLLM costs ~30-60s + extra time for CUDA-kernel
#    warmup. Calling `python -m eval.run_eval` once per (model, mode) would
#    pay this cost SIX times for no reason — each Python process gets a
#    fresh vLLM engine and tears it down at exit.
#
#    Instead we launch vLLM exactly twice (once per model) and run all
#    requested modes inside the same process via comma-separated
#    `--nestful-mode structural,execute,multiturn`. The runner reuses the
#    cached engine in `eval.model_adapter` between modes, so structural ->
#    execute -> multiturn share one warm engine.
#
#  GPU policy (default):
#    CUDA_VISIBLE_DEVICES=0,1 — the script reserves both GPUs for the eval
#    pod, but vLLM here uses tensor_parallel_size=1, so only GPU 0 is
#    actually computing. GPU 1 is held idle to prevent another job
#    grabbing it mid-run (matches our 2-GPU sandbox allocation).
#
#  Output files (suffix per mode so reruns never overwrite each other):
#
#    structural baseline   -> results/nestful/baseline_predictions.jsonl
#    structural finetuned  -> results/nestful/finetuned_predictions.jsonl
#    execute    baseline   -> results/nestful/baseline_execute_predictions.jsonl
#    execute    finetuned  -> results/nestful/finetuned_execute_predictions.jsonl
#    multiturn  baseline   -> results/nestful/baseline_multiturn_predictions.jsonl
#    multiturn  finetuned  -> results/nestful/finetuned_multiturn_predictions.jsonl
#
#  Usage:
#    bash eval/scripts/run_nestful_all_modes.sh
#
#  Environment variables (all optional):
#    BASELINE_MODEL          - HF id or local path of the untrained model.
#                              default: Qwen/Qwen3-4B-Instruct-2507
#    FINETUNED_MODEL         - local checkpoint directory of the trained model.
#                              default: qwen3-4b-tool-r0/iter3_solver/checkpoint-50
#    BASELINE_CONFIG         - YAML config for baseline run
#                              default: eval/configs/baseline.yaml
#    FINETUNED_CONFIG        - YAML config for finetuned run
#                              default: eval/configs/finetuned.yaml
#    OUTPUT_DIR              - where to write predictions/summaries
#                              default: eval/results/nestful
#    CUDA_DEVICES            - GPUs to expose to vLLM (default: 0,1)
#    MAX_TASKS               - cap tasks per run (default: full 1861); useful for pilots
#    NESTFUL_MAX_STEPS       - multiturn step limit per task (default: 10)
#    SKIP_STRUCTURAL=1       - skip structural mode (already ran it)
#    SKIP_EXECUTE=1          - skip execute mode
#    SKIP_MULTITURN=1        - skip multiturn mode (slowest)
#    SKIP_BASELINE=1         - skip the baseline run entirely
#    SKIP_FINETUNED=1        - skip the finetuned run entirely
#    USE_JUDGE=1             - opt in to the LLM judge fallback for
#                              executor mismatches (off by default; the
#                              IBM funcs make this unnecessary). Requires
#                              OPENAI_API_KEY when enabled.
#    NESTFUL_NO_JUDGE=1      - DEPRECATED no-op; kept for backwards compat.
#    NESTFUL_REPO_DIR        - alternative location of the IBM repo
#                              (default: data/nestful_repo)
# ==========================================================================

set -uo pipefail

BASELINE_MODEL="${BASELINE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
FINETUNED_MODEL="${FINETUNED_MODEL:-qwen3-4b-tool-r0/iter3_solver/checkpoint-50}"
BASELINE_CONFIG="${BASELINE_CONFIG:-eval/configs/baseline.yaml}"
FINETUNED_CONFIG="${FINETUNED_CONFIG:-eval/configs/finetuned.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-eval/results/nestful}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
NESTFUL_MAX_STEPS="${NESTFUL_MAX_STEPS:-10}"
NESTFUL_REPO_DIR="${NESTFUL_REPO_DIR:-data/nestful_repo}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NESTFUL_REPO_DIR

# ---- 1. Idempotently fetch the IBM functions before touching vLLM. -------
SENTINEL="${NESTFUL_REPO_DIR}/data_v2/executable_functions/func_file_map.json"
if [[ ! -f "$SENTINEL" ]]; then
    echo "[run_nestful_all_modes] IBM functions not found at $SENTINEL"
    echo "[run_nestful_all_modes] Calling scripts/setup_nestful_funcs.sh ..."
    if ! bash scripts/setup_nestful_funcs.sh; then
        echo "[run_nestful_all_modes] WARNING: setup script failed; continuing"
        echo "  with primitives-only execution. Many tasks will end up in"
        echo "  the unknown_function bucket."
    fi
else
    echo "[run_nestful_all_modes] IBM functions present: $SENTINEL"
fi

MAX_TASKS_FLAG=""
if [[ -n "${MAX_TASKS:-}" ]]; then
    MAX_TASKS_FLAG="--max-tasks $MAX_TASKS"
fi

# Default: judge OFF (rely on IBM funcs). Opt in via USE_JUDGE=1.
JUDGE_FLAG=""
JUDGE_NOTE="disabled (IBM funcs handle the long tail)"
if [[ "${USE_JUDGE:-0}" == "1" ]]; then
    JUDGE_FLAG="--nestful-use-judge"
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        JUDGE_NOTE="ENABLED but OPENAI_API_KEY missing (judge will skip tasks)"
    else
        JUDGE_NOTE="enabled (USE_JUDGE=1)"
    fi
fi
if [[ "${NESTFUL_NO_JUDGE:-0}" == "1" ]]; then
    echo "[run_nestful_all_modes] NESTFUL_NO_JUDGE=1 is now a no-op (judge OFF by default)."
fi

# Build the comma-separated mode list once based on SKIP_* flags.
# Empty => nothing to run.
MODES=()
[[ "${SKIP_STRUCTURAL:-0}" != "1" ]] && MODES+=("structural")
[[ "${SKIP_EXECUTE:-0}"    != "1" ]] && MODES+=("execute")
[[ "${SKIP_MULTITURN:-0}"  != "1" ]] && MODES+=("multiturn")

if [[ ${#MODES[@]} -eq 0 ]]; then
    echo "ERROR: All modes are skipped (SKIP_STRUCTURAL/SKIP_EXECUTE/SKIP_MULTITURN). Nothing to do."
    exit 1
fi

MODES_CSV=$(IFS=,; echo "${MODES[*]}")

PASS=0
FAIL=0
RESULTS=()

echo "=================================================================="
echo "  Tool-R0 NESTFUL evaluation - all modes (single vLLM per model)"
echo "=================================================================="
echo "  Baseline model       : $BASELINE_MODEL"
echo "  Baseline config      : $BASELINE_CONFIG"
echo "  Finetuned model      : $FINETUNED_MODEL"
echo "  Finetuned config     : $FINETUNED_CONFIG"
echo "  Output dir           : $OUTPUT_DIR"
echo "  GPUs (CUDA_VISIBLE)  : $CUDA_DEVICES (vLLM uses tensor_parallel=1)"
echo "  IBM repo             : $NESTFUL_REPO_DIR"
echo "  Max tasks            : ${MAX_TASKS:-full (1861)}"
echo "  Multiturn max_steps  : $NESTFUL_MAX_STEPS"
echo "  LLM judge            : $JUDGE_NOTE"
echo "  Modes (in order)     : $MODES_CSV"
echo "  Skip baseline        : ${SKIP_BASELINE:-0}"
echo "  Skip finetuned       : ${SKIP_FINETUNED:-0}"
echo "=================================================================="

mkdir -p "$OUTPUT_DIR"

run_model_all_modes() {
    # $1 label, $2 config, $3 model_path, $4 profile_name
    local LABEL="$1"
    local CONFIG="$2"
    local MODEL="$3"
    local PROFILE="$4"

    echo ""
    echo "============================================================"
    echo ">>> $LABEL"
    echo "    modes=$MODES_CSV  profile=$PROFILE  model=$MODEL"
    echo "    (single Python process; vLLM is loaded once and shared)"
    echo "============================================================"

    python -m eval.run_eval \
        --benchmark nestful \
        --config "$CONFIG" \
        --model-path "$MODEL" \
        --profile-name "$PROFILE" \
        --output-dir "$OUTPUT_DIR" \
        --nestful-mode "$MODES_CSV" \
        --nestful-max-steps "$NESTFUL_MAX_STEPS" \
        $JUDGE_FLAG $MAX_TASKS_FLAG
    local RC=$?

    sleep 2

    if [[ $RC -eq 0 ]]; then
        echo ""
        echo "    [OK] $LABEL"
        PASS=$((PASS + 1))
        RESULTS+=("OK   | $LABEL")
    else
        echo ""
        echo "    [FAIL] $LABEL (exit $RC)"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL | $LABEL")
    fi

    echo "<<< end $LABEL"
}

# -------------------------------------------------------------- baseline run
if [[ "${SKIP_BASELINE:-0}" != "1" ]]; then
    run_model_all_modes \
        "NESTFUL all-modes - baseline" \
        "$BASELINE_CONFIG" \
        "$BASELINE_MODEL" \
        baseline
else
    echo ""
    echo "[SKIP] baseline run (SKIP_BASELINE=1)"
fi

# ------------------------------------------------------------- finetuned run
if [[ "${SKIP_FINETUNED:-0}" != "1" ]]; then
    run_model_all_modes \
        "NESTFUL all-modes - finetuned" \
        "$FINETUNED_CONFIG" \
        "$FINETUNED_MODEL" \
        finetuned
else
    echo ""
    echo "[SKIP] finetuned run (SKIP_FINETUNED=1)"
fi

# ---------------------------------------------------- structural baseline-vs-ft
echo ""
echo "============================================================"
echo "  POST: paper-aligned baseline-vs-finetuned compare"
echo "============================================================"
BASE_SUM="$OUTPUT_DIR/baseline_summary.json"
TUNED_SUM="$OUTPUT_DIR/finetuned_summary.json"
if [[ -f "$BASE_SUM" && -f "$TUNED_SUM" ]]; then
    python -m eval.scripts.compare \
        --baseline  "$BASE_SUM" \
        --finetuned "$TUNED_SUM" \
        --output    "$OUTPUT_DIR/comparison.json" \
        --table     "$OUTPUT_DIR/comparison.md" \
        || echo "  [WARN] compare script failed; summaries are still on disk."
    echo ""
    echo "  Structural comparison: $OUTPUT_DIR/comparison.md"
else
    echo "  Skipping structural compare (missing $BASE_SUM or $TUNED_SUM)"
fi

# Lightweight side-by-side summaries for the new modes.
print_metric() {
    local LABEL="$1"
    local PATH_JSON="$2"
    local KEY="$3"
    if [[ -f "$PATH_JSON" ]]; then
        local VAL
        VAL=$(python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get(sys.argv[2],'?'))" "$PATH_JSON" "$KEY" 2>/dev/null || echo '?')
        printf "    %-40s %s\n" "$LABEL" "$VAL"
    else
        printf "    %-40s (missing)\n" "$LABEL"
    fi
}

print_breakdown() {
    # Pretty-print execution_class_breakdown from a summary JSON.
    local LABEL="$1"
    local PATH_JSON="$2"
    if [[ ! -f "$PATH_JSON" ]]; then
        printf "    %-40s (missing)\n" "$LABEL"
        return
    fi
    local LINE
    LINE=$(python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('execution_class_breakdown', {}))" "$PATH_JSON" 2>/dev/null || echo '{}')
    printf "    %-40s %s\n" "$LABEL" "$LINE"
}

echo ""
echo "============================================================"
echo "  Final summary"
echo "============================================================"
echo "  PASS=$PASS  FAIL=$FAIL"
for line in "${RESULTS[@]}"; do
    echo "    $line"
done
echo ""
echo "  Structural (partial_match_accuracy_percent):"
print_metric "baseline"  "$OUTPUT_DIR/baseline_summary.json"            partial_match_accuracy_percent
print_metric "finetuned" "$OUTPUT_DIR/finetuned_summary.json"           partial_match_accuracy_percent
echo ""
echo "  Execute (final_answer_accuracy_percent):"
print_metric "baseline"  "$OUTPUT_DIR/baseline_execute_summary.json"    final_answer_accuracy_percent
print_metric "finetuned" "$OUTPUT_DIR/finetuned_execute_summary.json"   final_answer_accuracy_percent
echo ""
echo "  Multiturn (final_answer_accuracy_percent):"
print_metric "baseline"  "$OUTPUT_DIR/baseline_multiturn_summary.json"  final_answer_accuracy_percent
print_metric "finetuned" "$OUTPUT_DIR/finetuned_multiturn_summary.json" final_answer_accuracy_percent
echo ""
echo "  Execute - execution_class_breakdown (primitives vs IBM vs errors):"
print_breakdown "baseline"  "$OUTPUT_DIR/baseline_execute_summary.json"
print_breakdown "finetuned" "$OUTPUT_DIR/finetuned_execute_summary.json"
echo ""
echo "  Multiturn - execution_class_breakdown:"
print_breakdown "baseline"  "$OUTPUT_DIR/baseline_multiturn_summary.json"
print_breakdown "finetuned" "$OUTPUT_DIR/finetuned_multiturn_summary.json"

echo ""
echo "  Predictions and per-mode summaries are in: $OUTPUT_DIR"
echo "  Done."

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
exit 0
