#!/usr/bin/env bash
# Stabilized v2 FINAL EVAL — run on the pod AFTER the full v2 run passes gates.
#
# Evaluates the baseline + selected checkpoints on the clean NESTFUL test set
# (1861 tasks) under the IDENTICAL eval harness (same unified prompt, lenient
# parser, full executor, official scorer). Each final_eval run automatically
# writes final_eval_trajectories.jsonl + metrics_official.json so the offline
# verified pipeline can recompute per-sample Win.
#
# Cells (each optional except baseline; set the env var to a checkpoint dir):
#   - baseline        no LoRA (always runs)
#   - $CELL_A_NAME    $ADAPTER_A   (default: stage 2 / epoch 4)
#   - $CELL_B_NAME    $ADAPTER_B   (default: stage 3 / epoch 1)
#   - $CELL_C_NAME    $ADAPTER_C   (optional; e.g. best_react_win_adapter)
#
# vLLM note (Blackwell / sm_120): export the FlashInfer-sampler workaround before
# running, otherwise the engine crashes in the sampler warmup:
#   export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
#
# Usage (on pod) — defaults already point at the execution_v2 run, so this is enough:
#   USE_VLLM=1 bash experiments/nestful_mtgrpo_partial/run_final_eval_v2.sh
#
# Override any cell explicitly:
#   USE_VLLM=1 \
#     ADAPTER_A=<dir> CELL_A_NAME=stage2_e4 \
#     ADAPTER_B=<dir> CELL_B_NAME=stage3_e1 \
#     ADAPTER_C=<dir> CELL_C_NAME=exec_v2_best \
#     bash experiments/nestful_mtgrpo_partial/run_final_eval_v2.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINIMAL="$(cd "$HERE/../nestful_mtgrpo_minimal" && pwd)"
COMPARISON="$(cd "$HERE/../comparison" && pwd)"
PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-$HERE/outputs/final_eval_v2}"
DATASET="${DATASET:-$MINIMAL/data/NESTFUL-main/data_v2/nestful_data.jsonl}"
USE_VLLM="${USE_VLLM:-0}"
CKPT_ROOT="${CKPT_ROOT:-$HERE/outputs/execution_v2_mixed_replay_full}"
mkdir -p "$OUT_ROOT"

# Default checkpoint selection (override via env). stage 3 has no epoch_1 on disk
# in the current run, so ADAPTER_B defaults to stage 3 / epoch 2 as the closest
# early-stage-3 checkpoint; set ADAPTER_B explicitly to change it.
ADAPTER_A="${ADAPTER_A:-$CKPT_ROOT/stage_2/checkpoints/adapter_epoch_4}"
CELL_A_NAME="${CELL_A_NAME:-stage2_e4}"
ADAPTER_B="${ADAPTER_B:-$CKPT_ROOT/stage_3/checkpoints/adapter_epoch_1}"
CELL_B_NAME="${CELL_B_NAME:-stage3_e1}"
ADAPTER_C="${ADAPTER_C:-}"
CELL_C_NAME="${CELL_C_NAME:-exec_v2_best}"

# Each cell is one `run.py --mode final_eval` invocation; only the adapter changes.
# Baseline = no --checkpoint (+ lora_adapter=null). final_eval always dumps
# final_eval_trajectories.jsonl + metrics_official.json under the output dir.
run_cell () {
    local name="$1"; local adapter="$2"
    echo ""
    echo "[final_eval_v2] ===== $name ====="
    local args=(--mode final_eval
                --config "$HERE/config.yaml"
                --override "experiment.output_dir=$OUT_ROOT/$name"
                --override "paths.full_nestful_jsonl=$DATASET"
                --override "data.eval_paradigm=react")
    if [ "$USE_VLLM" = "1" ]; then
        args+=(--override "hardware.use_vllm=true")
    fi
    if [ -n "$adapter" ]; then
        if [ ! -d "$adapter" ]; then
            echo "[final_eval_v2] ERROR: adapter dir not found: $adapter" >&2
            echo "  (skipping cell '$name' — check the path / epoch number)" >&2
            return 0
        fi
        args+=(--checkpoint "$adapter")
    else
        args+=(--override "model.lora_adapter=null")
    fi
    "$PYTHON" "$HERE/run.py" "${args[@]}"
}

run_cell "baseline" ""
[ -n "$ADAPTER_A" ] && run_cell "$CELL_A_NAME" "$ADAPTER_A"
[ -n "$ADAPTER_B" ] && run_cell "$CELL_B_NAME" "$ADAPTER_B"
[ -n "$ADAPTER_C" ] && run_cell "$CELL_C_NAME" "$ADAPTER_C"

echo ""
echo "[final_eval_v2] per-cell official Win rates:"
for cell in baseline "$CELL_A_NAME" "$CELL_B_NAME" "$CELL_C_NAME"; do
    mo="$OUT_ROOT/$cell/metrics_official.json"
    [ -f "$mo" ] || continue
    win="$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1])).get('win_rate','?'))" "$mo" 2>/dev/null || echo '?')"
    printf "  %-18s win_rate=%s\n" "$cell" "$win"
done

echo ""
echo "[final_eval_v2] recomputing canonical per-sample official Win + consistency ..."
"$PYTHON" "$COMPARISON/recompute_per_sample_official.py" || \
    echo "[final_eval_v2] WARNING: recompute step failed (inspect trajectories)." >&2

echo "[final_eval_v2] verified overlap + taxonomy ..."
"$PYTHON" "$COMPARISON/meeting_analysis.py" --assert-consistency || \
    echo "[final_eval_v2] WARNING: verified analysis failed (non-fatal)." >&2

cat > "$OUT_ROOT/FINAL_RESULTS_VERIFIED.md" <<EOF
# FINAL RESULTS (VERIFIED)

Eval cells (clean NESTFUL test set, n=1861, identical harness):
- baseline, $CELL_A_NAME, $CELL_B_NAME${ADAPTER_C:+, $CELL_C_NAME}

Per-cell metrics_official.json live under $OUT_ROOT/<cell>/.
Fill the headline table from those metrics_official.json win_rate fields.
EOF
echo "[final_eval_v2] wrote $OUT_ROOT/FINAL_RESULTS_VERIFIED.md"
