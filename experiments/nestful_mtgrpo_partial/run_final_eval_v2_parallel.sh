#!/usr/bin/env bash
# Stabilized v2 FINAL EVAL — PARALLEL launcher (GPU pool over a checkpoint list).
#
# final_eval is single-engine by design (the data-parallel pool is train-only),
# so to use all GPUs we run the INDEPENDENT eval cells concurrently, each pinned
# to its own GPU via CUDA_VISIBLE_DEVICES. Cells write to distinct output dirs,
# so this is safe. When there are more cells than GPUs, cells are queued and run
# in waves (at most one cell per GPU at a time).
#
# Blackwell / sm_120: the FlashInfer-sampler workaround is exported below so the
# vLLM engine doesn't crash in the sampler warmup.
#
# Usage (on pod, ideally inside tmux):
#   bash experiments/nestful_mtgrpo_partial/run_final_eval_v2_parallel.sh
#
# Override the GPU set or the checkpoint list via env:
#   GPUS="0 1 2 3" bash .../run_final_eval_v2_parallel.sh
#   CELLS="baseline= stage2_e2=$CKPT_ROOT/stage_2/checkpoints/adapter_epoch_2" \
#     bash .../run_final_eval_v2_parallel.sh
set -uo pipefail

# ── Blackwell / sm_120 vLLM workaround ────────────────────────────────────────
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINIMAL="$(cd "$HERE/../nestful_mtgrpo_minimal" && pwd)"
PYTHON="${PYTHON:-python}"
RUN="$HERE/run.py"
CFG="${CONFIG:-$HERE/config.yaml}"
OUT_ROOT="${OUT_ROOT:-$HERE/outputs/final_eval_v2}"
CKPT_ROOT="${CKPT_ROOT:-$HERE/outputs/execution_v2_mixed_replay_full}"
# Phase-0 re-eval uses the FULL NESTFUL (1861) to compare against the prior
# 0.544 baseline. For a Phase-1 run whose checkpoints were SELECTED on the dev
# split, report on the disjoint test set instead:
#   DATASET=$MINIMAL/data/splits/nestful_test.jsonl bash .../run_final_eval_v2_parallel.sh
DATASET="${DATASET:-$MINIMAL/data/NESTFUL-main/data_v2/nestful_data.jsonl}"
mkdir -p "$OUT_ROOT"

# GPU pool (all 4 by default; eval can use GPU 0 too since no training runs).
read -r -a GPU_ARR <<< "${GPUS:-0 1 2 3}"

# Checkpoint cells: "name=adapter_dir" (empty adapter => baseline, no LoRA).
# Default set = every checkpoint relevant to the degradation analysis, incl. the
# auto-selected best_react_win_adapter (stage 2 / epoch 2). Non-existent adapter
# dirs are skipped automatically.
if [ -n "${CELLS:-}" ]; then
    read -r -a CELL_ARR <<< "$CELLS"
else
    CELL_ARR=(
        "baseline="
        "best_react_win=$CKPT_ROOT/best_react_win_adapter"
        "stage1_e3=$CKPT_ROOT/stage_1/checkpoints/adapter_epoch_3"
        "stage2_e2=$CKPT_ROOT/stage_2/checkpoints/adapter_epoch_2"
        "stage2_e3=$CKPT_ROOT/stage_2/checkpoints/adapter_epoch_3"
        "stage2_e4=$CKPT_ROOT/stage_2/checkpoints/adapter_epoch_4"
        "stage3_e1=$CKPT_ROOT/stage_3/checkpoints/adapter_epoch_1"
        "stage3_e2=$CKPT_ROOT/stage_3/checkpoints/adapter_epoch_2"
    )
fi

# ── Kill any previous eval run so it doesn't hog a GPU ─────────────────────────
echo "[parallel] killing any stale eval processes ..."
pkill -f VLLM::EngineCore 2>/dev/null || true
pkill -f "$RUN" 2>/dev/null || true
sleep 4

COMMON=(--mode final_eval
        --config "$CFG"
        --override "hardware.use_vllm=true"
        --override "data.eval_paradigm=react"
        --override "paths.full_nestful_jsonl=$DATASET")

run_cell () {
    local gpu="$1"; local name="$2"; local adapter="$3"
    local log="$OUT_ROOT/$name.log"
    local args=("${COMMON[@]}" --override "experiment.output_dir=$OUT_ROOT/$name")
    if [ -n "$adapter" ]; then
        if [ ! -d "$adapter" ]; then
            echo "[parallel] SKIP $name — adapter not found: $adapter" >&2
            return 0
        fi
        args+=(--checkpoint "$adapter")
    else
        args+=(--override "model.lora_adapter=null")
    fi
    echo "[parallel] launching $name on GPU $gpu (log: $log)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUN" "${args[@]}" > "$log" 2>&1
}

# ── GPU job pool: keep at most ${#GPU_ARR[@]} cells running at once ────────────
declare -A GPU_PID     # gpu_id -> pid of the cell currently on it
launched_names=()

wait_for_free_gpu () {
    # Block until at least one GPU is free; echo the free gpu id.
    while true; do
        for gpu in "${GPU_ARR[@]}"; do
            local pid="${GPU_PID[$gpu]:-}"
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
                echo "$gpu"; return 0
            fi
        done
        sleep 5
    done
}

for cell in "${CELL_ARR[@]}"; do
    name="${cell%%=*}"
    adapter="${cell#*=}"
    launched_names+=("$name")
    gpu="$(wait_for_free_gpu)"
    run_cell "$gpu" "$name" "$adapter" &
    GPU_PID[$gpu]=$!
    sleep 2   # small stagger so vLLM engine inits don't collide
done

echo "[parallel] all cells scheduled; waiting for completion ..."
wait

echo ""
echo "[parallel] ===== DONE — official Win rates ====="
for name in "${launched_names[@]}"; do
    mo="$OUT_ROOT/$name/metrics_official.json"
    [ -f "$mo" ] || { printf "  %-18s (no metrics_official.json)\n" "$name"; continue; }
    win="$("$PYTHON" -c "import json,sys;print(json.load(open(sys.argv[1])).get('win_rate','?'))" "$mo" 2>/dev/null || echo '?')"
    printf "  %-18s win_rate=%s\n" "$name" "$win"
done

echo ""
echo "[parallel] building CHECKPOINT_REEVAL_REPORT.md ..."
"$PYTHON" "$HERE/../comparison/checkpoint_reeval_report.py" --eval-root "$OUT_ROOT" || \
    echo "[parallel] WARNING: report generation failed (inspect $OUT_ROOT)." >&2
