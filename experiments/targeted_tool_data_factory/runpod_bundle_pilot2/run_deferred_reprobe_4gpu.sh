#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  Deferred Phase-2 re-probe (OPTIONAL, never started by the Phase-1 canary)
#
#  Re-runs the inference-only signal probe on deferred_phase2_tasks.jsonl
#  (the ~80 tasks not selected for Phase-1). Same BF16 base model, A4
#  dispatch and factory executor as the original probe. No training.
#
#  Usage (from repo root, AFTER reading the C0 vs C1 Phase-1 report):
#      bash experiments/targeted_tool_data_factory/runpod_bundle_pilot2/run_deferred_reprobe_4gpu.sh
#      bash .../run_deferred_reprobe_4gpu.sh --dry-run
#      bash .../run_deferred_reprobe_4gpu.sh --resume
# ═══════════════════════════════════════════════════════════════════════════
set -Eeuo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FACTORY="$(cd "$BUNDLE/.." && pwd)"
REPO="$(cd "$FACTORY/../.." && pwd)"
PY="${PYTHON:-python3}"

DRY_RUN=0
RESUME_FLAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --resume)  RESUME_FLAG="--resume"; shift ;;
    *) echo "[deferred] unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Locate the deferred file written by the original signal probe.
if [ -f "$FACTORY/outputs/runpod_pilot2/signal_probe/deferred_phase2_tasks.jsonl" ]; then
  DEFERRED="$FACTORY/outputs/runpod_pilot2/signal_probe/deferred_phase2_tasks.jsonl"
elif [ -f "$FACTORY/outputs/runpod_pilot2/signal_probe_from_zip/signal_probe/deferred_phase2_tasks.jsonl" ]; then
  DEFERRED="$FACTORY/outputs/runpod_pilot2/signal_probe_from_zip/signal_probe/deferred_phase2_tasks.jsonl"
else
  echo "[deferred] ABORT: deferred_phase2_tasks.jsonl not found" >&2
  exit 1
fi

OUT="$FACTORY/outputs/runpod_pilot2/signal_probe_deferred"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SEED="${SEED:-20260726}"
GPUS="${GPUS:-0,1,2,3}"
export SYNTHETIC_TOOLS_DIR="$FACTORY/trainer_adapter"
export PYTHONUNBUFFERED=1

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N_GPUS=${#GPU_ARR[@]}

mkdir -p "$OUT"
cd "$REPO"

echo "[deferred] data: $DEFERRED"
echo "[deferred] out:  $OUT"
echo "[deferred] model: $MODEL (bfloat16, no LoRA update)"
echo "[deferred] NO training will be performed"

if [ "$DRY_RUN" = "1" ]; then
  echo "[deferred] DRY RUN — planning P2 on deferred tasks only"
fi

# P2 only: all deferred tasks × 4 rollouts across the 4 GPUs.
pids=(); i=0
for gpu in "${GPU_ARR[@]}"; do
  out="$OUT/shard_p2_${i}.jsonl"
  cmd=("$PY" "$BUNDLE/signal_probe_worker.py"
       --data "$DEFERRED" --out "$out"
       --phase P2 --rollouts 4
       --shard-index "$i" --shard-count "$N_GPUS"
       --model "$MODEL" --seed "$SEED" --backend vllm)
  [ -n "$RESUME_FLAG" ] && cmd+=("$RESUME_FLAG")
  [ "$DRY_RUN" = "1" ] && cmd+=(--dry-run)
  echo "+ CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" &
  pids+=($!)
  i=$((i + 1))
done
rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
[ "$rc" = "0" ] || { echo "[deferred] worker failure" >&2; exit 1; }

if [ "$DRY_RUN" = "0" ]; then
  "$PY" "$BUNDLE/signal_probe_analyze.py" --mode report \
    --probe-dir "$OUT" --data "$DEFERRED"
fi

echo "[deferred] report: $OUT/SIGNAL_PROBE_REPORT.md"
echo "[deferred] done (still no training)"
