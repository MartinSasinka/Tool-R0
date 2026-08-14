#!/usr/bin/env bash
# Sequential night eval: 3 NESTFUL T=0 runs on 4 GPU vLLM.
#
#  1) baseline, gold+10 tool slack
#  2) latest checkpoint, gold+10 tool slack
#  3) latest checkpoint, original gold+0 tool slack
#
# From nestful_mtgrpo_minimal (or repo root):
#   export CHECKPOINT=outputs/.../checkpoints/adapter_epoch_XX
#   bash scripts/night_eval_triple.sh
#
# Required:
#   CHECKPOINT   LoRA adapter dir of the latest trained ckpt
#
# Optional:
#   OUT_ROOT     default: outputs/evals/p43_nestful_t0_night
#   CONFIG       default: continue750 enrich30 yaml
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   USE_VLLM=1 EVAL_TP=4 VLLM_GPU_UTIL=0.85 MAX_MODEL_LEN=12288
#   GPU_FREE_MIB=2048 GPU_WAIT_TIMEOUT_S=600 GPU_SETTLE_S=15
set -euo pipefail

MINIMAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_SH="$MINIMAL/scripts/final_eval.sh"
CFG="${CONFIG:-$MINIMAL/configs/qwen3_p43_profile1000_dynamic_online_continue750_enrich30.yaml}"
OUT_ROOT="${OUT_ROOT:-$MINIMAL/outputs/evals/p43_nestful_t0_night}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the latest adapter dir (adapter_config.json)}"

if [[ ! -x "$EVAL_SH" && ! -f "$EVAL_SH" ]]; then
  echo "[night_eval] ERROR: missing $EVAL_SH" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT/adapter_config.json" ]]; then
  echo "[night_eval] ERROR: not a LoRA adapter: $CHECKPOINT" >&2
  exit 1
fi

export WANDB_MODE="${WANDB_MODE:-disabled}"
export USE_VLLM="${USE_VLLM:-1}"
export EVAL_TP="${EVAL_TP:-4}"
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
export CONFIG="$CFG"
unset SYNTHETIC_TOOLS_DIR || true

CKPT_ABS="$(cd "$CHECKPOINT" && pwd)"
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
LOG="$OUT_ROOT/night_eval.log"
GPU_FREE_MIB="${GPU_FREE_MIB:-2048}"
GPU_WAIT_TIMEOUT_S="${GPU_WAIT_TIMEOUT_S:-600}"
GPU_SETTLE_S="${GPU_SETTLE_S:-15}"

gpu_ids() {
  echo "${CUDA_VISIBLE_DEVICES}" | tr ',' ' '
}

gpus_are_free() {
  local id used
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[night_eval] WARNING: nvidia-smi missing — skip GPU wait" | tee -a "$LOG"
    return 0
  fi
  for id in $(gpu_ids); do
    used="$(nvidia-smi -i "$id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    if [[ -z "$used" || "$used" -gt "$GPU_FREE_MIB" ]]; then
      return 1
    fi
    if nvidia-smi -i "$id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
      return 1
    fi
  done
  return 0
}

wait_gpus_free() {
  local why="$1"
  local start=$SECONDS
  echo "[night_eval] waiting for GPUs ${CUDA_VISIBLE_DEVICES} free (${why}; used<=${GPU_FREE_MIB} MiB, no compute pids)" | tee -a "$LOG"
  while true; do
    if gpus_are_free; then
      echo "[night_eval] GPUs free — settle ${GPU_SETTLE_S}s" | tee -a "$LOG"
      sleep "$GPU_SETTLE_S"
      if gpus_are_free; then
        return 0
      fi
      echo "[night_eval] GPUs busy again after settle — keep waiting" | tee -a "$LOG"
    fi
    if (( SECONDS - start > GPU_WAIT_TIMEOUT_S )); then
      echo "[night_eval] ERROR: GPUs still busy after ${GPU_WAIT_TIMEOUT_S}s" | tee -a "$LOG"
      nvidia-smi | tee -a "$LOG" || true
      exit 1
    fi
    sleep 5
  done
}

run_one() {
  local n="$1" label="$2" slack="$3" cap="$4" out="$5" ckpt="${6:-}"
  echo "==============================================================" | tee -a "$LOG"
  echo "[night_eval] $n/3  label=$label  slack=$slack  cap=$cap" | tee -a "$LOG"
  echo "[night_eval] CHECKPOINT=${ckpt:-<base model>}" | tee -a "$LOG"
  echo "[night_eval] OUT_DIR=$out" | tee -a "$LOG"
  echo "==============================================================" | tee -a "$LOG"
  wait_gpus_free "before eval $n/3"
  if [[ -n "$ckpt" ]]; then
    LABEL="$label" CHECKPOINT="$ckpt" OUT_DIR="$out" \
      TOOL_CALL_SLACK="$slack" TOOL_CALL_SLACK_CAP="$cap" \
      bash "$EVAL_SH" 2>&1 | tee -a "$LOG"
  else
    LABEL="$label" OUT_DIR="$out" \
      TOOL_CALL_SLACK="$slack" TOOL_CALL_SLACK_CAP="$cap" \
      env -u CHECKPOINT \
      bash "$EVAL_SH" 2>&1 | tee -a "$LOG"
  fi
  echo "[night_eval] $n/3 DONE  $out/metrics_official.json" | tee -a "$LOG"
  wait_gpus_free "after eval $n/3"
}

echo "[night_eval] start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$LOG"
echo "[night_eval] CHECKPOINT=$CKPT_ABS" | tee -a "$LOG"
echo "[night_eval] OUT_ROOT=$OUT_ROOT" | tee -a "$LOG"
echo "[night_eval] CONFIG=$CFG" | tee -a "$LOG"

# 1) baseline, current gold+10
run_one 1 baseline 10 10 "$OUT_ROOT/baseline_goldp10"

# 2) latest ckpt, current gold+10
run_one 2 final 10 10 "$OUT_ROOT/ckpt_goldp10" "$CKPT_ABS"

# 3) latest ckpt, original gold+0 (historical eval budget)
run_one 3 final 0 4 "$OUT_ROOT/ckpt_goldp0" "$CKPT_ABS"

echo "==============================================================" | tee -a "$LOG"
echo "[night_eval] ALL 3 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
echo "[night_eval] 1) $OUT_ROOT/baseline_goldp10/metrics_official.json" | tee -a "$LOG"
echo "[night_eval] 2) $OUT_ROOT/ckpt_goldp10/metrics_official.json" | tee -a "$LOG"
echo "[night_eval] 3) $OUT_ROOT/ckpt_goldp0/metrics_official.json" | tee -a "$LOG"
echo "==============================================================" | tee -a "$LOG"
