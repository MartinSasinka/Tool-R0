#!/usr/bin/env bash
# Launch P43 PROFILE_1000 online-dynamic MT-GRPO with the correct executor registry.
#
# From repo root (Linux pod):
#   bash experiments/nestful_mtgrpo_minimal/scripts/run_p43_dynamic_online.sh
#
# Requires: 4 GPUs, dataset under targeted_tool_data_factory/outputs/...,
#           factory trainer_adapter_p43 synced to the pod.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MINIMAL="$REPO/experiments/nestful_mtgrpo_minimal"
FACTORY="$REPO/experiments/targeted_tool_data_factory"
ADAPTER="$FACTORY/trainer_adapter_p43"
DATA="$FACTORY/outputs/pilot4_3_nestful_profile_1000/train_nestful_profile_1000.jsonl"
CFG="${CFG:-$MINIMAL/configs/qwen3_p43_profile1000_dynamic_online_samplingfix.yaml}"
if [[ ! -f "$CFG" ]]; then
  CFG="$MINIMAL/configs/qwen3_p43_profile1000_dynamic_online.yaml"
fi

export SYNTHETIC_TOOLS_DIR="$ADAPTER"
export CANARY_TRAJ_LOG="${CANARY_TRAJ_LOG:-1}"
export PYTHONPATH="${MINIMAL}:${FACTORY}/src:${ADAPTER}:${PYTHONPATH:-}"

if [[ ! -d "$ADAPTER/lib" ]]; then
  echo "[p43] ERROR: missing adapter at $ADAPTER" >&2
  exit 1
fi
if [[ ! -f "$DATA" ]]; then
  echo "[p43] ERROR: missing dataset $DATA" >&2
  exit 1
fi

echo "[p43] preflight gold replay..."
python "$ADAPTER/preflight_gold_replay.py" --data "$DATA" \
  --report "$FACTORY/outputs/pilot4_3_nestful_profile_1000/preflight_p43_gold_replay.json"

cd "$MINIMAL"
echo "[p43] SYNTHETIC_TOOLS_DIR=$SYNTHETIC_TOOLS_DIR"
echo "[p43] CANARY_TRAJ_LOG=$CANARY_TRAJ_LOG"
echo "[p43] starting train..."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
python run.py --mode train \
  --config "$CFG" \
  --override hardware.use_vllm=true \
  --override hardware.rollout_data_parallel_gpus=1,2,3 \
  --override hardware.vllm_gpu_memory_utilization=0.45 \
  --override hardware.vllm_gpu_memory_utilization_dp=0.70 \
  --override hardware.vllm_enforce_eager=true \
  --override logging.use_wandb=true \
  "$@"
