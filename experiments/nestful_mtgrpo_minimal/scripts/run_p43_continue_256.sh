#!/usr/bin/env bash
# Continue P43 PROFILE_1000 online-dynamic MT-GRPO: checkpoint@127 → target 256.
#
# From repo root (Linux / RunPod):
#   bash experiments/nestful_mtgrpo_minimal/scripts/run_p43_continue_256.sh
#
# Prerequisites:
#   - Original run dir present with adapter_epoch_13 (+ sampler_state.json)
#   - Updated run.py / grpo_train.py with continuous resume wiring
#   - Same dataset + trainer_adapter_p43 as the first run
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MINIMAL="$REPO/experiments/nestful_mtgrpo_minimal"
FACTORY="$REPO/experiments/targeted_tool_data_factory"
ADAPTER="$FACTORY/trainer_adapter_p43"
DATA="$FACTORY/outputs/pilot4_3_nestful_profile_1000/train_nestful_profile_1000.jsonl"
CFG="$MINIMAL/configs/qwen3_p43_profile1000_dynamic_online_continue256.yaml"

# Last valid saved adapter from the samplingfix run (trainer_state.global_step=127).
SRC_RUN="${SRC_RUN:-$MINIMAL/outputs/qwen3_p43_profile1000_dynamic_online_samplingfix}"
CKPT="${CKPT:-$SRC_RUN/checkpoints/adapter_epoch_13}"

export SYNTHETIC_TOOLS_DIR="$ADAPTER"
export CANARY_TRAJ_LOG="${CANARY_TRAJ_LOG:-0}"
export PYTHONPATH="${MINIMAL}:${FACTORY}/src:${ADAPTER}:${PYTHONPATH:-}"

if [[ ! -f "$CFG" ]]; then
  echo "[p43-continue] ERROR: missing config $CFG" >&2
  exit 1
fi
if [[ ! -d "$ADAPTER/lib" ]]; then
  echo "[p43-continue] ERROR: missing adapter at $ADAPTER" >&2
  exit 1
fi
if [[ ! -f "$DATA" ]]; then
  echo "[p43-continue] ERROR: missing dataset $DATA" >&2
  exit 1
fi
if [[ ! -f "$CKPT/adapter_config.json" ]]; then
  echo "[p43-continue] ERROR: resume checkpoint invalid: $CKPT" >&2
  exit 1
fi
if [[ ! -f "$CKPT/sampler_state.json" ]]; then
  echo "[p43-continue] ERROR: missing sampler_state.json in $CKPT" >&2
  exit 1
fi
if [[ ! -f "$CKPT/trainer_state.json" ]]; then
  echo "[p43-continue] ERROR: missing trainer_state.json in $CKPT" >&2
  exit 1
fi

echo "[p43-continue] SRC_RUN=$SRC_RUN"
echo "[p43-continue] CKPT=$CKPT"
echo "[p43-continue] CFG=$CFG"
echo "[p43-continue] SYNTHETIC_TOOLS_DIR=$SYNTHETIC_TOOLS_DIR"
python - <<'PY' "$CKPT"
import json, sys
ckpt = sys.argv[1]
ts = json.load(open(f"{ckpt}/trainer_state.json", encoding="utf-8"))
ss = json.load(open(f"{ckpt}/sampler_state.json", encoding="utf-8"))
print(f"[p43-continue] precheck trainer_state.global_step={ts.get('global_step')}")
print(f"[p43-continue] precheck sampler bootstrap_complete={ss.get('bootstrap_complete')} "
      f"n_observed={ss.get('n_observed')}")
assert int(ts.get("global_step") or -1) == 127, ts.get("global_step")
assert ss.get("bootstrap_complete") is True
print("[p43-continue] precheck OK")
PY

cd "$MINIMAL"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
python run.py --mode train \
  --config "$CFG" \
  --checkpoint "$CKPT" \
  --override hardware.use_vllm=true \
  --override hardware.rollout_data_parallel_gpus=1,2,3 \
  --override hardware.vllm_gpu_memory_utilization=0.45 \
  --override hardware.vllm_gpu_memory_utilization_dp=0.70 \
  "$@"
