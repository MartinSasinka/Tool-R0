#!/usr/bin/env bash
# Continue P43 PROFILE_1000 + 30% enrichment: checkpoint@200 → target 350.
#
# From repo root (Linux / RunPod):
#   bash experiments/nestful_mtgrpo_minimal/scripts/run_p43_continue_350_enrich30.sh
#
# Smoke first (2 steps, separate output_dir):
#   bash experiments/nestful_mtgrpo_minimal/scripts/smoke_p43_continue_350_enrich30.sh
#
# Speed vs continue256 launch:
#   - does NOT override vllm_enforce_eager (yaml=false → CUDA graphs ON)
#   - CANARY_TRAJ_LOG default 0 / log_canary_trajectories=false
# Prefetch-during-HF-update: NOT implemented. Expect modest speedup, not 5×.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MINIMAL="$REPO/experiments/nestful_mtgrpo_minimal"
FACTORY="$REPO/experiments/targeted_tool_data_factory"
ADAPTER="$FACTORY/trainer_adapter_p43"
PROFILE="$FACTORY/outputs/pilot4_3_nestful_profile_1000/train_nestful_profile_1000.jsonl"
ENRICH="$FACTORY/outputs/pilot4_3_nestful_profile_1000/train_nestful_enrichment_500.jsonl"
CFG="$MINIMAL/configs/qwen3_p43_profile1000_dynamic_online_continue350_enrich30.yaml"

SRC_RUN="${SRC_RUN:-$MINIMAL/outputs/qwen3_p43_profile1000_dynamic_online_continue256}"
CKPT="${CKPT:-$SRC_RUN/checkpoints/adapter_epoch_8}"

export SYNTHETIC_TOOLS_DIR="$ADAPTER"
export CANARY_TRAJ_LOG="${CANARY_TRAJ_LOG:-0}"
export PYTHONPATH="${MINIMAL}:${FACTORY}/src:${ADAPTER}:${PYTHONPATH:-}"

if [[ ! -f "$CFG" ]]; then
  echo "[p43-continue350] ERROR: missing config $CFG" >&2
  exit 1
fi
if [[ ! -d "$ADAPTER/lib" ]]; then
  echo "[p43-continue350] ERROR: missing adapter at $ADAPTER" >&2
  exit 1
fi
if [[ ! -f "$PROFILE" ]]; then
  echo "[p43-continue350] ERROR: missing profile dataset $PROFILE" >&2
  exit 1
fi
if [[ ! -f "$ENRICH" ]]; then
  echo "[p43-continue350] ERROR: missing enrichment dataset $ENRICH" >&2
  exit 1
fi
if [[ ! -f "$CKPT/adapter_config.json" ]]; then
  echo "[p43-continue350] ERROR: resume checkpoint invalid: $CKPT" >&2
  exit 1
fi
if [[ ! -f "$CKPT/sampler_state.json" ]]; then
  echo "[p43-continue350] ERROR: missing sampler_state.json in $CKPT" >&2
  exit 1
fi
if [[ ! -f "$CKPT/trainer_state.json" ]]; then
  echo "[p43-continue350] ERROR: missing trainer_state.json in $CKPT" >&2
  exit 1
fi
if [[ ! -f "$CKPT/optimizer.pt" ]]; then
  echo "[p43-continue350] ERROR: missing optimizer.pt in $CKPT" >&2
  exit 1
fi

echo "[p43-continue350] SRC_RUN=$SRC_RUN"
echo "[p43-continue350] CKPT=$CKPT"
echo "[p43-continue350] CFG=$CFG"
echo "[p43-continue350] SYNTHETIC_TOOLS_DIR=$SYNTHETIC_TOOLS_DIR"
echo "[p43-continue350] CANARY_TRAJ_LOG=$CANARY_TRAJ_LOG (expect 0)"
echo "[p43-continue350] mix: profile=0.70 enrichment=0.30 → target 350"
echo "[p43-continue350] vllm_enforce_eager: NOT overridden (yaml false)"
python - <<'PY' "$CKPT" "$PROFILE" "$ENRICH"
import json, sys
from pathlib import Path
ckpt, profile, enrich = sys.argv[1], sys.argv[2], sys.argv[3]
ts = json.load(open(f"{ckpt}/trainer_state.json", encoding="utf-8"))
ss = json.load(open(f"{ckpt}/sampler_state.json", encoding="utf-8"))
print(f"[p43-continue350] precheck trainer_state.global_step={ts.get('global_step')}")
print(f"[p43-continue350] precheck sampler bootstrap_complete={ss.get('bootstrap_complete')} "
      f"n_observed={ss.get('n_observed')} restored_mode={ss.get('sampler_mode')}")
assert int(ts.get("global_step") or -1) == 200, ts.get("global_step")
assert ss.get("bootstrap_complete") is True
assert Path(ckpt, "optimizer.pt").is_file()
n_prof = sum(1 for _ in open(profile, encoding="utf-8") if _.strip())
n_enr = sum(1 for _ in open(enrich, encoding="utf-8") if _.strip())
print(f"[p43-continue350] precheck pools profile={n_prof} enrichment={n_enr}")
assert n_prof == 1000 and n_enr == 500, (n_prof, n_enr)
print("[p43-continue350] precheck OK")
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
