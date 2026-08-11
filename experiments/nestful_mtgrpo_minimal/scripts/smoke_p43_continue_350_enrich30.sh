#!/usr/bin/env bash
# RunPod smoke: continue350 enrich30 on 4 GPUs + vLLM (few optimizer steps only).
#
# From repo root (Linux / RunPod):
#   bash experiments/nestful_mtgrpo_minimal/scripts/smoke_p43_continue_350_enrich30.sh
#
# Defaults:
#   SMOKE_STEPS=2  → target_optimizer_updates = 200 + 2 = 202
#   separate smoke output_dir (does not touch full continue350 artifacts)
#
# Expect in logs:
#   [sampler] override restored mode '...' -> 'dynamic_profile_plus_enrichment'
#   CANARY_TRAJ_LOG=0
#   vLLM DP workers on GPUs 1,2,3 (train on 0)
#   no hardware.vllm_enforce_eager=true override
#
# NOT included: prefetch-during-HF-update (still not implemented).
# Expectation: modest speedup vs continue256 leftovers, not a 5× miracle.
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
SMOKE_STEPS="${SMOKE_STEPS:-2}"
SMOKE_TARGET=$((200 + SMOKE_STEPS))
SMOKE_NAME="${SMOKE_NAME:-qwen3_p43_continue350_enrich30_smoke}"
SMOKE_OUT="outputs/${SMOKE_NAME}"

export SYNTHETIC_TOOLS_DIR="$ADAPTER"
export CANARY_TRAJ_LOG="${CANARY_TRAJ_LOG:-0}"
export PYTHONPATH="${MINIMAL}:${FACTORY}/src:${ADAPTER}:${PYTHONPATH:-}"

if [[ ! -f "$CFG" ]]; then
  echo "[smoke350] ERROR: missing config $CFG" >&2
  exit 1
fi
if [[ ! -d "$ADAPTER/lib" ]]; then
  echo "[smoke350] ERROR: missing adapter at $ADAPTER" >&2
  exit 1
fi
if [[ ! -f "$PROFILE" || ! -f "$ENRICH" ]]; then
  echo "[smoke350] ERROR: missing profile/enrichment jsonl" >&2
  exit 1
fi
if [[ ! -f "$CKPT/adapter_config.json" || ! -f "$CKPT/optimizer.pt" ]]; then
  echo "[smoke350] ERROR: resume checkpoint invalid: $CKPT" >&2
  exit 1
fi

echo "[smoke350] SRC_RUN=$SRC_RUN"
echo "[smoke350] CKPT=$CKPT"
echo "[smoke350] SMOKE_STEPS=$SMOKE_STEPS → target_optimizer_updates=$SMOKE_TARGET"
echo "[smoke350] SMOKE_OUT=$SMOKE_OUT"
echo "[smoke350] CANARY_TRAJ_LOG=$CANARY_TRAJ_LOG (expect 0)"
echo "[smoke350] mix: profile=0.70 enrichment=0.30"
echo "[smoke350] 4GPU vLLM: CUDA_VISIBLE_DEVICES=0,1,2,3 DP=1,2,3 enforce_eager=NOT overridden"
python - <<'PY' "$CKPT" "$PROFILE" "$ENRICH" "$SMOKE_TARGET"
import json, sys
from pathlib import Path
ckpt, profile, enrich, target = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ts = json.load(open(f"{ckpt}/trainer_state.json", encoding="utf-8"))
ss = json.load(open(f"{ckpt}/sampler_state.json", encoding="utf-8"))
gs = int(ts.get("global_step") or -1)
print(f"[smoke350] precheck global_step={gs} bootstrap_complete={ss.get('bootstrap_complete')} "
      f"restored_mode={ss.get('sampler_mode')}")
assert gs == 200, gs
assert ss.get("bootstrap_complete") is True
assert Path(ckpt, "optimizer.pt").is_file()
assert target > 200, target
n_prof = sum(1 for _ in open(profile, encoding="utf-8") if _.strip())
n_enr = sum(1 for _ in open(enrich, encoding="utf-8") if _.strip())
print(f"[smoke350] precheck pools profile={n_prof} enrichment={n_enr}")
assert n_prof == 1000 and n_enr == 500, (n_prof, n_enr)
print("[smoke350] precheck OK")
PY

cd "$MINIMAL"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
python run.py --mode train \
  --config "$CFG" \
  --checkpoint "$CKPT" \
  --override "experiment.name=${SMOKE_NAME}" \
  --override "experiment.output_dir=${SMOKE_OUT}" \
  --override "model.output_adapter_dir=${SMOKE_OUT}/checkpoints" \
  --override "logging.observability_dir=${SMOKE_OUT}/observability" \
  --override "training.target_optimizer_updates=${SMOKE_TARGET}" \
  --override "logging.timing_profile_warmup_groups=1" \
  --override "logging.timing_profile_groups=4" \
  --override hardware.use_vllm=true \
  --override hardware.rollout_data_parallel_gpus=1,2,3 \
  --override hardware.vllm_gpu_memory_utilization=0.45 \
  --override hardware.vllm_gpu_memory_utilization_dp=0.70 \
  "$@"

echo "[smoke350] DONE — check logs for sampler override + timing_profile; artifacts in $MINIMAL/$SMOKE_OUT"
echo "[smoke350] Full run after smoke OK:"
echo "  bash experiments/nestful_mtgrpo_minimal/scripts/run_p43_continue_350_enrich30.sh"
