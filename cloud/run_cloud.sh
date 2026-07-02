#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run the FULL Tool-R0 curriculum (stages 1-6, from scratch) on all visible
# GPUs of a rented 80GB node (8× H100 SXM or A100-80GB on RunPod / Lambda).
#
# Prerequisites (run setup_cloud.sh first):
#   - Python env with all packages installed
#   - Data transferred via transfer_data.sh:
#       curricullum/data/filtered_toolr0_synthetic/  (training JSONL)
#       eval/data/NESTFUL-main/                      (eval dataset)
#       helper_calculations/output/                  (call distribution JSON)
#       nestful_repo/                                (IBM tool execution)
#
# Usage:
#   export WANDB_API_KEY=<key>    # optional but strongly recommended
#   export HF_TOKEN=<token>       # only if the base model needs auth
#   bash cloud/run_cloud.sh
#
# To RESUME from a specific stage (e.g. you already ran stages 1-3):
#   RESUME_STAGE=4 bash cloud/run_cloud.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."          # repo root
REPO_ROOT="$(pwd)"
echo "[run] repo root: $REPO_ROOT"

# ── Environment ────────────────────────────────────────────────────────────
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export TOKENIZERS_PARALLELISM=false

# ── GPU detection ──────────────────────────────────────────────────────────
NGPU="$(python3 -c 'import torch; print(torch.cuda.device_count())')"
if [ "$NGPU" -lt 1 ]; then
  echo "[run] ERROR: no CUDA GPUs visible. Did setup_cloud.sh finish correctly?"
  exit 1
fi
GPU_IDS="$(python3 -c "print(','.join(str(i) for i in range($NGPU)))")"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TRAIN_GPUS="$GPU_IDS"
echo "[run] using $NGPU GPU(s): $GPU_IDS"

for i in $(seq 0 $((NGPU-1))); do
  MEM="$(python3 -c "import torch; print(round(torch.cuda.get_device_properties($i).total_memory/1e9, 0), 'GB')" 2>/dev/null || echo '?')"
  NAME="$(python3 -c "import torch; print(torch.cuda.get_device_properties($i).name)" 2>/dev/null || echo '?')"
  echo "[run]   GPU $i: $NAME  $MEM"
done

# ── Preflight checks ───────────────────────────────────────────────────────
MISSING=0
for path in \
  "curricullum/data/filtered_toolr0_synthetic/epoch_1_1call.jsonl" \
  "curricullum/data/filtered_toolr0_synthetic/epoch_4_4call.jsonl" \
  "eval/data/NESTFUL-main/data_v2/nestful_data.jsonl" \
  "helper_calculations/output/nestful_call_distribution.json"
do
  if [ ! -f "$path" ]; then
    echo "[run] ERROR: missing required file: $path"
    MISSING=1
  fi
done
if [ "$MISSING" -eq 1 ]; then
  echo "[run] Run transfer_data.sh from the DGX first, then retry."
  exit 1
fi

BASELINE_FLAG=()
if [ -f "curricullum/training/results/baseline_nestful.json" ]; then
  echo "[run] baseline cache found — skipping baseline eval (~1 h saved)"
  BASELINE_FLAG=(--baseline_cache curricullum/training/results/baseline_nestful.json)
else
  echo "[run] no baseline cache — will run baseline eval on all NESTFUL groups (~1 h)"
fi

# ── Resume control ─────────────────────────────────────────────────────────
RESUME_STAGE="${RESUME_STAGE:-1}"
if [ "$RESUME_STAGE" -gt 1 ]; then
  MANIFEST="curricullum/checkpoints/qwen3_4b_curriculum_v2/checkpoints.json"
  if [ ! -f "$MANIFEST" ]; then
    echo "[run] ERROR: --resume_stage $RESUME_STAGE requires $MANIFEST."
    echo "      Transfer the checkpoint manifest from the DGX, or set RESUME_STAGE=1."
    exit 1
  fi
  STAGE_CKPT="curricullum/checkpoints/qwen3_4b_curriculum_v2/stage_$((RESUME_STAGE-1))_epoch1"
  if [ ! -d "$STAGE_CKPT" ]; then
    echo "[run] WARNING: stage $((RESUME_STAGE-1)) checkpoint not found at $STAGE_CKPT"
    echo "      Training may start from the base model. Transfer the checkpoint or set RESUME_STAGE=1."
  fi
fi

# ── Build STAGES flag (1..6) ───────────────────────────────────────────────
STAGES="$(python3 -c "print(','.join(str(i) for i in range($RESUME_STAGE, 7)))")"
echo "[run] stages to run: $STAGES  (resume_stage=$RESUME_STAGE)"

# ── Launch ─────────────────────────────────────────────────────────────────
mkdir -p curricullum/training/logs
LOG="curricullum/training/logs/cloud_run_$(date +%Y%m%d_%H%M%S).log"
echo "[run] logging to: $LOG"
echo "[run] tail -f $LOG   ← follow progress in another terminal"
echo

nohup python3 -u curricullum/train/run_curriculum_training.py \
  --config     curricullum/train/configs/qwen3_4b_curriculum_v2.yaml \
  --deepspeed_config configs/deepseed_zero2_8gpu.yaml \
  --wandb_project nestful-curriculum-toolr0 \
  --run_group  qwen3-4b-curriculum-v2-cloud \
  --resume_stage "$RESUME_STAGE" \
  --stages     "$STAGES" \
  --num_processes "$NGPU" \
  --train_gpus "$GPU_IDS" \
  --nestful_path       eval/data/NESTFUL-main/data_v2/nestful_data.jsonl \
  --call_dist_path     helper_calculations/output/nestful_call_distribution.json \
  "${BASELINE_FLAG[@]}" \
  > "$LOG" 2>&1 &

PID=$!
echo "[run] launched — PID $PID"
echo "[run] to kill:   kill $PID"
echo
echo "  Estimated wall time on 8×H100 SXM 80GB:"
echo "    ~9-11 h for stages 1-6 (all stages, from scratch)"
echo "    ~4-6 h for stages 4-6 (with RESUME_STAGE=4)"
echo
echo "  Cost at \$3.29/GPU/hr × 8 GPUs:"
echo "    stages 1-6:  ~\$230-290"
echo "    stages 4-6:  ~\$105-158"
