#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  Pilot2 SIGNAL PROBE — inference only, 4 GPUs
#
#  Answers ONE question before any GPU hour is spent on GRPO: do the 160 frozen
#  Pilot2 train tasks produce a usable rollout/reward signal for the exact BF16
#  Qwen/Qwen3-4B-Instruct-2507 base checkpoint?
#
#  This script NEVER trains. No optimizer, no gradients, no LoRA update, no
#  checkpoint write — only forward passes through the same rollout code,
#  executor and reward dispatch the planned D1 run will use.
#
#    P2   all 160 train tasks x 4 rollouts   (broad signal census)
#    P3   <= 64 boundary tasks x 8 rollouts  (resolve the borderline groups)
#
#  Same as D1: executor.mode=synthetic on the factory trainer adapter,
#  reward arm A4_GATED_VERIFIABLE, temperature 0.7, top_p 0.95, react paradigm.
#  Different from D1 on purpose: BF16 base weights (no 4-bit quantisation) and
#  no adapter, because the probe measures the UNMODIFIED starting policy.
#
#  Usage (from repo root):
#      export HF_TOKEN=...
#      bash experiments/targeted_tool_data_factory/runpod_bundle_pilot2/run_signal_probe_4gpu.sh
#
#  Flags:
#      --dry-run        resolve everything, print every command, generate nothing
#      --resume         reuse cached rollouts (content-hash cache) and continue
#      --stage STAGE    run only one of: p2 select p3 report
#      --backend B      vllm (default) | hf
#      --p2-rollouts N  default 4
#      --p3-rollouts N  default 8
#      --p3-limit N     default 64
# ═══════════════════════════════════════════════════════════════════════════
set -Eeuo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FACTORY="$(cd "$BUNDLE/.." && pwd)"
EXPERIMENTS="$(cd "$FACTORY/.." && pwd)"
REPO="$(cd "$EXPERIMENTS/.." && pwd)"
PY="${PYTHON:-python3}"

DRY_RUN=0
RESUME_FLAG=""
ONLY_STAGE=""
BACKEND="vllm"
P2_ROLLOUTS=4
P3_ROLLOUTS=8
P3_LIMIT=64
GPUS="${GPUS:-0,1,2,3}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SEED="${SEED:-20260726}"

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)      DRY_RUN=1; shift ;;
    --resume)       RESUME_FLAG="--resume"; shift ;;
    --stage)        ONLY_STAGE="$2"; shift 2 ;;
    --backend)      BACKEND="$2"; shift 2 ;;
    --p2-rollouts)  P2_ROLLOUTS="$2"; shift 2 ;;
    --p3-rollouts)  P3_ROLLOUTS="$2"; shift 2 ;;
    --p3-limit)     P3_LIMIT="$2"; shift 2 ;;
    *) echo "[probe] unknown arg: $1" >&2; exit 1 ;;
  esac
done

DATA="$BUNDLE/data/train_grpo_pilot2.jsonl"
ADAPTER="$FACTORY/trainer_adapter"
RESULTS="${RESULTS:-$FACTORY/outputs/runpod_pilot2}"
PROBE_DIR="$RESULTS/signal_probe"

# The trainer resolves its executable registry from this variable. The probe
# must run pilot2 tasks on the FACTORY adapter, never the legacy Stage-3 one.
export SYNTHETIC_TOOLS_DIR="$ADAPTER"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$PROBE_DIR"
LOG="$PROBE_DIR/run_signal_probe_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

banner() {
  echo ""
  echo "══════════════════════════════════════════════════════════════════"
  echo "[probe] $1"
  echo "══════════════════════════════════════════════════════════════════"
}

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = "0" ]; then "$@"; fi
}

want() { [ -z "$ONLY_STAGE" ] || [ "$ONLY_STAGE" = "$1" ]; }

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N_GPUS=${#GPU_ARR[@]}

cd "$REPO"

echo "[probe] mode:       inference only (NO training, NO optimizer, NO LoRA update)"
echo "[probe] model:      $MODEL (bfloat16, no 4-bit, no adapter)"
echo "[probe] data:       $DATA"
echo "[probe] registry:   $SYNTHETIC_TOOLS_DIR"
echo "[probe] reward arm: A4_GATED_VERIFIABLE"
echo "[probe] workers:    $N_GPUS parallel inference workers on GPUs $GPUS"
echo "[probe] P2:         all tasks x $P2_ROLLOUTS rollouts"
echo "[probe] P3:         <= $P3_LIMIT boundary tasks x $P3_ROLLOUTS rollouts"
echo "[probe] out:        $PROBE_DIR"

# ── 1. GPUs ────────────────────────────────────────────────────────────────
banner "1/6  GPU check ($N_GPUS parallel inference workers)"
DRY_RUN="$DRY_RUN" N_GPUS="$N_GPUS" "$PY" - <<'PYEOF'
import os
import sys
try:
    import torch
except ImportError:
    print("[probe] torch not importable — install dependencies first "
          "(bash runpod_bundle_pilot2/install.sh)")
    sys.exit(0 if os.environ.get("DRY_RUN") == "1" else 1)
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name} {p.total_memory/1e9:.1f} GB")
print(f"[probe] visible GPUs: {n}")
need = int(os.environ.get("N_GPUS", "4"))
if n < need:
    if os.environ.get("DRY_RUN") == "1":
        print(f"[probe] dry run: continuing without {need} GPUs")
        sys.exit(0)
    print(f"[probe] ABORT: {need} GPUs required", file=sys.stderr)
    sys.exit(1)
PYEOF

# ── 2. frozen data + registry ──────────────────────────────────────────────
# Read-only, so it runs even under --dry-run: a dry run should catch a pod with
# the wrong frozen data or the legacy tool registry BEFORE the GPUs are booked.
banner "2/6  verify frozen dataset hashes and the factory registry"
"$PY" "$BUNDLE/verify_hashes.py" --manifest "$BUNDLE/MANIFEST.sha256.json"
"$PY" - <<PYEOF
import os, sys
sys.path.insert(0, "$EXPERIMENTS/nestful_mtgrpo_minimal")
from synthetic_tool_registry import load_synthetic_tools_module
mod = load_synthetic_tools_module(os.environ["SYNTHETIC_TOOLS_DIR"])
print(f"[probe] registry {mod.REGISTRY_VERSION} hash={mod.registry_hash()[:16]}… "
      f"n_tools={len(mod.TOOLS)}")
assert getattr(mod, "REGISTRY_SOURCE", "") == "targeted_tool_data_factory", \
    "probe must run on the factory adapter, not the legacy Stage-3 registry"
PYEOF

# ── worker launcher ────────────────────────────────────────────────────────
# One process per GPU, contiguous strided shards over the frozen file order.
launch_shards () {           # $1=phase  $2=rollouts  $3=optional --task-ids file
  local phase="$1" rollouts="$2" task_ids="${3:-}"
  local pids=() labels=() i=0
  for gpu in "${GPU_ARR[@]}"; do
    local out="$PROBE_DIR/shard_$(echo "$phase" | tr 'A-Z' 'a-z')_${i}.jsonl"
    local -a cmd=(
      "$PY" "$BUNDLE/signal_probe_worker.py"
      --data "$DATA" --out "$out"
      --phase "$phase" --rollouts "$rollouts"
      --shard-index "$i" --shard-count "$N_GPUS"
      --model "$MODEL" --seed "$SEED" --backend "$BACKEND"
    )
    [ -n "$task_ids" ] && cmd+=(--task-ids "$task_ids")
    [ -n "$RESUME_FLAG" ] && cmd+=("$RESUME_FLAG")
    [ "$DRY_RUN" = "1" ] && cmd+=(--dry-run)
    echo "+ CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" &
    pids+=($!)
    labels+=("$phase/gpu$gpu")
    i=$((i + 1))
  done
  local rc=0 k=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[probe] FAILED worker ${labels[$k]}" >&2
      rc=1
    fi
    k=$((k + 1))
  done
  return $rc
}

# ── 3. P2 ──────────────────────────────────────────────────────────────────
if want p2; then
banner "3/6  P2 — all train tasks x $P2_ROLLOUTS rollouts on $N_GPUS GPUs"
launch_shards P2 "$P2_ROLLOUTS"
fi

# ── 4. P3 selection ────────────────────────────────────────────────────────
if want select; then
banner "4/6  select <= $P3_LIMIT boundary tasks for P3"
run "$PY" "$BUNDLE/signal_probe_analyze.py" --mode select-p3 \
  --probe-dir "$PROBE_DIR" --data "$DATA" --p3-limit "$P3_LIMIT" \
  $([ "$DRY_RUN" = "1" ] && echo --dry-run)
fi

# ── 5. P3 ──────────────────────────────────────────────────────────────────
if want p3; then
banner "5/6  P3 — boundary tasks x $P3_ROLLOUTS rollouts on $N_GPUS GPUs"
P3_IDS="$PROBE_DIR/p3_task_ids.txt"
if [ "$DRY_RUN" = "1" ] && [ ! -f "$P3_IDS" ]; then
  echo "[probe] dry run: $P3_IDS not present yet (produced by stage 4) — "
  echo "        P3 workers would be launched with --task-ids $P3_IDS"
  launch_shards P3 "$P3_ROLLOUTS"
else
  if [ ! -f "$P3_IDS" ]; then
    echo "[probe] ABORT: $P3_IDS missing — run stage 'select' first" >&2
    exit 1
  fi
  echo "[probe] P3 tasks: $(wc -l < "$P3_IDS")"
  launch_shards P3 "$P3_ROLLOUTS" "$P3_IDS"
fi
fi

# ── 6. report ──────────────────────────────────────────────────────────────
if want report; then
banner "6/6  group metrics, reward-ordering audit, Phase-1 subset, verdict"
run "$PY" "$BUNDLE/signal_probe_analyze.py" --mode report \
  --probe-dir "$PROBE_DIR" --data "$DATA" \
  $([ "$DRY_RUN" = "1" ] && echo --dry-run)
fi

banner "done"
echo "[probe] no training was performed by this script"
echo "[probe] report:   $PROBE_DIR/SIGNAL_PROBE_REPORT.md"
echo "[probe] json:     $PROBE_DIR/SIGNAL_PROBE_REPORT.json"
echo "[probe] rollouts: $PROBE_DIR/rollouts.jsonl"
echo "[probe] groups:   $PROBE_DIR/groups.jsonl"
echo "[probe] phase 1:  $PROBE_DIR/recommended_phase1_train.jsonl"
echo "[probe] deferred: $PROBE_DIR/deferred_phase2_tasks.jsonl"
echo "[probe] read the VERDICT before starting GRPO:"
echo "        PASS -> train on recommended_phase1_train.jsonl"
echo "        CONDITIONAL -> A4 only, expect slow movement"
echo "        STOP -> fix reward or task mix first, do not train"
