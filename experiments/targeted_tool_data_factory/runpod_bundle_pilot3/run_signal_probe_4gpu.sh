#!/usr/bin/env bash
# =============================================================================
#  Pilot3 SIGNAL PROBE — inference only, 4 GPUs
#
#  All 600 frozen train tasks x 4 rollouts (P2), then a boundary subset x 8
#  (P3). Selects a NESTFUL-matched Phase-1 train subset with terminal/process
#  mixed signal. NEVER trains. Reuses the pilot2 probe workers unchanged.
#
#  Usage (from repo root):
#      export HF_TOKEN=...
#      bash experiments/targeted_tool_data_factory/runpod_bundle_pilot3/run_signal_probe_4gpu.sh
#
#  Flags: --dry-run | --resume | --stage p2|select|p3|report
# =============================================================================
set -Eeuo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FACTORY="$(cd "$BUNDLE/.." && pwd)"
PROBE_LIB="$(cd "$BUNDLE/../runpod_bundle_pilot2" && pwd)"
EXPERIMENTS="$(cd "$FACTORY/.." && pwd)"
REPO="$(cd "$EXPERIMENTS/.." && pwd)"
PY="${PYTHON:-python3}"

# Git Bash + Windows CPython path fix
if command -v cygpath >/dev/null 2>&1; then
  BUNDLE="$(cygpath -m "$BUNDLE")"
  FACTORY="$(cygpath -m "$FACTORY")"
  PROBE_LIB="$(cygpath -m "$PROBE_LIB")"
  REPO="$(cygpath -m "$REPO")"
fi

DRY_RUN=0
RESUME_FLAG=""
ONLY_STAGE=""
BACKEND="vllm"
P2_ROLLOUTS=4
P3_ROLLOUTS=8
P3_LIMIT=120
PHASE1_TARGET=400
PHASE1_MIN=300
PHASE1_MAX=500
GPUS="${GPUS:-0,1,2,3}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SEED="${SEED:-20260727}"

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)      DRY_RUN=1; shift ;;
    --resume)       RESUME_FLAG="--resume"; shift ;;
    --stage)        ONLY_STAGE="$2"; shift 2 ;;
    --backend)      BACKEND="$2"; shift 2 ;;
    --p2-rollouts)  P2_ROLLOUTS="$2"; shift 2 ;;
    --p3-rollouts)  P3_ROLLOUTS="$2"; shift 2 ;;
    --p3-limit)     P3_LIMIT="$2"; shift 2 ;;
    --phase1-target) PHASE1_TARGET="$2"; shift 2 ;;
    *) echo "[probe3] unknown arg: $1" >&2; exit 1 ;;
  esac
done

DATA="$BUNDLE/data/train_grpo_pilot3.jsonl"
ADAPTER="$FACTORY/trainer_adapter"
RESULTS="${RESULTS:-$FACTORY/outputs/runpod_pilot3}"
PROBE_DIR="$RESULTS/signal_probe"
export SYNTHETIC_TOOLS_DIR="$ADAPTER"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$PROBE_DIR"
LOG="$PROBE_DIR/run_signal_probe_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

banner() {
  echo ""
  echo "=================================================================="
  echo "[probe3] $1"
  echo "=================================================================="
}
run() {
  echo "+ $*"
  if [ "$DRY_RUN" = "0" ]; then "$@"; else
    case " $* " in *" --dry-run "*) "$@" ;; esac
  fi
}
want() { [ -z "$ONLY_STAGE" ] || [ "$ONLY_STAGE" = "$1" ]; }

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N_GPUS=${#GPU_ARR[@]}
cd "$REPO"

echo "[probe3] train data: $DATA (600 tasks)"
echo "[probe3] model: $MODEL  backend=$BACKEND"
echo "[probe3] NO training / NO LoRA update / NO NESTFUL-1661"

[ -f "$DATA" ] || { echo "[probe3] ABORT: missing $DATA — run build_bundle.py" >&2; exit 1; }
"$PY" "$BUNDLE/verify_hashes.py" || true

# ── P2 ────────────────────────────────────────────────────────────────────
if want p2; then
banner "P2 — all 600 train tasks x $P2_ROLLOUTS rollouts on $N_GPUS GPUs"
pids=(); i=0
for gpu in "${GPU_ARR[@]}"; do
  out="$PROBE_DIR/shard_p2_${i}.jsonl"
  cmd=("$PY" "$PROBE_LIB/signal_probe_worker.py"
       --data "$DATA" --out "$out"
       --phase P2 --rollouts "$P2_ROLLOUTS"
       --shard-index "$i" --shard-count "$N_GPUS"
       --model "$MODEL" --seed "$SEED" --backend "$BACKEND")
  [ -n "$RESUME_FLAG" ] && cmd+=("$RESUME_FLAG")
  [ "$DRY_RUN" = "1" ] && cmd+=(--dry-run)
  echo "+ CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
  if [ "$DRY_RUN" = "1" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" &
  else
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" &
  fi
  pids+=($!)
  i=$((i + 1))
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
[ "$rc" = "0" ] || { echo "[probe3] P2 worker failure" >&2; exit 1; }
fi

# ── select P3 ─────────────────────────────────────────────────────────────
if want select; then
banner "select P3 boundary tasks (limit=$P3_LIMIT)"
run "$PY" "$PROBE_LIB/signal_probe_analyze.py" --mode select-p3 \
  --probe-dir "$PROBE_DIR" --data "$DATA" --p3-limit "$P3_LIMIT"
fi

# ── P3 ────────────────────────────────────────────────────────────────────
if want p3; then
banner "P3 — boundary tasks x $P3_ROLLOUTS"
P3_IDS="$PROBE_DIR/p3_task_ids.txt"
if [ ! -f "$P3_IDS" ] && [ "$DRY_RUN" = "1" ]; then
  echo "[probe3] dry-run: P3 ids not present yet"
else
  [ -f "$P3_IDS" ] || { echo "[probe3] ABORT: missing $P3_IDS" >&2; exit 1; }
  pids=(); i=0
  for gpu in "${GPU_ARR[@]}"; do
    out="$PROBE_DIR/shard_p3_${i}.jsonl"
    cmd=("$PY" "$PROBE_LIB/signal_probe_worker.py"
         --data "$DATA" --out "$out"
         --phase P3 --rollouts "$P3_ROLLOUTS"
         --task-ids "$P3_IDS"
         --shard-index "$i" --shard-count "$N_GPUS"
         --model "$MODEL" --seed "$SEED" --backend "$BACKEND")
    [ -n "$RESUME_FLAG" ] && cmd+=("$RESUME_FLAG")
    [ "$DRY_RUN" = "1" ] && cmd+=(--dry-run)
    echo "+ CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" &
    pids+=($!)
    i=$((i + 1))
  done
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  [ "$rc" = "0" ] || { echo "[probe3] P3 worker failure" >&2; exit 1; }
fi
fi

# ── report + Phase-1 selection (NESTFUL-matched, mixed-signal) ────────────
if want report; then
banner "report + Phase-1 selection (target=$PHASE1_TARGET, keep hard buckets)"
run "$PY" "$PROBE_LIB/signal_probe_analyze.py" --mode report \
  --probe-dir "$PROBE_DIR" --data "$DATA" \
  --phase1-target "$PHASE1_TARGET" \
  --phase1-min "$PHASE1_MIN" \
  --phase1-max "$PHASE1_MAX"
echo "[probe3] report: $PROBE_DIR/SIGNAL_PROBE_REPORT.md"
echo "[probe3] phase1: $PROBE_DIR/recommended_phase1_train.jsonl"
fi

banner "done"
echo "[probe3] GRPO was NOT started"
