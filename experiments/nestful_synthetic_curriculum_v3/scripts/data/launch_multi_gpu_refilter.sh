#!/usr/bin/env bash
# Launch N parallel rollout-gate refilter workers (one per GPU), each probing
# ONE agentic worker shard (gpu0..gpuN-1), then merge kept/rejected outputs.
#
# Each worker re-runs the NEW multi-turn probe_rollout_signal() on rows that
# were already accepted at generation time (old single-shot gate). No OpenRouter.
#
# Usage (repo root, agentic venv active):
#   export WEAK_SOLVER_BACKEND=local
#   bash experiments/nestful_synthetic_curriculum_v3/scripts/data/launch_multi_gpu_refilter.sh
#
# Env overrides:
#   NUM_GPUS=4
#   WORKERS_BASE=<v3>/data/agentic_workers     source shards gpu0..gpu3
#   OUT_BASE=<v3>/data/.../refilter_mt_gate   per-GPU output refilter_gpu$i
#   BASE_SEED=42
#   DRY_RUN=1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
V3_ROOT="$REPO_ROOT/experiments/nestful_synthetic_curriculum_v3"
cd "$REPO_ROOT"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[refilter-launch] ERROR: tmux required" >&2
  exit 1
fi

DETECTED_GPUS="$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | wc -l || echo 1)"
NUM_GPUS="${NUM_GPUS:-$DETECTED_GPUS}"
WORKERS_BASE="${WORKERS_BASE:-$V3_ROOT/data/agentic_workers}"
OUT_BASE="${OUT_BASE:-$V3_ROOT/data/curriculum_v4_nestful_like_agentic_openrouter/refilter_mt_gate_$(date -u +%Y%m%d)}"
BASE_SEED="${BASE_SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON="${PYTHON:-$V3_ROOT/.venv/bin/python}"
REFILTER="$V3_ROOT/scripts/data/refilter_agentic_rollout_gate.py"
MERGE="$V3_ROOT/scripts/data/merge_refilter_shards.py"

export WEAK_SOLVER_BACKEND="${WEAK_SOLVER_BACKEND:-local}"
export LOCAL_WEAK_MODEL="${LOCAL_WEAK_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
export LOCAL_WEAK_4BIT="${LOCAL_WEAK_4BIT:-0}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-0}"

echo "======================================================================="
echo "[refilter-launch] num_gpus     = $NUM_GPUS"
echo "[refilter-launch] workers_base = $WORKERS_BASE"
echo "[refilter-launch] out_base     = $OUT_BASE"
echo "[refilter-launch] weak model   = $LOCAL_WEAK_MODEL (4bit=$LOCAL_WEAK_4BIT)"
echo "[refilter-launch] rollout      = N=$ROLLOUT_N T=$ROLLOUT_TEMPERATURE"
echo "======================================================================="

mkdir -p "$OUT_BASE"

for i in $(seq 0 $((NUM_GPUS - 1))); do
  WDIR="$WORKERS_BASE/gpu$i"
  ODIR="$OUT_BASE/refilter_gpu$i"
  SESSION="refilter_gpu$i"

  if [[ ! -d "$WDIR" ]]; then
    echo "[refilter-launch] WARN: missing $WDIR — skipping gpu$i"
    continue
  fi

  CMD="cd '$REPO_ROOT' && \
export WEAK_SOLVER_BACKEND='$WEAK_SOLVER_BACKEND' && \
export LOCAL_WEAK_MODEL='$LOCAL_WEAK_MODEL' && \
export LOCAL_WEAK_4BIT='$LOCAL_WEAK_4BIT' && \
export LOCAL_WEAK_DEVICE=cuda:0 && \
export CUDA_VISIBLE_DEVICES=$i && \
export ROLLOUT_N='$ROLLOUT_N' && \
export ROLLOUT_TEMPERATURE='$ROLLOUT_TEMPERATURE' && \
export ROLLOUT_MAX_TOKENS='$ROLLOUT_MAX_TOKENS' && \
'$PYTHON' '$REFILTER' \
  --workers '$WDIR' \
  --output-dir '$ODIR' \
  --seed $((BASE_SEED + i)) \
  --no-dedup \
  --log-every 10 \
  && echo DONE > '$ODIR/DONE' \
  || echo FAIL > '$ODIR/FAIL'"

  echo "[refilter-launch] gpu$i -> $WDIR -> $ODIR (tmux:$SESSION)"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  $CMD"
    continue
  fi

  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SESSION" "$CMD"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[refilter-launch] DRY RUN — no tmux sessions started"
  exit 0
fi

echo
echo "[refilter-launch] tmux sessions: refilter_gpu0 .. refilter_gpu$((NUM_GPUS - 1))"
echo "[refilter-launch] monitor:  tmux attach -t refilter_gpu0"
echo "[refilter-launch] progress: ls -l $OUT_BASE/refilter_gpu*/DONE"
echo
echo "After ALL workers finish, merge:"
echo "  python $MERGE \\"
echo "    --shards-glob '$OUT_BASE/refilter_gpu*' \\"
echo "    --output-dir '$OUT_BASE/merged'"
