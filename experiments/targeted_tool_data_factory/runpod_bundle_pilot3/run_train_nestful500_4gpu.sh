#!/usr/bin/env bash
# =============================================================================
#  Pilot3 — GRPO train on frozen train tasks, then NESTFUL-500 eval only
#
#  - Train: first N rows of data/train_grpo_pilot3.jsonl (default N=600)
#  - Eval:  ONLY the new checkpoint on nestful_diagnostic_500,
#           sharded across GPUs 0-3 (~125 tasks/GPU), then merged
#  - Skips: signal probe, C0 eval, held-out eval, NESTFUL-1661
#
#  Usage (repo root on RunPod):
#      export HF_TOKEN=...
#      bash experiments/targeted_tool_data_factory/runpod_bundle_pilot3/run_train_nestful500_4gpu.sh
#      bash .../run_train_nestful500_4gpu.sh --train-n 300
#      # re-eval an existing checkpoint only:
#      bash .../run_train_nestful500_4gpu.sh --train-n 300 --stage eval
#
#  Flags: --dry-run | --resume | --stage train|eval | --train-n N
#  Env:   EVAL_GPUS=0,1,2,3   (default)
# =============================================================================
set -Eeuo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FACTORY="$(cd "$BUNDLE/.." && pwd)"
P2="$(cd "$BUNDLE/../runpod_bundle_pilot2" && pwd)"
EXPERIMENTS="$(cd "$FACTORY/.." && pwd)"
V3="$EXPERIMENTS/nestful_synthetic_curriculum_v3"
PARTIAL="$EXPERIMENTS/nestful_mtgrpo_partial"
REPO="$(cd "$EXPERIMENTS/.." && pwd)"
PY="${PYTHON:-python3}"

if command -v cygpath >/dev/null 2>&1; then
  BUNDLE="$(cygpath -m "$BUNDLE")"
  FACTORY="$(cygpath -m "$FACTORY")"
  P2="$(cygpath -m "$P2")"
  EXPERIMENTS="$(cygpath -m "$EXPERIMENTS")"
  V3="$(cygpath -m "$V3")"
  PARTIAL="$(cygpath -m "$PARTIAL")"
  REPO="$(cygpath -m "$REPO")"
fi

DRY_RUN=0
RESUME_FLAG=""
ONLY_STAGE=""
TRAIN_N=600
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)  DRY_RUN=1; shift ;;
    --resume)   RESUME_FLAG="--resume"; shift ;;
    --stage)    ONLY_STAGE="$2"; shift 2 ;;
    --train-n)  TRAIN_N="$2"; shift 2 ;;
    *) echo "[p3train] unknown arg: $1" >&2; exit 1 ;;
  esac
done

SEED="${SEED:-20260727}"
REWARD_ARM="${REWARD_ARM:-A4_GATED_VERIFIABLE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$V3/outputs/runs}"
RESULTS="${RESULTS:-$FACTORY/outputs/runpod_pilot3/train_nestful500}"
RUN_ID="${RUN_ID:-pilot3_D1_seed${SEED}_n${TRAIN_N}}"
TRAIN_FULL="$BUNDLE/data/train_grpo_pilot3.jsonl"
DIAG="$P2/data/nestful_diagnostic_500.jsonl"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"

export USE_VLLM="${USE_VLLM:-1}"
export ROLLOUT_DP_GPUS="${ROLLOUT_DP_GPUS:-1,2,3}"
export CANARY_TRAJ_LOG=1
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export SYNTHETIC_TOOLS_DIR="$FACTORY/trainer_adapter"

mkdir -p "$RESULTS"
LOG="$RESULTS/run_train_nestful500_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

banner() {
  echo ""
  echo "=================================================================="
  echo "[p3train] $1"
  echo "=================================================================="
}
run() {
  echo "+ $*"
  if [ "$DRY_RUN" = "0" ]; then "$@"; fi
}
want() { [ -z "$ONLY_STAGE" ] || [ "$ONLY_STAGE" = "$1" ]; }

cd "$REPO"

echo "[p3train] run_id:  $RUN_ID"
echo "[p3train] train_n: $TRAIN_N  (full frozen train file = 600)"
echo "[p3train] reward:  $REWARD_ARM"
echo "[p3train] eval:    NESTFUL-500 sharded on GPUs $EVAL_GPUS (no C0)"

banner "1/4  GPU check"
DRY_RUN="$DRY_RUN" "$PY" - <<'PYEOF'
import os, sys
try:
    import torch
except ImportError:
    sys.exit(0 if os.environ.get("DRY_RUN") == "1" else 1)
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name} {p.total_memory/1e9:.1f} GB")
if n < 4 and os.environ.get("DRY_RUN") != "1":
    raise SystemExit("[p3train] ABORT: 4 GPUs required")
print(f"[p3train] visible GPUs: {n}")
PYEOF

banner "2/4  hashes + inputs"
# Fail-fast on frozen artefact drift. For a pure --stage eval re-run you may
# set SKIP_HASH=1 if only the eval helper changed and train data must stay as
# the original freeze used for the checkpoint.
if [ "${SKIP_HASH:-0}" != "1" ]; then
  if ! "$PY" "$BUNDLE/verify_hashes.py" --manifest "$BUNDLE/MANIFEST.sha256.json"; then
    echo "[p3train] ABORT: hash mismatch — sync runpod_bundle_pilot3/ from the" >&2
    echo "         machine that ran build_bundle.py (README/scripts were updated)." >&2
    echo "         For eval-only on an already-trained ckpt: SKIP_HASH=1 ... --stage eval" >&2
    exit 1
  fi
else
  echo "[p3train] SKIP_HASH=1 — not verifying MANIFEST"
fi
[ -f "$DIAG" ] || { echo "[p3train] ABORT: missing $DIAG" >&2; exit 1; }

TRAIN_SUBSET="$RESULTS/train_subset_${TRAIN_N}.jsonl"
if want train; then
[ -f "$TRAIN_FULL" ] || { echo "[p3train] ABORT: missing $TRAIN_FULL" >&2; exit 1; }
"$PY" - "$TRAIN_FULL" "$TRAIN_SUBSET" "$TRAIN_N" <<'PYEOF'
import sys
from pathlib import Path
src, dst, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
if not (1 <= n <= len(rows)):
    raise SystemExit(f"[p3train] ABORT: --train-n {n} out of range 1..{len(rows)}")
dst.write_text("\n".join(rows[:n]) + "\n", encoding="utf-8")
print(f"[p3train] wrote {n}/{len(rows)} -> {dst}")
PYEOF

run "$PY" "$FACTORY/trainer_adapter/preflight_gold_replay.py" \
  --data "$TRAIN_SUBSET" --expect "$TRAIN_N" \
  --report "$RESULTS/preflight_gold_replay.json"

banner "3/4  GRPO train (GPU0 learner / GPU1-3 rollouts; skip built-in eval)"
run "$PY" "$V3/scripts/ablation/run_reward_ablation.py" \
  --round 3 \
  --reward-arm "$REWARD_ARM" \
  --seed "$SEED" \
  --train-subset "$TRAIN_SUBSET" \
  --expected-rows "$TRAIN_N" \
  --skip-eval --skip-c0-eval \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID" \
  --wandb-project "${WANDB_PROJECT:-ttdf-pilot3}" \
  --wandb-group "pilot3_train_nestful500" \
  $RESUME_FLAG
else
  echo "[p3train] skip train subset / gold replay / GRPO (--stage eval)"
fi

if want eval; then
banner "4/4  eval new checkpoint on NESTFUL-500 (4-GPU sharded, no C0)"
# If a previous single-GPU eval is still writing into this dir, use a fresh stamp
# only when EVAL_OUT is unset; default path stays stable for the report.
OUT="${EVAL_OUT:-$RESULTS/eval/D1_nestful500}"
mkdir -p "$OUT"
# Stop a leftover single-GPU eval on this same out dir if the user re-runs
# --stage eval while an old job is still alive (best-effort, non-fatal).
if [ "${KILL_STALE_EVAL:-1}" = "1" ] && [ "$DRY_RUN" = "0" ]; then
  pkill -f "paths.full_nestful_jsonl=.*nestful_diagnostic_500" 2>/dev/null || true
  pkill -f "eval_nestful500_sharded.py" 2>/dev/null || true
  sleep 2
fi
run "$PY" "$BUNDLE/eval_nestful500_sharded.py" \
  --run-dir "$OUTPUT_ROOT/$RUN_ID" \
  --diagnostic "$DIAG" \
  --out-dir "$OUT" \
  --run-py "$V3/run.py" \
  --config "$PARTIAL/config.yaml" \
  --gpus "$EVAL_GPUS" \
  $([ "$DRY_RUN" = "1" ] && echo --dry-run)
fi

banner "done"
echo "[p3train] ckpt:    $OUTPUT_ROOT/$RUN_ID"
echo "[p3train] results: ${EVAL_OUT:-$RESULTS/eval/D1_nestful500}"
echo "[p3train] C0 was NOT evaluated — compare to your previous C0 nestful500 numbers"
