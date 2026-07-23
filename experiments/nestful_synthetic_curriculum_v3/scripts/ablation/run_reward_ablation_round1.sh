#!/usr/bin/env bash
# Reward ablation — Round 1 RunPod launcher (reports/reward_ablation/ABLATION_PLAN.md §16).
#
# Runs one or all five reward arms SEQUENTIALLY (never in parallel on the
# same GPUs) via scripts/ablation/run_reward_ablation.py, which drives the
# EXISTING TwoPhaseTrainSession trainer / vLLM DP rollout pool / synthetic
# executor — this launcher adds no new training code, only orchestration,
# environment checks, and logging.
#
# GPU topology (1 pod, 4 GPUs): GPU0=learner, GPU1-3=rollout workers. Eval
# runs only after the training session closes (learner+optimizer released) —
# enforced inside run_reward_ablation.py, not here.
#
# Usage:
#   # all 5 arms, sequentially, fresh run:
#   bash experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation_round1.sh
#
#   # one arm only:
#   bash experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation_round1.sh \
#     --arm A2_R3_OUTCOME_FIRST --seed 20260724
#
#   # resume an interrupted arm:
#   bash experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation_round1.sh \
#     --arm A2_R3_OUTCOME_FIRST --seed 20260724 --resume
#
#   # smoke test (8 tasks x 8 rollouts, 20 eval tasks) before committing GPU time:
#   bash experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation_round1.sh --smoke
#
# Env overrides: SEED, WANDB_PROJECT, WANDB_GROUP, OUTPUT_ROOT, TRAIN_SUBSET,
# EVAL_SUBSET, USE_VLLM, ROLLOUT_DP_GPUS, EVAL_TP, VLLM_GPU_UTIL,
# CUDA_VISIBLE_DEVICES, HF_TOKEN / HUGGING_FACE_HUB_TOKEN, WANDB_API_KEY.
set -Eeuo pipefail

# ── locate repo / experiment roots (works from any cwd) ────────────────────
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V3="$(cd "$_HERE/../.." && pwd)"
REPO="$(cd "$V3/../.." && pwd)"
MINIMAL="$(cd "$V3/../nestful_mtgrpo_minimal" && pwd)"
PY="${PYTHON:-python3}"

banner() {
  echo "──────────────────────────────────────────────────────────────"
  echo "[reward-ablation] $1"
  echo "──────────────────────────────────────────────────────────────"
}
require_file() {
  if [ ! -f "$1" ]; then
    echo "[reward-ablation] ERROR: $2 not found: $1" >&2
    exit 1
  fi
}

ALL_ARMS=(A0_R0_CURRENT A1_OUTCOME_ONLY A2_R3_OUTCOME_FIRST A3_VERIFIABLE_PROCESS A4_GATED_VERIFIABLE)

ARM=""
SMOKE=0
RESUME=0
ROUND=1
while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --round) ROUND="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --resume) RESUME=1; shift ;;
    *) echo "[reward-ablation] ERROR: unknown arg $1" >&2; exit 1 ;;
  esac
done

SEED="${SEED:-20260724}"
WANDB_PROJECT="${WANDB_PROJECT:-nestful-reward-ablation}"
WANDB_GROUP="${WANDB_GROUP:-reward_ablation_round${ROUND}_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$V3/outputs/runs}"
TRAIN_SUBSET="${TRAIN_SUBSET:-$V3/reports/reward_ablation/data/train_subset_160.jsonl}"
EVAL_SUBSET="${EVAL_SUBSET:-$V3/reports/reward_ablation/data/nestful_diagnostic_500_ids.json}"

# ── 1. repo root sanity ─────────────────────────────────────────────────────
banner "repo root check"
[ -d "$V3" ] || { echo "[reward-ablation] ERROR: $V3 not found — run from inside Tool-R0" >&2; exit 1; }
[ -d "$REPO/.git" ] || echo "[reward-ablation] WARNING: $REPO has no .git — not running from a repo checkout?" >&2
echo "[reward-ablation] REPO=$REPO"
echo "[reward-ablation] V3=$V3"

# ── 2. GPU check ─────────────────────────────────────────────────────────────
banner "GPU check"
command -v "$PY" >/dev/null || { echo "[reward-ablation] ERROR: python missing" >&2; exit 1; }
if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "[reward-ablation] ERROR: CUDA not available" >&2
  exit 1
fi
N_GPU=$("$PY" -c "import torch; print(torch.cuda.device_count())")
if [ "${N_GPU:-0}" -lt 4 ] && [ "${ALLOW_FEWER_GPUS:-0}" != "1" ]; then
  echo "[reward-ablation] ERROR: need 4 GPUs, found $N_GPU (set ALLOW_FEWER_GPUS=1 to override)" >&2
  exit 1
fi
echo "[reward-ablation] N_GPU=$N_GPU"

# ── 3. dataset / freeze hash checks ─────────────────────────────────────────
banner "dataset + frozen-reward-spec checks"
require_file "$TRAIN_SUBSET" "TRAIN_SUBSET (run prepare_train_subset_160.py first)"
require_file "$EVAL_SUBSET" "EVAL_SUBSET (run prepare_nestful_diagnostic_500.py first)"
FROZEN="$V3/reports/reward_ablation/FROZEN_REWARD_SPECS.json"
require_file "$FROZEN" "FROZEN_REWARD_SPECS.json (run scripts/ablation/freeze_reward_specs.py --backend vllm first)"
IS_REAL=$("$PY" -c "import json; print(json.load(open(r'$FROZEN',encoding='utf-8'))['probe']['is_real_calibration'])")
if [ "$IS_REAL" != "True" ] && [ "${ALLOW_STUB_FREEZE:-0}" != "1" ]; then
  echo "[reward-ablation] ERROR: FROZEN_REWARD_SPECS.json was frozen with backend=stub (CPU self-test)," >&2
  echo "[reward-ablation]        not a real GPU probe. Re-run freeze_reward_specs.py --backend vllm on this pod," >&2
  echo "[reward-ablation]        or set ALLOW_STUB_FREEZE=1 to proceed anyway (NOT recommended for real Round 1)." >&2
  exit 1
fi
TRAIN_HASH=$("$PY" -c "
import hashlib
h=hashlib.sha256()
with open(r'$TRAIN_SUBSET','rb') as fh:
    for c in iter(lambda: fh.read(1<<20), b''):
        h.update(c)
print(h.hexdigest())
")
echo "[reward-ablation] train_subset_160.jsonl sha256=$TRAIN_HASH"

# ── 4. C0 checkpoint reachability (base model, not a local file — just assert model id resolves) ──
banner "base model / C0 check"
BASE_MODEL=$("$PY" -c "import yaml; print(yaml.safe_load(open(r'$V3/configs/reward_ablation/round1_base.yaml',encoding='utf-8'))['model']['base_model'])")
echo "[reward-ablation] base_model=$BASE_MODEL"

# ── 5. W&B check ─────────────────────────────────────────────────────────────
banner "W&B check"
if [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "disabled" ] && [ "${WANDB_MODE:-}" != "offline" ]; then
  echo "[reward-ablation] WARNING: WANDB_API_KEY unset (runs will fail to log unless WANDB_MODE=offline/disabled)" >&2
fi
echo "[reward-ablation] WANDB_PROJECT=$WANDB_PROJECT"
echo "[reward-ablation] WANDB_GROUP=$WANDB_GROUP"
# NEVER print WANDB_API_KEY / HF_TOKEN values themselves.

# ── 6. disk space check ──────────────────────────────────────────────────────
banner "disk space check"
FREE_KB=$(df -Pk "$OUTPUT_ROOT" 2>/dev/null | awk 'NR==2{print $4}' || df -Pk "$V3" | awk 'NR==2{print $4}')
if [ "${FREE_KB:-0}" -lt 10000000 ]; then
  echo "[reward-ablation] WARNING: low free disk (${FREE_KB} KB) — each arm's checkpoint + eval can need several GB" >&2
fi
echo "[reward-ablation] free_kb=$FREE_KB"

mkdir -p "$OUTPUT_ROOT"

export SEED
export DATA_SEED="$SEED"
export ROLLOUT_SEED="$SEED"
export WANDB_PROJECT
export WANDB_RUN_GROUP="$WANDB_GROUP"
export USE_VLLM="${USE_VLLM:-1}"
export ROLLOUT_DP_GPUS="${ROLLOUT_DP_GPUS:-1,2,3}"
export EVAL_TP="${EVAL_TP:-4}"
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

ARMS_TO_RUN=("${ALL_ARMS[@]}")
if [ -n "$ARM" ]; then
  ARMS_TO_RUN=("$ARM")
fi

_cleanup() {
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[reward-ablation] ERROR: aborted (exit $rc) — see per-arm logs under $OUTPUT_ROOT/*/logs/" >&2
  fi
}
trap _cleanup EXIT

RESULTS=()
for arm in "${ARMS_TO_RUN[@]}"; do
  banner "arm=$arm seed=$SEED round=$ROUND smoke=$SMOKE resume=$RESUME"
  RUN_ID="reward_ablation_r${ROUND}_${arm}_seed${SEED}"
  RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
  mkdir -p "$RUN_DIR/logs"
  LOG="$RUN_DIR/logs/console.log"

  ARGS=("$V3/scripts/ablation/run_reward_ablation.py"
    --round "$ROUND" --reward-arm "$arm" --seed "$SEED"
    --train-subset "$TRAIN_SUBSET" --eval-subset "$EVAL_SUBSET"
    --run-id "$RUN_ID" --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP"
    --output-root "$OUTPUT_ROOT")
  # NOTE: deliberately `if`-guarded, not `[ ... ] && ARGS+=(...)` — under
  # `set -e`, a bare `test && cmd` list whose test is FALSE (the common
  # case here) would exit the whole script, since the list's exit status
  # would be the test's non-zero status.
  if [ "$SMOKE" = "1" ]; then
    ARGS+=(--smoke)
  fi
  if [ "$RESUME" = "1" ]; then
    ARGS+=(--resume)
  fi

  set +e
  "$PY" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
  RC=${PIPESTATUS[0]}
  set -e

  if [ "$RC" -eq 0 ] && [ -f "$RUN_DIR/SUCCESS" ]; then
    echo "[reward-ablation] arm=$arm -> SUCCESS ($RUN_DIR)"
    RESULTS+=("$arm=SUCCESS")
  else
    echo "[reward-ablation] arm=$arm -> FAILED (rc=$RC); see $LOG" >&2
    RESULTS+=("$arm=FAILED")
    if [ "${STOP_ON_FAILURE:-1}" = "1" ]; then
      banner "SUMMARY (stopped early)"
      printf '  %s\n' "${RESULTS[@]}"
      exit 1
    fi
  fi
done

banner "Round $ROUND summary"
printf '  %s\n' "${RESULTS[@]}"
echo "[reward-ablation] all requested arms finished — see $OUTPUT_ROOT/reward_ablation_r${ROUND}_*/  "
echo "[reward-ablation] next: python $V3/scripts/ablation/summarize_reward_ablation.py round-summary --round $ROUND ..."
