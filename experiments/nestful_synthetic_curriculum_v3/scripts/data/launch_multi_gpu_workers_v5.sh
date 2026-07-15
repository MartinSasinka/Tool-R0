#!/usr/bin/env bash
# Launch N parallel v5 agentic-data-generation workers, one per GPU, each in
# its own tmux session, then (after they all finish) merge the results with
# merge_agentic_workers_v5.py.
#
# v5 counterpart of launch_multi_gpu_workers.sh: same tmux-per-GPU design,
# but drives build_curriculum_v5_agentic_openrouter.py — the versioned
# executable v5 tool registry (lib/synthetic_tools.py, ~163 tools), REAL
# executor.mode=synthetic execution everywhere (challenger verify, weak/
# strong solvers, rollout probe — never gold_replay), and best-of-N candidate
# selection per challenger batch. Only CUDA_VISIBLE_DEVICES, --seed and
# --output-dir differ per worker, so hard-trace validation / semantic checks
# / GRPO-signal probe / diversity caps / best-of-N run identically to a
# single-GPU run, just on 1/N of the target and on a dedicated GPU.
#
# Run on a multi-GPU box (e.g. a RunPod pod with 4 GPUs). Requires
# OPENROUTER_API_KEY (challenger/strong/judge calls); the weak solver runs
# LOCALLY on each worker's own GPU (WEAK_SOLVER_BACKEND=local by default) —
# this is the exact target Qwen3-4B checkpoint, needed for the rollout
# GRPO-signal probe to be meaningful.
#
# Usage (repo root, inside the activated agentic venv):
#   bash experiments/nestful_synthetic_curriculum_v3/scripts/data/launch_multi_gpu_workers_v5.sh
#
# Env overrides (all optional):
#   NUM_GPUS=4                          workers to launch (default: nvidia-smi count, capped 4)
#   STAGES="stage2_2call_agentic_openrouter stage3_3call_agentic_openrouter stage4_4to6call_agentic_openrouter"
#   TOTAL_PER_STAGE=60                  TOTAL accepted rows wanted PER STAGE, summed over ALL workers
#   BASE_SEED=45                        worker i uses seed BASE_SEED+i
#   OUT_BASE=<v3_root>/data/agentic_v5_workers   worker i writes to $OUT_BASE/gpu$i
#   TOTAL_SPEND_USD=20                  OpenRouter budget, split evenly across workers
#   TOTAL_REQUESTS=4000                 OpenRouter request cap, split evenly across workers
#   BEST_OF_N_ENABLED=1 BEST_OF_N_MAX_ACCEPTS_PER_BATCH=1   candidate-selection knobs
#   LOCAL_WEAK_MODEL / LOCAL_WEAK_4BIT / OPENROUTER_CHALLENGER_MODEL / etc.
#   Defaults mirror lib/agentic_data/env_defaults.py — override via env.
#   DRY_RUN=1                           print the per-worker commands, launch nothing
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
V3_ROOT="$REPO_ROOT/experiments/nestful_synthetic_curriculum_v3"
cd "$REPO_ROOT"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[launch-v5] ERROR: tmux not found. On RunPod: apt-get update && apt-get install -y tmux" >&2
  exit 1
fi

DETECTED_GPUS="$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | wc -l || echo 1)"
NUM_GPUS="${NUM_GPUS:-$DETECTED_GPUS}"
STAGES="${STAGES:-stage2_2call_agentic_openrouter stage3_3call_agentic_openrouter stage4_4to6call_agentic_openrouter}"
TOTAL_PER_STAGE="${TOTAL_PER_STAGE:-60}"
BASE_SEED="${BASE_SEED:-45}"
OUT_BASE="${OUT_BASE:-$V3_ROOT/data/agentic_v5_workers}"
TOTAL_SPEND_USD="${TOTAL_SPEND_USD:-20}"
TOTAL_REQUESTS="${TOTAL_REQUESTS:-4000}"
DRY_RUN="${DRY_RUN:-0}"

PER_WORKER_TARGET=$(( (TOTAL_PER_STAGE + NUM_GPUS - 1) / NUM_GPUS ))
PER_WORKER_SPEND=$(python3 -c "print(round(${TOTAL_SPEND_USD} / ${NUM_GPUS}, 2))")
PER_WORKER_REQUESTS=$(( TOTAL_REQUESTS / NUM_GPUS ))

echo "======================================================================="
echo "[launch-v5] repo            = $REPO_ROOT"
echo "[launch-v5] registry        = lib/synthetic_tools.py (v5, ~163 tools)"
echo "[launch-v5] executor.mode   = synthetic (REAL execution, everywhere)"
echo "[launch-v5] num_gpus        = $NUM_GPUS"
echo "[launch-v5] stages          = $STAGES"
echo "[launch-v5] total/stage     = $TOTAL_PER_STAGE  ->  ${PER_WORKER_TARGET}/worker/stage"
echo "[launch-v5] base_seed       = $BASE_SEED (worker i = BASE_SEED+i)"
echo "[launch-v5] out_base        = $OUT_BASE"
echo "[launch-v5] spend_usd/worker= $PER_WORKER_SPEND  (total $TOTAL_SPEND_USD)"
echo "[launch-v5] requests/worker = $PER_WORKER_REQUESTS  (total $TOTAL_REQUESTS)"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "[launch-v5] ERROR: OPENROUTER_API_KEY not set (export it or put it in .env + source it)" >&2
  exit 1
fi

if [[ "$PER_WORKER_TARGET" -gt 50 ]]; then
  export CONFIRM_FULL_AGENTIC_GENERATION=1
  echo "[launch-v5] per-worker target ($PER_WORKER_TARGET) > 50 -> CONFIRM_FULL_AGENTIC_GENERATION=1 set for all workers"
fi

# Values captured here — tmux spawns a fresh shell that does NOT inherit the
# parent env unless we embed them explicitly in each worker CMD.
# Defaults mirror lib/agentic_data/env_defaults.py — keep in sync.
MAX_ITERS="${OPENROUTER_MAX_ITERATIONS_PER_STAGE:-2000}"
CONFIRM_VAL="${CONFIRM_FULL_AGENTIC_GENERATION:-0}"
AGENTIC_SOLVER_GAP_MODE_VAL="${AGENTIC_SOLVER_GAP_MODE:-multiturn}"
AGENTIC_ROLLOUT_MODE_VAL="${AGENTIC_ROLLOUT_MODE:-multiturn}"
ROLLOUT_N_VAL="${ROLLOUT_N:-8}"
ROLLOUT_TEMPERATURE_VAL="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_MAX_TOKENS_VAL="${ROLLOUT_MAX_TOKENS:-0}"
ROLLOUT_REQUIRE_ACHIEVABLE_WIN_VAL="${ROLLOUT_REQUIRE_ACHIEVABLE_WIN:-0}"
SOLVER_MT_WEAK_TEMPERATURE_VAL="${SOLVER_MT_WEAK_TEMPERATURE:-1.0}"
SOLVER_MT_STRONG_TEMPERATURE_VAL="${SOLVER_MT_STRONG_TEMPERATURE:-0.7}"
MIN_ACCEPT_RATE_VAL="${MIN_ACCEPT_RATE:-0}"
WARMUP_BATCHES_VAL="${WARMUP_BATCHES:-999999}"
RESUME_MIN_ITERATIONS_VAL="${RESUME_MIN_ITERATIONS:-999999}"
LOCAL_WEAK_4BIT_VAL="${LOCAL_WEAK_4BIT:-0}"
LOCAL_WEAK_MODEL_VAL="${LOCAL_WEAK_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
WEAK_SOLVER_BACKEND_VAL="${WEAK_SOLVER_BACKEND:-local}"
DIVERSITY_MAX_SAME_WEAK_SCORE_VAL="${DIVERSITY_MAX_SAME_WEAK_SCORE:-0.55}"
DIVERSITY_MAX_SAME_FAILURE_TYPE_VAL="${DIVERSITY_MAX_SAME_FAILURE_TYPE:-0.55}"
DIVERSITY_ENFORCE_AFTER_VAL="${DIVERSITY_ENFORCE_AFTER:-50}"
CANDIDATES_PER_REQUEST_VAL="${CANDIDATES_PER_REQUEST:-5}"
BEST_OF_N_ENABLED_VAL="${BEST_OF_N_ENABLED:-1}"
BEST_OF_N_MAX_ACCEPTS_PER_BATCH_VAL="${BEST_OF_N_MAX_ACCEPTS_PER_BATCH:-1}"
BEST_OF_N_WEIGHT_GAP_VAL="${BEST_OF_N_WEIGHT_GAP:-0.5}"
BEST_OF_N_WEIGHT_NOVELTY_VAL="${BEST_OF_N_WEIGHT_NOVELTY:-0.3}"
BEST_OF_N_WEIGHT_SIGNAL_VAL="${BEST_OF_N_WEIGHT_SIGNAL:-0.2}"

echo "[launch-v5] rollout        = T=$ROLLOUT_TEMPERATURE_VAL N=$ROLLOUT_N_VAL achievable_win=$ROLLOUT_REQUIRE_ACHIEVABLE_WIN_VAL"
echo "[launch-v5] weak_solver    = $WEAK_SOLVER_BACKEND_VAL $LOCAL_WEAK_MODEL_VAL (4bit=$LOCAL_WEAK_4BIT_VAL) weak_T=$SOLVER_MT_WEAK_TEMPERATURE_VAL"
echo "[launch-v5] diversity      = enforce_after=$DIVERSITY_ENFORCE_AFTER_VAL weak_score_cap=$DIVERSITY_MAX_SAME_WEAK_SCORE_VAL failure_cap=$DIVERSITY_MAX_SAME_FAILURE_TYPE_VAL"
echo "[launch-v5] best_of_n      = enabled=$BEST_OF_N_ENABLED_VAL max_accepts/batch=$BEST_OF_N_MAX_ACCEPTS_PER_BATCH_VAL candidates/batch=$CANDIDATES_PER_REQUEST_VAL weights(gap/novelty/signal)=$BEST_OF_N_WEIGHT_GAP_VAL/$BEST_OF_N_WEIGHT_NOVELTY_VAL/$BEST_OF_N_WEIGHT_SIGNAL_VAL"
echo "[launch-v5] early-stop     = MIN_ACCEPT_RATE=$MIN_ACCEPT_RATE_VAL (0=disabled) WARMUP=$WARMUP_BATCHES_VAL"
echo "======================================================================="

mkdir -p "$OUT_BASE"

for i in $(seq 0 $((NUM_GPUS - 1))); do
  WDIR="$OUT_BASE/gpu$i"
  mkdir -p "$WDIR"
  SEED=$((BASE_SEED + i))
  SESSION="agentic_v5_gpu$i"
  LOG="$OUT_BASE/gpu$i.log"

  CMD=$(cat <<EOF
cd "$REPO_ROOT" && \
source "$V3_ROOT/.venv/bin/activate" && \
export OPENROUTER_API_KEY="$OPENROUTER_API_KEY" && \
export CONFIRM_FULL_AGENTIC_GENERATION="$CONFIRM_VAL" && \
export CUDA_VISIBLE_DEVICES="$i" && \
export WEAK_SOLVER_BACKEND="$WEAK_SOLVER_BACKEND_VAL" && \
export LOCAL_WEAK_DEVICE="cuda:0" && \
export LOCAL_WEAK_MODEL="$LOCAL_WEAK_MODEL_VAL" && \
export LOCAL_WEAK_4BIT="$LOCAL_WEAK_4BIT_VAL" && \
export OPENROUTER_CHALLENGER_MODEL="\${OPENROUTER_CHALLENGER_MODEL:-deepseek/deepseek-v3.2}" && \
export OPENROUTER_STRONG_MODEL="\${OPENROUTER_STRONG_MODEL:-qwen/qwen3-235b-a22b-2507}" && \
export OPENROUTER_JUDGE_MODEL="\${OPENROUTER_JUDGE_MODEL:-deepseek/deepseek-v3.2}" && \
export AGENTIC_REWARD_POLICY="\${AGENTIC_REWARD_POLICY:-execution_aware_v3_2_dense}" && \
export AGENTIC_SOLVER_GAP_MODE="$AGENTIC_SOLVER_GAP_MODE_VAL" && \
export AGENTIC_ROLLOUT_MODE="$AGENTIC_ROLLOUT_MODE_VAL" && \
export ROLLOUT_N="$ROLLOUT_N_VAL" && \
export ROLLOUT_TEMPERATURE="$ROLLOUT_TEMPERATURE_VAL" && \
export ROLLOUT_MAX_TOKENS="$ROLLOUT_MAX_TOKENS_VAL" && \
export ROLLOUT_REQUIRE_ACHIEVABLE_WIN="$ROLLOUT_REQUIRE_ACHIEVABLE_WIN_VAL" && \
export SOLVER_MT_WEAK_TEMPERATURE="$SOLVER_MT_WEAK_TEMPERATURE_VAL" && \
export SOLVER_MT_STRONG_TEMPERATURE="$SOLVER_MT_STRONG_TEMPERATURE_VAL" && \
export MIN_ACCEPT_RATE="$MIN_ACCEPT_RATE_VAL" && \
export WARMUP_BATCHES="$WARMUP_BATCHES_VAL" && \
export RESUME_MIN_ITERATIONS="$RESUME_MIN_ITERATIONS_VAL" && \
export DIVERSITY_MAX_SAME_WEAK_SCORE="$DIVERSITY_MAX_SAME_WEAK_SCORE_VAL" && \
export DIVERSITY_MAX_SAME_FAILURE_TYPE="$DIVERSITY_MAX_SAME_FAILURE_TYPE_VAL" && \
export DIVERSITY_ENFORCE_AFTER="$DIVERSITY_ENFORCE_AFTER_VAL" && \
export CANDIDATES_PER_REQUEST="$CANDIDATES_PER_REQUEST_VAL" && \
export BEST_OF_N_ENABLED="$BEST_OF_N_ENABLED_VAL" && \
export BEST_OF_N_MAX_ACCEPTS_PER_BATCH="$BEST_OF_N_MAX_ACCEPTS_PER_BATCH_VAL" && \
export BEST_OF_N_WEIGHT_GAP="$BEST_OF_N_WEIGHT_GAP_VAL" && \
export BEST_OF_N_WEIGHT_NOVELTY="$BEST_OF_N_WEIGHT_NOVELTY_VAL" && \
export BEST_OF_N_WEIGHT_SIGNAL="$BEST_OF_N_WEIGHT_SIGNAL_VAL" && \
export OPENROUTER_MAX_SPEND_USD="$PER_WORKER_SPEND" && \
export OPENROUTER_MAX_REQUESTS="$PER_WORKER_REQUESTS" && \
export OPENROUTER_MAX_ITERATIONS_PER_STAGE="$MAX_ITERS" && \
export OPENROUTER_CACHE=1 OPENROUTER_SAVE_RAW=1 && \
python experiments/nestful_synthetic_curriculum_v3/scripts/data/build_curriculum_v5_agentic_openrouter.py \
  --stages $STAGES \
  --seed $SEED \
  --output-dir "$WDIR" \
  --max-accepted-per-stage $PER_WORKER_TARGET \
  2>&1 | tee "$LOG"; \
echo "[gpu$i] DONE exit=\$?" >> "$LOG"
EOF
)

  echo "[launch-v5] worker $i: GPU=$i seed=$SEED target/stage=$PER_WORKER_TARGET out=$WDIR"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "---- gpu$i command ----"
    echo "$CMD"
    continue
  fi
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SESSION" "bash -lc '$CMD'"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[launch-v5] DRY_RUN=1 — nothing launched."
  exit 0
fi

echo
echo "[launch-v5] all $NUM_GPUS workers launched in tmux sessions agentic_v5_gpu0..agentic_v5_gpu$((NUM_GPUS - 1))"
echo "[launch-v5] attach with:      tmux attach -t agentic_v5_gpu0"
echo "[launch-v5] detach with:      Ctrl+b then d"
echo "[launch-v5] tail all logs:    tail -f $OUT_BASE/gpu*.log"
echo "[launch-v5] list sessions:    tmux ls"
echo "[launch-v5] check completion: grep -l DONE $OUT_BASE/gpu*.log"
echo
echo "[launch-v5] once ALL workers show 'DONE' in their log, merge with:"
echo "  python experiments/nestful_synthetic_curriculum_v3/scripts/data/merge_agentic_workers_v5.py \\"
echo "    --workers-glob \"$OUT_BASE/gpu*\" \\"
echo "    --output-dir experiments/nestful_synthetic_curriculum_v3/data/curriculum_v5_agentic_synthetic"
