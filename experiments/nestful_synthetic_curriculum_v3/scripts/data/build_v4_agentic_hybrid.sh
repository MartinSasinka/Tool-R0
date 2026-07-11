#!/usr/bin/env bash
# Hybrid agentic generation: local Qwen3-4B weak solver + OpenRouter for other roles.
# Run from repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

export WEAK_SOLVER_BACKEND="${WEAK_SOLVER_BACKEND:-local}"
export LOCAL_WEAK_MODEL="${LOCAL_WEAK_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
export LOCAL_WEAK_4BIT="${LOCAL_WEAK_4BIT:-1}"

export OPENROUTER_CHALLENGER_MODEL="${OPENROUTER_CHALLENGER_MODEL:-deepseek/deepseek-v3.2}"
export OPENROUTER_STRONG_MODEL="${OPENROUTER_STRONG_MODEL:-qwen/qwen3-235b-a22b-2507}"
export OPENROUTER_JUDGE_MODEL="${OPENROUTER_JUDGE_MODEL:-deepseek/deepseek-v3.2}"

export WEAK_SOLVER_MODE="${WEAK_SOLVER_MODE:-minimal}"
export STRONG_SOLVER_MODE="${STRONG_SOLVER_MODE:-scaffolded}"
export STRONG_PASS_POLICY="${STRONG_PASS_POLICY:-exact_win}"

export OPENROUTER_MAX_ITERATIONS_PER_STAGE="${OPENROUTER_MAX_ITERATIONS_PER_STAGE:-8000}"
export OPENROUTER_MAX_REQUESTS="${OPENROUTER_MAX_REQUESTS:-50000}"
export OPENROUTER_MAX_SPEND_USD="${OPENROUTER_MAX_SPEND_USD:-35}"
export OPENROUTER_CACHE="${OPENROUTER_CACHE:-1}"
export OPENROUTER_SAVE_RAW="${OPENROUTER_SAVE_RAW:-1}"

STAGES="${STAGES:-stage2_2call_agentic_openrouter}"
SEED="${SEED:-44}"
RESUME="${RESUME:-0}"
PILOT="${PILOT:-0}"
DRY_RUN="${DRY_RUN:-0}"

echo "[hybrid] repo=$REPO_ROOT"
echo "[hybrid] weak_solver=LOCAL $LOCAL_WEAK_MODEL (4bit=$LOCAL_WEAK_4BIT)"
echo "[hybrid] challenger=$OPENROUTER_CHALLENGER_MODEL"
echo "[hybrid] strong_solver=$OPENROUTER_STRONG_MODEL"
echo "[hybrid] judge=$OPENROUTER_JUDGE_MODEL"
echo "[hybrid] verifier=deterministic executor (no LLM)"
echo "[hybrid] stages=$STAGES seed=$SEED resume=$RESUME pilot=$PILOT"

ARGS=(--stages "$STAGES" --seed "$SEED")
if [[ "$RESUME" == "1" ]]; then ARGS+=(--resume); fi
if [[ "$PILOT" == "1" ]]; then ARGS+=(--pilot); fi
if [[ "$DRY_RUN" == "1" ]]; then ARGS+=(--dry-run); fi

if [[ -z "${OPENROUTER_API_KEY:-}" && "$DRY_RUN" != "1" ]]; then
  echo "[hybrid] ERROR: OPENROUTER_API_KEY required for challenger/strong/judge" >&2
  exit 1
fi

if [[ "$PILOT" != "1" && "${CONFIRM_FULL_AGENTIC_GENERATION:-}" != "1" ]]; then
  echo "[hybrid] Set CONFIRM_FULL_AGENTIC_GENERATION=1 for full run (or PILOT=1)" >&2
  exit 1
fi

python experiments/nestful_synthetic_curriculum_v3/scripts/data/build_curriculum_v4_agentic_openrouter.py "${ARGS[@]}"
