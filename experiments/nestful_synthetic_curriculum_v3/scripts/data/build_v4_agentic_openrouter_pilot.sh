#!/usr/bin/env bash
# Tiny agentic pilot (default 10 accepted/stage) via OpenRouter.
# Requires OPENROUTER_API_KEY in the environment (NEVER printed, never hardcoded).
#
# Usage (repo root):
#   MAX_ACCEPTED_PER_STAGE=10 OPENROUTER_MAX_REQUESTS=200 \
#     bash experiments/nestful_synthetic_curriculum_v3/scripts/data/build_v4_agentic_openrouter_pilot.sh
#   DRY_RUN=1  ... (print plan only)   MOCK=1 ... (offline mock, no API cost)
if grep -q $'\r' "$0" 2>/dev/null; then exec /bin/bash <(sed 's/\r$//' "$0") "$@"; fi
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
V3="$REPO/experiments/nestful_synthetic_curriculum_v3"
PYTHON="${PYTHON:-python}"

MAX_ACCEPTED_PER_STAGE="${MAX_ACCEPTED_PER_STAGE:-10}"
export OPENROUTER_MAX_REQUESTS="${OPENROUTER_MAX_REQUESTS:-200}"
export OPENROUTER_MAX_SPEND_USD="${OPENROUTER_MAX_SPEND_USD:-5}"
export OPENROUTER_CACHE="${OPENROUTER_CACHE:-1}"
export OPENROUTER_SAVE_RAW="${OPENROUTER_SAVE_RAW:-1}"
OUT_DIR="${OUT_DIR:-$V3/data/curriculum_v4_nestful_like_agentic_openrouter}"
STAGES="${STAGES:-stage2_2call_agentic_openrouter stage3_3call_agentic_openrouter stage4_4to6call_agentic_openrouter}"
DRY_RUN="${DRY_RUN:-0}"
MOCK="${MOCK:-0}"

echo "[pilot] repo            = $REPO"
echo "[pilot] output_dir      = $OUT_DIR"
echo "[pilot] stages          = $STAGES"
echo "[pilot] accepted/stage  = $MAX_ACCEPTED_PER_STAGE"
echo "[pilot] max_requests    = $OPENROUTER_MAX_REQUESTS  max_spend_usd = $OPENROUTER_MAX_SPEND_USD"
echo "[pilot] challenger      = ${OPENROUTER_CHALLENGER_MODEL:-deepseek/deepseek-chat}"
echo "[pilot] weak/strong     = ${OPENROUTER_WEAK_MODEL:-deepseek/deepseek-chat} / ${OPENROUTER_STRONG_MODEL:-deepseek/deepseek-chat}"
echo "[pilot] judge           = ${OPENROUTER_JUDGE_MODEL:-deepseek/deepseek-chat}"
if [ -n "${OPENROUTER_API_KEY:-}" ]; then echo "[pilot] api_key         = set (redacted)"; else echo "[pilot] api_key         = NOT SET"; fi

ARGS=(--pilot --max-accepted-per-stage "$MAX_ACCEPTED_PER_STAGE" --output-dir "$OUT_DIR")
# shellcheck disable=SC2086
ARGS+=(--stages $STAGES)
[ "$DRY_RUN" = "1" ] && ARGS+=(--dry-run)
[ "$MOCK" = "1" ] && ARGS+=(--mock)

cd "$REPO"
"$PYTHON" "$V3/scripts/data/build_curriculum_v4_agentic_openrouter.py" "${ARGS[@]}"

if [ "$DRY_RUN" != "1" ]; then
  echo "[pilot] scoring the pilot dataset ..."
  SCORE_DIR="$OUT_DIR"
  [ "$MOCK" = "1" ] && SCORE_DIR="${OUT_DIR}_mock"
  "$PYTHON" "$V3/scripts/data/score_dataset_quality.py" --dataset-dir "$SCORE_DIR"
fi
echo "[pilot] done. No training was launched."
