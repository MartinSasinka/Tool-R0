#!/usr/bin/env bash
# FULL agentic generation (mirrors deterministic v4 per-stage counts; real
# API cost). Hard-requires CONFIRM_FULL_AGENTIC_GENERATION=1.
# Requires OPENROUTER_API_KEY in the environment (NEVER printed).
if grep -q $'\r' "$0" 2>/dev/null; then exec /bin/bash <(sed 's/\r$//' "$0") "$@"; fi
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
V3="$REPO/experiments/nestful_synthetic_curriculum_v3"
PYTHON="${PYTHON:-python}"

if [ "${CONFIRM_FULL_AGENTIC_GENERATION:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  echo "[full] ABORT: full generation costs real money and takes hours." >&2
  echo "       Set CONFIRM_FULL_AGENTIC_GENERATION=1 to proceed (run the pilot" >&2
  echo "       + scoring + stage probe FIRST — see docs/AGENTIC_DATA_GENERATION.md)." >&2
  exit 3
fi

export OPENROUTER_MAX_REQUESTS="${OPENROUTER_MAX_REQUESTS:-60000}"
export OPENROUTER_MAX_SPEND_USD="${OPENROUTER_MAX_SPEND_USD:-20}"
export OPENROUTER_MAX_ACCEPTED_PER_STAGE="${OPENROUTER_MAX_ACCEPTED_PER_STAGE:-800}"
export OPENROUTER_CACHE="${OPENROUTER_CACHE:-1}"
export OPENROUTER_SAVE_RAW="${OPENROUTER_SAVE_RAW:-1}"
export CONFIRM_FULL_AGENTIC_GENERATION="${CONFIRM_FULL_AGENTIC_GENERATION:-0}"
OUT_DIR="${OUT_DIR:-$V3/data/curriculum_v4_nestful_like_agentic_openrouter}"
DRY_RUN="${DRY_RUN:-0}"

echo "[full] repo             = $REPO"
echo "[full] output_dir       = $OUT_DIR"
echo "[full] max_requests     = $OPENROUTER_MAX_REQUESTS  max_spend_usd = $OPENROUTER_MAX_SPEND_USD"
echo "[full] accepted cap     = $OPENROUTER_MAX_ACCEPTED_PER_STAGE / stage"
echo "[full] challenger       = ${OPENROUTER_CHALLENGER_MODEL:-deepseek/deepseek-chat}"
echo "[full] weak/strong      = ${OPENROUTER_WEAK_MODEL:-deepseek/deepseek-chat} / ${OPENROUTER_STRONG_MODEL:-deepseek/deepseek-chat}"
echo "[full] judge            = ${OPENROUTER_JUDGE_MODEL:-deepseek/deepseek-chat}"
if [ -n "${OPENROUTER_API_KEY:-}" ]; then echo "[full] api_key          = set (redacted)"; else echo "[full] api_key          = NOT SET"; fi

ARGS=(--output-dir "$OUT_DIR")
[ "$DRY_RUN" = "1" ] && ARGS+=(--dry-run)

cd "$REPO"
"$PYTHON" "$V3/scripts/data/build_curriculum_v4_agentic_openrouter.py" "${ARGS[@]}"

if [ "$DRY_RUN" != "1" ]; then
  echo "[full] scoring the dataset ..."
  "$PYTHON" "$V3/scripts/data/score_dataset_quality.py" --dataset-dir "$OUT_DIR"
fi
echo "[full] done. No training was launched."
