#!/usr/bin/env bash
# Score the agentic dataset (validity / contamination / distribution /
# solver gap / GRPO signal). Read-only; no API calls, no training.
if grep -q $'\r' "$0" 2>/dev/null; then exec /bin/bash <(sed 's/\r$//' "$0") "$@"; fi
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
V3="$REPO/experiments/nestful_synthetic_curriculum_v3"
PYTHON="${PYTHON:-python}"
DATASET_DIR="${DATASET_DIR:-$V3/data/curriculum_v4_nestful_like_agentic_openrouter}"

echo "[score] repo        = $REPO"
echo "[score] dataset_dir = $DATASET_DIR"

cd "$REPO"
"$PYTHON" "$V3/scripts/data/score_dataset_quality.py" --dataset-dir "$DATASET_DIR" "$@"
