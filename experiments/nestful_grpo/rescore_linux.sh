#!/usr/bin/env bash
# Re-score ALL curriculum trajectories with official NESTFUL metrics INCLUDING Win Rate.
# Requires Linux (SIGALRM in IBM scorer). No GPU. ~2 min total.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python curricullum/evaluation/rescore_official.py \
  --reparse \
  --out experiments/nestful_grpo/rescored_official.json
python experiments/nestful_grpo/consolidate.py
echo "Done. See experiments/nestful_grpo/curriculum_official_metrics.csv"
