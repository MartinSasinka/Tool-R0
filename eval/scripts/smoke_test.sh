#!/bin/bash
# Quick smoke test – runs BFCL with the dummy backend.
# No GPU, no API keys needed.
#
# Usage:
#   bash eval/scripts/smoke_test.sh

set -euo pipefail

echo "============================================================"
echo "  Tool-R0 Eval Smoke Test"
echo "============================================================"

echo ""
echo "--- BFCL smoke test ---"
python -m eval.run_eval \
    --benchmark bfcl \
    --config eval/configs/smoke_test.yaml \
    --output-dir eval/results/smoke_test/bfcl \
    --profile-name smoke \
    --dry-run

echo ""
echo "Smoke test complete. Check eval/results/smoke_test/"
