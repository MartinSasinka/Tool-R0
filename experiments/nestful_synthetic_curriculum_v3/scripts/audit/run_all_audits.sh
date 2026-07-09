#!/usr/bin/env bash
# Re-run the audit extractors (P0 remediation). Read-only w.r.t. code/datasets;
# regenerates the machine-generated audit artifacts under audits/:
#   DATASET_AUDIT.json  (dataset_audit.py)
#   RUN_AUDIT.json/.csv (run_audit.py)
#   failure-mode summary to stdout (failure_summary.py)
# The .md audit reports are hand-written analysis and are NOT overwritten.
#
# Usage (from repo root):
#   bash experiments/nestful_synthetic_curriculum_v3/scripts/audit/run_all_audits.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
TOOLS="experiments/nestful_synthetic_curriculum_v3/audits/tools"
AUDITS="experiments/nestful_synthetic_curriculum_v3/audits"

echo "=== run_all_audits ==="
echo "repo root : $REPO_ROOT"
echo "tools     : $TOOLS"
echo "outputs   : $AUDITS"
echo

echo "--- 1/3 dataset audit ---"
python "$TOOLS/dataset_audit.py"
echo

echo "--- 2/3 run audit ---"
python "$TOOLS/run_audit.py"
echo

echo "--- 3/3 failure-mode summary ---"
python "$TOOLS/failure_summary.py"
echo

echo "=== run_all_audits done — regenerated JSON/CSV under $AUDITS ==="
