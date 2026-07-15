#!/usr/bin/env bash
# 2/9 — Generate a SMALL curriculum-v5 dataset pilot (40 rows/stage).
#
# Every row is replayed through the REAL trainer executor before it is
# written; a dataset manifest (registry version + hash, per-file sha256)
# lands in $OUTPUT_DIR/manifests/.
#
# Env:  PYTHON=python3
#       OUTPUT_DIR=<v3>/data/curriculum_v5_registry_pilot
#       SEED=42
#       MAX_TOOL_SHARE=0.08    diversity threshold (max share of one tool)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

OUTPUT_DIR="${OUTPUT_DIR:-$V3/data/curriculum_v5_registry_pilot}"
SEED="${SEED:-42}"
MAX_TOOL_SHARE="${MAX_TOOL_SHARE:-0.08}"

banner "dataset pilot generation"
print_env PYTHON OUTPUT_DIR SEED MAX_TOOL_SHARE

cd "$V3"
"$PY" scripts/data/build_v5_dataset.py \
  --pilot \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR" \
  --max-tool-share "$MAX_TOOL_SHARE"

banner "pilot written to $OUTPUT_DIR"
