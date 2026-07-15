#!/usr/bin/env bash
# 3/9 — Score dataset quality & diversity.
#
# Reports unique tools, tool frequency/share, family & domain distribution,
# arity/type/output distributions, offered-tool counts, dependency motifs and
# the replay pass rate; fails when configurable thresholds are violated.
#
# Usage: score_dataset.sh <file-or-glob> [more files ...]
# Env:   PYTHON=python3
#        REPORT_OUT=            optional JSON report path
#        MAX_TOOL_SHARE=0.10    fail if one tool exceeds this share
#        MIN_UNIQUE_TOOLS=60    fail below this many distinct tools
#        MIN_REPLAY_PASS=1.0    fail below this replay pass rate
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [ $# -lt 1 ]; then
  echo "usage: $0 <dataset.jsonl or glob> [...]" >&2
  exit 2
fi

banner "dataset quality scoring"
print_env PYTHON REPORT_OUT MAX_TOOL_SHARE MIN_UNIQUE_TOOLS MIN_REPLAY_PASS
echo "[v5] inputs: $*"

cd "$V3"
ARGS=(scripts/data/score_v5_dataset.py "$@"
      --max-tool-share "${MAX_TOOL_SHARE:-0.10}"
      --min-unique-tools "${MIN_UNIQUE_TOOLS:-60}"
      --min-replay-pass "${MIN_REPLAY_PASS:-1.0}")
if [ -n "${REPORT_OUT:-}" ]; then
  ARGS+=(--out "$REPORT_OUT")
fi
"$PY" "${ARGS[@]}"
