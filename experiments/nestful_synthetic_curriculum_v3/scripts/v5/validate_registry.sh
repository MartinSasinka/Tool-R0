#!/usr/bin/env bash
# 1/9 — Validate the executable synthetic tool registry.
#
# Runs the registry self-check (determinism probes, output types, behavioural
# duplicate detection) and the executor regression tests.
#
# Env:  PYTHON=python3   interpreter to use
#       SKIP_TESTS=0     set to 1 to skip the pytest regression suite
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

banner "registry validation"
print_env PYTHON SKIP_TESTS
require_file "$V3/lib/synthetic_tools.py" "registry"

cd "$V3"
"$PY" -m lib.synthetic_tools

if [ "${SKIP_TESTS:-0}" != "1" ]; then
  banner "executor regression tests"
  cd "$MINIMAL"
  "$PY" -m pytest tests/test_synthetic_executor.py -q
fi

banner "registry OK"
