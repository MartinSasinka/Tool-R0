#!/usr/bin/env bash
# Dry-run gate for --c0-vs-d1: D0 trainer must never be invoked, C0 eval must stay.
set -Eeuo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FACTORY="$(cd "$BUNDLE/.." && pwd)"
REPO="$(cd "$FACTORY/../.." && pwd)"
PY="${PYTHON:-python3}"
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

cd "$REPO"
export PYTHON="$PY"

set +e
bash "$BUNDLE/run_all_4gpu.sh" --c0-vs-d1 --dry-run >"$LOG" 2>&1
RC=$?
set -e

echo "──── dry-run log (tail) ────"
tail -n 40 "$LOG"
echo "────────────────────────────"

fail=0
assert_contains() {
  local pat="$1" msg="$2"
  # `--` stops option parsing so patterns like `--arms C0,D1` are not flags.
  if ! grep -Eq -- "$pat" "$LOG"; then
    echo "FAIL: $msg (pattern: $pat)" >&2
    fail=1
  else
    echo "ok: $msg"
  fi
}
assert_absent() {
  local pat="$1" msg="$2"
  if grep -Eq -- "$pat" "$LOG"; then
    echo "FAIL: $msg (pattern matched: $pat)" >&2
    grep -En -- "$pat" "$LOG" | head -n 5 >&2 || true
    fail=1
  else
    echo "ok: $msg"
  fi
}

if [ "$RC" -ne 0 ]; then
  echo "FAIL: dry-run exited $RC" >&2
  fail=1
else
  echo "ok: dry-run exited 0"
fi

assert_contains 'mode: C0 vs D1' "mode banner announces C0 vs D1"
assert_contains 'SKIP D0 training|--c0-vs-d1|D0 trainer NOT invoked' \
  "D0 training is explicitly skipped"
assert_contains 'train D1' "D1 training is scheduled"
assert_contains 'train_grpo_pilot2\.jsonl' "D1 uses frozen pilot2 train set"
assert_contains 'run_eval_all\.py' "evaluation is scheduled"
assert_contains '--arms C0,D1' "eval arms are C0,D1 (no D0)"
assert_contains 'C0_VS_D1_REPORT\.md' "C0 vs D1 report path is used"
assert_contains 'make_paired_report\.py' "paired report is scheduled"
assert_contains '--baseline C0 --treatment D1' "report contrasts C0 vs D1"

# The D0 trainer is the reward-ablation invocation with run-id pilot2_D0_*.
# Canary / parity / hash steps mentioning "D0" in config filenames are fine.
assert_absent 'run_reward_ablation\.py.*--run-id pilot2_D0_' \
  "D0 trainer (run_reward_ablation --run-id pilot2_D0_*) is never invoked"
assert_absent '--d0-run ' \
  "eval does not require a D0 run id"
assert_absent 'CONFIRM_FULL_NESTFUL=yes' \
  "full NESTFUL 1661 is not executed"

# Positive: C0 evaluation remains (base model, no adapter) via --arms C0,D1
# and the report baseline C0.
assert_contains 'evaluation C0 \+ D1' "C0 evaluation stage stays"

if [ "$fail" -ne 0 ]; then
  echo "[test_c0_vs_d1] FAIL" >&2
  exit 1
fi
echo "[test_c0_vs_d1] PASS"
exit 0
