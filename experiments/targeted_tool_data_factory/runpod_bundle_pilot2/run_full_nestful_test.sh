#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  FULL NESTFUL test (1661 tasks) — DISABLED BY DEFAULT. NOT part of run_all.
#
#  Run this ONLY after reading D0_VS_D1_REPORT.md. The diagnostic-500 is the
#  decision set; the full test is the confirmation set. Running it first and
#  then choosing what to report would turn it into a fishing expedition.
#
#  Requires an explicit acknowledgement:
#      CONFIRM_FULL_NESTFUL=yes bash run_full_nestful_test.sh
# ═══════════════════════════════════════════════════════════════════════════
set -Eeuo pipefail

if [ "${CONFIRM_FULL_NESTFUL:-no}" != "yes" ]; then
  cat >&2 <<'MSG'
[full_nestful] refusing to run.

  This is the confirmation set (1661 tasks, ~4 GPU-hours). It is deliberately
  not wired into run_all_4gpu.sh. Read the paired D0 vs D1 report first, then:

      CONFIRM_FULL_NESTFUL=yes bash run_full_nestful_test.sh
MSG
  exit 3
fi

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FACTORY="$(cd "$BUNDLE/.." && pwd)"
V3="$(cd "$FACTORY/../nestful_synthetic_curriculum_v3" && pwd)"
PY="${PYTHON:-python3}"
SEED="${SEED:-20260726}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$V3/outputs/runs}"
RESULTS="${RESULTS:-$FACTORY/outputs/runpod_pilot2}/full_nestful"
NESTFUL_TEST="${NESTFUL_TEST:-$V3/data/splits/nestful_test.jsonl}"

mkdir -p "$RESULTS"
for label in C0 D0 D1; do
  case "$label" in
    C0) CK_ARG="" ;;
    *)  CK="$($PY - "$OUTPUT_ROOT/pilot2_${label}_seed${SEED}" <<'PYEOF'
import sys, pathlib
root = pathlib.Path(sys.argv[1])
hits = sorted(root.rglob("adapter_config.json"))
print(hits[-1].parent if hits else "", end="")
PYEOF
)"
        [ -n "$CK" ] || { echo "[full_nestful] no adapter for $label" >&2; exit 2; }
        CK_ARG="--checkpoint $CK" ;;
  esac
  echo "[full_nestful] $label"
  $PY "$V3/scripts/eval/final_eval_v5.py" run \
    --label "$label" $CK_ARG \
    --eval-set "$NESTFUL_TEST" \
    --out-dir "$RESULTS/$label"
done

$PY "$V3/scripts/eval/final_eval_v5.py" compare \
  --baseline "$RESULTS/C0" --best "$RESULTS/D1" --final "$RESULTS/D0" \
  --out "$RESULTS/FULL_NESTFUL_COMPARE.md"
echo "[full_nestful] report: $RESULTS/FULL_NESTFUL_COMPARE.md"
