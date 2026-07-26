#!/usr/bin/env bash
# =============================================================================
#  Pilot2 Phase-1 GRPO canary (4 GPUs) — after the signal probe
#
#  Does NOT train full D1 on 160 tasks. Does NOT run full NESTFUL-1661.
#
#  Pipeline:
#    1. GPU check (GPU0 learner, GPU1-3 rollout workers)
#    2. verify frozen hashes
#    3. offline reward audit over stored probe rollouts (no new inference)
#    4. verify recommended_phase1_train.jsonl (80, replay, leakage, NESTFUL JSD)
#    5. train C1 on the 80-task subset with the selected reward variant
#       (~20 optimizer steps, 8 rollouts)
#    6. evaluate C0 + C1 on structural held-out 80 and NESTFUL-500
#    7. write C0_VS_C1_PHASE1_REPORT.md/json + canary diagnostics
#
#  After C1, a SEPARATE re-probe command for deferred_phase2_tasks.jsonl is
#  printed but NEVER started automatically.
#
#  Usage (from repo root):
#      export HF_TOKEN=...
#      bash experiments/targeted_tool_data_factory/runpod_bundle_pilot2/run_phase1_canary_4gpu.sh
#
#  Flags:
#      --dry-run     validate + print every command, train nothing
#      --resume      continue interrupted training / eval
#      --stage S     one of: audit verify train eval report
#      --probe-dir D path to signal_probe outputs (rollouts + phase1 jsonl)
# =============================================================================
set -Eeuo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FACTORY="$(cd "$BUNDLE/.." && pwd)"
EXPERIMENTS="$(cd "$FACTORY/.." && pwd)"
V3="$EXPERIMENTS/nestful_synthetic_curriculum_v3"
REPO="$(cd "$EXPERIMENTS/.." && pwd)"
PY="${PYTHON:-python3}"

# Git Bash passes /c/Users/... paths that Windows CPython cannot open.
# Prefer mixed Windows paths (C:/Users/...) when cygpath is available.
if command -v cygpath >/dev/null 2>&1; then
  BUNDLE="$(cygpath -m "$BUNDLE")"
  FACTORY="$(cygpath -m "$FACTORY")"
  EXPERIMENTS="$(cygpath -m "$EXPERIMENTS")"
  V3="$(cygpath -m "$V3")"
  REPO="$(cygpath -m "$REPO")"
fi

DRY_RUN=0
RESUME_FLAG=""
ONLY_STAGE=""
PROBE_DIR_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)   DRY_RUN=1; shift ;;
    --resume)    RESUME_FLAG="--resume"; shift ;;
    --stage)     ONLY_STAGE="$2"; shift 2 ;;
    --probe-dir) PROBE_DIR_ARG="$2"; shift 2 ;;
    *) echo "[phase1] unknown arg: $1" >&2; exit 1 ;;
  esac
done

SEED="${SEED:-20260726}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$V3/outputs/runs}"
RESULTS="${RESULTS:-$FACTORY/outputs/runpod_pilot2/phase1_canary}"
DATA="$BUNDLE/data"
FACTORY_TOOLS="$FACTORY/trainer_adapter"
export USE_VLLM="${USE_VLLM:-1}"
export ROLLOUT_DP_GPUS="${ROLLOUT_DP_GPUS:-1,2,3}"
export CANARY_TRAJ_LOG=1
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export SYNTHETIC_TOOLS_DIR="$FACTORY_TOOLS"

# Resolve the probe directory. Prefer an explicit flag, then the canonical
# location, then the locally extracted zip used for offline validation.
if [ -n "$PROBE_DIR_ARG" ]; then
  PROBE_DIR="$PROBE_DIR_ARG"
elif [ -d "$FACTORY/outputs/runpod_pilot2/signal_probe" ]; then
  PROBE_DIR="$FACTORY/outputs/runpod_pilot2/signal_probe"
elif [ -d "$FACTORY/outputs/runpod_pilot2/signal_probe_from_zip/signal_probe" ]; then
  PROBE_DIR="$FACTORY/outputs/runpod_pilot2/signal_probe_from_zip/signal_probe"
else
  PROBE_DIR="$FACTORY/outputs/runpod_pilot2/signal_probe"
fi

PHASE1_JSONL="$PROBE_DIR/recommended_phase1_train.jsonl"
DEFERRED_JSONL="$PROBE_DIR/deferred_phase2_tasks.jsonl"
VARIANT_FILE="$PROBE_DIR/SELECTED_REWARD_VARIANT.json"
C1_RUN_ID="pilot2_C1_phase1_seed${SEED}"
REPORT_MD="$RESULTS/C0_VS_C1_PHASE1_REPORT.md"
REPORT_JSON="$RESULTS/C0_VS_C1_PHASE1_REPORT.json"

mkdir -p "$RESULTS"
LOG="$RESULTS/run_phase1_canary_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

banner() {
  echo ""
  echo "=================================================================="
  echo "[phase1] $1"
  echo "=================================================================="
}

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = "0" ]; then
    "$@"
  else
    # Execute planning-only subcommands so --dry-run still validates gates.
    case " $* " in
      *" --dry-run "*) "$@" ;;
    esac
  fi
}

want() { [ -z "$ONLY_STAGE" ] || [ "$ONLY_STAGE" = "$1" ]; }

fail() { echo "[phase1] ABORT: $*" >&2; exit 1; }

cd "$REPO"

echo "[phase1] probe_dir: $PROBE_DIR"
echo "[phase1] results:   $RESULTS"
echo "[phase1] C1 run id: $C1_RUN_ID"
echo "[phase1] full NESTFUL-1661: DISABLED"
echo "[phase1] full D1-160:       DISABLED"

# ── 1. GPUs ────────────────────────────────────────────────────────────────
banner "1/7  GPU check (GPU0 learner, GPU1-3 rollout workers)"
DRY_RUN="$DRY_RUN" "$PY" - <<'PYEOF'
import os, sys
try:
    import torch
except ImportError:
    print("[phase1] torch not importable yet")
    sys.exit(0 if os.environ.get("DRY_RUN") == "1" else 1)
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name} {p.total_memory/1e9:.1f} GB")
print(f"[phase1] visible GPUs: {n}")
if n < 4:
    if os.environ.get("DRY_RUN") == "1":
        print("[phase1] dry run: continuing without 4 GPUs")
        sys.exit(0)
    print("[phase1] ABORT: 4 GPUs required", file=sys.stderr)
    sys.exit(1)
PYEOF

# ── 2. hashes (always run — fail-fast even under --dry-run) ────────────────
banner "2/7  verify frozen dataset hashes + factory registry"
"$PY" "$BUNDLE/verify_hashes.py" --manifest "$BUNDLE/MANIFEST.sha256.json"
"$PY" "$BUNDLE/verify_factory_registry.py"

# ── 3. offline reward audit ────────────────────────────────────────────────
if want audit; then
banner "3/7  offline reward audit (no new inference)"
[ -d "$PROBE_DIR" ] || fail "probe dir missing: $PROBE_DIR"
run "$PY" "$BUNDLE/offline_reward_audit.py" \
  --probe-dir "$PROBE_DIR" \
  --data "$DATA/train_grpo_pilot2.jsonl" \
  --out-dir "$RESULTS/offline_reward_audit" \
  $([ "$DRY_RUN" = "1" ] && echo --dry-run)
# Prefer the freshly written audit artefact under RESULTS, then probe root.
if [ -f "$RESULTS/offline_reward_audit/SELECTED_REWARD_VARIANT.json" ]; then
  cp -f "$RESULTS/offline_reward_audit/SELECTED_REWARD_VARIANT.json" \
        "$PROBE_DIR/SELECTED_REWARD_VARIANT.json" 2>/dev/null || true
  VARIANT_FILE="$RESULTS/offline_reward_audit/SELECTED_REWARD_VARIANT.json"
fi
if [ -f "$VARIANT_FILE" ]; then
  "$PY" - "$VARIANT_FILE" "$DRY_RUN" <<'PYEOF'
import json, sys
sel = json.loads(open(sys.argv[1], encoding="utf-8").read())
dry = sys.argv[2] == "1"
prefix = "[phase1] dry-run selected variant:" if dry else "[phase1] selected variant:"
print(prefix, sel.get("selected"), sel.get("reason"))
if not dry and (sel.get("hard_gate") != "PASS" or not sel.get("selected")):
    sys.exit("hard gate FAIL — refusing to train")
PYEOF
elif [ "$DRY_RUN" = "0" ]; then
  fail "missing $VARIANT_FILE after audit"
else
  echo "[phase1] dry-run: SELECTED_REWARD_VARIANT.json not present yet (ok)"
fi
fi

# ── 4. Phase-1 subset verification ─────────────────────────────────────────
if want verify; then
banner "4/7  verify recommended_phase1_train.jsonl (80 / replay / leakage / NESTFUL JSD)"
[ -f "$PHASE1_JSONL" ] || fail "missing $PHASE1_JSONL — run the signal probe first"
run "$PY" "$BUNDLE/verify_phase1_subset.py" \
  --phase1 "$PHASE1_JSONL" \
  --deferred "$DEFERRED_JSONL" \
  --heldout "$DATA/heldout_grpo_pilot2.jsonl" \
  --reserve "$DATA/reserve_canonical_pilot2.jsonl" \
  --selected "$DATA/canonical_pilot2.jsonl" \
  --nestful-profile "$DATA/nestful_profile.json" \
  --out-dir "$RESULTS/phase1_verification" \
  $([ "$DRY_RUN" = "1" ] && echo --dry-run)
# Fail-fast: also gold-replay via the stock preflight for the 80-row file.
run "$PY" "$FACTORY/trainer_adapter/preflight_gold_replay.py" \
  --data "$PHASE1_JSONL" --expect 80 \
  --report "$RESULTS/preflight_phase1_gold_replay.json"
fi

# ── 5. train C1 ────────────────────────────────────────────────────────────
if want train; then
banner "5/7  train C1 on 80 Phase-1 tasks (selected reward, 8 rollouts, ~20 opt steps)"
[ -f "$VARIANT_FILE" ] || fail "missing $VARIANT_FILE — run --stage audit first"
run "$PY" "$BUNDLE/run_phase1_train.py" \
  --train-subset "$PHASE1_JSONL" \
  --variant-file "$VARIANT_FILE" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$C1_RUN_ID" \
  --seed "$SEED" \
  --wandb-project "${WANDB_PROJECT:-ttdf-pilot2}" \
  --wandb-group "pilot2_phase1_canary" \
  $RESUME_FLAG \
  $([ "$DRY_RUN" = "1" ] && echo --dry-run)
fi

# ── 6. eval C0 + C1 ────────────────────────────────────────────────────────
if want eval; then
banner "6/7  evaluate C0 + C1 (held-out 80 + NESTFUL-500; NOT 1661)"
run "$PY" "$BUNDLE/run_eval_all.py" \
  --output-root "$OUTPUT_ROOT" \
  --c1-run "$C1_RUN_ID" \
  --d1-run "$C1_RUN_ID" \
  --heldout "$DATA/heldout_nestful_pilot2.jsonl" \
  --diagnostic "$DATA/nestful_diagnostic_500.jsonl" \
  --results "$RESULTS" \
  --arms C0,C1 \
  --gpus 0,1,2,3 \
  $([ "$DRY_RUN" = "1" ] && echo --dry-run)
fi

# ── 7. report + diagnostics ────────────────────────────────────────────────
if want report; then
banner "7/7  C0 vs C1 report + canary diagnostics"
run "$PY" "$BUNDLE/make_paired_report.py" \
  --results "$RESULTS" \
  --out "$REPORT_MD" \
  --json-out "$REPORT_JSON" \
  --baseline C0 --treatment C1 \
  --title "C0 vs C1 Phase-1 canary"
run "$PY" "$BUNDLE/collect_canary_diagnostics.py" \
  --run-dir "$OUTPUT_ROOT/$C1_RUN_ID" \
  --out "$RESULTS/C1_CANARY_DIAGNOSTICS.json" \
  $([ "$DRY_RUN" = "1" ] && echo --dry-run)
fi

banner "done"
echo "[phase1] report:  $REPORT_MD"
echo "[phase1] json:    $REPORT_JSON"
echo "[phase1] variant: $VARIANT_FILE"
echo "[phase1] C1 ckpt: $OUTPUT_ROOT/$C1_RUN_ID"
echo "[phase1] full NESTFUL-1661 was NOT started"
echo "[phase1] full D1-160 was NOT started"
echo ""
echo "[phase1] OPTIONAL later re-probe of deferred Phase-2 tasks"
echo "         (NOT started by this script — run only after reading the report):"
echo "  bash $BUNDLE/run_deferred_reprobe_4gpu.sh"
