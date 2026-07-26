#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  Pilot2 RunPod entry point (4 GPUs)
#
#  Default mode — D0 vs D1:
#    D0 = 160 OLD Stage-3 tasks       (the data that did not transfer)
#    D1 = 160 pilot2 factory tasks    (target-conditioned, executor-verified)
#    Both train sequentially from the same C0, then C0/D0/D1 are evaluated.
#
#  --c0-vs-d1 mode:
#    Keeps GPU check, install, SHA256, config parity, gold-replay preflight
#    and the A1/A4 canary. After a PASS canary, SKIPS D0 training entirely,
#    trains only D1 on frozen data/train_grpo_pilot2.jsonl (160 tasks), and
#    evaluates C0 vs D1. Full NESTFUL 1661 is never started automatically.
#
#  Usage (from repo root):
#      export HF_TOKEN=...            # required
#      export WANDB_API_KEY=...       # optional
#      bash experiments/targeted_tool_data_factory/runpod_bundle_pilot2/run_all_4gpu.sh
#      bash experiments/targeted_tool_data_factory/runpod_bundle_pilot2/run_all_4gpu.sh --c0-vs-d1
#
#  Flags:
#      --c0-vs-d1       skip D0 training; evaluate C0 vs D1 only
#      --dry-run        validate everything, print every command, train nothing
#      --skip-canary    reuse an earlier PASS canary
#      --resume         continue interrupted runs
#      --stage STAGE    run only one stage of: preflight canary d0 d1 eval report
# ═══════════════════════════════════════════════════════════════════════════
set -Eeuo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FACTORY="$(cd "$BUNDLE/.." && pwd)"
EXPERIMENTS="$(cd "$FACTORY/.." && pwd)"
V3="$EXPERIMENTS/nestful_synthetic_curriculum_v3"
REPO="$(cd "$EXPERIMENTS/.." && pwd)"
PY="${PYTHON:-python3}"

DRY_RUN=0
SKIP_CANARY=0
RESUME_FLAG=""
ONLY_STAGE=""
C0_VS_D1=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)     DRY_RUN=1; shift ;;
    --skip-canary) SKIP_CANARY=1; shift ;;
    --resume)      RESUME_FLAG="--resume"; shift ;;
    --c0-vs-d1)    C0_VS_D1=1; shift ;;
    --stage)       ONLY_STAGE="$2"; shift 2 ;;
    *) echo "[run_all] unknown arg: $1" >&2; exit 1 ;;
  esac
done

SEED="${SEED:-20260726}"
REWARD_ARM="${REWARD_ARM:-A4_GATED_VERIFIABLE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$V3/outputs/runs}"
RESULTS="${RESULTS:-$FACTORY/outputs/runpod_pilot2}"
DATA="$BUNDLE/data"
# The tool registry is a property of the DATASET, not of the run: D0's Stage-3
# tasks only exist in the legacy registry, D1's tasks only in the factory one.
# It is therefore set per stage and never exported globally — a global value
# would make D0 fail on every single tool call.
FACTORY_TOOLS="$FACTORY/trainer_adapter"
LEGACY_TOOLS="$V3"
export USE_VLLM="${USE_VLLM:-1}"
export ROLLOUT_DP_GPUS="${ROLLOUT_DP_GPUS:-1,2,3}"
export CANARY_TRAJ_LOG=1
export PYTHONUNBUFFERED=1

if [ "$C0_VS_D1" = "1" ]; then
  MODE_LABEL="C0 vs D1"
  REPORT_MD="$RESULTS/C0_VS_D1_REPORT.md"
  REPORT_JSON="$RESULTS/C0_VS_D1_REPORT.json"
  WANDB_GROUP_TRAIN="pilot2_c0_vs_d1"
else
  MODE_LABEL="D0 vs D1"
  REPORT_MD="$RESULTS/D0_VS_D1_REPORT.md"
  REPORT_JSON="$RESULTS/D0_VS_D1_REPORT.json"
  WANDB_GROUP_TRAIN="pilot2_d0_vs_d1"
fi

mkdir -p "$RESULTS"
LOG="$RESULTS/run_all_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

banner() {
  echo ""
  echo "══════════════════════════════════════════════════════════════════"
  echo "[run_all] $1"
  echo "══════════════════════════════════════════════════════════════════"
}

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = "0" ]; then "$@"; fi
}

want() { [ -z "$ONLY_STAGE" ] || [ "$ONLY_STAGE" = "$1" ]; }

cd "$REPO"

echo "[run_all] mode: $MODE_LABEL"
if [ "$C0_VS_D1" = "1" ]; then
  echo "[run_all] D0 training will be SKIPPED; C0 evaluation is kept"
fi

# ── 1. GPUs ────────────────────────────────────────────────────────────────
banner "1/9  GPU check (4 required: GPU0 learner, GPU1-3 rollout workers)"
DRY_RUN="$DRY_RUN" "$PY" - <<'PYEOF'
import os
import sys
try:
    import torch
except ImportError:
    print("[run_all] torch not importable yet (install step follows)")
    sys.exit(0)
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name} {p.total_memory/1e9:.1f} GB")
print(f"[run_all] visible GPUs: {n}")
if n < 4:
    # A dry run is meant to be validated on a laptop, so the count is advisory
    # there. Any real run stops here rather than silently training on one GPU.
    if os.environ.get("DRY_RUN") == "1":
        print("[run_all] dry run: continuing without 4 GPUs")
        sys.exit(0)
    print("[run_all] ABORT: 4 GPUs required", file=sys.stderr)
    sys.exit(1)
PYEOF

# ── 2. dependencies ────────────────────────────────────────────────────────
banner "2/9  install dependencies"
run bash "$BUNDLE/install.sh"

# ── 3. hashes ──────────────────────────────────────────────────────────────
banner "3/9  verify frozen dataset hashes"
run "$PY" "$BUNDLE/verify_hashes.py" --manifest "$BUNDLE/MANIFEST.sha256.json"

banner "3b/9 config parity (D0 vs D1 differ ONLY in the dataset + registry)"
run "$PY" "$BUNDLE/check_config_parity.py" \
  --d0 "$BUNDLE/configs/d0_stage3_old.json" \
  --d1 "$BUNDLE/configs/d1_pilot2.json" \
  --report "$RESULTS/config_parity.json"

# ── 4. gold-replay preflight ───────────────────────────────────────────────
if want preflight; then
banner "4/9  gold-replay preflight through the REAL trainer executor"
run "$PY" "$FACTORY/trainer_adapter/preflight_gold_replay.py" \
  --data "$DATA/train_grpo_pilot2.jsonl"   --expect 160 \
  --data "$DATA/heldout_grpo_pilot2.jsonl" --expect 80 \
  --report "$RESULTS/preflight_gold_replay.json"
fi

# ── 5. dispatch / executor canary ──────────────────────────────────────────
if want canary && [ "$SKIP_CANARY" = "0" ]; then
banner "5/9  dispatch + executor canary (24 pilot2 tasks x 8 rollouts, A1 then A4)"
export SYNTHETIC_TOOLS_DIR="$FACTORY_TOOLS"     # canary runs on pilot2 tasks
for arm in A1_OUTCOME_ONLY A4_GATED_VERIFIABLE; do
  echo "[run_all] canary arm=$arm (registry: $SYNTHETIC_TOOLS_DIR)"
  run "$PY" "$V3/scripts/ablation/run_reward_ablation.py" \
    --round 2 --reward-arm "$arm" --seed "$SEED" --canary \
    --train-subset "$DATA/canary_pilot2_24.jsonl" \
    --output-root "$OUTPUT_ROOT" \
    --wandb-project "${WANDB_PROJECT:-ttdf-pilot2}" \
    --wandb-group "pilot2_canary" $RESUME_FLAG
done

banner "6/9  canary gates (resolved policy, distinct A1/A4 rewards, NaN/Inf, logging)"
# Gate A: the existing reward-dispatch validator, unmodified.
run "$PY" "$V3/scripts/ablation/validate_dispatch_canary.py" \
  --output-root "$OUTPUT_ROOT" --seed "$SEED" \
  --report "$RESULTS/canary_dispatch_gate.json"
# Gate B: pilot2 executor gates (factory adapter really executed, no legacy fallback).
run "$PY" "$BUNDLE/check_canary_gates.py" \
  --output-root "$OUTPUT_ROOT" --seed "$SEED" \
  --report "$RESULTS/canary_executor_gate.json"
if [ "$C0_VS_D1" = "1" ]; then
  echo "[run_all] canary PASS — D0 training skipped; D1 may start"
else
  echo "[run_all] canary PASS — D0/D1 may start"
fi
fi

# ── 7. training ────────────────────────────────────────────────────────────
train_arm () {          # $1 = label (D0/D1), $2 = dataset, $3 = banner, $4 = registry
  export SYNTHETIC_TOOLS_DIR="$4"
  banner "$3  train $1 (reward=$REWARD_ARM seed=$SEED, GPU0 learner / GPU1-3 rollouts)"
  echo "[run_all] tool registry: $SYNTHETIC_TOOLS_DIR"
  run "$PY" "$V3/scripts/ablation/run_reward_ablation.py" \
    --round 3 --reward-arm "$REWARD_ARM" --seed "$SEED" \
    --train-subset "$2" \
    --skip-eval --skip-c0-eval \
    --output-root "$OUTPUT_ROOT" \
    --run-id "pilot2_${1}_seed${SEED}" \
    --wandb-project "${WANDB_PROJECT:-ttdf-pilot2}" \
    --wandb-group "$WANDB_GROUP_TRAIN" $RESUME_FLAG
}

if [ "$C0_VS_D1" = "1" ]; then
  if want d0; then
    banner "7/9  SKIP D0 training (--c0-vs-d1)"
    echo "[run_all] D0 trainer NOT invoked"
  fi
else
  if want d0; then train_arm D0 "$DATA/d0_stage3_train_160.jsonl" "7/9" "$LEGACY_TOOLS"; fi
fi

if want d1; then train_arm D1 "$DATA/train_grpo_pilot2.jsonl" "8/9" "$FACTORY_TOOLS"; fi

# ── 9. evaluation on GPU0-3 in parallel, then the paired report ────────────
if want eval; then
  if [ "$C0_VS_D1" = "1" ]; then
    banner "9/9  evaluation C0 + D1 (structural held-out 80, G/A separately, NESTFUL diagnostic-500)"
    run "$PY" "$BUNDLE/run_eval_all.py" \
      --output-root "$OUTPUT_ROOT" \
      --d1-run "pilot2_D1_seed${SEED}" \
      --heldout "$DATA/heldout_nestful_pilot2.jsonl" \
      --diagnostic "$DATA/nestful_diagnostic_500.jsonl" \
      --results "$RESULTS" \
      --arms C0,D1 \
      --gpus 0,1,2,3 $([ "$DRY_RUN" = "1" ] && echo --dry-run)
  else
    banner "9/9  evaluation (structural held-out 80, G/A separately, NESTFUL diagnostic-500)"
    run "$PY" "$BUNDLE/run_eval_all.py" \
      --output-root "$OUTPUT_ROOT" \
      --d0-run "pilot2_D0_seed${SEED}" \
      --d1-run "pilot2_D1_seed${SEED}" \
      --heldout "$DATA/heldout_nestful_pilot2.jsonl" \
      --diagnostic "$DATA/nestful_diagnostic_500.jsonl" \
      --results "$RESULTS" \
      --arms C0,D0,D1 \
      --gpus 0,1,2,3 $([ "$DRY_RUN" = "1" ] && echo --dry-run)
  fi
fi

if want report; then
  if [ "$C0_VS_D1" = "1" ]; then
    banner "paired C0 vs D1 report"
    run "$PY" "$BUNDLE/make_paired_report.py" \
      --results "$RESULTS" \
      --out "$REPORT_MD" \
      --json-out "$REPORT_JSON" \
      --baseline C0 --treatment D1 \
      --title "C0 vs D1"
  else
    banner "paired D0 vs D1 report"
    run "$PY" "$BUNDLE/make_paired_report.py" \
      --results "$RESULTS" \
      --out "$REPORT_MD" \
      --json-out "$REPORT_JSON" \
      --baseline D0 --treatment D1 \
      --title "D0 vs D1"
  fi
fi

banner "done"
echo "[run_all] mode:    $MODE_LABEL"
echo "[run_all] results: $RESULTS"
echo "[run_all] report:  $REPORT_MD"
echo "[run_all] json:    $REPORT_JSON"
echo "[run_all] full NESTFUL test (1661) is DISABLED by default; run it only after"
echo "          reading the paired report:  bash $BUNDLE/run_full_nestful_test.sh"
