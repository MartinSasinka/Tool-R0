#!/usr/bin/env bash
# Pure Stage 3 — two continuous GRPO epochs from base (overnight).
#
# ALWAYS export RUN_DIR and mkdir BEFORE tee:
#   export RUN_DIR="outputs/runs/pure_stage3_2ep_$(date +%Y%m%d_%H%M%S)"
#   mkdir -p "$RUN_DIR"
#   bash scripts/v5/run_pure_stage3_two_epoch_overnight.sh 2>&1 | tee "$RUN_DIR/console.log"
#
# Resume (phase-level; Adam not on disk):
#   export RUN_DIR="outputs/runs/<existing>"
#   export RESUME=1
#   bash scripts/v5/run_pure_stage3_two_epoch_overnight.sh 2>&1 | tee -a "$RUN_DIR/console.log"
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

RUN_DIR="${RUN_DIR:?export RUN_DIR first (see header comments)}"
mkdir -p "$RUN_DIR" "$RUN_DIR/logs"

STAGE3_SRC="${STAGE3_SOURCE:-$V3/data/training_ready_v5/filtered/phase2_stage3_plus_stage2_replay.jsonl}"
STAGE3_PURE="${STAGE3_DATASET:-$V3/data/training_ready_v5/filtered/stage3_train_ready.jsonl}"
DEV_SET="${DEV_SET:-$MINIMAL/data/splits/nestful_dev.jsonl}"
TEST_SET="${TEST_SET:-$MINIMAL/data/splits/nestful_test.jsonl}"
AUDIT_DIR="${AUDIT_DIR:-$V3/reports/stage3_syntax_audit}"

require_file "$STAGE3_SRC" "STAGE3_SOURCE (phase2 mix)"
require_file "$DEV_SET" "DEV_SET"
require_file "$TEST_SET" "TEST_SET"

if [ -f "$RUN_DIR/SUCCESS" ] && [ "${RESUME:-0}" != "1" ]; then
  echo "[pure-s3] ERROR: SUCCESS marker exists at $RUN_DIR — pick new RUN_DIR" >&2
  exit 1
fi

# ── environment assertions ──────────────────────────────────────────────────
banner "environment assertions"
command -v "$PY" >/dev/null || { echo "python missing"; exit 1; }
if ! "$PY" -c "import torch; assert torch.cuda.is_available()"; then
  echo "[pure-s3] ERROR: CUDA not available" >&2
  exit 1
fi
N_GPU=$("$PY" -c "import torch; print(torch.cuda.device_count())")
if [ "${N_GPU:-0}" -lt 4 ] && [ "${ALLOW_FEWER_GPUS:-0}" != "1" ]; then
  echo "[pure-s3] ERROR: need 4 GPUs, found $N_GPU (set ALLOW_FEWER_GPUS=1 to override)" >&2
  exit 1
fi
if [ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "[pure-s3] WARNING: HF_TOKEN unset — gated model download may fail" >&2
fi
if [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "disabled" ] && [ "${WANDB_MODE:-}" != "offline" ]; then
  echo "[pure-s3] WARNING: WANDB_API_KEY unset" >&2
fi
IBM_DIR="$MINIMAL/data/NESTFUL-main/data_v2/executable_functions"
if [ ! -f "$IBM_DIR/func_file_map.json" ]; then
  echo "[pure-s3] ERROR: IBM executable_functions missing at $IBM_DIR" >&2
  exit 1
fi
# disk space (need ~20GB free for adapters + eval)
FREE_KB=$(df -Pk "$RUN_DIR" 2>/dev/null | awk 'NR==2{print $4}' || echo 999999999)
if [ "${FREE_KB:-0}" -lt 10000000 ]; then
  echo "[pure-s3] WARNING: low free disk (${FREE_KB} KB)" >&2
fi

export SEED="${SEED:-42}"
export DATA_SEED="${DATA_SEED:-42}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export WANDB_PROJECT="${WANDB_PROJECT:-nestful-v5-pure-stage3}"
export WANDB_RUN_GROUP="${WANDB_GROUP:-$(basename "$RUN_DIR")}"
export USE_VLLM="${USE_VLLM:-1}"
export ROLLOUT_DP_GPUS="${ROLLOUT_DP_GPUS:-1,2,3}"
export EVAL_TP="${EVAL_TP:-4}"
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export EPOCHS="${EPOCHS:-2}"

# Targeted cleanup of our rollout workers only
_cleanup() {
  local pidfile="$RUN_DIR/logs/rollout_worker_pids.json"
  if [ -f "$pidfile" ]; then
    "$PY" - <<'PY' "$pidfile" 2>/dev/null || true
import json, os, signal, sys
path = sys.argv[1]
try:
    pids = json.load(open(path, encoding="utf-8"))
except Exception:
    sys.exit(0)
for pid in pids:
    try:
        os.kill(int(pid), signal.SIGTERM)
    except (ProcessLookupError, ValueError, PermissionError):
        pass
PY
  fi
}
trap _cleanup EXIT INT TERM

print_env RUN_DIR STAGE3_PURE DEV_SET TEST_SET SEED DATA_SEED ROLLOUT_SEED \
          NUM_GENERATIONS LEARNING_RATE KL_BETA TEMPERATURE TOP_P \
          MAX_TRAIN_TASKS DEV_MAX_TASKS USE_VLLM ROLLOUT_DP_GPUS EVAL_TP \
          VLLM_GPU_UTIL WANDB_PROJECT WANDB_RUN_GROUP RESUME EPOCHS

cd "$V3"

# ── 1. Materialize pure Stage 3 (326) ───────────────────────────────────────
banner "materialize pure Stage 3"
"$PY" scripts/data/materialize_pure_stage3.py \
  --source "$STAGE3_SRC" --out "$STAGE3_PURE"
require_file "$STAGE3_PURE" "STAGE3_PURE"

# ── 2. Syntax audit ─────────────────────────────────────────────────────────
banner "NESTFUL syntax audit"
"$PY" scripts/data/audit_stage3_nestful_syntax.py \
  --input "$STAGE3_PURE" \
  --report-dir "$AUDIT_DIR"
AUDIT_JSON="$AUDIT_DIR/stage3_nestful_syntax_audit.json"
require_file "$AUDIT_JSON" "syntax audit report"
VERDICT=$("$PY" -c "import json; print(json.load(open(r'$AUDIT_JSON',encoding='utf-8'))['verdict'])")
echo "[pure-s3] syntax audit verdict=$VERDICT"
if [ "$VERDICT" = "AMBIGUOUS" ]; then
  echo "[pure-s3] ERROR: syntax audit AMBIGUOUS — abort before training" >&2
  echo FAILED > "$RUN_DIR/FAILED"
  exit 2
fi
# Prefer derived dataset if audit wrote one
DERIVED=$("$PY" -c "import json; print(json.load(open(r'$AUDIT_JSON',encoding='utf-8')).get('output_path') or '')")
TRAIN_DS="$STAGE3_PURE"
if [ -n "$DERIVED" ] && [ -f "$DERIVED" ]; then
  TRAIN_DS="$DERIVED"
  echo "[pure-s3] using derived dataset: $TRAIN_DS"
else
  echo "[pure-s3] syntax normalization no-op; using $TRAIN_DS"
fi

# ── 3. Full preflight ───────────────────────────────────────────────────────
banner "preflight"
"$PY" scripts/training/preflight_training_datasets.py \
  "$TRAIN_DS" \
  --report "$RUN_DIR/preflight_report.json"

# ── 4. Dry-run manifest ─────────────────────────────────────────────────────
banner "dry-run manifest"
DRY_ARGS=(scripts/training/run_pure_stage3_two_epoch.py
  --run-dir "$RUN_DIR"
  --dataset "$TRAIN_DS"
  --dev-set "$DEV_SET"
  --test-set "$TEST_SET"
  --syntax-audit-verdict "$VERDICT"
  --syntax-audit-path "$AUDIT_JSON"
  --dry-run)
[ "${RESUME:-0}" = "1" ] && DRY_ARGS+=(--resume)
"$PY" "${DRY_ARGS[@]}"

# ── 5. Full overnight orchestrator ──────────────────────────────────────────
banner "pure Stage 3 two-epoch training + eval"
ARGS=(scripts/training/run_pure_stage3_two_epoch.py
  --run-dir "$RUN_DIR"
  --dataset "$TRAIN_DS"
  --dev-set "$DEV_SET"
  --test-set "$TEST_SET"
  --num-generations "${NUM_GENERATIONS:-8}"
  --learning-rate "${LEARNING_RATE:-3e-7}"
  --kl-beta "${KL_BETA:-0.15}"
  --temperature "${TEMPERATURE:-1.0}"
  --top-p "${TOP_P:-0.95}"
  --reward-policy "${REWARD_POLICY:-execution_aware_v3_2_dense}"
  --max-train-tasks "${MAX_TRAIN_TASKS:-0}"
  --dev-max-tasks "${DEV_MAX_TASKS:-0}"
  --test-max-tasks "${TEST_MAX_TASKS:-0}"
  --syntax-audit-verdict "$VERDICT"
  --syntax-audit-path "$AUDIT_JSON")
[ "${SKIP_PREFLIGHT:-0}" = "1" ] && ARGS+=(--skip-preflight)
[ "${SKIP_BASELINE_EVAL:-0}" = "1" ] && ARGS+=(--skip-baseline-eval)
[ "${SKIP_TEST_EVAL:-0}" = "1" ] && ARGS+=(--skip-test-eval)
[ "${RESUME:-0}" = "1" ] && ARGS+=(--resume)

"$PY" "${ARGS[@]}"

# ── 6. Artifact hashes ──────────────────────────────────────────────────────
banner "SHA-256 artefacts"
"$PY" - <<PY
import hashlib, json, os
from pathlib import Path
run = Path(r"$RUN_DIR")
paths = []
for p in sorted(run.rglob("*")):
    if p.is_file() and p.suffix in {".json", ".jsonl", ".md", ".safetensors"}:
        if p.stat().st_size > 200_000_000:
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        paths.append({"path": str(p.relative_to(run)), "sha256": h, "bytes": p.stat().st_size})
out = run / "artefact_sha256.json"
out.write_text(json.dumps({"n": len(paths), "files": paths}, indent=2), encoding="utf-8")
print(f"[pure-s3] wrote {out} ({len(paths)} files)")
PY

if [ -f "$RUN_DIR/SUCCESS" ]; then
  banner "SUCCESS: $RUN_DIR"
  exit 0
fi
echo FAILED > "$RUN_DIR/FAILED"
banner "FAILED: see $RUN_DIR/console.log"
exit 1
