#!/usr/bin/env bash
# =============================================================================
#  Pilot3 — GRPO train on frozen train tasks, then NESTFUL-500 eval only
#
#  - Train: first N rows of data/train_grpo_pilot3.jsonl (default N=600)
#  - Eval:  ONLY the new checkpoint on nestful_diagnostic_500
#  - Skips: signal probe, C0 eval, held-out eval, NESTFUL-1661
#
#  Usage (repo root on RunPod):
#      export HF_TOKEN=...
#      bash experiments/targeted_tool_data_factory/runpod_bundle_pilot3/run_train_nestful500_4gpu.sh
#      bash .../run_train_nestful500_4gpu.sh --train-n 200   # smaller train
#
#  Flags: --dry-run | --resume | --stage train|eval | --train-n N
# =============================================================================
set -Eeuo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FACTORY="$(cd "$BUNDLE/.." && pwd)"
P2="$(cd "$BUNDLE/../runpod_bundle_pilot2" && pwd)"
EXPERIMENTS="$(cd "$FACTORY/.." && pwd)"
V3="$EXPERIMENTS/nestful_synthetic_curriculum_v3"
PARTIAL="$EXPERIMENTS/nestful_mtgrpo_partial"
REPO="$(cd "$EXPERIMENTS/.." && pwd)"
PY="${PYTHON:-python3}"

if command -v cygpath >/dev/null 2>&1; then
  BUNDLE="$(cygpath -m "$BUNDLE")"
  FACTORY="$(cygpath -m "$FACTORY")"
  P2="$(cygpath -m "$P2")"
  EXPERIMENTS="$(cygpath -m "$EXPERIMENTS")"
  V3="$(cygpath -m "$V3")"
  PARTIAL="$(cygpath -m "$PARTIAL")"
  REPO="$(cygpath -m "$REPO")"
fi

DRY_RUN=0
RESUME_FLAG=""
ONLY_STAGE=""
TRAIN_N=600
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)  DRY_RUN=1; shift ;;
    --resume)   RESUME_FLAG="--resume"; shift ;;
    --stage)    ONLY_STAGE="$2"; shift 2 ;;
    --train-n)  TRAIN_N="$2"; shift 2 ;;
    *) echo "[p3train] unknown arg: $1" >&2; exit 1 ;;
  esac
done

SEED="${SEED:-20260727}"
REWARD_ARM="${REWARD_ARM:-A4_GATED_VERIFIABLE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$V3/outputs/runs}"
RESULTS="${RESULTS:-$FACTORY/outputs/runpod_pilot3/train_nestful500}"
RUN_ID="${RUN_ID:-pilot3_D1_seed${SEED}_n${TRAIN_N}}"
TRAIN_FULL="$BUNDLE/data/train_grpo_pilot3.jsonl"
DIAG="$P2/data/nestful_diagnostic_500.jsonl"
EVAL_GPU="${EVAL_GPU:-0}"

export USE_VLLM="${USE_VLLM:-1}"
export ROLLOUT_DP_GPUS="${ROLLOUT_DP_GPUS:-1,2,3}"
export CANARY_TRAJ_LOG=1
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export SYNTHETIC_TOOLS_DIR="$FACTORY/trainer_adapter"

mkdir -p "$RESULTS"
LOG="$RESULTS/run_train_nestful500_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

banner() {
  echo ""
  echo "=================================================================="
  echo "[p3train] $1"
  echo "=================================================================="
}
run() {
  echo "+ $*"
  if [ "$DRY_RUN" = "0" ]; then "$@"; fi
}
want() { [ -z "$ONLY_STAGE" ] || [ "$ONLY_STAGE" = "$1" ]; }

cd "$REPO"

echo "[p3train] run_id:  $RUN_ID"
echo "[p3train] train_n: $TRAIN_N  (full frozen train file = 600)"
echo "[p3train] reward:  $REWARD_ARM"
echo "[p3train] eval:    NESTFUL-500 only on GPU$EVAL_GPU (no C0)"

banner "1/4  GPU check"
DRY_RUN="$DRY_RUN" "$PY" - <<'PYEOF'
import os, sys
try:
    import torch
except ImportError:
    sys.exit(0 if os.environ.get("DRY_RUN") == "1" else 1)
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name} {p.total_memory/1e9:.1f} GB")
if n < 4 and os.environ.get("DRY_RUN") != "1":
    raise SystemExit("[p3train] ABORT: 4 GPUs required")
print(f"[p3train] visible GPUs: {n}")
PYEOF

banner "2/4  hashes + train subset + gold replay"
"$PY" "$BUNDLE/verify_hashes.py" --manifest "$BUNDLE/MANIFEST.sha256.json"
[ -f "$TRAIN_FULL" ] || { echo "[p3train] ABORT: missing $TRAIN_FULL" >&2; exit 1; }
[ -f "$DIAG" ] || { echo "[p3train] ABORT: missing $DIAG" >&2; exit 1; }

TRAIN_SUBSET="$RESULTS/train_subset_${TRAIN_N}.jsonl"
"$PY" - "$TRAIN_FULL" "$TRAIN_SUBSET" "$TRAIN_N" <<'PYEOF'
import sys
from pathlib import Path
src, dst, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
if not (1 <= n <= len(rows)):
    raise SystemExit(f"[p3train] ABORT: --train-n {n} out of range 1..{len(rows)}")
dst.write_text("\n".join(rows[:n]) + "\n", encoding="utf-8")
print(f"[p3train] wrote {n}/{len(rows)} -> {dst}")
PYEOF

run "$PY" "$FACTORY/trainer_adapter/preflight_gold_replay.py" \
  --data "$TRAIN_SUBSET" --expect "$TRAIN_N" \
  --report "$RESULTS/preflight_gold_replay.json"

if want train; then
banner "3/4  GRPO train (GPU0 learner / GPU1-3 rollouts; skip built-in eval)"
run "$PY" "$V3/scripts/ablation/run_reward_ablation.py" \
  --round 3 \
  --reward-arm "$REWARD_ARM" \
  --seed "$SEED" \
  --train-subset "$TRAIN_SUBSET" \
  --expected-rows "$TRAIN_N" \
  --skip-eval --skip-c0-eval \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID" \
  --wandb-project "${WANDB_PROJECT:-ttdf-pilot3}" \
  --wandb-group "pilot3_train_nestful500" \
  $RESUME_FLAG
fi

if want eval; then
banner "4/4  eval new checkpoint on NESTFUL-500 only (no C0)"
OUT="$RESULTS/eval/D1_nestful500"
mkdir -p "$OUT"
export P3_CKPT_ROOT="$OUTPUT_ROOT/$RUN_ID"
export P3_DIAG="$DIAG"
export P3_OUT="$OUT"
export P3_DRY="$DRY_RUN"
export P3_EVAL_GPU="$EVAL_GPU"
export P3_RUN_PY="$V3/run.py"
export P3_CONFIG="$PARTIAL/config.yaml"
run "$PY" - <<'PYEOF'
import json, os, subprocess, sys
from pathlib import Path

run_dir = Path(os.environ["P3_CKPT_ROOT"])
diag = Path(os.environ["P3_DIAG"])
out = Path(os.environ["P3_OUT"])
dry = os.environ["P3_DRY"] == "1"
run_py = Path(os.environ["P3_RUN_PY"])
config = Path(os.environ["P3_CONFIG"])

def find_checkpoint(rd: Path):
    for cand in (rd / "checkpoints" / "final", rd / "final",
                 rd / "checkpoints" / "FINAL"):
        if (cand / "adapter_config.json").is_file():
            return cand
    hits = sorted(rd.rglob("adapter_config.json"))
    return hits[-1].parent if hits else None

ck = find_checkpoint(run_dir)
if ck is None and not dry:
    raise SystemExit(f"[p3train] ABORT: no final adapter under {run_dir}")
cmd = [
    sys.executable, str(run_py), "--mode", "final_eval",
    "--config", str(config),
    "--override", f"experiment.output_dir={out}",
    "--override", f"paths.full_nestful_jsonl={diag}",
    "--override", "generation.temperature=0.0",
    "--override", "generation.top_p=1.0",
    "--override", "data.num_eval_rollouts=1",
    "--override", "data.eval_paradigm=react",
]
if ck is not None:
    cmd += ["--checkpoint", str(ck)]
print("[p3train] +", " ".join(cmd))
(out / "eval_manifest.json").write_text(
    json.dumps({"checkpoint": str(ck), "diagnostic": str(diag), "arm": "D1"},
               indent=2),
    encoding="utf-8")
if dry:
    raise SystemExit(0)
env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = os.environ.get("P3_EVAL_GPU", "0")
env.pop("SYNTHETIC_TOOLS_DIR", None)
with (out / "eval.log").open("w", encoding="utf-8") as log:
    rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
print(f"[p3train] nestful500 rc={rc}")
raise SystemExit(rc)
PYEOF
fi

banner "done"
echo "[p3train] ckpt:    $OUTPUT_ROOT/$RUN_ID"
echo "[p3train] results: $RESULTS/eval/D1_nestful500"
echo "[p3train] C0 was NOT evaluated — compare to your previous C0 nestful500 numbers"
