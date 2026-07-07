#!/usr/bin/env bash
# NESTFUL v3.1 FIXED curriculum smoke pilot (post-audit).
#
# Runs the gated stage1→2→3 diagnostic pilot with the FIXED training stack:
#   * reward dispatch verified in parent AND DP workers (no silent fallback)
#   * per-position (between-completion) advantages — no turn-position artifact
#   * checkpoint eligibility guard (0-step adapters can never be crowned)
#   * regression guard ON, global best persisted across stages
#   * replay_ratio semantics (0.20 => 20% previous / 80% current)
#   * hard stage-advancement gates: stage N+1 runs ONLY if stage N passed
#   * stage4 NEVER runs; NESTFUL test split NEVER touched
#
# Usage:
#   REWARD_POLICY=execution_aware_v3_1_stepwise \
#     bash experiments/nestful_synthetic_curriculum_v3/scripts/pilot/run_v3_1_fixed_stage123_smoke.sh
#
#   # ablation: motif reward, stage1 only
#   REWARD_POLICY=execution_aware_v2_1_motif STAGES="1" \
#     bash experiments/nestful_synthetic_curriculum_v3/scripts/pilot/run_v3_1_fixed_stage123_smoke.sh
#
#   # preflight only (no training)
#   PREFLIGHT_ONLY=1 bash .../run_v3_1_fixed_stage123_smoke.sh

# Re-exec without CRLF when checked out on Windows.
if grep -q $'\r' "$0" 2>/dev/null; then
  exec /bin/bash <(sed 's/\r$//' "$0") "$@"
fi
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."

export CURRICULUM_VERSION="${CURRICULUM_VERSION:-v3_1}"
export ALLOW_PROTOTYPE_TRAINING="${ALLOW_PROTOTYPE_TRAINING:-1}"
export REGRESSION_GUARD="${REGRESSION_GUARD:-1}"
export REGRESSION_EARLY_ABORT="${REGRESSION_EARLY_ABORT:-1}"
export ALLOW_NO_REGRESSION_GUARD="${ALLOW_NO_REGRESSION_GUARD:-0}"
export ALLOW_STRICT_REWARD_FALLBACK="${ALLOW_STRICT_REWARD_FALLBACK:-0}"
export STAGE_GATES="${STAGE_GATES:-1}"

export REWARD_POLICY="${REWARD_POLICY:-execution_aware_v3_1_stepwise}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-1.0}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
export VAL_SUBSET_SIZE="${VAL_SUBSET_SIZE:-200}"
export MAX_EPOCHS_PER_STAGE="${MAX_EPOCHS_PER_STAGE:-1}"
export EVAL_EVERY_EPOCH="${EVAL_EVERY_EPOCH:-1}"

export USE_VLLM="${USE_VLLM:-1}"
export ROLLOUT_DP_GPUS="${ROLLOUT_DP_GPUS:-1,2,3}"
export DP_LEARNER_GPU="${DP_LEARNER_GPU:-0}"

export STAGES="${STAGES:-1 2 3}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

V3="experiments/nestful_synthetic_curriculum_v3"
PYTHON="${PYTHON:-python}"

if echo "$STAGES" | grep -qE '(^| )4( |$)'; then
  echo "[pilot] ERROR: stage 4 is FORBIDDEN in this smoke pilot" >&2
  exit 1
fi

echo "============================================================"
echo "NESTFUL v3.1 fixed curriculum smoke pilot"
echo "CURRICULUM_VERSION=${CURRICULUM_VERSION}"
echo "REWARD_POLICY=${REWARD_POLICY}          (configured)"
echo "STAGES=${STAGES}"
echo "MAX_EPOCHS_PER_STAGE=${MAX_EPOCHS_PER_STAGE}"
echo "NUM_GENERATIONS=${NUM_GENERATIONS}"
echo "TRAIN_TEMPERATURE=${TRAIN_TEMPERATURE}  EVAL_TEMPERATURE=${EVAL_TEMPERATURE}"
echo "VAL_SUBSET_SIZE=${VAL_SUBSET_SIZE}"
echo "REGRESSION_GUARD=${REGRESSION_GUARD}  STAGE_GATES=${STAGE_GATES}"
echo "ALLOW_STRICT_REWARD_FALLBACK=${ALLOW_STRICT_REWARD_FALLBACK}"
echo "ROLLOUT_DP_GPUS=${ROLLOUT_DP_GPUS}  DP_LEARNER_GPU=${DP_LEARNER_GPU}"
echo "PREFLIGHT_ONLY=${PREFLIGHT_ONLY}"
echo "NO stage4. NO test split. Diagnostic smoke pilot only."
echo "============================================================"

echo "[1/3] Running dataset audits + reward/training-stack preflight ..."

"$PYTHON" "$V3/scripts/final_dataset_audit_v3_1.py"
"$PYTHON" "$V3/scripts/analyze_dataset_uniqueness_v3_1.py"
"$PYTHON" "$V3/scripts/validate_question_trace_alignment_v3_1.py"
"$PYTHON" "$V3/scripts/validate_curriculum_integrity_v3_1.py"
"$PYTHON" "$V3/scripts/replay_synthetic_gold_traces_v3_1.py"
"$PYTHON" "$V3/scripts/run_tool_family_realism_v3_1.py"

"$PYTHON" "$V3/scripts/smoke_test_reward_dispatch_v3_1.py" \
  --reward-policy "${REWARD_POLICY}"

"$PYTHON" "$V3/scripts/preflight_reward_training_stack_v3_1.py" \
  --reward-policy "${REWARD_POLICY}" \
  --curriculum-version "${CURRICULUM_VERSION}"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "[pilot] PREFLIGHT_ONLY=1 — preflight passed, stopping before training."
  exit 0
fi

echo "[2/3] Starting gated curriculum smoke pilot ..."
echo "  Stage2 runs ONLY if stage1 passes gates; stage3 ONLY if stage2 passes."
echo "  (gates enforced INSIDE run_curriculum.sh via check_stage_gates.py)"

# Trainer overrides: reward policy + rollout sampling. run_curriculum_v3.sh
# appends the reward override automatically if missing; set it explicitly
# anyway so the config trail is unambiguous.
EXTRA_TRAIN_OVERRIDES_STR="${EXTRA_TRAIN_OVERRIDES_STR:-}"
EXTRA_TRAIN_OVERRIDES_STR="${EXTRA_TRAIN_OVERRIDES_STR} --override reward.train_policy=${REWARD_POLICY}"
EXTRA_TRAIN_OVERRIDES_STR="${EXTRA_TRAIN_OVERRIDES_STR} --override generation.num_generations=${NUM_GENERATIONS}"
EXTRA_TRAIN_OVERRIDES_STR="${EXTRA_TRAIN_OVERRIDES_STR} --override generation.temperature=${TRAIN_TEMPERATURE}"
export EXTRA_TRAIN_OVERRIDES_STR

set +e
bash "$V3/scripts/run_curriculum_v3.sh"
TRAIN_RC=$?
set -e

if [ "$TRAIN_RC" -eq 4 ]; then
  echo "[pilot] a stage FAILED its advancement gates — pilot stopped (expected behavior)."
elif [ "$TRAIN_RC" -ne 0 ]; then
  echo "[pilot] training exited with rc=$TRAIN_RC — inspect logs before re-running." >&2
fi

echo "[3/3] Building final smoke pilot report ..."

"$PYTHON" "$V3/scripts/build_fixed_reward_pilot_report.py" \
  --latest \
  --curriculum-version "${CURRICULUM_VERSION}" \
  --reward-policy "${REWARD_POLICY}" || true

echo "============================================================"
echo "Smoke pilot finished (training rc=$TRAIN_RC)."
echo "Check $V3/outputs/runs/<latest_run>/FIXED_REWARD_PILOT_REPORT.md"
echo "Do not use the test split."
echo "Do not claim transfer unless dev gates pass."
echo "============================================================"
exit "$TRAIN_RC"
