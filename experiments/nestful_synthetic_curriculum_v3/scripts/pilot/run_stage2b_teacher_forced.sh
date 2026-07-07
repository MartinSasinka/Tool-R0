#!/usr/bin/env bash
# Stage2b — teacher-forced continuation training (follow-up to a Stage1+2 pilot).
#
# Motivation (see the Stage2 dead_group_rate diagnosis): after Stage1 (1-call
# tasks), the policy develops a strong "stop after the first call" prior. On
# Stage2 (2-call) tasks it then systematically emits `too_few_calls`, which
# caps the reward THE SAME WAY for every one of the num_generations rollouts
# in a group -> zero within-group variance -> dead group -> no GRPO gradient
# for exactly the behavior Stage2 is supposed to teach.
#
# Stage2b removes that degree of freedom: the first N gold calls (default 1)
# + their REAL observations are teacher-forced (replayed, not generated) into
# every training episode, so the policy only has to generate the
# CONTINUATION. Forced tokens get NO gradient (see
# rollout.build_teacher_forced_prefix / vllm_dp_pool.run_episode_collect —
# r_seq is sliced to drop the forced-prefix entries before pairing with
# turn_token_ids). Evaluation is COMPLETELY UNCHANGED (full free generation,
# no forcing — rollout.run_episode never forces), so Stage2b's effect on real
# task-solving ability is measured honestly, not just on the forced setup.
#
# This is a SHORT, TARGETED follow-up — it does NOT re-run stage1 or stage3,
# and it does NOT replace the main curriculum. Treat it as a diagnostic /
# ablation: does removing "when to stop" from the policy's degrees of freedom
# fix Stage2's dead-group problem and improve real (non-forced) dev Win?
#
# Usage:
#   CHECKPOINT_IN=<stage2a adapter dir, e.g. .../stage_2/checkpoints/adapter_epoch_1> \
#   BEFORE_RUN_DIR=<stage2a's outputs/runs/<id> dir, for the before/after report> \
#     bash experiments/nestful_synthetic_curriculum_v3/scripts/pilot/run_stage2b_teacher_forced.sh
#
# Optional knobs (same meaning as run_v3_1_fixed_stage123_smoke.sh):
#   TEACHER_FORCED_PREFIX_CALLS (default 1), REWARD_POLICY, NUM_GENERATIONS,
#   TRAIN_TEMPERATURE, EVAL_TEMPERATURE, VAL_SUBSET_SIZE, MAX_EPOCHS_PER_STAGE,
#   PREV_STAGE_DEV_WIN (stage2a's dev Win, for a stricter stage gate check),
#   PREFLIGHT_ONLY=1 (stop before training).

# Re-exec without CRLF when checked out on Windows.
if grep -q $'\r' "$0" 2>/dev/null; then
  exec /bin/bash <(sed 's/\r$//' "$0") "$@"
fi
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."

if [ -z "${CHECKPOINT_IN:-}" ]; then
  echo "[stage2b] ERROR: CHECKPOINT_IN is required — point it at the Stage2a" >&2
  echo "  checkpoint to continue training from, e.g." >&2
  echo "  CHECKPOINT_IN=experiments/nestful_synthetic_curriculum_v3/outputs/runs/<id>/stage_2/checkpoints/adapter_epoch_1" >&2
  exit 1
fi
if [ ! -f "$CHECKPOINT_IN/adapter_config.json" ]; then
  echo "[stage2b] ERROR: CHECKPOINT_IN is not a valid LoRA adapter dir: $CHECKPOINT_IN" >&2
  exit 1
fi

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

# The core intervention: force only the FIRST gold call by default (2a already
# does that turn well; the goal is training the SECOND, i.e. the continuation).
export TEACHER_FORCED_PREFIX_CALLS="${TEACHER_FORCED_PREFIX_CALLS:-1}"
export STAGES="2"
export INIT_FROM="checkpoint"
export CHECKPOINT_IN
export PREV_STAGE_DEV_WIN="${PREV_STAGE_DEV_WIN:-}"

# Own OUTPUT_ROOT — never reuse the Stage2a run dir. Forcing changes what
# "stage 2 training" means, so this must stay a separate, auditable run.
if [ -z "${OUTPUT_ROOT:-}" ]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  export OUTPUT_ROOT="experiments/nestful_synthetic_curriculum_v3/outputs/runs/${TS}_stage2b_teacher_forced"
fi

V3="experiments/nestful_synthetic_curriculum_v3"
PYTHON="${PYTHON:-python}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

echo "============================================================"
echo "NESTFUL Stage2b — teacher-forced continuation training"
echo "CHECKPOINT_IN (stage2a)      = ${CHECKPOINT_IN}"
echo "BEFORE_RUN_DIR (for report)  = ${BEFORE_RUN_DIR:-<none>}"
echo "TEACHER_FORCED_PREFIX_CALLS  = ${TEACHER_FORCED_PREFIX_CALLS}"
echo "REWARD_POLICY                = ${REWARD_POLICY}"
echo "NUM_GENERATIONS              = ${NUM_GENERATIONS}"
echo "TRAIN_TEMPERATURE / EVAL_TEMPERATURE = ${TRAIN_TEMPERATURE} / ${EVAL_TEMPERATURE}"
echo "REGRESSION_GUARD=${REGRESSION_GUARD}  STAGE_GATES=${STAGE_GATES}"
echo "OUTPUT_ROOT                  = ${OUTPUT_ROOT}"
echo "Eval is ALWAYS full free generation (no forcing) — numbers below reflect"
echo "REAL, non-forced continuation ability."
echo "============================================================"

echo "[1/3] Running dataset audits + reward/training-stack preflight ..."
"$PYTHON" "$V3/scripts/final_dataset_audit_v3_1.py"
"$PYTHON" "$V3/scripts/analyze_dataset_uniqueness_v3_1.py"
"$PYTHON" "$V3/scripts/validate_question_trace_alignment_v3_1.py"
"$PYTHON" "$V3/scripts/validate_curriculum_integrity_v3_1.py"
"$PYTHON" "$V3/scripts/replay_synthetic_gold_traces_v3_1.py"
"$PYTHON" "$V3/scripts/run_tool_family_realism_v3_1.py"
"$PYTHON" "$V3/scripts/smoke_test_reward_dispatch_v3_1.py" --reward-policy "${REWARD_POLICY}"
"$PYTHON" "$V3/scripts/preflight_reward_training_stack_v3_1.py" \
  --reward-policy "${REWARD_POLICY}" --curriculum-version "${CURRICULUM_VERSION}"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "[stage2b] PREFLIGHT_ONLY=1 — preflight passed, stopping before training."
  exit 0
fi

echo "[2/3] Training Stage2 WITH teacher-forced continuation ..."
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
  echo "[stage2b] stage gates FAILED — Stage2b did not produce an eligible checkpoint."
elif [ "$TRAIN_RC" -ne 0 ]; then
  echo "[stage2b] training exited with rc=$TRAIN_RC — inspect logs before drawing conclusions." >&2
fi

echo "[3/3] Building before/after comparison report ..."
REPORT_ARGS=(--after-run-dir "$OUTPUT_ROOT")
if [ -n "${BEFORE_RUN_DIR:-}" ]; then
  REPORT_ARGS+=(--before-run-dir "$BEFORE_RUN_DIR")
fi
"$PYTHON" "$V3/scripts/build_stage2b_teacher_forced_report.py" "${REPORT_ARGS[@]}" || true

echo "============================================================"
echo "Stage2b finished (training rc=$TRAIN_RC)."
echo "Report: $OUTPUT_ROOT/STAGE2B_TEACHER_FORCED_REPORT.md"
echo "Eval used FULL free generation (no forcing) — numbers are honest."
echo "Do not use the test split. Do not claim transfer beyond the dev subset."
echo "============================================================"
exit "$TRAIN_RC"
