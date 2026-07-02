#!/usr/bin/env bash
# run_all.sh — one entry point to (a) resume curriculum training from ANY
# checkpoint and (b) run the FINAL evaluation on the whole NESTFUL benchmark in
# both paradigms (ReAct + Direct). Thin, opinionated wrapper around
# run_curriculum.sh so you don't have to remember the env-var combinations.
#
# It inherits every fix in this folder: tool-observation truncation, the vLLM /
# HF prompt-overflow guards (no more crashes on long histories) and the 4×GPU
# split (learner on 1 GPU, eval tensor-parallel across all GPUs).
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  MODES (env MODE=...):                                                   │
# │    resume   resume curriculum training from CHECKPOINT_IN, evaluating    │
# │             after every epoch. Pick up where a run stopped.              │
# │    final    final_eval ONLY on the full NESTFUL benchmark (1861 tasks),  │
# │             ReAct + Direct, TP across all GPUs, graceful + incremental.  │
# │    all       resume training, then final_eval (default).                 │
# └─────────────────────────────────────────────────────────────────────────┘
#
# Common env (everything run_curriculum.sh accepts also works here):
#   CHECKPOINT_IN   adapter/checkpoint to resume from / evaluate (required for
#                   resume + final; for `all` it can be empty to train from base)
#   STAGES          stages to (resume) train, e.g. "2 3 4"          (default "1 2 3 4")
#   START_EPOCH     epoch to resume the FIRST stage at               (default 1)
#   PARADIGMS       final eval paradigms: "react direct" | react | direct (default both)
#   PROFILE         pilot | curriculum                               (default curriculum)
#   USE_VLLM        1 to use vLLM (strongly recommended)             (default 1)
#   TRAIN_GPUS / EVAL_GPUS / VLLM_TP_TRAIN / VLLM_TP_EVAL  (see run_curriculum.sh)
#
# Examples:
#   # Resume training from a stage-2 epoch-2 checkpoint across stages 2-4, then final eval
#   CHECKPOINT_IN=outputs/curriculum/stage_2/checkpoints/adapter_epoch_2 \
#     STAGES="2 3 4" START_EPOCH=3 USE_VLLM=1 bash run_all.sh
#
#   # Final eval only, both paradigms, on 4 GPUs
#   MODE=final CHECKPOINT_IN=outputs/curriculum/stage_4/checkpoints/adapter_epoch_4 \
#     USE_VLLM=1 bash run_all.sh
#
#   # Dry-run to inspect the exact commands
#   DRY_RUN=1 MODE=all CHECKPOINT_IN=... bash run_all.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Allow a sibling experiment (partial) to reuse this script by overriding RUN_PY/
# CONFIG/OUTPUT_ROOT before exec (same pattern as run_curriculum.sh).
RUNNER="${CURRICULUM_RUNNER:-$ROOT/run_curriculum.sh}"

MODE="${MODE:-all}"
PARADIGMS="${PARADIGMS:-both}"
USE_VLLM="${USE_VLLM:-1}"
export USE_VLLM

# Map PARADIGMS -> FINAL_EVAL_PARADIGM understood by run_curriculum.sh.
case "$PARADIGMS" in
    both|"react direct"|"direct react") _FEP="both" ;;
    react)  _FEP="react" ;;
    direct) _FEP="direct" ;;
    *)      _FEP="$PARADIGMS" ;;
esac

echo "════════════════════════════════════════════════════════════════"
echo "  nestful_mtgrpo — run_all"
echo "    MODE          : $MODE"
echo "    CHECKPOINT_IN : ${CHECKPOINT_IN:-<none>}"
echo "    STAGES        : ${STAGES:-<default>}"
echo "    PARADIGMS     : $_FEP"
echo "    runner        : $RUNNER"
echo "════════════════════════════════════════════════════════════════"

case "$MODE" in
    resume)
        # Resume training only (eval after each epoch is built into the loop).
        PROFILE="${PROFILE:-curriculum}" \
        USE_VLLM="$USE_VLLM" \
        RUN_FINAL_EVAL=0 \
        exec bash "$RUNNER"
        ;;

    final)
        # Final eval ONLY on the full NESTFUL benchmark, both paradigms.
        if [ -z "${CHECKPOINT_IN:-}" ]; then
            echo "[run_all] ERROR: MODE=final requires CHECKPOINT_IN=<adapter path>" >&2
            exit 1
        fi
        PROFILE="${PROFILE:-curriculum}" \
        USE_VLLM="$USE_VLLM" \
        ONLY_FINAL_EVAL=1 \
        RUN_FINAL_EVAL=1 \
        FINAL_EVAL_PARADIGM="$_FEP" \
        FINAL_CHECKPOINT="${FINAL_CHECKPOINT:-$CHECKPOINT_IN}" \
        MAX_EVAL_TASKS="${MAX_EVAL_TASKS:-}" \
        exec bash "$RUNNER"
        ;;

    all)
        # Resume training, then final eval on the full benchmark.
        PROFILE="${PROFILE:-curriculum}" \
        USE_VLLM="$USE_VLLM" \
        RUN_FINAL_EVAL=1 \
        FINAL_EVAL_PARADIGM="$_FEP" \
        exec bash "$RUNNER"
        ;;

    *)
        echo "[run_all] ERROR: unknown MODE='$MODE'. Use: resume | final | all" >&2
        exit 1
        ;;
esac
