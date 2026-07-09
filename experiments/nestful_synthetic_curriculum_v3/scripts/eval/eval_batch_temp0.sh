#!/usr/bin/env bash
# Deterministic temp0 NESTFUL eval batch (P0 remediation).
# Thin wrapper over run_eval_batch.py — baseline cell is MANDATORY, official scorer
# verified per cell, one flat batch dir, manifest + BATCH_REPORT.md emitted.
#
# Usage (from repo root):
#   CELLS="baseline,s3_e1=<adapter_dir>,s3_e2=<adapter_dir>" \
#   DATASET=nestful_test \
#   bash experiments/nestful_synthetic_curriculum_v3/scripts/eval/eval_batch_temp0.sh
#
# Env vars:
#   CELLS       (required) comma list: 'baseline' and/or 'name=checkpoint_dir'
#   DATASET     nestful_test | nestful_full | nestful_dev | path   (default nestful_test)
#   BATCH_NAME  batch dir prefix                                    (default eval_batch)
#   MAX_TASKS   smoke-test cap (empty = full dataset)
#   PARALLEL    1 = run cells concurrently, one per GPU             (default 0)
#   GPUS        GPU ids for PARALLEL=1, space or comma separated    (default "0")
#   MAX_PARALLEL cap on concurrent cells                            (default #GPUS)
#   DRY_RUN     1 = print commands only, run nothing                (default 0)
#   EXTRA_ARGS  passed through to run_eval_batch.py verbatim
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

CELLS="${CELLS:?Set CELLS, e.g. CELLS='baseline,s3_e1=outputs/.../adapter_epoch_1'}"
DATASET="${DATASET:-nestful_test}"
BATCH_NAME="${BATCH_NAME:-eval_batch}"
MAX_TASKS="${MAX_TASKS:-}"
PARALLEL="${PARALLEL:-0}"
GPUS="${GPUS:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "[eval_batch_temp0] repo root : $REPO_ROOT"
echo "[eval_batch_temp0] cells     : $CELLS"
echo "[eval_batch_temp0] dataset   : $DATASET"
echo "[eval_batch_temp0] batch name: $BATCH_NAME"

ARGS=( --cells "$CELLS" --dataset "$DATASET" --batch-name "$BATCH_NAME" --temperature 0.0 )
[[ -n "$MAX_TASKS" ]] && ARGS+=( --max-tasks "$MAX_TASKS" )
if [[ "$PARALLEL" == "1" ]]; then
  ARGS+=( --parallel --gpus "${GPUS// /,}" )
  [[ -n "$MAX_PARALLEL" ]] && ARGS+=( --max-parallel "$MAX_PARALLEL" )
fi
[[ "$DRY_RUN" == "1" ]] && ARGS+=( --dry-run )
[[ -n "$EXTRA_ARGS" ]] && ARGS+=( $EXTRA_ARGS )

exec python "$SCRIPT_DIR/run_eval_batch.py" "${ARGS[@]}"
