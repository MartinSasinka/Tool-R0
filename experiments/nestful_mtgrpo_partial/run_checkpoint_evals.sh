#!/usr/bin/env bash
# =============================================================================
#  run_checkpoint_evals.sh  (PARTIAL experiment)
#
#  Full NESTFUL final_eval (Direct + ReAct) pro vybrané curriculum checkpointy.
#  Checkpointy (uživatelský výběr):
#    partial_s1_e4  stage_1/adapter_epoch_4
#    partial_s2_e2  stage_2/adapter_epoch_2
#    partial_s3_e2  stage_3/adapter_epoch_2
#    partial_s4_e1  stage_4/adapter_epoch_1
#
#  Pro každý: paradigma direct + react  =>  8 evalů celkem.
#  Eval nastavení (stejně jako curriculum final_eval / minimal):
#    - vLLM tensor parallel na všech GPU (default TP=4)
#    - 1 rollout na task, temperature=0 (greedy)
#
#    tmux new -s partial_eval
#    cd /workspace/nestful_mtgrpo_partial
#    CUDA_VISIBLE_DEVICES=0,1,2,3 USE_VLLM=1 bash run_checkpoint_evals.sh
# =============================================================================
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PYTHON="${PYTHON:-python}"
USE_VLLM="${USE_VLLM:-true}"
GPU_UTIL="${GPU_UTIL:-0.85}"
VLLM_TP="${VLLM_TP:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0}"
NUM_EVAL_ROLLOUTS="${NUM_EVAL_ROLLOUTS:-1}"
DIRECT_MAX_NEW="${DIRECT_MAX_NEW:-2048}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="outputs/final_eval/_logs_${TS}"
mkdir -p "$LOG_ROOT"

CKPTS=(
  "partial_s1_e4:outputs/curriculum/stage_1/checkpoints/adapter_epoch_4"
  "partial_s2_e2:outputs/curriculum/stage_2/checkpoints/adapter_epoch_2"
  "partial_s3_e2:outputs/curriculum/stage_3/checkpoints/adapter_epoch_2"
  "partial_s4_e1:outputs/curriculum/stage_4/checkpoints/adapter_epoch_1"
)
PARADIGMS=(direct react)

echo "=============================================================="
echo " NESTFUL PARTIAL checkpoint evals (4 ckpts × direct + react = 8 runs)"
echo "   python      : $PYTHON"
echo "   use_vllm    : $USE_VLLM  (gpu_util=$GPU_UTIL  tp=$VLLM_TP)"
echo "   GPUs        : $CUDA_VISIBLE_DEVICES"
echo "   rollouts    : $NUM_EVAL_ROLLOUTS per task  temperature=$EVAL_TEMPERATURE"
echo "   direct toks : $DIRECT_MAX_NEW"
echo "   logy        : $LOG_ROOT"
echo "=============================================================="

if ! "$PYTHON" -c "import jsonlines, sklearn" >/dev/null 2>&1; then
  echo "[deps] doinstalovávám jsonlines + scikit-learn ..."
  "$PYTHON" -m pip install "jsonlines>=4.0" "scikit-learn>=1.3"
fi

run_one() {
  local label="$1" ckpt="$2" paradigm="$3"
  local out_dir="outputs/final_eval/${label}_${paradigm}"
  local log="${LOG_ROOT}/${label}_${paradigm}.log"
  mkdir -p "$out_dir"

  echo ""
  echo "[$(date +%H:%M:%S)] >>> ${label} / ${paradigm}"
  echo "    checkpoint : ${ckpt}"
  echo "    out_dir    : ${out_dir}"

  if [[ ! -f "${ckpt}/adapter_config.json" ]]; then
    echo "    SKIP: checkpoint nenalezen (chybí adapter_config.json)"
    return 2
  fi

  env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$PYTHON" run.py --mode final_eval \
    --checkpoint "$ckpt" \
    --override data.eval_paradigm="$paradigm" \
    --override experiment.output_dir="$out_dir" \
    --override hardware.use_vllm="$USE_VLLM" \
    --override hardware.vllm_gpu_memory_utilization="$GPU_UTIL" \
    --override hardware.vllm_tensor_parallel_size="$VLLM_TP" \
    --override data.num_eval_rollouts="$NUM_EVAL_ROLLOUTS" \
    --override generation.temperature="$EVAL_TEMPERATURE" \
    --override generation.max_new_tokens_direct="$DIRECT_MAX_NEW" \
    2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}

  if [[ $rc -ne 0 ]]; then
    echo "    FAILED (rc=$rc)"
  elif [[ -f "${out_dir}/metrics_official.json" ]]; then
    echo "    OK — metrics_official.json"
  fi
  return $rc
}

declare -a SUMMARY=()
for entry in "${CKPTS[@]}"; do
  label="${entry%%:*}"
  ckpt="${entry#*:}"
  for paradigm in "${PARADIGMS[@]}"; do
    if run_one "$label" "$ckpt" "$paradigm"; then
      SUMMARY+=("OK    ${label}_${paradigm}")
    else
      SUMMARY+=("FAIL  ${label}_${paradigm}")
    fi
  done
done

echo ""
echo "=============================================================="
echo " SOUHRN"
for s in "${SUMMARY[@]}"; do echo "   $s"; done
echo " Výstupy: outputs/final_eval/<label>_<paradigm>/metrics_official.json"
echo "=============================================================="
