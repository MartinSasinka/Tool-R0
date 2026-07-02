#!/usr/bin/env bash
# =============================================================================
#  run_checkpoint_evals.sh
#  Spustí Direct + ReAct NESTFUL eval pro vybrané curriculum checkpointy.
#
#  Checkpointy (podle audit mapy train-s{N}-e{E}):
#    - train-s1-e4  ->  outputs/curriculum/stage_1/checkpoints/adapter_epoch_4
#    - train-s2-e4  ->  outputs/curriculum/stage_2/checkpoints/adapter_epoch_4
#
#  Pro každý: paradigma `direct` i `react`  =>  4 evaly celkem.
#  Každý eval má vlastní výstupní složku v outputs/final_eval/<label>_<paradigm>/.
#
#  Odolnost:
#    - Selhání jednoho evalu NEUKONČÍ ostatní (běží dál).
#    - Predikce se ukládají PRŮBĚŽNĚ:
#        direct -> direct_predictions.jsonl (hned po generaci)
#        react  -> final_eval_predictions.partial.jsonl (po každém vzorku)
#      Takže ani pád scoringu po hodině generování nezahodí práci.
#    - Win Rate se na Linuxu (RunPod) počítá automaticky (SIGALRM dostupné).
#
#  Použití (z této složky):   bash run_checkpoint_evals.sh
#  Viz instrukce pro tmux na konci výpisu / v chatu.
# =============================================================================
set -uo pipefail

# --- vždy běž z adresáře skriptu (kde leží run.py) ---------------------------
cd "$(dirname "$(readlink -f "$0")")"

# --- konfigurace (lze přepsat env proměnnou) ---------------------------------
PYTHON="${PYTHON:-python}"
USE_VLLM="${USE_VLLM:-true}"
GPU_UTIL="${GPU_UTIL:-0.85}"
DIRECT_MAX_NEW="${DIRECT_MAX_NEW:-2048}"   # 1024 default je u delších sekvencí těsný -> clipping -> parse fail
TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="outputs/final_eval/_logs_${TS}"
mkdir -p "$LOG_ROOT"

# label:cesta_k_adaptéru
CKPTS=(
  "stage1_epoch4:outputs/curriculum/stage_1/checkpoints/adapter_epoch_4"
  "stage2_epoch4:outputs/curriculum/stage_2/checkpoints/adapter_epoch_4"
)
PARADIGMS=(direct react)

echo "=============================================================="
echo " NESTFUL checkpoint evals"
echo "   python      : $PYTHON"
echo "   use_vllm    : $USE_VLLM  (gpu_util=$GPU_UTIL)"
echo "   direct toks : $DIRECT_MAX_NEW"
echo "   logy        : $LOG_ROOT"
echo "=============================================================="

# --- závislosti oficiálního scoreru (s uvozovkami! '>=1.3' bez nich vyrobí soubor) ---
if ! "$PYTHON" -c "import jsonlines, sklearn" >/dev/null 2>&1; then
  echo "[deps] doinstalovávám jsonlines + scikit-learn ..."
  "$PYTHON" -m pip install "jsonlines>=4.0" "scikit-learn>=1.3"
fi

# --- jeden eval --------------------------------------------------------------
run_one() {
  local label="$1" ckpt="$2" paradigm="$3"
  local out_dir="outputs/final_eval/${label}_${paradigm}"
  local log="${LOG_ROOT}/${label}_${paradigm}.log"
  mkdir -p "$out_dir"

  echo ""
  echo "[$(date +%H:%M:%S)] >>> ${label} / ${paradigm}"
  echo "    checkpoint : ${ckpt}"
  echo "    out_dir    : ${out_dir}"
  echo "    log        : ${log}"

  if [[ ! -f "${ckpt}/adapter_config.json" ]]; then
    echo "    SKIP: checkpoint nenalezen (chybí adapter_config.json)"
    return 2
  fi

  "$PYTHON" run.py --mode final_eval \
    --checkpoint "$ckpt" \
    --override data.eval_paradigm="$paradigm" \
    --override experiment.output_dir="$out_dir" \
    --override hardware.use_vllm="$USE_VLLM" \
    --override hardware.vllm_gpu_memory_utilization="$GPU_UTIL" \
    --override generation.max_new_tokens_direct="$DIRECT_MAX_NEW" \
    2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}

  if [[ $rc -ne 0 ]]; then
    echo "    FAILED (rc=$rc) — predikce zůstaly v ${out_dir} (lze re-scórovat, viz níže)"
  elif [[ -f "${out_dir}/metrics_official.json" ]]; then
    echo "    OK — metrics_official.json hotové"
  else
    echo "    DONE bez metrics_official.json — re-scóruj z uložených predikcí (viz níže)"
  fi
  return $rc
}

# --- hlavní smyčka: nikdy nepřeruš celou dávku -------------------------------
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

# --- souhrn ------------------------------------------------------------------
echo ""
echo "=============================================================="
echo " SOUHRN"
for s in "${SUMMARY[@]}"; do echo "   $s"; done
echo "--------------------------------------------------------------"
echo " Výstupy: outputs/final_eval/<label>_<paradigm>/metrics_official.json"
echo " Logy:    ${LOG_ROOT}/"
echo ""
echo " Re-score (kdyby metrics_official.json chybělo):"
echo "   direct: python nestful_official_score.py --direct-predictions \\"
echo "             outputs/final_eval/<label>_direct/direct_predictions.jsonl \\"
echo "             --out outputs/final_eval/<label>_direct/metrics_official.json"
echo "   react : python nestful_official_score.py --direct-predictions \\"
echo "             outputs/final_eval/<label>_react/final_eval_predictions.partial.jsonl \\"
echo "             --out outputs/final_eval/<label>_react/metrics_official.json"
echo "=============================================================="
