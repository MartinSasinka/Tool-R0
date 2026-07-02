#!/usr/bin/env bash
# =============================================================================
#  install_deps.sh — instalace závislostí na čistém RunPod podu.
#
#  Předpoklad: torch je UŽ nainstalovaný (typicky 2.8.0+cu128 na H100 image).
#  Tento skript torch NEPŘEINSTALOVÁVÁ — jen doplní zbytek + vLLM kompatibilní
#  s torch 2.8 / CUDA 12.8, a na konci ověří, že CUDA zůstala funkční.
#
#  Použití:   bash install_deps.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PY="${PYTHON:-python}"
VLLM_VERSION="${VLLM_VERSION:-0.11.1}"   # 0.11.x = CUDA 12.8 + torch 2.8 (sedí na cu128)
TORCH_IDX="https://download.pytorch.org/whl/cu128"

echo "=============================================================="
echo " Instalace závislostí (torch ponecháváme tak, jak je)"
echo "=============================================================="

echo "[0/5] stávající torch:"
"$PY" -c "import torch; print('   torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "   CHYBA: torch není importovatelný — nejdřív vyřeš torch."; exit 1; }

# --- core deps (BEZ torch; transformers dost nový kvůli Qwen3-4B-Instruct-2507) ---
echo "[1/5] core deps ..."
"$PY" -m pip install --no-cache-dir -U \
  "transformers>=4.53" \
  "accelerate>=0.33" \
  "datasets>=2.20" \
  "peft>=0.12" \
  "trl>=0.9" \
  "bitsandbytes>=0.43" \
  "pyyaml>=6.0" "tqdm>=4.66" "numpy>=1.24" "safetensors>=0.4"

# --- official NESTFUL scorer deps (uvozovky! '>=1.3' bez nich vyrobí soubor) ---
echo "[2/5] scorer deps (scikit-learn, jsonlines) ..."
"$PY" -m pip install --no-cache-dir -U "scikit-learn>=1.3" "jsonlines>=4.0"

# --- logging + HF hub fast download ------------------------------------------
# RunPod images often export HF_HUB_ENABLE_HF_TRANSFER=1 but omit hf_transfer;
# model download then crashes unless we install it OR unset that env var.
# wandb is optional at runtime but expected when WANDB_API_KEY is set.
echo "[2b/5] wandb + hf_transfer ..."
"$PY" -m pip install --no-cache-dir -U "wandb>=0.16" "hf_transfer>=0.1"

# --- vLLM kompatibilní s torch 2.8 / cu128 -----------------------------------
# extra-index-url cu128 zajistí, že pokud by pip sahal na torch, vezme GPU wheel,
# ne CPU build z PyPI (klasická past, která tiše zabije CUDA).
echo "[3/5] vLLM ${VLLM_VERSION} (cu128) ..."
"$PY" -m pip install --no-cache-dir "vllm==${VLLM_VERSION}" --extra-index-url "$TORCH_IDX"

# --- pojistka: vLLM mohl přetáhnout torch; pokud zmizela CUDA, vrať cu128 build ---
echo "[4/5] kontrola, že CUDA přežila instalaci vLLM ..."
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "   !! CUDA zmizela — vLLM přepsal torch na CPU. Vracím torch 2.8.0+cu128 ..."
  "$PY" -m pip install --no-cache-dir --force-reinstall "torch==2.8.0" --index-url "$TORCH_IDX"
fi

# --- finální ověření ---------------------------------------------------------
echo "[5/5] finální ověření ..."
"$PY" - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "CUDA NENÍ dostupná!"
print("  torch        ", torch.__version__, "| cuda", torch.version.cuda, "| OK")
import transformers, peft, trl, accelerate, bitsandbytes, sklearn, jsonlines
import wandb, hf_transfer  # noqa: F401
print("  transformers ", transformers.__version__)
print("  peft         ", peft.__version__)
print("  trl          ", trl.__version__)
print("  bitsandbytes ", bitsandbytes.__version__)
print("  wandb        ", wandb.__version__)
print("  hf_transfer  ", hf_transfer.__version__)
from vllm import LLM  # noqa: F401
import vllm
print("  vllm         ", vllm.__version__, "| import OK")
print("  GPU          ", torch.cuda.get_device_name(0))
PYEOF

echo "=============================================================="
echo " HOTOVO. Teď můžeš spustit curriculum / eval."
echo "=============================================================="
