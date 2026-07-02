#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-time environment setup for running Tool-R0 curriculum on a rented
# 80GB node (RunPod / Lambda / Vast — H100 SXM or A100-80GB).
#
# Versions are pinned to the set proven working on the DGX (torch 2.9.0+cu128).
# flash-attn is intentionally NOT installed: the upstream wheel for torch 2.9.0
# has an undefined symbol bug (Dao-AILab#2002). The trainer falls back to SDPA
# automatically (our resolve_attn_implementation gracefully handles this), which
# is fine on 80GB with the bounded max_prompt_length and gradient checkpointing.
# vLLM uses its own bundled FA2 for generation, so generation is not affected.
#
# Usage (from repo root or any directory):
#   bash cloud/setup_cloud.sh
#   # then:
#   export WANDB_API_KEY=<your_key>
#   export HF_TOKEN=<your_hf_token>   # if Qwen model requires auth
#   bash cloud/run_cloud.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."    # repo root
echo "[setup] repo root: $(pwd)"

PYTHON="${PYTHON_BIN:-python3}"
echo "[setup] python:  $($PYTHON --version)"
echo "[setup] pip:     $($PYTHON -m pip --version | head -1)"

# Upgrade pip/setuptools silently
$PYTHON -m pip install -q --upgrade pip setuptools wheel

# ── Step 1: vLLM — pulls its own matching torch 2.9.0+cu128 ───────────────
# Install vllm first so its torch requirement wins. Everything else pins on top.
echo "[setup] [1/5] installing vllm==0.12.0 + torch 2.9.0+cu128 ..."
$PYTHON -m pip install "vllm==0.12.0"

# Confirm torch version and CUDA device visibility
$PYTHON - <<'PY'
import torch
print(f"  torch {torch.__version__}  |  CUDA {torch.version.cuda}  |  GPUs: {torch.cuda.device_count()}")
assert "2.9" in torch.__version__, f"Expected torch 2.9.x, got {torch.__version__}"
PY

# ── Step 2: Training stack ─────────────────────────────────────────────────
echo "[setup] [2/5] installing training stack (trl / peft / transformers / accelerate / deepspeed) ..."
$PYTHON -m pip install \
  "trl==0.29.0" \
  "peft==0.18.1" \
  "transformers==4.57.6" \
  "accelerate==1.13.0" \
  "deepspeed==0.18.7" \
  "datasets==4.7.0" \
  "wandb==0.25.1" \
  "safetensors>=0.7.0" \
  "ninja>=1.11"   # required by DeepSpeed JIT compilation

# ── Step 3: Misc runtime deps ──────────────────────────────────────────────
echo "[setup] [3/5] installing misc deps ..."
$PYTHON -m pip install \
  "pyyaml>=6.0" \
  "packaging>=24.0" \
  "tabulate" \
  "rich"

# ── Step 4: Repo-internal paths ────────────────────────────────────────────
echo "[setup] [4/5] verifying repo structure ..."
for d in curricullum/train curricullum/data eval; do
  if [ ! -d "$d" ]; then
    echo "[setup] WARNING: directory '$d' not found — did you git clone + transfer data?"
  fi
done

# nestful_evaluation module (imported by prepare_dataset_toolr0): the training
# script auto-clones it to nestful_repo/ on first run. Make sure git is available.
if ! command -v git &>/dev/null; then
  echo "[setup] WARNING: git not found — nestful_repo auto-clone may fail. Install git."
fi

# ── Step 5: Full sanity check ──────────────────────────────────────────────
echo "[setup] [5/5] full import check ..."
$PYTHON - <<'PY'
import torch, vllm, trl, peft, transformers, accelerate, deepspeed

def v(pkg): return getattr(pkg, "__version__", "?")
print(f"  torch        {v(torch)}   CUDA={torch.version.cuda}  GPUs={torch.cuda.device_count()}")
print(f"  vllm         {v(vllm)}")
print(f"  trl          {v(trl)}")
print(f"  transformers {v(transformers)}")
print(f"  accelerate   {v(accelerate)}")
print(f"  deepspeed    {v(deepspeed)}")
print(f"  peft         {v(peft)}")

assert torch.cuda.device_count() >= 1, "No CUDA GPUs visible — check CUDA_VISIBLE_DEVICES"
assert "0.12" in v(vllm),    f"vLLM 0.12.x expected, got {v(vllm)}"
assert "0.29" in v(trl),     f"TRL 0.29.x expected, got {v(trl)}"
assert "2.9"  in v(torch),   f"torch 2.9.x expected, got {v(torch)}"
print()
print("  ✓ all imports OK")
PY

echo
echo "════════════════════════════════════════════════════════════"
echo "  [setup] DONE — environment ready."
echo "════════════════════════════════════════════════════════════"
echo
echo "  Next steps:"
echo "    1. Transfer data (if not done):  bash cloud/transfer_data.sh   (run on the DGX)"
echo "    2. Set credentials:"
echo "         export WANDB_API_KEY=<key>"
echo "         export HF_TOKEN=<token>   # only if Qwen needs auth"
echo "    3. Launch training:"
echo "         bash cloud/run_cloud.sh"
echo
