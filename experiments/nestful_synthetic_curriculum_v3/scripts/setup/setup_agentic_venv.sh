#!/usr/bin/env bash
# Create and populate the agentic data-generation venv (Linux/macOS).
# Usage (repo root):
#   bash experiments/nestful_synthetic_curriculum_v3/scripts/setup/setup_agentic_venv.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
V3_ROOT="$REPO_ROOT/experiments/nestful_synthetic_curriculum_v3"
VENV_DIR="$V3_ROOT/.venv"
REQ_FILE="$V3_ROOT/requirements-agentic.txt"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "=== agentic venv setup ==="
echo "repo : $REPO_ROOT"
echo "venv : $VENV_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV_DIR/bin/pip" install torch --index-url "$CUDA_INDEX"
"$VENV_DIR/bin/pip" install -r "$REQ_FILE"

echo
echo "--- verify ---"
"$VENV_DIR/bin/python" - <<'PY'
import importlib
for m in ("torch", "transformers", "bitsandbytes", "accelerate", "pytest", "dotenv"):
    try:
        importlib.import_module(m)
        print(f"  {m}: OK")
    except Exception as exc:
        print(f"  {m}: FAIL ({exc})")
import torch
print(f"  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
PY

echo
echo "Activate:"
echo "  source experiments/nestful_synthetic_curriculum_v3/.venv/bin/activate"
