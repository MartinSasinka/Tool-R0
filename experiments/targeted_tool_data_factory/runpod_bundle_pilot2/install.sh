#!/usr/bin/env bash
# Idempotent dependency install for the pilot2 D0-vs-D1 RunPod run.
# Reuses the v3 trainer environment; only adds what the factory adapter needs.
set -Eeuo pipefail
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V3="$(cd "$BUNDLE/../../nestful_synthetic_curriculum_v3" && pwd)"
PY="${PYTHON:-python3}"

echo "[install] python: $($PY -V)"

if [ -f "$V3/requirements.txt" ]; then
  echo "[install] trainer requirements"
  $PY -m pip install -q -r "$V3/requirements.txt"
fi

echo "[install] bundle requirements"
$PY -m pip install -q -r "$BUNDLE/requirements.txt"

echo "[install] import check"
$PY - <<'PYEOF'
import importlib, sys
missing = []
for m in ("torch", "transformers", "peft", "trl", "datasets", "numpy", "scipy", "yaml"):
    try:
        importlib.import_module(m)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{m}: {exc}")
if missing:
    print("[install] MISSING:", *missing, sep="\n  ", file=sys.stderr)
    sys.exit(1)
import torch
print(f"[install] torch {torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}")
PYEOF
echo "[install] ok"
