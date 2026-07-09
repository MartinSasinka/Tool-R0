#!/usr/bin/env bash
# Environment / repo sanity check (P0 remediation). Read-only; safe anywhere.
# Verifies: python deps, GPU, IBM executable-functions dir (needed by the official
# scorer), canonical dataset presence + SHAs, and flags configs still defaulting
# to the LEGACY dataset B. Informational sections never abort the script.
#
# Usage (from repo root):
#   bash experiments/nestful_synthetic_curriculum_v3/scripts/setup/check_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
V3="experiments/nestful_synthetic_curriculum_v3"
MIN="experiments/nestful_mtgrpo_minimal"

echo "=== check_env ==="
echo "repo root : $REPO_ROOT"
echo "python    : $(command -v python || echo MISSING)"
python --version 2>&1 || true
echo

echo "--- python deps (training/eval stack) ---"
python - <<'PY'
import importlib.util
for mod in ("torch", "transformers", "peft", "bitsandbytes", "vllm", "yaml",
            "datasets", "sklearn", "wandb", "jsonlines"):
    spec = importlib.util.find_spec(mod)
    ver = "?"
    if spec:
        try:
            import importlib.metadata as md
            ver = md.version({"yaml": "pyyaml", "sklearn": "scikit-learn"}.get(mod, mod))
        except Exception:
            pass
    print(f"  {mod:14s} {'OK  ' + ver if spec else 'MISSING'}")
try:
    import torch
    print(f"  cuda available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.device_count()} device(s))" if torch.cuda.is_available() else ""))
except Exception as exc:
    print(f"  cuda check failed: {exc}")
PY
echo

echo "--- canonical datasets (A) + NESTFUL splits ---"
python "$V3/scripts/lib/paths.py"
echo

echo "--- IBM executable functions (required for OFFICIAL win rate) ---"
IBM_DIR="$MIN/data/NESTFUL-main/data_v2/executable_functions"
if [ -f "$IBM_DIR/func_file_map.json" ] && [ -f "$IBM_DIR/basic_functions.py" ]; then
  echo "  OK: $IBM_DIR (func_file_map.json + basic_functions.py present)"
else
  echo "  MISSING or incomplete: $IBM_DIR"
  echo "  -> official_nestful_win_rate CANNOT be computed; final_eval will emit"
  echo "     metrics_official.json without win_rate and the eval batch runner will fail."
fi
echo

echo "--- legacy dataset-B footgun (audits/DATASET_AUDIT.md) ---"
for cfg in "$MIN/config.yaml" "experiments/nestful_mtgrpo_partial/config.yaml"; do
  # only NON-comment lines count (warning comments about legacy B are fine)
  if [ -f "$cfg" ] && grep -q "^[^#]*filtered_toolr0_synthetic" "$cfg"; then
    echo "  WARNING: $cfg still defaults to LEGACY dataset B (filtered_toolr0_synthetic)."
    echo "           Do not rely on its defaults; pass explicit --override paths."
  else
    echo "  OK: $cfg has no legacy-B default"
  fi
done
echo
echo "=== check_env done ==="
