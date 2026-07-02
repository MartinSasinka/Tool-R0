#!/bin/bash
# ==========================================================================
#  setup_nestful_funcs.sh
#
#  Idempotently fetch the IBM/NESTFUL repository into data/nestful_repo/
#  so the eval can dispatch unknown tool calls into the dataset's own Python
#  implementations (data_v2/executable_functions/*.py).
#
#  Why git clone (not submodule / pip): we only ever read from this folder,
#  it doesn't track our commits, and 4350+ small Python files don't need to
#  bloat our git history. --depth 1 keeps the clone fast (~30 MB).
#
#  Safe to call repeatedly. The script:
#    1. Returns immediately if a valid checkout is already on disk.
#    2. Removes a partial / malformed checkout and re-clones it.
#    3. Fails fast with a clear error if the network is unreachable.
#
#  Called automatically by eval/scripts/run_nestful_all_modes.sh, but also
#  safe to run by hand:
#
#      bash scripts/setup_nestful_funcs.sh
# ==========================================================================

set -euo pipefail

REPO_URL="https://github.com/IBM/NESTFUL.git"
TARGET_DIR="${NESTFUL_REPO_DIR:-data/nestful_repo}"
SENTINEL="${TARGET_DIR}/data_v2/executable_functions/func_file_map.json"
BASIC_FUNCTIONS="${TARGET_DIR}/data_v2/executable_functions/basic_functions.py"

is_valid_checkout() {
    [[ -f "$SENTINEL" && -f "$BASIC_FUNCTIONS" ]]
}

if is_valid_checkout; then
    echo "[setup_nestful_funcs] OK: $TARGET_DIR is already populated."
    echo "                       (sentinel: $SENTINEL)"
    exit 0
fi

if [[ -d "$TARGET_DIR" ]]; then
    echo "[setup_nestful_funcs] WARN: $TARGET_DIR exists but is incomplete; removing."
    rm -rf "$TARGET_DIR"
fi

mkdir -p "$(dirname "$TARGET_DIR")"

echo "[setup_nestful_funcs] Cloning $REPO_URL into $TARGET_DIR (depth=1)..."
if ! git clone --depth 1 "$REPO_URL" "$TARGET_DIR"; then
    echo "" >&2
    echo "[setup_nestful_funcs] ERROR: git clone failed." >&2
    echo "  Check network access on this host. Manual fallback:" >&2
    echo "    git clone --depth 1 $REPO_URL $TARGET_DIR" >&2
    echo "  Or download the ZIP from https://github.com/IBM/NESTFUL and" >&2
    echo "  unpack it so that the file" >&2
    echo "    $SENTINEL" >&2
    echo "  exists." >&2
    exit 1
fi

if ! is_valid_checkout; then
    echo "[setup_nestful_funcs] ERROR: clone succeeded but sentinel is missing." >&2
    echo "  Expected: $SENTINEL" >&2
    echo "  The IBM repo layout may have changed; please verify upstream." >&2
    exit 1
fi

NUM_FUNCS=$(python -c "import json; print(len(json.load(open('$SENTINEL'))))" 2>/dev/null || echo "?")
echo "[setup_nestful_funcs] Done. $NUM_FUNCS functions registered in $SENTINEL"
