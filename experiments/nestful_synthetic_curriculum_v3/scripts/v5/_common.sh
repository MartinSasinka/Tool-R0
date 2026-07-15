# Shared helpers for the curriculum-v5 entry points. Source, do not execute.
set -euo pipefail

_V5_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V3="$(cd "$_V5_SCRIPTS/../.." && pwd)"
MINIMAL="$(cd "$V3/../nestful_mtgrpo_minimal" && pwd)"
PY="${PYTHON:-python3}"

banner() {
  echo "──────────────────────────────────────────────────────────────"
  echo "[v5] $1"
  echo "──────────────────────────────────────────────────────────────"
}

require_file() {  # require_file <path> <what>
  if [ ! -f "$1" ]; then
    echo "[v5] ERROR: $2 not found: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [ ! -d "$1" ]; then
    echo "[v5] ERROR: $2 not found: $1" >&2
    exit 1
  fi
}

require_adapter() {  # require_adapter <dir> <what>
  require_dir "$1" "$2"
  require_file "$1/adapter_config.json" "$2 adapter_config.json"
}

print_env() {  # print_env VAR1 VAR2 ...
  echo "[v5] environment:"
  for v in "$@"; do
    echo "       $v=${!v:-<unset>}"
  done
}
