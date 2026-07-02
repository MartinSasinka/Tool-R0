#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -q -r demo_api/requirements-demo-api.txt
exec python -m uvicorn demo_api.server:app --host 127.0.0.1 --port 8000 "$@"
