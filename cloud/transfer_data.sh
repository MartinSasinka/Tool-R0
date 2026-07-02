#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Transfer all data needed to run the Tool-R0 curriculum on a rented cloud node.
# Run this script ON THE DGX (or any machine that has the data).
#
# Required: SSH access to the RunPod pod.
# RunPod gives you SSH connection details in the pod page:
#   Pods → your pod → Connect → SSH over exposed TCP
#   Format: ssh root@<host> -p <port> -i ~/.ssh/id_rsa
#
# Usage:
#   POD_HOST=ssh.runpod.io POD_PORT=12345 POD_USER=root bash cloud/transfer_data.sh
#   # or edit the variables below directly
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration — fill these in or pass as environment variables ──────────
POD_HOST="${POD_HOST:-}"           # e.g. 213.173.99.72  or  ssh.runpod.io
POD_PORT="${POD_PORT:-22}"         # RunPod TCP port (shown in Connect → SSH)
POD_USER="${POD_USER:-root}"       # default for RunPod
POD_DIR="${POD_DIR:-/workspace}"   # where to put the repo on the pod
SSH_KEY="${SSH_KEY:-}"             # path to private key, e.g. ~/.ssh/id_rsa

if [ -z "$POD_HOST" ]; then
  echo "ERROR: set POD_HOST (and optionally POD_PORT/POD_USER/POD_DIR/SSH_KEY)."
  echo "  Example:"
  echo "    POD_HOST=ssh.runpod.io POD_PORT=12345 bash cloud/transfer_data.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."     # DGX repo root
DGX_ROOT="$(pwd)"

# Build SSH / rsync flags
SSH_OPTS="-p $POD_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15"
[ -n "$SSH_KEY" ] && SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
RSYNC="rsync -avz --progress -e \"ssh $SSH_OPTS\""
DEST="${POD_USER}@${POD_HOST}:${POD_DIR}/Tool-R0"

echo "════════════════════════════════════════════════════════════"
echo "  Transfer: DGX → $DEST"
echo "  SSH opts: $SSH_OPTS"
echo "════════════════════════════════════════════════════════════"

# ── 1. Clone (or update) the Git repo on the pod ───────────────────────────
# Assumes the repo is already cloned there, or clones it fresh.
REPO_URL="${REPO_URL:-}"
if [ -n "$REPO_URL" ]; then
  echo "[transfer] cloning repo on pod ..."
  ssh $SSH_OPTS "${POD_USER}@${POD_HOST}" \
    "mkdir -p ${POD_DIR} && cd ${POD_DIR} && \
     if [ -d Tool-R0/.git ]; then cd Tool-R0 && git pull; \
     else git clone ${REPO_URL} Tool-R0; fi"
else
  echo "[transfer] REPO_URL not set — skipping git clone."
  echo "           Create the Tool-R0 directory manually on the pod first:"
  echo "           ssh ... 'mkdir -p ${POD_DIR}/Tool-R0'"
fi

echo "[transfer] ensuring destination directories exist ..."
ssh $SSH_OPTS "${POD_USER}@${POD_HOST}" "mkdir -p \
  ${POD_DIR}/Tool-R0/curricullum/data/filtered_toolr0_synthetic \
  ${POD_DIR}/Tool-R0/curricullum/training/results \
  ${POD_DIR}/Tool-R0/curricullum/checkpoints/qwen3_4b_curriculum_v2 \
  ${POD_DIR}/Tool-R0/eval/data \
  ${POD_DIR}/Tool-R0/helper_calculations/output \
  ${POD_DIR}/Tool-R0/nestful_repo"

# ── 2. Training data (JSONL, ~50 MB) ───────────────────────────────────────
echo "[transfer] [1/6] training JSONL data ..."
eval $RSYNC \
  curricullum/data/filtered_toolr0_synthetic/ \
  "${DEST}/curricullum/data/filtered_toolr0_synthetic/"

# ── 3. NESTFUL evaluation dataset ──────────────────────────────────────────
echo "[transfer] [2/6] NESTFUL evaluation dataset ..."
eval $RSYNC \
  eval/data/NESTFUL-main/ \
  "${DEST}/eval/data/NESTFUL-main/"

# ── 4. Call distribution JSON ──────────────────────────────────────────────
echo "[transfer] [3/6] call distribution JSON ..."
eval $RSYNC \
  helper_calculations/output/ \
  "${DEST}/helper_calculations/output/"

# ── 5. IBM nestful_repo (tool execution registry, ~2 MB) ───────────────────
echo "[transfer] [4/6] nestful_repo (IBM tool execution) ..."
eval $RSYNC \
  nestful_repo/ \
  "${DEST}/nestful_repo/"

# ── 6. Baseline eval cache (saves ~1 h on first run) ──────────────────────
if [ -f "curricullum/training/results/baseline_nestful.json" ]; then
  echo "[transfer] [5/6] baseline cache ..."
  eval $RSYNC \
    curricullum/training/results/baseline_nestful.json \
    "${DEST}/curricullum/training/results/baseline_nestful.json"
else
  echo "[transfer] [5/6] no baseline cache found — will be recomputed on the pod (~1 h)"
fi

# ── 7. Stage checkpoints (only if resuming mid-curriculum) ─────────────────
RESUME_STAGE="${RESUME_STAGE:-1}"
if [ "$RESUME_STAGE" -gt 1 ]; then
  echo "[transfer] [6/6] stage checkpoints (resuming from stage $RESUME_STAGE) ..."
  eval $RSYNC \
    curricullum/checkpoints/qwen3_4b_curriculum_v2/ \
    "${DEST}/curricullum/checkpoints/qwen3_4b_curriculum_v2/"
else
  echo "[transfer] [6/6] starting from scratch — skipping checkpoint transfer"
fi

# ── Done ───────────────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════"
echo "  [transfer] DONE"
echo "════════════════════════════════════════════════════════════"
echo
echo "  On the pod, run:"
echo "    cd ${POD_DIR}/Tool-R0"
echo "    bash cloud/setup_cloud.sh"
echo "    export WANDB_API_KEY=<key>"
echo "    export HF_TOKEN=<token>"
echo "    bash cloud/run_cloud.sh"
