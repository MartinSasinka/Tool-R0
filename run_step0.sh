#!/bin/bash
set -euo pipefail

MODEL_PATH=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
PORT=${2:-5000}
TOKENIZER_PATH=${3:-"$MODEL_PATH"}
STEP_NAME="step0_host_solver"
STEP_DIR="${TOOL_R0_RUN_DIR:-.}/iter${TOOL_R0_ITERATION:-unknown}/${STEP_NAME}"
mkdir -p "$STEP_DIR"
export TOOL_R0_STEP_NAME="$STEP_NAME"
export TOOL_R0_STEP_DIR="$STEP_DIR"
STEP_LOG_FILE="${STEP_DIR}/step.log"
exec > >(tee -a "$STEP_LOG_FILE") 2>&1
PID_FILE="${STEP_DIR}/vllm_server.pid"
META_FILE="${STEP_DIR}/vllm_server.meta"
VLLM_CMD_MARKER="vllm.entrypoints.openai.api_server"

get_proc_start_time() {
  local pid="$1"
  if [ -r "/proc/${pid}/stat" ]; then
    awk '{print $22}' "/proc/${pid}/stat" 2>/dev/null || true
  fi
}

get_proc_cmdline() {
  local pid="$1"
  if [ -r "/proc/${pid}/cmdline" ]; then
    tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true
  fi
}

stop_if_owned_vllm() {
  local pid="$1"
  local expected_start="$2"

  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  local cmdline
  cmdline="$(get_proc_cmdline "$pid")"
  if [[ "$cmdline" != *"$VLLM_CMD_MARKER"* ]]; then
    echo "Skip stop for PID=$pid (cmdline mismatch): $cmdline"
    return 1
  fi

  local current_start
  current_start="$(get_proc_start_time "$pid")"
  if [ -n "$expected_start" ] && [ -n "$current_start" ] && [ "$expected_start" != "$current_start" ]; then
    echo "Skip stop for PID=$pid (start_time mismatch expected=$expected_start current=$current_start)"
    return 1
  fi

  echo "Stopping previous owned vLLM process PID=$pid"
  kill "$pid" 2>/dev/null || true
  sleep 2
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  return 0
}

# export VLLM_DISABLE_COMPILE_CACHE=1
echo "Starting vLLM on port $PORT with model: $MODEL_PATH"
echo "Step dir: $STEP_DIR"
echo "PID file: $PID_FILE"
echo "Meta file: $META_FILE"

# Auto-fix checkpoint for vLLM before loading
BASE_HUB="${TOOL_R0_BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
if [ -d "$MODEL_PATH" ]; then
    echo "Pre-flight: fixing checkpoint for vLLM compatibility..."
    python -c "
from grpo_processing import fix_checkpoint_for_vllm
fix_checkpoint_for_vllm('$MODEL_PATH', '$BASE_HUB')
"
fi

STEP0_GPU="${STEP0_GPU:-2}"
STEP0_TP="${STEP0_TP:-1}"
STEP0_GPU_MEM_UTIL="${STEP0_GPU_MEM_UTIL:-0.80}"

if [ -f "$PID_FILE" ]; then
  OLD_PID=$(cat "$PID_FILE" 2>/dev/null || true)
  OLD_START=""
  if [ -f "$META_FILE" ]; then
    OLD_START=$(grep '^start_time=' "$META_FILE" | head -n 1 | cut -d= -f2-)
  fi

  if [ -n "$OLD_PID" ]; then
    stop_if_owned_vllm "$OLD_PID" "$OLD_START" || true
  fi
fi

CUDA_VISIBLE_DEVICES="$STEP0_GPU" python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --tokenizer "$TOKENIZER_PATH" \
  --served-model-name solver \
  --port $PORT \
  --enforce-eager \
  --tensor-parallel-size "$STEP0_TP" \
  --gpu-memory-utilization "$STEP0_GPU_MEM_UTIL" \
  --trust-remote-code \
  --language-model-only > "${STEP_DIR}/vllm_server.log" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > "$PID_FILE"
echo "Started vLLM PID=$VLLM_PID"
VLLM_START_TIME="$(get_proc_start_time "$VLLM_PID")"
VLLM_CMDLINE="$(get_proc_cmdline "$VLLM_PID")"
cat > "$META_FILE" <<EOF
pid=$VLLM_PID
start_time=$VLLM_START_TIME
cmdline=$VLLM_CMDLINE
EOF

echo "Waiting for vLLM to initialize..."
MAX_WAIT=300
ELAPSED=0
until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "WARNING: vLLM did not become healthy after ${MAX_WAIT}s, continuing anyway..."
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
echo "vLLM ready after ${ELAPSED}s (or timed out)"