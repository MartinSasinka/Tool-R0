#!/bin/bash
MODEL_PATH=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
PORT=${2:-5000}

# export VLLM_DISABLE_COMPILE_CACHE=1
echo "Starting vLLM on port $PORT with model: $MODEL_PATH"

export VLLM_USE_V1=0

CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name solver \
  --port $PORT \
  --enforce-eager \
  --tensor-parallel-size 1 > vllm_server.log 2>&1 &

echo "Waiting for vLLM to initialize..."
sleep 60