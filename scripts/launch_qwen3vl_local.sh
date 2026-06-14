#!/bin/bash
# Launch Qwen3-VL-8B-Thinking on local machine (RTX 5090), serving OpenAI-compatible API on port 8200.
# Run from local machine; cml18 connects to http://192.168.0.87:8200/v1/chat/completions
#
# Requires ~/.venv-gemma with:
#   vllm==0.23.0
#   fastapi==0.115.12   (pinned: fastapi>=0.116 introduces _IncludedRouter which breaks prometheus middleware)
#   VLLM_USE_FLASHINFER_SAMPLER=0  (no CUDA toolkit / nvcc on this machine)
#
# Usage:
#   bash scripts/launch_qwen3vl_local.sh
set -e

MODEL="Qwen/Qwen3-VL-8B-Thinking"
PORT=8200
GPU=0

echo "Starting vLLM server for $MODEL on GPU $GPU, port $PORT..."

CUDA_VISIBLE_DEVICES=$GPU \
VLLM_USE_FLASHINFER_SAMPLER=0 \
~/.venv-gemma/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --port $PORT \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
