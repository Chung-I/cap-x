#!/bin/bash
# Launch Gemma 4 31B IT on cml30 GPU 0, serving an OpenAI-compatible API on port 8200.
#
# Usage (run on cml30):
#   bash ~/Codes/cap-x/scripts/launch_gemma_cml30.sh
#
# Or from another machine:
#   ssh cml30.csie.ntu.edu.tw "bash ~/Codes/cap-x/scripts/launch_gemma_cml30.sh"
#
# The server stays in the foreground. Use tmux or nohup to keep it alive:
#   ssh cml30.csie.ntu.edu.tw "tmux new-session -d -s gemma 'bash ~/Codes/cap-x/scripts/launch_gemma_cml30.sh'"
#
# Verify it's up:
#   curl http://cml30.csie.ntu.edu.tw:8200/v1/models

set -e

MODEL_DIR="${HF_MODEL_DIR:-$HOME/.cache/huggingface/hub/models--google--gemma-4-31b-it/}"
GPU="${GEMMA_GPU:-0}"
PORT="${GEMMA_PORT:-8200}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    echo "Download it first:"
    echo "  source ~/.venv-gemma/bin/activate"
    echo "  huggingface-cli download google/gemma-4-31b-it --local-dir $MODEL_DIR"
    exit 1
fi

echo "=== Launching Gemma 4 31B IT ==="
echo "  GPU:   $GPU"
echo "  Port:  $PORT"
echo "  Model: $MODEL_DIR"

CUDA_VISIBLE_DEVICES=$GPU \
~/.venv-gemma/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name google/gemma-4-31b-it \
    --port "$PORT" \
    --host 0.0.0.0 \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
