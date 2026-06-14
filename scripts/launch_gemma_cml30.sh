#!/bin/bash
# Launch Gemma 4 31B IT on cml30, serving an OpenAI-compatible API on port 8200.
#
# Model is ~100 GB in BF16 (multimodal: vision+audio+text). Uses 2× RTX A6000 (49 GB each)
# with INT8 quantization via bitsandbytes and tensor parallelism.
#
# Usage (run on cml30):
#   bash ~/Codes/cap-x/scripts/launch_gemma_cml30.sh
#
# Or from another machine:
#   ssh cml30.csie.ntu.edu.tw "tmux new-session -d -s gemma 'bash ~/Codes/cap-x/scripts/launch_gemma_cml30.sh'"
#
# Verify it's up:
#   curl http://cml30.csie.ntu.edu.tw:8200/v1/models

set -e

# Uses shell ~-expansion so the path works whether HOME=/home/chungyili or /home/ra/chungyili
ACTUAL_HOME=$(echo ~)
MODEL_DIR="${HF_MODEL_DIR:-${ACTUAL_HOME}/.cache/huggingface/hub/models--google--gemma-4-31b-it/}"
GPUS="${GEMMA_GPUS:-0,1}"
PORT="${GEMMA_PORT:-8200}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    echo "Download it first: run ~/download_gemma.py with ~/.venv-gemma/bin/python"
    exit 1
fi

# Count number of GPUs in GPUS
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

echo "=== Launching Gemma 4 31B IT ==="
echo "  GPUs:  $GPUS  (tensor-parallel-size $NUM_GPUS)"
echo "  Port:  $PORT"
echo "  Model: $MODEL_DIR"

CUDA_VISIBLE_DEVICES=$GPUS \
"${ACTUAL_HOME}/.venv-gemma/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name google/gemma-4-31b-it \
    --port "$PORT" \
    --host 0.0.0.0 \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --tensor-parallel-size "$NUM_GPUS" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
