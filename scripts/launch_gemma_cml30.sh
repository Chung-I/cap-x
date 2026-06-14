#!/bin/bash
# Launch Gemma 4 12B IT QAT (w4a16 compressed-tensors) on cml30.
# Serves an OpenAI-compatible API on port 8200.
#
# Model is ~7 GB in w4a16 QAT format. Runs comfortably on a single RTX A6000 (49 GB).
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
MODEL_DIR="${HF_MODEL_DIR:-${ACTUAL_HOME}/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-w4a16-ct/}"
GPUS="${GEMMA_GPUS:-0}"
PORT="${GEMMA_PORT:-8200}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    echo "Download it first:"
    echo "  ~/.venv-gemma/bin/huggingface-cli download google/gemma-4-12B-it-qat-w4a16-ct \\"
    echo "      --local-dir ~/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-w4a16-ct/"
    exit 1
fi

echo "=== Launching Gemma 4 12B IT QAT (w4a16) ==="
echo "  GPU:   $GPUS"
echo "  Port:  $PORT"
echo "  Model: $MODEL_DIR"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS \
"${ACTUAL_HOME}/.venv-gemma/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name google/gemma-4-12b-it \
    --port "$PORT" \
    --host 0.0.0.0 \
    --quantization compressed-tensors \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90
