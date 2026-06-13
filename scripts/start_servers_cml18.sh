#!/bin/bash
# Start all API servers on cml18 using GPU 3.
# Run this on cml18 BEFORE running run_cap_agent0_libero_pro.sh.
#
# Usage (from cml18):
#   bash /tmp3/chungyili/cap-x/scripts/start_servers_cml18.sh
set -e

CAP_X=/tmp3/chungyili/cap-x
cd "$CAP_X"
mkdir -p logs

echo "=== Starting API servers on GPU 3 ==="

# SAM3 (GPU 3)
if ! curl -sf -o /dev/null --connect-timeout 2 http://127.0.0.1:8114/ 2>/dev/null; then
    echo "Starting SAM3 on port 8114 (GPU 3)..."
    CUDA_VISIBLE_DEVICES=3 nohup .venv/bin/python -m capx.serving.launch_sam3_server \
        --device cuda --port 8114 --host 127.0.0.1 \
        > logs/sam3.log 2>&1 &
    echo "  SAM3 PID: $!"
else
    echo "  SAM3 already up on port 8114"
fi

# GraspNet (GPU 3)
if ! curl -sf -o /dev/null --connect-timeout 2 http://127.0.0.1:8115/ 2>/dev/null; then
    echo "Starting GraspNet on port 8115 (GPU 3)..."
    CUDA_VISIBLE_DEVICES=3 nohup .venv/bin/python -m capx.serving.launch_contact_graspnet_server \
        --port 8115 --host 127.0.0.1 \
        > logs/graspnet.log 2>&1 &
    echo "  GraspNet PID: $!"
else
    echo "  GraspNet already up on port 8115"
fi

# PyRoKi (CPU)
if ! curl -sf -o /dev/null --connect-timeout 2 http://127.0.0.1:8116/ 2>/dev/null; then
    echo "Starting PyRoKi on port 8116 (CPU)..."
    nohup .venv/bin/python -m capx.serving.launch_pyroki_server \
        --port 8116 --host 127.0.0.1 --robot panda_description --target-link panda_hand \
        > logs/pyroki.log 2>&1 &
    echo "  PyRoKi PID: $!"
else
    echo "  PyRoKi already up on port 8116"
fi

# LLM proxy (OpenRouter)
if ! curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 2 http://127.0.0.1:8110/health 2>/dev/null | grep -q 200; then
    echo "Starting LLM proxy on port 8110..."
    nohup .venv/bin/python -m capx.serving.openrouter_server \
        --key-file .openrouterkey --port 8110 \
        > logs/llm_proxy.log 2>&1 &
    echo "  LLM proxy PID: $!"
else
    echo "  LLM proxy already up on port 8110"
fi

echo ""
echo "Waiting 60s for models to load..."
sleep 60

echo ""
echo "=== Server health check ==="
for port in 8114 8115 8116; do
    code=$(curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 2 http://127.0.0.1:$port/ 2>/dev/null || echo "000")
    echo "  Port $port: $([ "$code" != "000" ] && echo "UP" || echo "DOWN (still loading?)")"
done
code=$(curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 2 http://127.0.0.1:8110/health 2>/dev/null || echo "000")
echo "  Port 8110 (LLM): $([ "$code" = "200" ] && echo "UP" || echo "DOWN")"

echo ""
echo "Monitor logs:"
echo "  tail -f $CAP_X/logs/sam3.log"
echo "  tail -f $CAP_X/logs/graspnet.log"
