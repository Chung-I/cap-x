#!/bin/bash
# Run CaP-Agent0 evaluation on all 6 LIBERO-PRO suites.
# 6 suites × 10 tasks × 50 trials = 3,000 total trials.
#
# Usage:
#   bash scripts/run_cap_agent0_libero_pro.sh [NUM_WORKERS]
#
# Model configuration (env vars or --args flags):
#   CAP_MODEL     Main policy model (code gen + ensemble)
#                 default: openrouter/meta-llama/llama-4-maverick:free
#   CAP_VDM_MODEL Visual differencing model (best free VLM)
#                 default: openrouter/google/gemma-4-31b-it:free
#
# Examples:
#   bash scripts/run_cap_agent0_libero_pro.sh 2
#   CAP_MODEL=openrouter/meta-llama/llama-4-maverick:free \
#   CAP_VDM_MODEL=openrouter/google/gemma-4-31b-it:free \
#   bash scripts/run_cap_agent0_libero_pro.sh 2
#
# Prerequisites: all API servers must be running before calling this script.
#   Start them with: bash scripts/start_servers_cml18.sh
#   Required ports: 8110 (LLM proxy), 8114 (SAM3), 8115 (GraspNet), 8116 (PyRoKi)
set -e

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

NUM_WORKERS=${1:-4}
CAP_MODEL=${CAP_MODEL:-openrouter/meta-llama/llama-4-maverick:free}
CAP_VDM_MODEL=${CAP_VDM_MODEL:-openrouter/google/gemma-4-31b-it:free}

echo "=== Checking required servers ==="
all_up=true
for port in 8114 8115 8116; do
    code=$(curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 2 http://127.0.0.1:$port/ 2>/dev/null || echo "000")
    if [ "$code" != "000" ]; then
        echo "  Port $port: UP"
    else
        echo "  Port $port: DOWN"
        all_up=false
    fi
done
code=$(curl -sf -o /dev/null -w "%{http_code}" --connect-timeout 2 http://127.0.0.1:8110/health 2>/dev/null || echo "000")
if [ "$code" = "200" ]; then
    echo "  Port 8110 (LLM proxy): UP"
else
    echo "  Port 8110 (LLM proxy): DOWN"
    all_up=false
fi

if [ "$all_up" = false ]; then
    echo ""
    echo "ERROR: Required servers are not reachable. Start them first:"
    echo "  bash scripts/start_servers_and_eval.sh"
    exit 1
fi

echo ""
echo "=== Launching CaP-Agent0 on LIBERO-PRO ==="
echo "Suites: libero_object_swap, libero_object_task,"
echo "        libero_goal_swap,   libero_goal_task,"
echo "        libero_spatial_swap, libero_spatial_task"
echo "Workers:    $NUM_WORKERS"
echo "Main model: $CAP_MODEL"
echo "VDM model:  $CAP_VDM_MODEL"
echo "Output:     ./outputs/cap_agent0_libero_pro/"
echo "Log:        logs/cap_agent0_libero_pro.log"
echo ""

source .venv-libero/bin/activate

MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python -m capx.envs.scripts.run_libero_batch \
    --args.suites libero_object_swap libero_object_task \
                  libero_goal_swap libero_goal_task \
                  libero_spatial_swap libero_spatial_task \
    --args.models "$CAP_MODEL" \
    --args.vdm-model "$CAP_VDM_MODEL" \
    --args.num-workers "$NUM_WORKERS" \
    --args.output-dir ./outputs/cap_agent0_libero_pro \
    2>&1 | tee logs/cap_agent0_libero_pro.log

echo ""
echo "=== Results ==="
for suite in libero_object_swap libero_object_task libero_goal_swap libero_goal_task libero_spatial_swap libero_spatial_task; do
    total=$(find "outputs/cap_agent0_libero_pro/$suite" -name "trial_*" -type d 2>/dev/null | wc -l)
    success=$(find "outputs/cap_agent0_libero_pro/$suite" -name "*taskcompleted_1*" -type d 2>/dev/null | wc -l)
    echo "  $suite: $success / $total"
done
