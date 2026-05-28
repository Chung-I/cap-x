#!/usr/bin/env bash
# Launch CaP-X baseline runs on VAB tasks. Each task runs in its own subshell
# with isolated GPU + perception-server ports, so all four tasks can execute
# in parallel.
#
# Usage:
#   ./scripts/run_vab_baseline.sh        # run all available tasks in parallel
#   TASK=object_all_variance ./scripts/run_vab_baseline.sh   # run one task
#
# Pre-requisites:
#   - .venv-libero is set up (uv sync --extra libero --extra contactgraspnet)
#   - VAB is installed into .venv-libero (uv pip install -e ../Variational-Automation-Benchmark)
#   - OpenRouter proxy running at 127.0.0.1:8110 (capx/serving/openrouter_server.py)
#   - HuggingFace is authenticated for SAM3 weights

set -u

CAPX_ROOT="/k8s-nfs/personal/haoru/gap-x/cap-x"
cd "$CAPX_ROOT"

if [[ ! -d .venv-libero ]]; then
    echo "ERROR: .venv-libero not found. Run \`uv venv .venv-libero --python 3.12\` and \`uv sync --active --extra libero --extra contactgraspnet\` first." >&2
    exit 1
fi

# Ensure OpenRouter proxy is up
if ! curl -s --max-time 2 http://127.0.0.1:8110/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"google/gemini-3.1-pro-preview","messages":[{"role":"user","content":"ping"}]}' \
        > /dev/null; then
    echo "OpenRouter proxy not responding on :8110; starting one now."
    nohup .venv-libero/bin/python capx/serving/openrouter_server.py \
        --key-file .openrouterkey --port 8110 \
        > /tmp/openrouter.log 2>&1 &
    disown
    sleep 6
fi

run_task() {
    local task="$1"
    local gpu="$2"
    local sam3_port="$3"
    local graspnet_port="$4"
    local pyroki_port="$5"
    local logfile="/tmp/vab_${task}.log"

    echo "[task=${task}] launching on GPU ${gpu}, ports SAM3=${sam3_port} GraspNet=${graspnet_port} PyRoKi=${pyroki_port}"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export SAM3_SERVICE_URL="http://127.0.0.1:${sam3_port}"
        export GRASPNET_SERVICE_URL="http://127.0.0.1:${graspnet_port}"
        export PYROKI_SERVICE_URL="http://127.0.0.1:${pyroki_port}"
        source .venv-libero/bin/activate
        uv run --no-sync --active capx/envs/launch.py \
            --config-path "env_configs/vab/${task}.yaml" \
            --model "google/gemini-3.1-pro-preview" \
            > "${logfile}" 2>&1
    ) &
    echo $! > "/tmp/vab_${task}.pid"
}

# Layout: task -> GPU + port_base
# port_base + 4 = SAM3, port_base + 5 = GraspNet, port_base + 6 = PyRoKi
# (matches each task YAML's api_servers ports)
declare -A TASK_GPU=(
    [object_all_variance]=0
    [object_packing]=1
    [popcorn_production]=2
    # crate_washing is bimanual; CaP-X 2-arm API is a separate work item.
    # Uncomment below once FrankaVabBimanualEnv + 2-arm FrankaLiberoApi are
    # implemented.
    # [crate_washing]=3
)

declare -A TASK_PORTS=(
    [object_all_variance]="8114 8115 8116"
    [object_packing]="8124 8125 8126"
    [popcorn_production]="8134 8135 8136"
    [crate_washing]="8144 8145 8146"
)

if [[ -n "${TASK:-}" ]]; then
    tasks=("$TASK")
else
    tasks=("${!TASK_GPU[@]}")
fi

for task in "${tasks[@]}"; do
    gpu="${TASK_GPU[$task]:-}"
    if [[ -z "$gpu" ]]; then
        echo "Skipping task=${task} (no GPU assignment)" >&2
        continue
    fi
    ports="${TASK_PORTS[$task]}"
    read -r sam3_port graspnet_port pyroki_port <<< "$ports"
    run_task "$task" "$gpu" "$sam3_port" "$graspnet_port" "$pyroki_port"
done

echo "All tasks launched. PIDs:"
for task in "${tasks[@]}"; do
    [[ -e "/tmp/vab_${task}.pid" ]] && echo "  ${task}: $(cat /tmp/vab_${task}.pid)  (log: /tmp/vab_${task}.log)"
done

echo ""
echo "Tail any log with: tail -f /tmp/vab_<task>.log"
echo "Wait for all to complete: wait \$(cat /tmp/vab_*.pid)"
