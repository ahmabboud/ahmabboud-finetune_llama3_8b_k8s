#!/bin/bash
set -e

# Set unlimited memlock for RDMA/InfiniBand
ulimit -l unlimited 2>/dev/null || echo "Warning: Could not set unlimited memlock"

# Default values
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}

echo "=============================================="
echo "GPU Cluster Acceptance Test"
echo "=============================================="
echo "Node: $(hostname)"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Nodes: $NNODES"
echo "Node rank: $NODE_RANK"
echo "Memlock: $(ulimit -l)"
echo "=============================================="

if [ "$NNODES" -gt 1 ]; then
    echo "Running distributed test..."
    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        /app/scripts/gpu_test.py "$@"
else
    echo "Running single-node test..."
    python /app/scripts/gpu_test.py "$@"
fi
