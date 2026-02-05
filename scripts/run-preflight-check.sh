#!/bin/bash
#
# Run preflight checks on the GPU cluster (2 nodes for NCCL/InfiniBand test)
# This submits a K8s job that verifies cluster readiness for multi-node training
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
JOB_NAME="preflight-check"
NAMESPACE="ray-cluster"

echo "🔍 Running preflight checks on GPU cluster (2 nodes)..."
echo "   This will test GPU health, NCCL, and InfiniBand connectivity."
echo ""

# Delete existing job and service if they exist
if kubectl get job "$JOB_NAME" -n "$NAMESPACE" &>/dev/null; then
    echo "Cleaning up previous preflight job..."
    kubectl delete job "$JOB_NAME" -n "$NAMESPACE" --wait=true 2>/dev/null || true
fi
if kubectl get svc "$JOB_NAME" -n "$NAMESPACE" &>/dev/null; then
    kubectl delete svc "$JOB_NAME" -n "$NAMESPACE" 2>/dev/null || true
fi

# Submit the job
echo "Submitting preflight check job (2 pods, 8 GPUs each)..."
kubectl apply -f "$PROJECT_DIR/k8s/preflight-check-job.yaml"

echo ""
echo "Waiting for pods to start (may trigger GPU node scale-up)..."
echo "This may take 5-10 minutes if GPU nodes need to scale from 0 to 2."
echo ""
echo "You can monitor node scaling with:"
echo "  kubectl get nodes -l nebius.com/gpu=true -w"
echo ""

# Wait for pods to be created
sleep 5

# Get pod names
for i in {1..120}; do
    PODS=$(kubectl get pods -n "$NAMESPACE" -l app="$JOB_NAME" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true)
    POD_COUNT=$(echo $PODS | wc -w | tr -d ' ')
    
    if [ "$POD_COUNT" -eq 2 ]; then
        echo "Both pods created: $PODS"
        break
    fi
    
    # Check if any pods exist
    if [ "$POD_COUNT" -ge 1 ]; then
        echo "$(date +%H:%M:%S) - $POD_COUNT/2 pods created..."
    else
        echo "$(date +%H:%M:%S) - Waiting for pods to be created..."
    fi
    sleep 5
done

# Wait for at least one pod to be running
echo ""
echo "Waiting for pods to be scheduled on GPU nodes..."

for i in {1..180}; do
    RUNNING=$(kubectl get pods -n "$NAMESPACE" -l app="$JOB_NAME" --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l | tr -d ' ')
    PENDING=$(kubectl get pods -n "$NAMESPACE" -l app="$JOB_NAME" --field-selector=status.phase=Pending --no-headers 2>/dev/null | wc -l | tr -d ' ')
    SUCCEEDED=$(kubectl get pods -n "$NAMESPACE" -l app="$JOB_NAME" --field-selector=status.phase=Succeeded --no-headers 2>/dev/null | wc -l | tr -d ' ')
    
    GPU_NODES=$(kubectl get nodes -l nebius.com/gpu=true --no-headers 2>/dev/null | wc -l | tr -d ' ')
    
    echo "$(date +%H:%M:%S) - GPU nodes: $GPU_NODES | Pods - Running: $RUNNING, Pending: $PENDING, Succeeded: $SUCCEEDED"
    
    if [ "$RUNNING" -ge 1 ] || [ "$SUCCEEDED" -ge 1 ]; then
        break
    fi
    
    sleep 10
done

echo ""
echo "=================================================="
echo " STREAMING LOGS FROM BOTH PODS"
echo "=================================================="
echo ""

# Get first pod name and stream its logs
POD1=$(kubectl get pods -n "$NAMESPACE" -l app="$JOB_NAME" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
POD2=$(kubectl get pods -n "$NAMESPACE" -l app="$JOB_NAME" -o jsonpath='{.items[1].metadata.name}' 2>/dev/null)

if [ -n "$POD1" ]; then
    echo "=== Logs from $POD1 ==="
    kubectl logs -n "$NAMESPACE" "$POD1" -f 2>/dev/null || kubectl logs -n "$NAMESPACE" "$POD1" 2>/dev/null || echo "Could not get logs from $POD1"
fi

echo ""

if [ -n "$POD2" ] && [ "$POD2" != "$POD1" ]; then
    echo "=== Logs from $POD2 ==="
    kubectl logs -n "$NAMESPACE" "$POD2" 2>/dev/null || echo "Could not get logs from $POD2"
fi

echo ""
echo "=================================================="

# Wait for job completion
echo "Waiting for job to complete..."
kubectl wait --for=condition=complete job/"$JOB_NAME" -n "$NAMESPACE" --timeout=600s 2>/dev/null || true

# Check final status
JOB_STATUS=$(kubectl get job "$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "0")

if [ "$JOB_STATUS" = "2" ]; then
    echo "✅ Preflight checks completed successfully on both nodes!"
else
    echo "⚠️  Preflight checks finished. Check logs above for details."
fi

echo ""
echo "Note: Job will auto-cleanup in 5 minutes, or run:"
echo "  kubectl delete job $JOB_NAME -n $NAMESPACE"
echo "  kubectl delete svc $JOB_NAME -n $NAMESPACE"
