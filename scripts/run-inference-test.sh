#!/bin/bash
###############################################################################
# Run Inference Test - Compare Fine-tuned vs Base Model
#
# This script runs the inference demo that compares the fine-tuned model
# against the base Llama-3 model on function calling tasks.
#
# Usage:
#   ./scripts/run-inference-test.sh                    # Use latest checkpoint
#   ./scripts/run-inference-test.sh checkpoint-600     # Use specific checkpoint
#   ./scripts/run-inference-test.sh --logs-only        # Just show logs of running job
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NAMESPACE="ray-cluster"
JOB_NAME="llama3-inference-demo"

# Handle --logs-only flag
if [ "$1" == "--logs-only" ]; then
    echo "Showing logs for existing job..."
    kubectl logs -n $NAMESPACE job/$JOB_NAME -f
    exit 0
fi

echo "=== Running Inference Test ==="

# Delete existing job if it exists
if kubectl get job $JOB_NAME -n $NAMESPACE &>/dev/null; then
    echo "Deleting existing job..."
    kubectl delete job $JOB_NAME -n $NAMESPACE
    sleep 3
fi

# Set checkpoint if provided
if [ -n "$1" ]; then
    export CHECKPOINT_NAME="$1"
    echo "Using checkpoint: $CHECKPOINT_NAME"
    envsubst < "$PROJECT_ROOT/k8s/inference-test-job.yaml" | kubectl apply -f -
else
    echo "Using auto-detected latest checkpoint"
    kubectl apply -f "$PROJECT_ROOT/k8s/inference-test-job.yaml"
fi

echo ""
echo "=== Job Submitted ==="
echo "Waiting for pod to start..."

# Wait for pod to be running
for i in {1..60}; do
    STATUS=$(kubectl get pods -n $NAMESPACE -l job-name=$JOB_NAME -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Pending")
    if [ "$STATUS" == "Running" ] || [ "$STATUS" == "Succeeded" ] || [ "$STATUS" == "Failed" ]; then
        break
    fi
    echo -n "."
    sleep 2
done
echo ""

# Stream logs
echo ""
echo "=== Inference Test Logs ==="
kubectl logs -n $NAMESPACE job/$JOB_NAME -f

# Show final status
echo ""
STATUS=$(kubectl get job $JOB_NAME -n $NAMESPACE -o jsonpath='{.status.conditions[0].type}' 2>/dev/null || echo "Unknown")
if [ "$STATUS" == "Complete" ]; then
    echo "✓ Inference test completed successfully!"
else
    echo "Job status: $STATUS"
fi
