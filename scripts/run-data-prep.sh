#!/bin/bash
#
# Run data preparation job on the Kubernetes cluster
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
JOB_NAME="prepare-training-data"
NAMESPACE="default"

echo "📦 Running data preparation job..."
echo ""

# Delete existing job if it exists
if kubectl get job "$JOB_NAME" -n "$NAMESPACE" &>/dev/null; then
    echo "Cleaning up previous job..."
    kubectl delete job "$JOB_NAME" -n "$NAMESPACE" --wait=true 2>/dev/null || true
fi

# Apply the job
kubectl apply -f "$PROJECT_DIR/k8s/data-prep-job.yaml"

echo ""
echo "Job submitted. Waiting for pod to start..."

# Wait for pod
for i in {1..60}; do
    POD=$(kubectl get pods -n "$NAMESPACE" -l app=data-preparation -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [ -n "$POD" ]; then
        STATUS=$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || true)
        if [ "$STATUS" = "Running" ] || [ "$STATUS" = "Succeeded" ]; then
            echo "Pod $POD is $STATUS"
            break
        fi
        echo "$(date +%H:%M:%S) - Pod status: $STATUS"
    fi
    sleep 5
done

echo ""
echo "Streaming logs..."
kubectl logs -f "job/$JOB_NAME" -n "$NAMESPACE"

# Wait for job to complete
echo ""
echo "Waiting for job to complete..."
kubectl wait --for=condition=complete --timeout=600s "job/$JOB_NAME" -n "$NAMESPACE" 2>/dev/null && {
    echo ""
    echo "✅ Data preparation completed successfully!"
    echo ""
    echo "Data is available at: /mnt/data/datasets/"
    echo "  - train.jsonl"
    echo "  - val.jsonl"
    exit 0
}

# Check if failed
JOB_FAILED=$(kubectl get job "$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || true)
if [ "$JOB_FAILED" = "True" ]; then
    echo ""
    echo "❌ Data preparation failed."
    exit 1
fi

echo ""
echo "⚠️  Job status unknown. Check manually with:"
echo "    kubectl get job $JOB_NAME -n $NAMESPACE"
exit 1
