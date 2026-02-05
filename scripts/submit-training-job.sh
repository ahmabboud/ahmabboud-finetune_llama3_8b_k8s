#!/bin/bash
###############################################################################
# Submit Ray Training Job with Wandb Integration
#
# This script properly substitutes the WANDB_API_KEY from the Kubernetes secret
# before applying the RayJob manifest.
#
# Usage:
#   ./scripts/submit-training-job.sh
#
# Prerequisites:
#   - kubectl configured for your cluster
#   - wandb-token secret exists: kubectl create secret generic wandb-token -n default --from-literal=key=$WANDB_API_KEY
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Submitting Ray Training Job ==="

# Check if wandb-token secret exists
if ! kubectl get secret wandb-token -n default &>/dev/null; then
    echo "ERROR: wandb-token secret not found in default namespace"
    echo "Create it with: kubectl create secret generic wandb-token -n default --from-literal=key=\$WANDB_API_KEY"
    exit 1
fi

# Get WANDB_API_KEY from Kubernetes secret
export WANDB_API_KEY=$(kubectl get secret wandb-token -n default -o jsonpath='{.data.key}' | base64 -d)

if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is empty"
    exit 1
fi

echo "✓ WANDB_API_KEY loaded from secret (length: ${#WANDB_API_KEY})"

# Delete existing job if it exists
if kubectl get rayjob llama3-finetuning -n ray-cluster &>/dev/null; then
    echo "Deleting existing job..."
    kubectl delete rayjob llama3-finetuning -n ray-cluster
    sleep 5
fi

# Apply the job with environment variable substitution
echo "Applying training job..."
envsubst < "$PROJECT_ROOT/k8s/ray-training-job.yaml" | kubectl apply -f -

echo ""
echo "=== Job Submitted ==="
echo ""
echo "=== Dashboards ==="
echo "Wandb (no setup needed):"
echo "  https://wandb.ai/ahm-abboud-solo/llama3-function-calling"
echo ""
echo "Ray Dashboard: http://localhost:8265"
echo "Grafana:       http://localhost:3000"
echo ""

# Start port-forwards in background if not already running
if ! pgrep -f "port-forward.*ray-cluster-head-svc.*8265" > /dev/null; then
    echo "Starting Ray Dashboard port-forward..."
    kubectl -n ray-cluster port-forward svc/ray-cluster-head-svc 8265:8265 &>/dev/null &
    echo "  ✓ Ray Dashboard available at http://localhost:8265"
else
    echo "  ✓ Ray Dashboard port-forward already running"
fi

if ! pgrep -f "port-forward.*ray-cluster-grafana.*3000" > /dev/null; then
    echo "Starting Grafana port-forward..."
    kubectl -n ray-cluster port-forward svc/ray-cluster-grafana 3000:80 &>/dev/null &
    echo "  ✓ Grafana available at http://localhost:3000"
else
    echo "  ✓ Grafana port-forward already running"
fi

echo ""
echo "=== Monitor Job ==="
echo "Status:  kubectl get rayjob -n ray-cluster -w"
echo "Logs:    kubectl exec -n ray-cluster \$(kubectl get pods -n ray-cluster -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}') -c ray-head -- ray job logs <job-id>"
