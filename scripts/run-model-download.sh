#!/bin/bash
set -e

echo "=== Downloading Llama-3 Model ==="

# Check if model already exists
MODEL_EXISTS=$(kubectl exec -n ray-cluster $(kubectl get pods -n ray-cluster -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}') -c ray-head -- test -d /mnt/data/models/llama3-8b-instruct && echo "yes" || echo "no")

if [ "$MODEL_EXISTS" = "yes" ]; then
    echo "✓ Model already exists at /mnt/data/models/llama3-8b-instruct"
    echo "  Skipping download. Delete the directory to re-download."
    exit 0
fi

# Check for HuggingFace token
HF_TOKEN=$(kubectl get secret hf-token -n default -o jsonpath='{.data.token}' 2>/dev/null | base64 -d || echo "")
if [ -z "$HF_TOKEN" ]; then
    echo "⚠ Warning: No HuggingFace token found"
    echo "  Create one with: kubectl create secret generic hf-token --from-literal=token=<your-token>"
    echo "  Required for gated models like Llama-3"
fi

# Delete previous job if exists
kubectl delete job model-download -n default --ignore-not-found

# Apply the job
kubectl apply -f k8s/model-download-job.yaml

echo ""
echo "=== Job Submitted ==="
echo "Waiting for download to complete (this may take 10-20 minutes)..."

# Wait for pod to start
echo -n "Waiting for pod"
for i in {1..60}; do
    POD=$(kubectl get pods -n default -l job-name=model-download -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    if [ -n "$POD" ]; then
        STATUS=$(kubectl get pod $POD -n default -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
        if [ "$STATUS" = "Running" ] || [ "$STATUS" = "Succeeded" ]; then
            echo ""
            break
        fi
    fi
    echo -n "."
    sleep 5
done

# Stream logs
echo ""
echo "=== Download Progress ==="
kubectl logs -n default -l job-name=model-download -f --tail=100 || true

# Check completion
kubectl wait --for=condition=complete job/model-download -n default --timeout=1800s

# Verify
echo ""
echo "=== Verifying Download ==="
kubectl exec -n ray-cluster $(kubectl get pods -n ray-cluster -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}') -c ray-head -- ls -la /mnt/data/models/llama3-8b-instruct/

echo ""
echo "✅ Model download complete!"
