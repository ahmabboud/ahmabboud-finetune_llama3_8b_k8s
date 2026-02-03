# Kubeflow Pipelines Setup Guide

Complete guide to setting up and using Kubeflow Pipelines for automated ML workflows.

## Table of Contents
- [What is Kubeflow Pipelines?](#what-is-kubeflow-pipelines)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

---

## What is Kubeflow Pipelines?

**Kubeflow Pipelines (KFP)** automates your ML workflows by:
- Orchestrating training stages automatically
- Handling dependencies between steps
- Retrying failed stages
- Tracking all runs and artifacts
- Providing a visual UI for monitoring

### Before Kubeflow (Manual)
```bash
# You run commands one by one
kubectl apply -f k8s/data-prep-job.yaml    # Wait 5min
kubectl apply -f k8s/ray-training-job.yaml # Wait 90min
kubectl apply -f k8s/inference-test-job.yaml # Wait 15min
# Check results, decide if good enough, deploy manually
```

### With Kubeflow (Automated)
```python
# Define pipeline once
pipeline = create_training_pipeline()

# Submit and forget - runs automatically
client.create_run_from_pipeline_func(pipeline)

# Pipeline handles:
# - Running stages in order
# - Waiting for completion
# - Conditional deployment if accuracy > 70%
# - Notifications when done
```

---

## Installation

### Prerequisites
- Kubernetes cluster with kubectl access
- At least 4GB RAM available on CPU nodes
- Persistent storage available

### Option 1: One-Command Install (Recommended)

```bash
./scripts/install_kubeflow_pipelines.sh
```

This script:
1. Installs Kubeflow Pipelines 2.0.5
2. Creates the `kubeflow` namespace
3. Deploys all required components (MySQL, MinIO, Argo, API Server, UI)
4. Creates service account for pipeline execution
5. Waits for all components to be ready

**Expected output:**
```
============================================
✓ Kubeflow Pipelines Installed Successfully!
============================================

Access the UI:
  kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
  Then open: http://localhost:8080
```

### Option 2: Manual Install

```bash
# Install cluster-scoped resources
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.0.5"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

# Install KFP components
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=2.0.5"

# Wait for pods to be ready
kubectl wait --for=condition=available --timeout=600s \
  -n kubeflow deployment/ml-pipeline \
  deployment/ml-pipeline-ui \
  deployment/mysql \
  deployment/minio
```

### Verify Installation

```bash
# Check all pods are running
kubectl get pods -n kubeflow

# Expected output (all Running):
# NAME                                     READY   STATUS
# ml-pipeline-xxx                          1/1     Running
# ml-pipeline-ui-xxx                       1/1     Running
# mysql-xxx                                1/1     Running
# minio-xxx                                1/1     Running
# workflow-controller-xxx                  1/1     Running
# ml-pipeline-persistenceagent-xxx         1/1     Running
# ml-pipeline-scheduledworkflow-xxx        1/1     Running
```

---

## Quick Start

### 1. Access the UI

```bash
# Port-forward to access UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

# Open in browser
open http://localhost:8080
```

### 2. Install KFP SDK

```bash
# Install Python SDK
pip install kfp==2.5.0

# Verify installation
python -c "import kfp; print(f'KFP version: {kfp.__version__}')"
```

### 3. Run Your First Pipeline

#### Option A: Upload Pre-compiled Pipeline (Easy)

```bash
# Compile the Llama-3 training pipeline
cd /path/to/ahmabboud-finetune_llama3_8b_k8s
python pipelines/llama3_training_pipeline.py --compile

# This creates: llama3_training_pipeline.yaml

# Upload to KFP UI:
# 1. Go to http://localhost:8080
# 2. Click "Pipelines" → "Upload pipeline"
# 3. Select llama3_training_pipeline.yaml
# 4. Click "Create run"
# 5. Fill in parameters (wandb_api_key optional)
# 6. Click "Start"
```

#### Option B: Submit Programmatically (Advanced)

```python
from kfp.client import Client

# Connect to KFP
client = Client(host='http://localhost:8080')

# Compile and submit
from pipelines.llama3_training_pipeline import llama3_training_pipeline

run = client.create_run_from_pipeline_func(
    llama3_training_pipeline,
    experiment_name='Llama-3 Fine-tuning',
    run_name='my-first-run',
    arguments={
        'wandb_api_key': 'your-key-here',  # Optional
        'accuracy_threshold': 0.7,
        'enable_deployment': True
    }
)

print(f"Run started: {run.run_id}")
```

#### Option C: Command Line (Quick)

```bash
# Submit pipeline from command line
python pipelines/llama3_training_pipeline.py \
  --submit \
  --host http://localhost:8080 \
  --wandb-key $WANDB_API_KEY \
  --run-name "test-run-$(date +%Y%m%d-%H%M%S)"
```

---

## Pipeline Architecture

### Visual Flow

```
┌───────────────────────────────────────────────────────────────┐
│               Llama-3 Training Pipeline                       │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  [1] Data Preparation                                         │
│       ├─ kubectl apply data-prep-job.yaml                     │
│       └─ Wait for completion                                  │
│              │                                                 │
│              ▼                                                 │
│  [2] Model Training                                           │
│       ├─ Submit Ray Train job                                 │
│       ├─ Train on 16 H100 GPUs                               │
│       └─ Save checkpoints to /mnt/data                       │
│              │                                                 │
│              ▼                                                 │
│  [3] Model Evaluation                                         │
│       ├─ Run inference demo                                   │
│       ├─ Test 6 function calling scenarios                    │
│       └─ Calculate success rate                              │
│              │                                                 │
│              ▼                                                 │
│  [4] Check Accuracy >= 70%?                                  │
│       ├─ YES → [5] Deploy Model                               │
│       │         └─ Copy to production storage                │
│       │                                                       │
│       └─ NO  → Skip deployment                               │
│              │                                                 │
│              ▼                                                 │
│  [6] Send Notification                                        │
│       └─ Summary of training results                         │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Component Details

| Component | Purpose | Duration | Output |
|-----------|---------|----------|--------|
| **Data Prep** | Download & process dataset | ~5 min | `/mnt/data/datasets/{train,val}.jsonl` |
| **Training** | Fine-tune with LoRA on 16 GPUs | ~90 min | `/mnt/data/checkpoints/llama3-*/final` |
| **Evaluation** | Test function calling accuracy | ~15 min | Success rate (e.g., 83%) |
| **Deploy** | Copy to production (conditional) | ~1 min | Model in prod storage |
| **Notify** | Send completion status | ~1 sec | Slack/Email message |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wandb_api_key` | string | "" | Weights & Biases API key (optional) |
| `accuracy_threshold` | float | 0.7 | Minimum success rate to deploy |
| `enable_deployment` | bool | true | Whether to deploy if threshold met |

---

## Usage

### View Pipeline Runs

**In UI:**
1. Go to http://localhost:8080
2. Click "Runs" in sidebar
3. See all pipeline executions with status
4. Click a run to see details and logs

**From Command Line:**
```bash
# List recent runs
python -c "
from kfp.client import Client
client = Client(host='http://localhost:8080')
runs = client.list_runs(experiment_name='Llama-3 Fine-tuning')
for run in runs.runs[:5]:
    print(f'{run.run.name}: {run.run.status}')
"
```

### Monitor a Running Pipeline

**In UI:**
- Click on a running pipeline
- See visual graph with each stage colored:
  - 🔵 Blue = Running
  - 🟢 Green = Completed
  - 🔴 Red = Failed
  - ⚪ Gray = Not started

**From Command Line:**
```bash
# Get run status
python -c "
from kfp.client import Client
client = Client(host='http://localhost:8080')
run = client.get_run('run-id-here')
print(f'Status: {run.run.status}')
"
```

### View Logs

**In UI:**
1. Click on a pipeline run
2. Click on any stage (e.g., "train_model_component")
3. Click "Logs" tab
4. See real-time stdout/stderr

**From Command Line:**
```bash
# Get workflow name from KFP run
export WORKFLOW_NAME=$(kubectl get workflow -n kubeflow -o name | head -1)

# View logs for specific pod
kubectl logs -n kubeflow $WORKFLOW_NAME-xxx
```

### Stop a Running Pipeline

**In UI:**
1. Click on the running pipeline
2. Click "Terminate run"

**From Command Line:**
```python
from kfp.client import Client
client = Client(host='http://localhost:8080')
client.terminate_run('run-id-here')
```

---

## Troubleshooting

### Issue: Pods Stuck in Pending

**Symptoms:**
```bash
kubectl get pods -n kubeflow
# ml-pipeline-xxx   0/1   Pending
```

**Solution:**
```bash
# Check why pod is pending
kubectl describe pod -n kubeflow ml-pipeline-xxx

# Common causes:
# 1. Insufficient resources
kubectl describe nodes | grep -A5 "Allocated resources"

# 2. PVC not bound
kubectl get pvc -n kubeflow

# 3. Node selector not matching
kubectl get nodes --show-labels
```

### Issue: UI Not Accessible

**Symptoms:**
- `kubectl port-forward` command runs but UI doesn't load
- Browser shows connection refused

**Solution:**
```bash
# Check if ml-pipeline-ui pod is running
kubectl get pods -n kubeflow -l app=ml-pipeline-ui

# If not running, check logs
kubectl logs -n kubeflow -l app=ml-pipeline-ui

# Try port-forward with different port
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 9090:80
# Then try http://localhost:9090
```

### Issue: Pipeline Fails at Training Stage

**Symptoms:**
- Pipeline shows training component failed
- Red error in UI

**Solution:**
```bash
# Get the workflow name
kubectl get workflows -n kubeflow

# Get logs from training pod
kubectl logs -n kubeflow <workflow-name>-train-xxx

# Common issues:
# 1. Ray cluster not available
kubectl get pods -n ray-cluster

# 2. Manifests not found at /mnt/data/k8s-manifests/
#    Solution: Copy manifests to shared storage
kubectl cp k8s/ray-training-job.yaml <any-pod>:/mnt/data/k8s-manifests/

# 3. Wandb key invalid
#    Solution: Update the wandb-token secret
kubectl create secret generic wandb-token \
  --from-literal=key=$WANDB_API_KEY \
  -n default --dry-run=client -o yaml | kubectl apply -f -
```

### Issue: Pipeline Components Can't Access Kubernetes

**Symptoms:**
```
Error: unable to execute kubectl command
```

**Solution:**

The components need proper ServiceAccount permissions:

```bash
# Verify service account exists
kubectl get sa pipeline-runner -n kubeflow

# Check permissions
kubectl auth can-i --list --as=system:serviceaccount:kubeflow:pipeline-runner

# If missing, recreate:
kubectl create serviceaccount pipeline-runner -n kubeflow
kubectl create clusterrolebinding pipeline-runner-binding \
  --clusterrole=cluster-admin \
  --serviceaccount=kubeflow:pipeline-runner
```

### Issue: Artifacts Not Saved

**Symptoms:**
- Pipeline completes but no artifacts in MinIO
- Can't view inputs/outputs in UI

**Solution:**
```bash
# Check MinIO is running
kubectl get pods -n kubeflow -l app=minio

# Access MinIO UI (for debugging)
kubectl port-forward -n kubeflow svc/minio-service 9000:9000
# Login at http://localhost:9000
# Default: minio / minio123

# Check MinIO logs
kubectl logs -n kubeflow -l app=minio
```

### Getting Help

```bash
# View all KFP resources
kubectl get all -n kubeflow

# Check KFP API server logs
kubectl logs -n kubeflow -l app=ml-pipeline

# Check Argo workflow controller logs
kubectl logs -n kubeflow -l app=workflow-controller

# Describe a failing workflow
kubectl describe workflow -n kubeflow <workflow-name>
```

---

## Next Steps

1. **Customize the Pipeline**: Edit `pipelines/llama3_training_pipeline.py` to add your own logic
2. **Add More Components**: Create new @dsl.component functions for additional stages
3. **Schedule Pipelines**: Use KFP's recurring runs feature for automated retraining
4. **Integrate with CI/CD**: Trigger pipelines from GitHub Actions or Jenkins
5. **Add Monitoring**: Integrate with Prometheus for pipeline metrics

## Related Documentation

- [Training Guide](Training_Guide.md) - Manual training workflow
- [Monitoring Guide](Monitoring_Guide.md) - Observability setup
- [Kubeflow Pipelines Official Docs](https://www.kubeflow.org/docs/components/pipelines/)

---

*Last updated: 2026-02-03*
