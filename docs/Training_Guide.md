# Training Guide

Multi-node fine-tuning of Llama-3-8B-Instruct on Kubernetes with distributed training.

## Training Methods

This project supports **two** distributed training methods:

| Method | Description | Best For |
|--------|-------------|----------|
| **StatefulSet + FSDP** | Native K8s deployment with PyTorch FSDP | Production training, fine-grained control |
| **KubeRay (Ray Train)** | Ray-managed distributed training | Easy scaling, fault tolerance, Ray ecosystem |

## Overview

This guide covers:
- [Architecture](#architecture) - How the distributed training works
- [Quick Start (StatefulSet)](#quick-start) - Deploy FSDP training
- [KubeRay Training](#kuberay-training) - Deploy Ray Train (recommended)
- [Scaling](#scaling) - Adjust from 1 to N nodes
- [Monitoring](#monitoring) - Track training progress
- [Debugging](#debugging) - Troubleshoot common issues
- [Configuration](#configuration) - Tune hyperparameters

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     StatefulSet                              │   │
│  │  ┌─────────────────────┐    ┌─────────────────────┐         │   │
│  │  │  llama3-training-0  │    │  llama3-training-1  │         │   │
│  │  │  (Master Node)      │◄──►│  (Worker Node)      │         │   │
│  │  │  8x H100 80GB       │ IB │  8x H100 80GB       │         │   │
│  │  │  Rank 0-7           │    │  Rank 8-15          │         │   │
│  │  └─────────────────────┘    └─────────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│  ┌───────────────────────────▼───────────────────────────────────┐ │
│  │                    Shared Storage (/mnt/data)                 │ │
│  │  ├── models/llama3-8b-instruct/   (cached model)             │ │
│  │  ├── datasets/{train,val}.jsonl   (preprocessed data)        │ │
│  │  └── checkpoints/                  (saved checkpoints)       │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Components:**
- **PyTorch FSDP** - Shards model across GPUs (Full Shard strategy)
- **LoRA (PEFT)** - Only trains 2% of parameters (167M / 8.2B)
- **TRL SFTTrainer** - Handles FSDP+LoRA integration automatically
- **InfiniBand** - High-speed inter-node communication (400 Gb/s)

## Quick Start

### Prerequisites

1. **Kubernetes cluster** with GPU nodes (see [Infrastructure Guide](Infrastructure_Quick_Start.md))
2. **HuggingFace token** with access to Llama-3 models
3. **Weights & Biases account** (optional, for logging)

### Step 1: Create Secrets

```bash
# HuggingFace token (required)
kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN

# Weights & Biases (optional)
kubectl create secret generic wandb-token --from-literal=key=$WANDB_API_KEY
```

### Step 2: Prepare Data & Model

```bash
# Download and cache the model
kubectl apply -f k8s/model-download-job.yaml
kubectl logs -f job/download-model

# Prepare training dataset  
kubectl apply -f k8s/data-prep-job.yaml
kubectl logs -f job/prepare-training-data
```

### Step 3: Run Smoke Test (Recommended)

Validate the environment before full training:

```bash
kubectl apply -f k8s/smoke-test-job.yaml
kubectl logs -f job/training-smoke-test
```

Expected output:
```
✅ Test 1/8 PASSED: PyTorch imports
✅ Test 2/8 PASSED: CUDA available (8 GPUs)
✅ Test 3/8 PASSED: Model loading
✅ Test 4/8 PASSED: PEFT/LoRA
✅ Test 5/8 PASSED: SFTConfig creation
✅ Test 6/8 PASSED: Training step simulation
✅ Test 7/8 PASSED: Llama tokenizer
✅ Test 8/8 PASSED: Dataset loading
✅ Test 9/8 PASSED: NCCL communication
✅ Test 10/8 PASSED: InfiniBand available
========================================
SMOKE TEST COMPLETE: 10/10 tests passed
```

The smoke test validates:
- Package imports and versions
- SFTConfig and LoraConfig creation
- Model loading and PEFT wrapping
- **NCCL communication** (multi-GPU AllReduce)
- **InfiniBand availability** (for multi-node)

### Step 4: Deploy Training

```bash
# Deploy FSDP training
kubectl apply -f k8s/training-fsdp.yaml

# Watch pods start
kubectl get pods -w
```

### Step 5: Monitor Progress

```bash
# Training logs
kubectl logs -f llama3-training-0

# GPU utilization
kubectl exec llama3-training-0 -- nvidia-smi

# Training metrics (loss, accuracy)
kubectl logs llama3-training-0 | grep -E "loss.*epoch"
```

---

## KubeRay Training

KubeRay provides Ray-managed distributed training with automatic scaling, fault tolerance, and easy configuration.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KubeRay Cluster (ray-cluster namespace)         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      Ray Head                                │   │
│  │  - Dashboard (port 8265)                                    │   │
│  │  - Job submission                                           │   │
│  │  - Cluster coordination                                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│         ┌────────────────────┴────────────────────┐                │
│         ▼                                         ▼                │
│  ┌─────────────────────┐    ┌─────────────────────┐               │
│  │  GPU Worker Node 1  │    │  GPU Worker Node 2  │               │
│  │  8x H100 80GB       │◄──►│  8x H100 80GB       │               │
│  │  Ray Train Workers  │ IB │  Ray Train Workers  │               │
│  │  (Rank 0-7)         │    │  (Rank 8-15)        │               │
│  └─────────────────────┘    └─────────────────────┘               │
│                              │                                      │
│  ┌───────────────────────────▼───────────────────────────────────┐ │
│  │                Shared Storage (/mnt/data)                     │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Prerequisites

KubeRay requires an **InfiniBand-enabled Ray image** for multi-node NCCL communication.

1. **Build and push the image** (one-time setup):

```bash
# Configure Docker for Nebius registry
nebius registry configure-helper

# Build for linux/amd64 and push
cd infra/modules/kuberay/kuberay-tests/ray-infiniband
docker buildx build --platform linux/amd64 \
  -t cr.eu-north1.nebius.cloud/<registry-id>/ray-gpu-infiniband:2.46.0-py310 \
  --push .
```

2. **Update Terraform** (`infra/k8s-installation/terraform.tfvars`):

```hcl
# Enable KubeRay
enable_kuberay = true

# Use InfiniBand-enabled image
kuberay_gpu_worker_image = "cr.eu-north1.nebius.cloud/<registry-id>/ray-gpu-infiniband:2.46.0-py310"

# GPU worker configuration  
kuberay_min_gpu_replicas = 2
kuberay_max_gpu_replicas = 2
kuberay_gpu_resources = {
  cpus   = 120
  gpus   = 8     # All 8 H100s per node
  memory = 1400  # ~1400 GB per node
}
```

3. **Apply Terraform**:

```bash
cd infra/k8s-installation
source environment.sh && terraform apply
```

### Step 1: Verify KubeRay Cluster

```bash
# Check Ray cluster status
kubectl get raycluster -n ray-cluster

# Check GPU workers are running
kubectl get pods -n ray-cluster -l ray.io/node-type=worker

# Verify InfiniBand in workers
kubectl exec -n ray-cluster <gpu-worker-pod> -- ibstat | head -20
```

Expected output:
```
CA 'mlx5_0'
    CA type: MT4126
    Port 1:
        State: Active
        Physical state: LinkUp
        Rate: 400
```

### Step 2: Submit Ray Training Job

**Without Wandb:**
```bash
kubectl apply -f k8s/ray-training-job.yaml
```

**With Wandb (recommended):**
```bash
# Export wandb API key and apply with envsubst
export WANDB_API_KEY=$(kubectl get secret wandb-token -n default -o jsonpath='{.data.key}' | base64 -d)
envsubst < k8s/ray-training-job.yaml | kubectl apply -f -
```

```bash
# Check job status
kubectl get rayjob -n ray-cluster

# Follow logs
kubectl logs -n ray-cluster -l job-name=llama3-finetuning -f
```

### Step 3: Monitor Training

```bash
# Access Ray Dashboard
kubectl -n ray-cluster port-forward svc/ray-cluster-kuberay-head-svc 8265:8265
# Open http://localhost:8265

# Check NCCL logs (should show InfiniBand)
kubectl logs -n ray-cluster <job-pod> | grep -E "NCCL|NET/IB"
```

Expected NCCL output (InfiniBand working):
```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB ... [7]mlx5_7:1/IB
NCCL INFO Using network IB
NCCL INFO Channel 01/0 : 0[0] -> 8[0] [receive] via NET/IB/4/GDRDMA
```

### Ray Training Configuration

The Ray training job ([k8s/ray-training-job.yaml](../k8s/ray-training-job.yaml)) includes:

```yaml
# ScalingConfig for 16 GPUs across 2 nodes
scaling_config=ScalingConfig(
    num_workers=16,          # Total GPU workers
    use_gpu=True,
    resources_per_worker={"GPU": 1, "CPU": 8},
)

# Runtime environment with pinned dependencies
runtimeEnvYAML: |
  pip:
    - torch==2.5.1
    - transformers==4.46.3
    - trl==0.11.4
    - peft==0.13.2
```

### Scaling Ray Training

Edit `num_workers` in [k8s/ray-training-job.yaml](../k8s/ray-training-job.yaml):

| Nodes | GPUs | num_workers |
|-------|------|-------------|
| 1 | 8 | 8 |
| 2 | 16 | 16 |
| 4 | 32 | 32 |

```python
# Example: 32 GPUs across 4 nodes
scaling_config=ScalingConfig(
    num_workers=32,
    use_gpu=True,
    resources_per_worker={"GPU": 1, "CPU": 8},
)
```

### KubeRay Job Management

```bash
# Check training status and progress
kubectl get rayjob -n ray-cluster
kubectl logs -n ray-cluster -l ray.io/job-name=llama3-finetuning --tail=50

# Stop/cancel training job
kubectl delete rayjob llama3-finetuning -n ray-cluster

# Check detailed job status
kubectl get rayjob -n ray-cluster llama3-finetuning -o yaml
```

### KubeRay Debugging

```bash
# Job pod logs
kubectl logs -n ray-cluster -l job-name=llama3-finetuning --tail=100

# Access Ray head for debugging
kubectl exec -it -n ray-cluster <ray-head-pod> -- ray status

# Check GPU worker resources
kubectl exec -n ray-cluster <ray-head-pod> -- ray status | grep GPU
```

---

## Scaling (StatefulSet/FSDP)

### Single Node (8 GPUs)

Edit [k8s/training-fsdp.yaml](../k8s/training-fsdp.yaml):

```yaml
# StatefulSet
spec:
  replicas: 1

# Environment variables  
- name: WORLD_SIZE
  value: "8"
- name: NNODES
  value: "1"

# Accelerate config (in ConfigMap)
num_machines: 1
num_processes: 8
```

### Multi-Node (16 GPUs - 2 Nodes)

```yaml
# StatefulSet
spec:
  replicas: 2

# Environment variables
- name: WORLD_SIZE
  value: "16"
- name: NNODES
  value: "2"

# Accelerate config
num_machines: 2
num_processes: 16
```

### Multi-Node (32 GPUs - 4 Nodes)

```yaml
spec:
  replicas: 4

- name: WORLD_SIZE
  value: "32"
- name: NNODES
  value: "4"

# Accelerate config
num_machines: 4
num_processes: 32
```

**Redeploy after changes:**
```bash
kubectl delete statefulset llama3-training --cascade=foreground
kubectl delete configmap training-config
kubectl apply -f k8s/training-fsdp.yaml
```

## Monitoring

### Available Dashboards

| Dashboard | What It Shows | Training Metrics Charts | URL |
|-----------|---------------|------------------------|-----|
| **Wandb** | Loss curves, LR schedule, grad norms | ✅ Yes (recommended) | https://wandb.ai |
| **Ray Dashboard** | Job status, logs, cluster resources | ❌ No (logs only) | localhost:8265 |
| **Grafana** | GPU metrics (temp, power, utilization) | ❌ No | localhost:8080 |
| **TensorBoard** | Training metrics | ❌ Not enabled by default | - |

### Weights & Biases (Recommended for Training Metrics)

Wandb provides real-time training visualization when enabled:
- **Loss curves** with smoothing
- **Learning rate schedule**
- **Gradient norms**
- **Epoch progress**
- **Run comparison** across experiments

**Enable Wandb for KubeRay:**
```bash
export WANDB_API_KEY=$(kubectl get secret wandb-token -n default -o jsonpath='{.data.key}' | base64 -d)
envsubst < k8s/ray-training-job.yaml | kubectl apply -f -
```

**View Dashboard:**
```
https://wandb.ai/<your-username>/llama3-function-calling
```

The RayJob includes wandb integration with:
- `WANDB_PROJECT`: llama3-function-calling
- `WANDB_RUN_NAME`: ray-h100-16x-lora
- Automatic login via `WANDB_API_KEY`

### Ray Dashboard (Job Management)

```bash
kubectl port-forward -n ray-cluster svc/ray-cluster-head-svc 8265:8265
# Open http://localhost:8265
```

Shows:
- Job status and logs
- Worker status
- Cluster resource usage
- Job submission history

**Note:** Ray Dashboard shows training metrics in logs but does not provide charts. Use Wandb for visualization.

### Training Logs

```bash
# KubeRay logs
kubectl logs -n ray-cluster -l job-name=llama3-finetuning -f

# StatefulSet logs
kubectl logs -f llama3-training-0

# Check specific rank logs
kubectl logs llama3-training-0 | grep "\[Rank 0\]"
kubectl logs llama3-training-1 | grep "\[Rank 8\]"
```

### Key Metrics to Watch

| Metric | Good Value | Concern |
|--------|------------|---------|
| `loss` | Decreasing | Flat or increasing |
| `mean_token_accuracy` | 70-85% | < 50% |
| `grad_norm` | 0.5-5.0 | > 10 (exploding), < 0.01 (vanishing) |
| `learning_rate` | Per schedule | Stuck at 0 |

Example log output:
```json
{'loss': '1.433', 'grad_norm': '2.876', 'learning_rate': '6.75e-06',
 'mean_token_accuracy': '0.7492', 'epoch': '0.02385'}
```

### GPU Utilization

```bash
# Single node
kubectl exec llama3-training-0 -- nvidia-smi

# KubeRay GPU workers
kubectl exec -n ray-cluster <gpu-worker-pod> -- nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv

# All StatefulSet nodes
for pod in llama3-training-0 llama3-training-1; do
  echo "=== $pod ==="
  kubectl exec $pod -- nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
done
```

**Expected H100 utilization:** 70-100% GPU, 20-80GB memory per GPU

### Grafana Dashboard (GPU & System Metrics)

```bash
kubectl port-forward -n o11y svc/grafana-and-prometheus 8080:80
# Open http://localhost:8080
# Login: admin / (see your .env for password)
```

Shows:
- NVIDIA DCGM metrics (GPU temperature, power, utilization)
- Node system metrics
- Loki logs

**Note:** Grafana shows infrastructure metrics, not training metrics. Use Wandb for loss/accuracy curves.

### TensorBoard (Optional)

TensorBoard is **not enabled by default**. To enable it, modify the training script:

```python
# In TrainingArguments
report_to=["wandb", "tensorboard"],  # Add tensorboard
logging_dir="/mnt/data/tensorboard_logs",
```

Then access via:
```bash
kubectl port-forward <training-pod> 6006:6006
# Open http://localhost:6006
```

> **For detailed monitoring instructions, troubleshooting, and alerting setup, see the [Monitoring Guide](Monitoring_Guide.md).**

## Debugging

### Check Pod Status

```bash
# Pod status
kubectl get pods -o wide

# Pod events (scheduling issues)
kubectl describe pod llama3-training-0

# Previous container logs (if crashed)
kubectl logs llama3-training-0 --previous
```

### Common Issues

#### 1. NCCL InfiniBand Error: "Cannot allocate memory"

**Error:**
```
NCCL WARN Call to ibv_create_qp failed with error Cannot allocate memory
```

**Cause:** Container lacks permission to pin memory for RDMA.

**Fix:** Ensure security context has required capabilities:
```yaml
securityContext:
  capabilities:
    add:
      - IPC_LOCK      # Memory locking for RDMA
      - SYS_RESOURCE  # Allows ulimit changes
```

And add to the container command:
```bash
ulimit -l unlimited
```

#### 2. OOM (Out of Memory)

**Error:**
```
CUDA out of memory. Tried to allocate X GiB
```

**Fixes:**
```yaml
# Reduce batch size
per_device_train_batch_size: 2  # Default: 4

# Enable gradient checkpointing (already enabled by default)
gradient_checkpointing: true

# Reduce sequence length
max_seq_length: 1024  # Default: 2048
```

#### 3. NCCL Timeout

**Error:**
```
NCCL timeout after 1800 seconds
```

**Fixes:**
```yaml
# Increase timeout
env:
  - name: NCCL_TIMEOUT
    value: "3600"  # 1 hour

# Check network connectivity
kubectl exec llama3-training-0 -- ping llama3-training-1.llama3-training
```

#### 4. Pods Stuck in Pending

```bash
# Check node resources
kubectl describe nodes | grep -A5 "Allocated resources"

# Check GPU availability
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu
```

#### 5. Model Download Fails

```bash
# Verify HF token
kubectl get secret hf-token -o jsonpath='{.data.token}' | base64 -d

# Check if model is gated
# Visit https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
# and accept the license agreement
```

### Debug Commands

```bash
# Interactive shell in training pod
kubectl exec -it llama3-training-0 -- bash

# Check NCCL environment
kubectl exec llama3-training-0 -- env | grep NCCL

# Check memory limits
kubectl exec llama3-training-0 -- cat /proc/1/limits | grep locked

# Test InfiniBand
kubectl exec llama3-training-0 -- ibv_devinfo

# Test inter-node connectivity
kubectl exec llama3-training-0 -- python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('llama3-training-1.llama3-training', 29500))
print('Connection successful!')
s.close()
"
```

## Configuration

### Training Hyperparameters

Edit the ConfigMap in [k8s/training-fsdp.yaml](../k8s/training-fsdp.yaml):

```yaml
train_config.yaml: |
  model:
    name: "meta-llama/Meta-Llama-3-8B-Instruct"
    local_path: "/mnt/data/models/llama3-8b-instruct"
  
  lora:
    r: 64                    # LoRA rank (higher = more params)
    lora_alpha: 128          # LoRA scaling factor
    lora_dropout: 0.05       # Dropout for regularization
    target_modules: "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
  
  data:
    train_path: "/mnt/data/datasets/train.jsonl"
    val_path: "/mnt/data/datasets/val.jsonl"
    max_seq_length: 2048     # Max tokens per example
  
  training:
    output_dir: "/mnt/data/checkpoints/llama3-function-calling"
    num_train_epochs: 3
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 4
    learning_rate: 1.5e-4
    lr_scheduler_type: "cosine"
    warmup_steps: 200
    weight_decay: 0.01
    save_steps: 200
    eval_steps: 200
    logging_steps: 10
    gradient_checkpointing: true
```

### Effective Batch Size

```
Effective Batch = per_device_batch × gradient_accumulation × num_gpus
                = 4 × 4 × 16
                = 256 samples per optimizer step
```

### FSDP Configuration

The Accelerate FSDP config uses recommended settings for PEFT:

```yaml
accelerate_config.yaml: |
  distributed_type: FSDP
  fsdp_config:
    fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
    fsdp_backward_prefetch: BACKWARD_PRE
    fsdp_cpu_ram_efficient_loading: true
    fsdp_sharding_strategy: FULL_SHARD
    fsdp_state_dict_type: SHARDED_STATE_DICT
    fsdp_sync_module_states: true
    fsdp_use_orig_params: false  # Required for PEFT
  mixed_precision: bf16
```

## Outputs

After training completes:

| Path | Description |
|------|-------------|
| `/mnt/data/checkpoints/llama3-function-calling/` | Final LoRA adapter weights |
| `/mnt/data/checkpoints/llama3-function-calling/checkpoint-*/` | Intermediate checkpoints |

### Merge LoRA Weights

To create a standalone model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = PeftModel.from_pretrained(base_model, "/mnt/data/checkpoints/llama3-function-calling")
merged = model.merge_and_unload()
merged.save_pretrained("/mnt/data/checkpoints/merged-model")
```

## Local Development

### Setup Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets accelerate peft trl wandb
```

### Run Smoke Test Locally

```bash
python scripts/smoke_test.py --cpu  # CPU-only validation
python scripts/smoke_test.py        # With GPU
```

### Test Configuration

```bash
python scripts/test_config_locally.py
```

## Files Reference

| File | Description |
|------|-------------|
| [k8s/training-fsdp.yaml](../k8s/training-fsdp.yaml) | StatefulSet FSDP training manifest |
| [k8s/ray-training-job.yaml](../k8s/ray-training-job.yaml) | KubeRay training job manifest |
| [k8s/smoke-test-job.yaml](../k8s/smoke-test-job.yaml) | Environment validation job |
| [k8s/data-prep-job.yaml](../k8s/data-prep-job.yaml) | Dataset preparation job |
| [k8s/model-download-job.yaml](../k8s/model-download-job.yaml) | Model download job |
| [scripts/smoke_test.py](../scripts/smoke_test.py) | Smoke test script |
| [infra/modules/kuberay/](../infra/modules/kuberay/) | KubeRay Terraform module |
| [infra/modules/kuberay/kuberay-tests/ray-infiniband/](../infra/modules/kuberay/kuberay-tests/ray-infiniband/) | InfiniBand-enabled Ray image |

## Appendix: Package Versions

### StatefulSet/FSDP Training

| Package | Version |
|---------|---------|
| torch | 2.6.0+cu124 |
| transformers | 5.0.0 |
| trl | 0.27.1 |
| peft | 0.18.1 |
| accelerate | 1.12.0 |
| datasets | latest |

### KubeRay Training

| Package | Version |
|---------|---------|
| ray | 2.46.0 |
| torch | 2.5.1 |
| transformers | 4.46.3 |
| trl | 0.11.4 |
| peft | 0.13.2 |
| accelerate | 1.1.1 |

> **Note**: KubeRay uses pinned dependency versions for compatibility with Ray Train.

## Appendix: Training Method Comparison

| Feature | StatefulSet + FSDP | KubeRay |
|---------|-------------------|---------|
| **Setup Complexity** | Medium | Low (after image setup) |
| **Scaling** | Manual YAML edits | Change `num_workers` |
| **Fault Tolerance** | None (manual restart) | Automatic retry |
| **Dashboard** | Grafana only | Ray Dashboard + Grafana |
| **Resource Management** | K8s native | Ray autoscaler |
| **Multi-node NCCL** | Works out of box | Requires InfiniBand image |
| **Best For** | Production, fine control | Experimentation, Ray ecosystem |

---

## Scaling to 512 H100 GPUs (64 Nodes)

This section provides step-by-step guidance for scaling from the current 16-GPU setup to 512 GPUs across 64 nodes.

### Quick Reference: Scaling Steps

| Step | What to Change | Where |
|------|----------------|-------|
| 1 | GPU node count | `terraform.tfvars` |
| 2 | KubeRay worker replicas | `terraform.tfvars` |
| 3 | Training worker count | `ray-training-job.yaml` or `training-fsdp.yaml` |
| 4 | Hyperparameters | Training config (batch size, LR) |

### Step 1: Scale Infrastructure (Terraform)

Edit `infra/k8s-installation/terraform.tfvars`:

```hcl
# Scale from 2 nodes to 64 nodes
gpu_nodes_count_per_group = 64    # Was: 2

# Storage: Scale proportionally (32TB for 64 nodes)
filestore_disk_size = 32 * (1024 * 1024 * 1024 * 1024)  # 32TB

# KubeRay: Scale GPU workers
kuberay_min_gpu_replicas = 64     # Was: 2
kuberay_max_gpu_replicas = 64     # Was: 2
```

Apply changes:
```bash
cd infra/k8s-installation
source environment.sh
terraform plan    # Review: should show 62 new GPU nodes
terraform apply   # Takes ~45-60 minutes for 64 nodes
```

### Step 2: Verify Infrastructure

```bash
# Verify all 64 GPU nodes are ready
kubectl get nodes -l nebius.com/gpu-h100-a=present | wc -l
# Expected: 64

# Check total GPU capacity
kubectl describe nodes | grep "nvidia.com/gpu" | grep -c "8"
# Expected: 64 (each node has 8 GPUs)

# Verify InfiniBand connectivity (spot check)
kubectl exec -n ray-cluster <any-gpu-worker> -- ibstat | grep -E "State|Rate"
# Expected: State: Active, Rate: 400
```

### Step 3: Scale Training (KubeRay - Recommended)

Edit `k8s/ray-training-job.yaml`:

```python
# Change num_workers from 16 to 512
trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(
        num_workers=512,         # Was: 16 (64 nodes × 8 GPUs)
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 8},
    ),
    ...
)
```

Deploy:
```bash
kubectl apply -f k8s/ray-training-job.yaml
kubectl get rayjob -n ray-cluster -w  # Watch until RUNNING
```

### Step 3 (Alternative): Scale Training (StatefulSet + FSDP)

Edit `k8s/training-fsdp.yaml`:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: llama3-training
spec:
  replicas: 64              # Was: 2
  ...
  containers:
  - name: trainer
    env:
    - name: WORLD_SIZE
      value: "512"          # Was: 16 (64 × 8)
    - name: NNODES  
      value: "64"           # Was: 2
```

Update the ConfigMap accelerate config:
```yaml
# In training-config ConfigMap
num_machines: 64            # Was: 2
num_processes: 512          # Was: 16
```

Deploy:
```bash
kubectl delete statefulset llama3-training --cascade=foreground
kubectl apply -f k8s/training-fsdp.yaml
```

### Step 4: Adjust Hyperparameters for Scale

When scaling to 512 GPUs, adjust these hyperparameters:

| Parameter | 16 GPUs | 512 GPUs | Reason |
|-----------|---------|----------|--------|
| `per_device_train_batch_size` | 4 | 4 | Keep same |
| `gradient_accumulation_steps` | 4 | 1 | Reduce - effective batch already large |
| `learning_rate` | 1.5e-4 | 4.5e-4 | Scale √(512/16) ≈ 5.6× |
| `warmup_ratio` | 0.03 | 0.05 | Increase for stability |
| `num_train_epochs` | 3 | 1 | Fewer epochs needed |

**Effective batch size calculation:**
- 16 GPUs: 4 × 4 × 16 = 256 samples/step
- 512 GPUs: 4 × 1 × 512 = 2048 samples/step (8× larger)

Updated TrainingArguments:
```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,    # Reduced from 4
    learning_rate=4.5e-4,             # Scaled up
    warmup_ratio=0.05,                # Increased
    num_train_epochs=1,               # Reduced
    ...
)
```

### Scaling Reference Table

| Nodes | GPUs | WORLD_SIZE | num_workers | Effective Batch | Est. Time (3 epochs) |
|-------|------|------------|-------------|-----------------|----------------------|
| 2 | 16 | 16 | 16 | 256 | ~30 min |
| 4 | 32 | 32 | 32 | 512 | ~15 min |
| 8 | 64 | 64 | 64 | 1024 | ~8 min |
| 16 | 128 | 128 | 128 | 2048 | ~4 min |
| 32 | 256 | 256 | 256 | 4096 | ~2 min |
| 64 | 512 | 512 | 512 | 8192 | ~1 min |

> **Note**: Times are estimates for the example dataset (~20K samples). Actual times depend on sequence length and model size.

### Storage Considerations at Scale

| Nodes | Recommended Storage | Checkpoint Size | Bandwidth Needed |
|-------|---------------------|-----------------|------------------|
| 2-8 | 2 TB | ~2 GB | 4 GiB/s sufficient |
| 16-32 | 8 TB | ~2 GB | May need tuning |
| 64+ | 32+ TB | ~2 GB | Consider staggered checkpoints |

**Checkpoint strategy for 512 GPUs:**
```python
# Reduce checkpoint frequency at scale
save_steps=500,            # Was: 200
save_total_limit=2,        # Was: 3 (reduce storage)
```

### Network Considerations

InfiniBand fabric selection becomes critical at scale:

```hcl
# terraform.tfvars
# Try different fabrics if capacity exhausted
infiniband_fabric = "fabric-2"   # or fabric-3, fabric-4, fabric-5, fabric-6
```

**Verify NCCL ring topology:**
```bash
# Check NCCL is using all 8 InfiniBand NICs per node
kubectl logs <training-pod> | grep "NET/IB"
# Expected: [0]mlx5_0:1/IB ... [7]mlx5_7:1/IB
```

### Troubleshooting at Scale

#### 1. NCCL Timeout
```
NCCL WARN Timeout waiting for ring...
```
**Solution:** Increase NCCL timeout:
```yaml
env:
- name: NCCL_TIMEOUT
  value: "1800"  # 30 minutes (default: 30 seconds)
```

#### 2. OOM on AllReduce
```
CUDA out of memory during AllReduce
```
**Solution:** Enable gradient checkpointing and reduce batch size:
```python
gradient_checkpointing=True,
per_device_train_batch_size=2,  # Reduce from 4
```

#### 3. Slow Checkpointing
**Solution:** Use async checkpointing or checkpoint less frequently:
```python
save_steps=1000,  # Reduce frequency
```

#### 4. Node Failures
KubeRay handles this automatically. For StatefulSet:
```bash
# Identify failed pod
kubectl get pods | grep -v Running

# Delete and let StatefulSet recreate
kubectl delete pod llama3-training-<N>
```

### Scaling Checklist

Before scaling to 512 GPUs:

- [ ] Verify InfiniBand fabric has capacity (`nebius compute v1 infiniband list`)
- [ ] Check Nebius GPU quota (contact support if needed)
- [ ] Increase storage size proportionally
- [ ] Adjust learning rate (√N scaling rule)
- [ ] Reduce gradient accumulation steps
- [ ] Consider reducing checkpoint frequency
- [ ] Test at intermediate scale first (32 → 128 → 512)
- [ ] Monitor first few training steps for NCCL issues

---

## Evaluate Results

After training completes, run the inference demo to evaluate the fine-tuned model:

### Run Inference Demo

```bash
# Deploy the inference demo job
kubectl apply -f k8s/inference-test-job.yaml

# Watch the demo output
kubectl logs -n ray-cluster -f job/llama3-inference-demo
```

### What the Demo Shows

The demo compares the **base Llama-3-8B-Instruct** vs the **fine-tuned model** on function calling tasks:

| Test Category | Example Query |
|--------------|---------------|
| API Calls | "What's the weather in San Francisco?" |
| Computation | "What is 15% of 850?" |
| Data Operations | "Find all customers from New York" |
| Communication | "Send an email to john@example.com" |
| Scheduling | "Schedule a 30-minute standup tomorrow" |
| Financial | "Convert 500 USD to Japanese yen" |

### Evaluation Metrics

For each test case, the demo evaluates:
- **JSON Validity** - Does the response contain valid JSON?
- **Function Name** - Is the correct function being called?
- **Arguments** - Are all required arguments present?

### Expected Output

```
═══════════════════════════════════════════════════════════════════════
                      EVALUATION SUMMARY
═══════════════════════════════════════════════════════════════════════

Fine-tuned Model:
  ✓ Passed: 6/6
  Success Rate: 100.0%

Base Model:
  ✓ Passed: 2/6
  Success Rate: 33.3%

═══════════════════════════════════════════════════════════════════════
  IMPROVEMENT FROM FINE-TUNING: +66.7%
═══════════════════════════════════════════════════════════════════════
```

---

*Last updated: 2026-02-03*
