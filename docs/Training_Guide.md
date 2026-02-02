# Training Guide

Multi-node fine-tuning of Llama-3-8B-Instruct on Kubernetes with PyTorch FSDP + LoRA.

## Overview

This guide covers:
- [Architecture](#architecture) - How the distributed training works
- [Quick Start](#quick-start) - Deploy training in minutes
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

## Scaling

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

### Training Logs

```bash
# Follow logs from master node
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

# All nodes
for pod in llama3-training-0 llama3-training-1; do
  echo "=== $pod ==="
  kubectl exec $pod -- nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
done
```

**Expected H100 utilization:** 70-100% GPU, 20-80GB memory per GPU

### Grafana Dashboard

```bash
kubectl port-forward -n o11y svc/grafana-and-prometheus 8080:80
# Open http://localhost:8080
# Login: admin / (see your .env for password)
```

### Weights & Biases

Training metrics are automatically logged to W&B if configured:
- Loss curves
- Learning rate schedule
- GPU memory usage
- Gradient norms

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
| [k8s/training-fsdp.yaml](../k8s/training-fsdp.yaml) | Main training manifest (StatefulSet + ConfigMap) |
| [k8s/smoke-test-job.yaml](../k8s/smoke-test-job.yaml) | Environment validation job |
| [k8s/data-prep-job.yaml](../k8s/data-prep-job.yaml) | Dataset preparation job |
| [k8s/model-download-job.yaml](../k8s/model-download-job.yaml) | Model download job |
| [scripts/smoke_test.py](../scripts/smoke_test.py) | Smoke test script |
| [src/training/fsdp_trainer.py](../src/training/fsdp_trainer.py) | FSDP trainer class |

## Appendix: Package Versions

Tested and validated versions:

| Package | Version |
|---------|---------|
| torch | 2.6.0+cu124 |
| transformers | 5.0.0 |
| trl | 0.27.1 |
| peft | 0.18.1 |
| accelerate | 1.12.0 |
| datasets | latest |

These are automatically installed by the training container from PyPI.
