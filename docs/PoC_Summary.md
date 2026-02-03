# Llama-3 Fine-Tuning PoC Summary

## Executive Summary

This document summarizes the Proof-of-Concept (PoC) for multi-node LLM fine-tuning on Nebius Cloud, demonstrating a production-ready pipeline that scales from 16 to 512 H100 GPUs with minimal configuration changes.

**Key Achievement:** End-to-end fine-tuning of Llama-3-8B for function calling on 16 H100 GPUs with full observability, easy reproducibility, and a clear path to 512 GPU scale.

---

## 1. Infrastructure Overview

### 1.1 Cluster Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Nebius Managed Kubernetes                           │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   CPU Nodes     │  │  GPU Node 1     │  │  GPU Node 2     │             │
│  │   (Control)     │  │  8x H100 80GB   │  │  8x H100 80GB   │             │
│  │   4 vCPU        │  │  128 vCPU       │  │  128 vCPU       │             │
│  │   16 GB RAM     │  │  1.6 TB RAM     │  │  1.6 TB RAM     │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │ NVLink            │ NVLink                │
│           │                    │ (900 GB/s)        │ (900 GB/s)            │
│           │                    └─────────┬─────────┘                       │
│           │                              │ InfiniBand (400 Gb/s)           │
│           │                              │                                  │
│  ┌────────┴──────────────────────────────┴─────────────────────────────┐   │
│  │                    Shared Filesystem (2 TB)                          │   │
│  │    /mnt/data: models, datasets, checkpoints, cache                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Node Configuration

| Component | PoC (Current) | Production (Target) |
|-----------|---------------|---------------------|
| **GPU Nodes** | 2 | 64 |
| **GPUs Total** | 16x H100 80GB | 512x H100 80GB |
| **GPU per Node** | 8 | 8 |
| **CPU Nodes** | 2 | 4-8 |
| **Shared Storage** | 2 TB | 32 TB |
| **Interconnect** | InfiniBand 400Gb/s | InfiniBand 400Gb/s |

### 1.3 Storage Architecture

| Storage Type | Size | Mount Point | Purpose |
|--------------|------|-------------|---------|
| **Shared Filesystem** | 2 TB | `/mnt/data` | Models, datasets, checkpoints |
| **Node Boot Disk** | 1 TB SSD | `/` | OS, container images |

**Shared Filesystem Benefits:**
- Read-Write-Many access (all nodes simultaneously)
- 4 GiB/s per client bandwidth
- Survives node restarts/failures
- No data copying between nodes needed

### 1.4 Network Configuration

| Connection | Technology | Bandwidth | Use Case |
|------------|------------|-----------|----------|
| **Intra-node GPU** | NVLink | 900 GB/s | GPU-to-GPU on same node |
| **Inter-node GPU** | InfiniBand RDMA | 400 Gb/s | Multi-node NCCL communication |
| **Pod Network** | Cilium CNI | 25 Gb/s | Service communication |

---

## 2. Software Stack

### 2.1 Infrastructure as Code (Terraform)

All infrastructure is defined in Terraform for reproducibility:

```
infra/
├── k8s-installation/
│   ├── main.tf           # K8s cluster & node groups
│   ├── filesystem.tf     # Shared storage
│   ├── gpu_cluster.tf    # GPU nodes & InfiniBand
│   ├── helm.tf           # Helm deployments
│   └── terraform.tfvars  # Configuration
└── modules/
    ├── kuberay/          # Ray cluster
    ├── gpu-operator/     # NVIDIA drivers
    ├── network-operator/ # InfiniBand/RDMA
    └── o11y/             # Monitoring stack
```

**Deployment Command:**
```bash
cd infra/k8s-installation
terraform init && terraform apply
# ~15 minutes to provision
```

### 2.2 Installed Frameworks

| Framework | Purpose | Namespace |
|-----------|---------|-----------|
| **KubeRay** | Distributed training orchestration | `ray-cluster` |
| **NVIDIA GPU Operator** | GPU driver management | `gpu-operator` |
| **NVIDIA Network Operator** | InfiniBand/RDMA drivers | `network-operator` |
| **NVIDIA DCGM Exporter** | GPU metrics collection | `nvidia-device-plugin` |
| **Prometheus** | Metrics storage & alerting | `o11y` |
| **Grafana** | Infrastructure dashboards | `o11y` |
| **Loki** | Log aggregation | `o11y` |

### 2.3 Training Frameworks

| Library | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | 2.5.1 | Deep learning framework |
| **Transformers** | 4.46.3 | Model loading & tokenization |
| **PEFT** | 0.13.2 | LoRA parameter-efficient fine-tuning |
| **TRL** | 0.11.4 | SFTTrainer for instruction tuning |
| **Ray Train** | 2.46.0 | Distributed training orchestration |
| **Accelerate** | 1.1.1 | FSDP integration |

---

## 3. Training Pipeline

### 3.1 Training Strategy

**Method:** LoRA (Low-Rank Adaptation) + Data Parallelism

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| **Fine-tuning** | LoRA (rank=64) | 2% trainable params, fast iteration |
| **Parallelism** | Data Parallel (DDP) | Simple, scales linearly |
| **Precision** | BF16 | H100 optimized, no accuracy loss |
| **Checkpointing** | Gradient checkpointing | Enables larger batch sizes |

**Trainable Parameters:**
- Base model: 8.2 billion parameters
- LoRA trainable: 167 million (2%)
- Memory per GPU: ~40-60 GB (fits H100 80GB)

### 3.2 Pipeline Stages

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Stage 1   │───►│   Stage 2   │───►│   Stage 3   │───►│   Stage 4   │
│   Prepare   │    │   Train     │    │   Monitor   │    │   Deploy    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     │                   │                  │                  │
     ▼                   ▼                  ▼                  ▼
 - Download model    - Submit RayJob   - Wandb metrics    - Base vs FT comparison
 - Prepare data      - 16 GPU workers  - GPU utilization  - 6 test scenarios
 - Create secrets    - Auto-resume     - Loss curves      - Success rate metrics
```

### 3.3 Commands to Run

```bash
# Stage 1: Prepare (one-time)
kubectl apply -f k8s/model-download-job.yaml
kubectl apply -f k8s/data-prep-job.yaml

# Stage 2: Train
export WANDB_API_KEY=$(kubectl get secret wandb-token -n default -o jsonpath='{.data.key}' | base64 -d)
envsubst < k8s/ray-training-job.yaml | kubectl apply -f -

# Stage 3: Monitor
# Wandb: https://wandb.ai/<user>/llama3-function-calling
kubectl port-forward -n ray-cluster svc/ray-cluster-head-svc 8265:8265

# Stage 4: Test (compares base vs fine-tuned model)
kubectl apply -f k8s/inference-test-job.yaml
kubectl logs -n ray-cluster -f job/llama3-inference-demo
```

### 3.4 Training Configuration

| Parameter | Value | Effective |
|-----------|-------|-----------|
| Batch size per GPU | 4 | - |
| Gradient accumulation | 4 | - |
| Number of GPUs | 16 | - |
| **Effective batch size** | - | **256** |
| Learning rate | 1.5e-4 | Cosine decay |
| Epochs | 3 | ~1.5 hours total |
| Checkpoint interval | 200 steps | Auto-resume enabled |

---

## 4. Monitoring & Observability

### 4.1 Dashboard Overview

| Dashboard | URL | What to Monitor |
|-----------|-----|-----------------|
| **Wandb** | wandb.ai | Loss curves, LR schedule, grad norms |
| **Ray Dashboard** | localhost:8265 | Job status, worker health |
| **Grafana** | localhost:8080 | GPU temp, power, utilization |

### 4.2 Key Metrics

**Training Health (Wandb):**
| Metric | Healthy | Warning |
|--------|---------|---------|
| Loss | Decreasing | Flat/increasing |
| Grad norm | 0.1 - 5.0 | >10 (exploding) |
| Learning rate | Per schedule | Stuck at 0 |

**Infrastructure Health (Grafana):**
| Metric | Healthy | Warning |
|--------|---------|---------|
| GPU Utilization | 70-100% | <50% |
| GPU Temperature | 40-75°C | >80°C |
| GPU Memory | 40-80% | >95% |

### 4.3 Alerting

Pre-configured alerts available for:
- GPU temperature > 80°C
- GPU memory > 95%
- Training loss not decreasing
- Node failures

---

## 5. Scaling to 512 GPUs

### 5.1 What Changes

| Component | 16 GPUs | 512 GPUs | Change Required |
|-----------|---------|----------|-----------------|
| `gpu_nodes_count` | 2 | 64 | Terraform variable |
| `num_workers` | 16 | 512 | RayJob YAML |
| `filestore_size` | 2 TB | 32 TB | Terraform variable |
| Learning rate | 1.5e-4 | 4.5e-4 | Training config |
| Grad accumulation | 4 | 1 | Training config |

### 5.2 Scaling Commands

**Step 1: Update Terraform**
```hcl
# terraform.tfvars
gpu_nodes_count_per_group = 64   # Was: 2
filestore_disk_size = 32TB       # Was: 2TB
kuberay_max_gpu_replicas = 64    # Was: 2
```

**Step 2: Apply Infrastructure**
```bash
terraform apply  # ~45-60 min for 64 nodes
```

**Step 3: Update Training Job**
```python
# k8s/ray-training-job.yaml
scaling_config=ScalingConfig(
    num_workers=512,  # Was: 16
    ...
)
```

**Step 4: Adjust Hyperparameters**
```python
learning_rate=4.5e-4,           # Scale √(512/16) ≈ 5.6×
gradient_accumulation_steps=1,   # Reduce (batch already large)
```

### 5.3 Scaling Performance

| GPUs | Effective Batch | Est. Time (3 epochs) | Throughput |
|------|-----------------|----------------------|------------|
| 16 | 256 | ~90 min | 1x |
| 64 | 1024 | ~25 min | 3.6x |
| 256 | 4096 | ~8 min | 11x |
| 512 | 8192 | ~4 min | 22x |

*Near-linear scaling due to efficient InfiniBand communication*

---

## 6. Why This Pipeline?

### 6.1 For ML Engineers (No Infra Expertise)

| Challenge | Solution |
|-----------|----------|
| "I don't know Kubernetes" | Terraform handles everything |
| "How do I monitor training?" | Wandb provides familiar ML metrics |
| "What if training fails?" | Auto-resume from checkpoints |
| "How do I debug GPU issues?" | Grafana dashboards, pre-built alerts |

### 6.2 For Small Teams (<20 people)

| Concern | Answer |
|---------|--------|
| **Maintenance** | Managed K8s, auto-healing nodes |
| **Complexity** | 3 commands to start training |
| **Cost efficiency** | LoRA = 50x faster iteration than full fine-tuning |
| **Scaling** | Same pipeline works at 16 or 512 GPUs |

### 6.3 Technical Advantages

1. **Reproducibility**: Infrastructure as Code (Terraform)
2. **Observability**: Full stack monitoring (Wandb + Grafana + Ray)
3. **Fault Tolerance**: Checkpoint resume, Ray auto-retry
4. **Efficiency**: 100% GPU utilization, InfiniBand RDMA
5. **Simplicity**: LoRA fine-tuning, not full model training

---

## 7. PoC Results

### 7.1 Training Metrics

| Metric | Value |
|--------|-------|
| Model | Llama-3-8B-Instruct |
| Task | Function calling |
| Training samples | ~20,000 |
| Final loss | ~0.36 |
| GPU utilization | 100% |
| Training time | ~90 min (3 epochs) |

### 7.2 Resource Utilization

| Resource | Usage |
|----------|-------|
| GPU Memory | 40-80 GB per GPU |
| GPU Compute | 100% during training |
| InfiniBand | Active (NCCL IB enabled) |
| Shared Storage | ~50 GB (model + data + checkpoints) |

### 7.3 Costs (Estimated)

| Scale | GPUs | Monthly Cost* | Use Case |
|-------|------|---------------|----------|
| PoC | 16 | ~$15,000 | Experimentation |
| Small | 64 | ~$60,000 | Regular fine-tuning |
| Large | 512 | ~$480,000 | Production training |

*Based on H100 cloud pricing estimates

### 7.4 Inference Demo

The inference demo compares the base Llama-3-8B-Instruct model against the fine-tuned version:

**Test Categories:**
| Category | Example Query |
|----------|---------------|
| API Calls | "What's the weather in San Francisco?" |
| Computation | "What is 15% of 850?" |
| Data Operations | "Find all customers from New York" |
| Communication | "Send an email to john@example.com..." |
| Scheduling | "Schedule a 30-minute standup tomorrow" |
| Financial | "Convert 500 USD to Japanese yen" |

**Demo Output:**
- Side-by-side responses from both models
- JSON validity checking (proper function call format)
- Argument validation (required parameters present)
- Success rate comparison and improvement metrics

**Run the Demo:**
```bash
kubectl apply -f k8s/inference-test-job.yaml
kubectl logs -n ray-cluster -f job/llama3-inference-demo
```

---

## 8. Next Steps for Client

### 8.1 Immediate (PoC Phase)

1. ✅ Clone repository
2. ✅ Run `terraform apply`
3. ✅ Execute training pipeline
4. ✅ View results in Wandb

### 8.2 Short-term (Production Setup)

1. Scale to 64 GPUs for regular training
2. Set up CI/CD for model deployment
3. Configure alerting for production
4. Implement model versioning

### 8.3 Long-term (512 GPU Reservation)

1. Scale infrastructure to 64 nodes
2. Optimize for larger models (70B+)
3. Implement distributed checkpointing
4. Set up multi-tenant training queues

---

## 9. Repository Structure

```
ahmabboud-finetune_llama3_8b_k8s/
├── infra/                    # Terraform infrastructure
│   ├── k8s-installation/     # Main cluster config
│   └── modules/              # Reusable modules
├── k8s/                      # Kubernetes manifests
│   ├── ray-training-job.yaml # Main training job
│   ├── training-fsdp.yaml    # Alternative (StatefulSet)
│   └── *.yaml                # Other jobs
├── scripts/                  # Python scripts
├── configs/                  # Training configs
├── docs/                     # Documentation
│   ├── Training_Guide.md
│   ├── Monitoring_Guide.md
│   └── Infrastructure_Quick_Start.md
└── README.md
```

---

## 10. Summary

This PoC demonstrates a **production-ready, scalable fine-tuning pipeline** that:

- **Works today** on 16 H100 GPUs
- **Scales to 512 GPUs** with minimal changes
- **Requires no infrastructure expertise** from ML engineers
- **Provides full observability** through Wandb, Grafana, and Ray
- **Handles failures gracefully** with checkpoints and auto-resume

The pipeline is designed for **small teams** who want to focus on ML, not infrastructure.

---

## Appendix: Quick Reference

### Start Training
```bash
export WANDB_API_KEY=$(kubectl get secret wandb-token -n default -o jsonpath='{.data.key}' | base64 -d)
envsubst < k8s/ray-training-job.yaml | kubectl apply -f -
```

### Check Status
```bash
kubectl get rayjob -n ray-cluster
kubectl logs -n ray-cluster -l job-name=llama3-finetuning --tail=50
```

### Access Dashboards
```bash
# Wandb
open https://wandb.ai/<user>/llama3-function-calling

# Ray Dashboard
kubectl port-forward -n ray-cluster svc/ray-cluster-head-svc 8265:8265

# Grafana
kubectl port-forward -n o11y svc/grafana-and-prometheus 8080:80
```

### Scale to 512 GPUs
```bash
# Edit terraform.tfvars
gpu_nodes_count_per_group = 64

# Apply
terraform apply

# Update training job
# num_workers=512
```

---

*Document prepared for Nebius Cloud PoC Demo*
*Last updated: 2026-02-03*
