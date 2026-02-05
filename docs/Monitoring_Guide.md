# Monitoring Guide

Comprehensive monitoring setup for Llama-3 fine-tuning on Kubernetes with 16x H100 GPUs.

## Overview

This project includes multiple monitoring tools, each serving a specific purpose:

| Tool | Purpose | Metrics | Access |
|------|---------|---------|--------|
| **Wandb** | Training metrics visualization | Loss, LR, grad norm, epoch | https://wandb.ai |
| **Ray Dashboard** | Job management & logs | Job status, workers, resources | localhost:8265 |
| **Grafana** | Infrastructure metrics | GPU temp, power, utilization | localhost:8080 |
| **Prometheus** | Metrics collection & storage | All infrastructure metrics | localhost:9090 |
| **NVIDIA DCGM** | GPU health & performance | Detailed GPU metrics | Via Prometheus/Grafana |

## Quick Access Commands

```bash
# Wandb - No port-forward needed (cloud hosted)
# https://wandb.ai/<your-username>/llama3-function-calling

# Ray Dashboard
kubectl port-forward -n ray-cluster svc/ray-cluster-head-svc 8265:8265
# Open http://localhost:8265

# Grafana
kubectl port-forward -n o11y svc/grafana-and-prometheus 8080:80
# Open http://localhost:8080
# Login: admin / run `terraform output -raw grafana_password` in infra/k8s-installation/

# Prometheus (direct access)
kubectl port-forward -n o11y svc/prometheus-server 9090:80
# Open http://localhost:9090
```

---

## 1. Weights & Biases (Wandb)

### What It Monitors
- **Training loss** - Primary indicator of model learning
- **Learning rate schedule** - Verifies LR warmup and decay
- **Gradient norms** - Detects exploding/vanishing gradients
- **Epoch progress** - Training completion tracking
- **Custom metrics** - Any metrics logged via `wandb.log()`

### Why Use It
Wandb is the **only tool that provides training metric charts**. While other tools show logs or infrastructure metrics, Wandb visualizes:
- Loss curves with smoothing
- Side-by-side run comparison
- Hyperparameter tracking
- Model artifact versioning

### Setup

**1. Create wandb secret (one-time):**
```bash
kubectl create secret generic wandb-token -n default --from-literal=key=$WANDB_API_KEY
```

**2. Deploy with wandb enabled:**
```bash
export WANDB_API_KEY=$(kubectl get secret wandb-token -n default -o jsonpath='{.data.key}' | base64 -d)
envsubst < k8s/ray-training-job.yaml | kubectl apply -f -
```

**3. View dashboard:**
```
https://wandb.ai/<your-username>/llama3-function-calling
```

### Key Metrics to Watch

| Metric | Healthy Range | Warning Signs | Action |
|--------|---------------|---------------|--------|
| `loss` | Decreasing trend | Flat or increasing | Check LR, data quality |
| `grad_norm` | 0.1 - 5.0 | > 10 (exploding) | Reduce LR, add gradient clipping |
| `grad_norm` | 0.1 - 5.0 | < 0.01 (vanishing) | Increase LR, check model init |
| `learning_rate` | Per schedule | Stuck at 0 | Check warmup config |
| `epoch` | Increasing | Stuck | Check for hangs/errors |

### Quick Actions

**Loss not decreasing:**
```bash
# Check for NaN/Inf in logs
kubectl logs -n ray-cluster -l job-name=llama3-finetuning | grep -E "nan|inf|NaN|Inf"

# Reduce learning rate
# Edit k8s/ray-training-job.yaml: learning_rate=5e-5 (was 1.5e-4)
```

**Gradient explosion (grad_norm > 10):**
```bash
# Add gradient clipping to TrainingArguments
max_grad_norm=1.0,
```

---

## 2. Ray Dashboard

### What It Monitors
- **Job status** - PENDING, RUNNING, SUCCEEDED, FAILED
- **Worker status** - Which workers are active
- **Resource usage** - CPU, GPU, memory allocation
- **Job logs** - Stdout/stderr from training
- **Cluster health** - Node availability

### Why Use It
Ray Dashboard is essential for:
- Verifying job submission succeeded
- Debugging job failures
- Monitoring cluster resource allocation
- Accessing job logs without kubectl

### Access

```bash
kubectl port-forward -n ray-cluster svc/ray-cluster-head-svc 8265:8265
# Open http://localhost:8265
```

### Key Pages

| Page | What to Check |
|------|---------------|
| **Jobs** | Job status, submission time, duration |
| **Cluster** | Node status, resource availability |
| **Logs** | Training output, errors |
| **Metrics** | Ray-specific metrics (tasks, actors) |

### Quick Actions

**Job stuck in PENDING:**
```bash
# Check if Ray cluster has resources
kubectl exec -n ray-cluster <ray-head-pod> -- ray status

# Expected: GPUs available
# If no GPUs: Check GPU worker pods
kubectl get pods -n ray-cluster -l ray.io/node-type=worker
```

**Job failed:**
```bash
# Get job logs
kubectl exec -n ray-cluster <ray-head-pod> -- ray job logs <job-id>

# Or via kubectl
kubectl logs -n ray-cluster -l job-name=llama3-finetuning --tail=200
```

**Workers not starting:**
```bash
# Check worker pod status
kubectl describe pod -n ray-cluster <gpu-worker-pod>

# Common issues:
# - Image pull errors: Check registry credentials
# - Resource limits: Verify GPU availability
# - Init container failures: Check InfiniBand setup
```

---

## 3. Grafana

### What It Monitors
- **GPU utilization** - SM (compute) activity percentage
- **GPU memory** - Used vs total VRAM
- **GPU temperature** - Thermal status
- **GPU power** - Power draw in watts
- **Node metrics** - CPU, RAM, disk, network

### Why Use It
Grafana provides infrastructure visibility:
- Verify GPUs are being utilized
- Detect thermal throttling
- Monitor memory pressure
- Track system health over time

### Access

```bash
kubectl port-forward -n o11y svc/grafana-and-prometheus 8080:80
# Open http://localhost:8080
# Login: admin / run `terraform output -raw grafana_password` in infra/k8s-installation/
```

### Pre-configured Dashboards

| Dashboard | Metrics |
|-----------|---------|
| **NVIDIA DCGM Exporter** | GPU utilization, memory, temp, power |
| **Node Exporter** | CPU, RAM, disk, network |
| **Kubernetes / Compute Resources** | Pod resource usage |

### Key Metrics to Watch

| Metric | Healthy Range | Warning Signs | Action |
|--------|---------------|---------------|--------|
| GPU Utilization | 70-100% | < 50% sustained | Check batch size, data loading |
| GPU Memory | 40-95% | > 98% or OOM | Reduce batch size, enable gradient checkpointing |
| GPU Temperature | 40-75°C | > 80°C | Check cooling, reduce workload |
| GPU Power | 300-700W (H100) | Near TDP limit | Normal under load |

### Quick Actions

**Low GPU utilization (< 50%):**
```bash
# Check if training is actually running
kubectl logs -n ray-cluster -l job-name=llama3-finetuning --tail=20

# Common causes:
# - Data loading bottleneck: Increase dataloader_num_workers
# - Small batch size: Increase per_device_train_batch_size
# - Evaluation running: Normal during eval steps
```

**GPU OOM (Out of Memory):**
```bash
# Reduce batch size
per_device_train_batch_size=2,  # Was 4

# Enable gradient checkpointing (should already be enabled)
gradient_checkpointing=True,

# Reduce sequence length
max_seq_length=1024,  # Was 2048
```

**High GPU temperature (> 80°C):**
```bash
# Check specific GPU
kubectl exec -n ray-cluster <gpu-worker-pod> -- nvidia-smi

# If sustained high temp:
# - Check datacenter cooling
# - Reduce batch size temporarily
# - Add delays between steps (not recommended for production)
```

---

## 4. Prometheus

### What It Monitors
Prometheus is the metrics backend that collects and stores:
- NVIDIA DCGM metrics (GPU)
- Node Exporter metrics (system)
- Kubernetes metrics (pods, services)
- Custom application metrics

### Why Use It
Direct Prometheus access is useful for:
- Custom queries (PromQL)
- Debugging metric collection
- Setting up alerts
- Verifying data is being scraped

### Access

```bash
kubectl port-forward -n o11y svc/prometheus-server 9090:80
# Open http://localhost:9090
```

### Useful PromQL Queries

**GPU Utilization (all GPUs):**
```promql
DCGM_FI_DEV_GPU_UTIL
```

**GPU Memory Usage:**
```promql
DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE * 100
```

**GPU Temperature:**
```promql
DCGM_FI_DEV_GPU_TEMP
```

**GPU Power Draw:**
```promql
DCGM_FI_DEV_POWER_USAGE
```

**Average GPU utilization across cluster:**
```promql
avg(DCGM_FI_DEV_GPU_UTIL)
```

### Quick Actions

**Metrics not appearing:**
```bash
# Check DCGM exporter pods
kubectl get pods -n nvidia-device-plugin -l app=nvidia-dcgm-exporter

# Check Prometheus targets
# Open http://localhost:9090/targets
# Look for dcgm-exporter targets - should be "UP"

# Check DCGM exporter logs
kubectl logs -n nvidia-device-plugin -l app=nvidia-dcgm-exporter --tail=50
```

---

## 5. NVIDIA DCGM (Data Center GPU Manager)

### What It Monitors
DCGM provides detailed GPU metrics:

| Metric | Description |
|--------|-------------|
| `DCGM_FI_DEV_GPU_UTIL` | GPU compute utilization % |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | Memory controller utilization % |
| `DCGM_FI_DEV_FB_USED` | Framebuffer (VRAM) used in MB |
| `DCGM_FI_DEV_FB_FREE` | Framebuffer free in MB |
| `DCGM_FI_DEV_GPU_TEMP` | GPU temperature in Celsius |
| `DCGM_FI_DEV_POWER_USAGE` | Power draw in Watts |
| `DCGM_FI_DEV_SM_CLOCK` | SM clock frequency in MHz |
| `DCGM_FI_DEV_MEM_CLOCK` | Memory clock frequency in MHz |
| `DCGM_FI_DEV_PCIE_TX_THROUGHPUT` | PCIe TX bandwidth |
| `DCGM_FI_DEV_PCIE_RX_THROUGHPUT` | PCIe RX bandwidth |
| `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` | NVLink bandwidth |

### Why Use It
DCGM metrics help diagnose:
- Performance bottlenecks (low utilization)
- Memory issues (OOM risk)
- Thermal throttling (high temp)
- Hardware problems (clock drops)

### Direct Access

```bash
# Run nvidia-smi on a GPU worker
kubectl exec -n ray-cluster <gpu-worker-pod> -- nvidia-smi

# Detailed query format
kubectl exec -n ray-cluster <gpu-worker-pod> -- nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv

# Continuous monitoring (every 2 seconds)
kubectl exec -n ray-cluster <gpu-worker-pod> -- nvidia-smi -l 2
```

### Quick Actions

**GPU clock throttling:**
```bash
# Check current clocks vs max
kubectl exec -n ray-cluster <gpu-worker-pod> -- nvidia-smi -q -d CLOCK

# If clocks are lower than expected:
# - Thermal throttling: Check temperature
# - Power throttling: Check power draw vs TDP
```

**NVLink issues:**
```bash
# Check NVLink status
kubectl exec -n ray-cluster <gpu-worker-pod> -- nvidia-smi nvlink -s

# Check NVLink errors
kubectl exec -n ray-cluster <gpu-worker-pod> -- nvidia-smi nvlink -e
```

---

## Monitoring Checklist

### Before Training

- [ ] Verify GPU workers are running: `kubectl get pods -n ray-cluster`
- [ ] Check GPU availability: `kubectl exec <gpu-pod> -- nvidia-smi`
- [ ] Verify InfiniBand: `kubectl exec <gpu-pod> -- ibstat | grep Active`
- [ ] Confirm wandb secret exists: `kubectl get secret wandb-token -n default`

### During Training

- [ ] Loss is decreasing (Wandb)
- [ ] GPU utilization > 70% (Grafana/nvidia-smi)
- [ ] No OOM errors (logs)
- [ ] Temperature < 80°C (Grafana/nvidia-smi)
- [ ] Job status is RUNNING (Ray Dashboard)

### After Training

- [ ] Job status is SUCCEEDED (Ray Dashboard)
- [ ] Final model saved: `kubectl exec <pod> -- ls /mnt/data/checkpoints/*/final`
- [ ] Wandb run finished cleanly
- [ ] No error logs

---

## Alerting (Optional)

### Prometheus Alertmanager

The o11y stack can be configured with alerts. Example alert rules:

```yaml
# GPU temperature alert
- alert: GPUHighTemperature
  expr: DCGM_FI_DEV_GPU_TEMP > 80
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "GPU {{ $labels.gpu }} temperature high"
    description: "GPU temperature is {{ $value }}°C"

# GPU memory alert
- alert: GPUMemoryHigh
  expr: (DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)) > 0.95
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "GPU {{ $labels.gpu }} memory > 95%"
```

### Wandb Alerts

Configure in Wandb UI:
1. Go to project settings
2. Add alert for `loss` not decreasing over N steps
3. Add alert for `grad_norm` > threshold

---

## Troubleshooting Quick Reference

| Symptom | Check | Action |
|---------|-------|--------|
| Job stuck PENDING | Ray Dashboard, `ray status` | Check GPU worker pods |
| Low GPU utilization | nvidia-smi, Grafana | Increase batch size, check data loading |
| OOM errors | Logs, nvidia-smi | Reduce batch size, enable gradient checkpointing |
| Loss not decreasing | Wandb | Check LR, data quality, gradient norms |
| Gradient explosion | Wandb (grad_norm) | Reduce LR, add gradient clipping |
| High GPU temp | Grafana, nvidia-smi | Check cooling, reduce workload |
| NCCL timeout | Logs | Check InfiniBand, increase timeout |
| Wandb not logging | Logs | Verify WANDB_API_KEY is set |

---

## Related Documentation

- [Training Guide](Training_Guide.md) - Full training instructions
- [Infrastructure Quick Start](Infrastructure_Quick_Start.md) - Cluster setup
- [INFRASTRUCTURE.md](INFRASTRUCTURE.md) - Detailed infrastructure docs

---

*Last updated: 2026-02-03*
