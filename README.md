# Llama-3 Function Calling Fine-tuning

Multi-node fine-tuning of Llama-3-8B-Instruct for function calling on 16x H100 GPUs.

## Overview

This project demonstrates efficient multi-node LLM fine-tuning using:
- **16 H100 GPUs** (2 nodes × 8 GPUs) with InfiniBand interconnect
- **PyTorch FSDP** for distributed training
- **LoRA (PEFT)** for parameter-efficient fine-tuning (2% trainable params)
- **TRL SFTTrainer** for supervised fine-tuning
- **Kubernetes StatefulSet** for orchestration

## Quick Start

### 1. Infrastructure Setup

```bash
cd infra/k8s-installation
source environment.sh
terraform init && terraform apply
```

See [docs/Infrastructure_Quick_Start.md](docs/Infrastructure_Quick_Start.md) for details.

### 2. Create Secrets

```bash
source .env
kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN
kubectl create secret generic wandb-token --from-literal=key=$WANDB_API_KEY
```

### 3. Prepare Data & Model

```bash
# Download model
kubectl apply -f k8s/model-download-job.yaml

# Prepare training data
kubectl apply -f k8s/data-prep-job.yaml
```

### 4. Run Training

```bash
# Run smoke test first (recommended)
kubectl apply -f k8s/smoke-test-job.yaml
kubectl logs -f job/training-smoke-test

# Deploy FSDP training (2 nodes, 16 GPUs)
kubectl apply -f k8s/training-fsdp.yaml

# Monitor
kubectl logs -f llama3-training-0
```

### 5. Evaluate Results

```bash
# Run inference demo (auto-detects latest checkpoint)
kubectl apply -f k8s/inference-test-job.yaml

# Watch the demo output
kubectl logs -n ray-cluster -f job/llama3-inference-demo
```

The demo automatically finds the latest checkpoint (or `final` model if training completed) and runs 6 diverse test cases:
- Side-by-side comparison of base model vs fine-tuned model
- JSON validity checking and argument validation
- Success rate metrics for both models
- Improvement percentage from fine-tuning

**Test categories:** Weather, Math, Database, Email, Scheduling, Currency

## Project Structure

```
├── src/
│   ├── data/           # Data loading & preprocessing
│   └── training/       # Distributed trainer & configs
├── scripts/
│   ├── prepare_data.py   # Download & process dataset
│   ├── train.py          # Main training script
│   ├── inference_demo.py # Enhanced inference demo (base vs fine-tuned)
│   ├── inference.py      # Simple inference test
│   ├── evaluate.py       # Measure accuracy
│   └── benchmark.py      # GPU performance test
├── configs/
│   └── train_config.yaml
├── k8s/                # Kubernetes manifests
├── infra/              # Terraform infrastructure
└── docs/               # Documentation
```

## Configuration

Key settings in `configs/train_config.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | Llama-3-8B-Instruct | Base model |
| LoRA rank | 64 | Trainable parameters |
| Batch size | 4 × 4 × 16 = 256 | Effective batch |
| Learning rate | 2e-4 | With warmup |
| Epochs | 3 | Training duration |

## Monitoring

### Training Metrics (Wandb - Recommended)

```bash
# Deploy with wandb enabled
export WANDB_API_KEY=$(kubectl get secret wandb-token -n default -o jsonpath='{.data.key}' | base64 -d)
envsubst < k8s/ray-training-job.yaml | kubectl apply -f -

# View dashboard at: https://wandb.ai/<your-username>/llama3-function-calling
```

### GPU Metrics (Grafana)

```bash
kubectl port-forward -n o11y svc/grafana-and-prometheus 8080:80
# Open http://localhost:8080 (admin / see .env for password)
```

### Ray Dashboard (Job Management)

```bash
kubectl port-forward -n ray-cluster svc/ray-cluster-head-svc 8265:8265
# Open http://localhost:8265
```

### Training Logs

```bash
kubectl logs llama3-training-0 | grep -E "loss.*epoch"
kubectl exec llama3-training-0 -- nvidia-smi
```

## Documentation

- [PoC Summary](docs/PoC_Summary.md) - Executive overview & scaling guide
- [Infrastructure Quick Start](docs/Infrastructure_Quick_Start.md)
- [Training Guide](docs/Training_Guide.md)
- [Monitoring Guide](docs/Monitoring_Guide.md)
- [Technical Implementation Plan](docs/Technical_Implementation_Plan.md)

## License

MIT
