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
kubectl apply -f k8s/inference-job.yaml
```

## Project Structure

```
├── src/
│   ├── data/           # Data loading & preprocessing
│   └── training/       # Distributed trainer & configs
├── scripts/
│   ├── prepare_data.py # Download & process dataset
│   ├── train.py        # Main training script
│   ├── inference.py    # Test fine-tuned model
│   ├── evaluate.py     # Measure accuracy
│   └── benchmark.py    # GPU performance test
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

```bash
# GPU metrics in Grafana
kubectl port-forward -n o11y svc/grafana-and-prometheus 8080:80
# Open http://localhost:8080 (admin / see .env for password)

# Training logs (loss, accuracy)
kubectl logs llama3-training-0 | grep -E "loss.*epoch"

# GPU utilization
kubectl exec llama3-training-0 -- nvidia-smi
```

## Documentation

- [Infrastructure Quick Start](docs/Infrastructure_Quick_Start.md)
- [Training Guide](docs/Training_Guide.md)
- [Technical Implementation Plan](docs/Technical_Implementation_Plan.md)

## License

MIT
