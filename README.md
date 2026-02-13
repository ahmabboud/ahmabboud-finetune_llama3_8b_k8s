# Llama-3 Function Calling Fine-tuning

Fine-tune Llama-3-8B-Instruct for function calling on 16x H100 GPUs using Ray Train.

## Overview

- **16 H100 80GB GPUs** (2 nodes × 8 GPUs) with InfiniBand
- **Ray Train + DDP** for distributed training
- **LoRA** for parameter-efficient fine-tuning
- **TRL SFTTrainer** for supervised fine-tuning
- **Kubernetes** with autoscaling GPU nodes

## Quick Start

### 1. Infrastructure

```bash
cd infra/k8s-installation
source environment.sh
terraform init && terraform apply
```

### 2. Secrets

```bash
source .env
kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN
kubectl create secret generic wandb-token --from-literal=key=$WANDB_API_KEY
```

### 3. Pipeline

```bash
# Step 1: Download model
./scripts/run-model-download.sh

# Step 2: Validate cluster (GPU health, NCCL, InfiniBand)
./scripts/run-preflight-check.sh

# Step 3: Prepare training data
./scripts/run-data-prep.sh

# Step 4: Run training
./scripts/submit-training-job.sh

# Step 5: Test inference (base vs fine-tuned)
./scripts/run-inference-test.sh
```

### Monitor Training

```bash
# Job status
kubectl get rayjob -n ray-cluster

# Dashboards
# Wandb: https://wandb.ai/<username>/llama3-function-calling
# Ray:   http://localhost:8265 (port-forward started by submit script)
```

## Project Structure

```
├── scripts/
│   ├── run-model-download.sh   # Download Llama-3-8B model
│   ├── run-preflight-check.sh  # Validate GPU cluster
│   ├── run-data-prep.sh        # Prepare training dataset
│   ├── submit-training-job.sh  # Submit Ray training job
│   ├── run-inference-test.sh   # Test fine-tuned model
│   └── inference_demo.py       # Inference comparison script
├── k8s/
│   ├── model-download-job.yaml
│   ├── preflight-check-job.yaml
│   ├── data-prep-job.yaml
│   ├── ray-training-job.yaml
│   └── inference-test-job.yaml
├── infra/                      # Terraform (Nebius MK8s)
└── docs/                       # Documentation
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | Llama-3-8B-Instruct |
| Dataset | glaiveai/glaive-function-calling-v2 |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| Batch size | 4 × 4 × 16 = 256 effective |
| Learning rate | 1.5e-4 |
| Epochs | 3 |
| GPUs | 16 (2 nodes × 8 H100) |

## Monitoring

| Dashboard | URL |
|-----------|-----|
| Wandb | https://wandb.ai/\<username\>/llama3-function-calling |
| Ray | http://localhost:8265 |
| Grafana | http://localhost:3000 |

## Results

Fine-tuned model achieves **100% success rate** on function calling tests vs **0%** for base model.

## Documentation

- [Infrastructure Quick Start](docs/Infrastructure_Quick_Start.md)
- [Training Guide](docs/Training_Guide.md)
- [Monitoring Guide](docs/Monitoring_Guide.md)


