# GPU Cluster Acceptance Test

Distributed training acceptance test for GPU clusters. Used by Cloud/Infrastructure engineers to validate GPU cluster readiness.

## Features

- **Portable**: Uses open-source model (GPT-2) and dataset (WikiText)
- **Lightweight**: Minimal dependencies, runs in minutes
- **Multi-GPU Support**: H100, H200, B200, and other NVIDIA GPUs
- **Distributed**: Tests single-node and multi-node training
- **CI Ready**: Includes container build and test scripts

## Tests Performed

| Test | Description |
|------|-------------|
| GPU Detection | nvidia-smi, ECC status, temperature |
| CUDA/PyTorch | Version compatibility, bf16 support |
| Memory Limits | memlock ulimit (required for RDMA) |
| InfiniBand | HCA devices, port status, link speed |
| NCCL | Version, P2P access, AllReduce bandwidth |
| Storage I/O | Read/write throughput |
| Training | Actual distributed training step with gradient sync |

## Quick Start

### Build Container

```bash
docker build -t gpu-cluster-test:latest .
```

### Run Single Node Test

```bash
kubectl apply -f k8s/single-node-test.yaml
kubectl logs -f gpu-cluster-test
```

### Run Multi-Node Test

```bash
kubectl apply -f k8s/multi-node-test.yaml
kubectl logs -f gpu-cluster-test-0
```

### Cleanup

```bash
# Single node
kubectl delete job gpu-cluster-test

# Multi-node
kubectl delete statefulset gpu-cluster-test
kubectl delete svc gpu-cluster-test
kubectl delete configmap gpu-cluster-test-config
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GPUS_PER_NODE` | 8 | Number of GPUs per node |
| `NNODES` | 1 | Number of nodes |
| `MODEL_NAME` | gpt2 | HuggingFace model to use |
| `TEST_STEPS` | 10 | Training steps to run |
| `BATCH_SIZE` | 4 | Per-GPU batch size |

## CI/CD

GitHub Actions workflow included:

```bash
# Trigger manually or on push
gh workflow run ci.yaml
```

## Directory Structure

```
gpu-cluster-test/
├── Dockerfile              # Container build
├── README.md
├── requirements.txt        # Python dependencies
├── scripts/
│   ├── gpu_test.py         # Main test script
│   └── entrypoint.sh       # Container entrypoint
├── k8s/
│   ├── single-node-test.yaml
│   └── multi-node-test.yaml
└── .github/
    └── workflows/
        └── ci.yaml         # CI pipeline
```

## Exit Codes

- `0`: All tests passed
- `1`: One or more tests failed

---
Nebius AI Infrastructure
