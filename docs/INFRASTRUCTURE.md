# Infrastructure Documentation

This document describes the infrastructure setup for the Llama-3 fine-tuning project on Nebius Cloud.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Nebius Cloud                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    Managed Kubernetes Cluster                       │     │
│  │                                                                     │     │
│  │   ┌─────────────────┐          ┌─────────────────┐                 │     │
│  │   │   GPU Node 1    │          │   GPU Node 2    │                 │     │
│  │   │   8x H100 80GB  │          │   8x H100 80GB  │                 │     │
│  │   │                 │          │                 │                 │     │
│  │   │  ┌───────────┐  │          │  ┌───────────┐  │                 │     │
│  │   │  │ Training  │  │◄────────►│  │ Training  │  │                 │     │
│  │   │  │  Pod 0    │  │  NCCL    │  │  Pod 1    │  │                 │     │
│  │   │  │ (rank 0-7)│  │InfiniBand│  │(rank 8-15)│  │                 │     │
│  │   │  └───────────┘  │          │  └───────────┘  │                 │     │
│  │   │        │        │          │        │        │                 │     │
│  │   │        ▼        │          │        ▼        │                 │     │
│  │   │   /mnt/data     │          │   /mnt/data     │                 │     │
│  │   └────────┬────────┘          └────────┬────────┘                 │     │
│  │            │                            │                          │     │
│  └────────────┼────────────────────────────┼──────────────────────────┘     │
│               │                            │                                 │
│               └──────────┬─────────────────┘                                 │
│                          │ virtiofs                                          │
│                          ▼                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │              Nebius Shared Filesystem (2 TB)                        │     │
│  │                                                                     │     │
│  │   /mnt/data/                                                        │     │
│  │   ├── models/           # Downloaded base models                    │     │
│  │   ├── datasets/         # Training data (train.jsonl, val.jsonl)   │     │
│  │   ├── checkpoints/      # Training checkpoints                      │     │
│  │   └── cache/            # HuggingFace cache                         │     │
│  │                                                                     │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Compute Resources

### GPU Nodes

| Property | Value |
|----------|-------|
| **Node Count** | 2 |
| **GPU per Node** | 8x NVIDIA H100 80GB HBM3 |
| **Total GPUs** | 16 |
| **Total GPU Memory** | 1.28 TB |
| **OS** | Ubuntu 24.04.3 LTS |
| **Kernel** | 6.11.0-1016-nvidia |
| **Container Runtime** | containerd 1.7.30 |
| **Kubernetes Version** | v1.32.9 |

### GPU Interconnect

| Property | Value |
|----------|-------|
| **Intra-node** | NVLink (900 GB/s per GPU) |
| **Inter-node** | InfiniBand (RDMA) |
| **NCCL Backend** | InfiniBand verbs |

## Storage

### Nebius Storage Options

Nebius offers several storage types. See [Nebius Storage Documentation](https://docs.nebius.com/compute/storage/types) for details.

#### Disk Types (Block Storage)

| Type | Max Read BW | Max Write BW | Max Read IOPS | Max Write IOPS | Reliability | Price/GiB/mo |
|------|-------------|--------------|---------------|----------------|-------------|--------------|
| **Network SSD** | 450 MiB/s | 450 MiB/s | 20,000 | 40,000 | Erasure coding (2 failures) | $0.071 |
| **Network SSD NRD** | 1 GiB/s | 1 GiB/s | 75,000 | 75,000 | None ❌ | $0.053 |
| **Network SSD IO M3** | 1 GiB/s | 1 GiB/s | 75,000 | 75,000 | 3-way replication ✅ | $0.118 |

> **Note**: Disks can only be attached to **one VM at a time** - not suitable for shared multi-node training.

#### Shared Filesystem (File Storage) - Used in This Project

| Type | Max BW/Client | Max Aggregate Read BW | Max Aggregate Write BW | Max Read IOPS | Max Write IOPS |
|------|---------------|----------------------|------------------------|---------------|----------------|
| **SSD (`network_ssd`)** | 4 GiB/s | 300 GiB/s | 100 GiB/s | 1,500,000 | 750,000 |

> **Note**: Shared filesystems only come in one type (`network_ssd`). SSD IO M3 is not available for shared filesystems.

#### Nebius Recommendations for ML Workloads

| Use Case | Recommended Storage |
|----------|---------------------|
| **Streaming datasets to workers** | SSD shared filesystems or Object Storage |
| **Sharing code between workers** | SSD shared filesystems |
| **Checkpoints during training** | SSD shared filesystems |
| **Long-term checkpoint storage** | Object Storage (S3) |
| **Inference weight sharing** | SSD shared filesystems |

### Current Configuration: Nebius Shared Filesystem

This project uses a Nebius Compute Filesystem - a network-attached, shared storage system accessible from all GPU nodes simultaneously.

| Property | Value |
|----------|-------|
| **Type** | Nebius Compute Filesystem (SSD) |
| **Terraform Type** | `NETWORK_SSD` |
| **Mount Protocol** | virtiofs |
| **Total Size** | 2 TB |
| **Mount Path** | `/mnt/data` |
| **Access Mode** | Read/Write from all nodes simultaneously |
| **Max Bandwidth per Client** | 4 GiB/s |
| **Reliability** | Erasure coding (tolerates 2 failures) |
| **Durability** | ✅ Survives node restarts/deletions |
| **Encryption** | ✅ Enabled by default |

### Data Layout

```
/mnt/data/
├── models/
│   └── llama3-8b-instruct/          # Meta-Llama-3-8B-Instruct (~16GB)
├── datasets/
│   ├── train.jsonl                   # Training data
│   └── val.jsonl                     # Validation data
├── checkpoints/
│   └── llama3-function-calling/
│       ├── checkpoint-600/
│       ├── checkpoint-800/
│       └── checkpoint-1000/          # Latest checkpoint (~2GB)
└── cache/
    └── huggingface/                  # HF model cache
```

### Storage Configuration (Terraform)

The filesystem is provisioned via Terraform in `infra/k8s-installation/filesystem.tf`:

```hcl
resource "nebius_compute_v1_filesystem" "shared-filesystem" {
  count            = var.enable_filestore ? 1 : 0
  parent_id        = var.parent_id
  name             = join("-", ["filesystem-tf", local.release-suffix])
  type             = var.filestore_disk_type
  size_bytes       = var.filestore_disk_size      # 2TB
  block_size_bytes = var.filestore_block_size
}
```

Configuration in `terraform.tfvars`:
```hcl
enable_filestore     = true
filestore_disk_size  = 2 * (1024 * 1024 * 1024 * 1024)  # 2TB
```

### Kubernetes Volume Mounting

All pods use `hostPath` volumes to access the shared filesystem:

```yaml
volumes:
  - name: data
    hostPath:
      path: /mnt/data
      type: Directory
```

This works because the Nebius Filesystem is mounted at `/mnt/data` on every node at the OS level (via cloud-init and virtiofs), so `hostPath` effectively references the shared network storage.

## Networking

### Kubernetes Services

| Service | Namespace | Purpose |
|---------|-----------|---------|
| `llama3-training` | default | Headless service for StatefulSet DNS |
| `grafana-and-prometheus` | o11y | Monitoring dashboard |

### Multi-Node Training Communication

Training pods discover each other via DNS:
- `llama3-training-0.llama3-training.default.svc.cluster.local`
- `llama3-training-1.llama3-training.default.svc.cluster.local`

NCCL uses InfiniBand RDMA for efficient GPU-to-GPU communication across nodes.

### Required Security Capabilities

For InfiniBand RDMA to work, pods need:

```yaml
securityContext:
  capabilities:
    add:
      - IPC_LOCK        # Lock memory for RDMA
      - SYS_RESOURCE    # Increase memlock limits
```

And the entrypoint must set:
```bash
ulimit -l unlimited
```

Without these, NCCL fails with: `ibv_create_qp failed with error Cannot allocate memory`

## Observability Stack

| Component | Purpose |
|-----------|---------|
| **Prometheus** | Metrics collection |
| **Grafana** | Dashboards and visualization |
| **Loki** | Log aggregation |
| **DCGM Exporter** | GPU metrics |

Access Grafana:
```bash
kubectl port-forward -n o11y svc/grafana-and-prometheus 8080:80
# Open http://localhost:8080
```

## Infrastructure as Code

### Directory Structure

```
infra/
├── k8s-installation/
│   ├── main.tf              # Main cluster configuration
│   ├── filesystem.tf        # Shared filesystem
│   ├── variables.tf         # Input variables
│   ├── terraform.tfvars     # Variable values
│   └── environment.sh       # Environment setup
└── modules/
    ├── instance/            # GPU node module
    ├── nfs-server/          # NFS server module (alternative)
    └── cilium-egress-gateway/
```

### Deploying Infrastructure

```bash
cd infra/k8s-installation
source environment.sh
terraform init
terraform plan
terraform apply
```

## Backup & Recovery

### Checkpoint Backup

The shared filesystem provides durability, but for additional safety, checkpoints can be backed up locally:

```bash
# Create helper pod
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: checkpoint-copy
spec:
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
  tolerations:
    - key: "nvidia.com/gpu"
      operator: "Exists"
      effect: "NoSchedule"
  containers:
  - name: copy
    image: busybox
    command: ["sleep", "3600"]
    volumeMounts:
    - name: data
      mountPath: /mnt/data
  volumes:
  - name: data
    hostPath:
      path: /mnt/data
  restartPolicy: Never
EOF

# Wait and copy
kubectl wait --for=condition=Ready pod/checkpoint-copy
kubectl exec checkpoint-copy -- tar czf - -C /mnt/data/checkpoints/llama3-function-calling checkpoint-1000 > checkpoint-1000.tar.gz

# Cleanup
kubectl delete pod checkpoint-copy
```

### Restoring Checkpoints

```bash
# Create helper pod (same as above)
# Then upload:
cat checkpoint-1000.tar.gz | kubectl exec -i checkpoint-copy -- tar xzf - -C /mnt/data/checkpoints/llama3-function-calling/
```

## Cost Considerations

| Resource | Approximate Cost Factor |
|----------|------------------------|
| 2x GPU Nodes (8x H100 each) | Highest cost |
| 2TB Shared Filesystem | Medium cost |
| Kubernetes Control Plane | Included |
| Network Egress | Per GB |

**Recommendation**: Delete GPU nodes when not training to minimize costs. The filesystem persists independently.

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| NCCL InfiniBand error | Add `IPC_LOCK`, `SYS_RESOURCE` capabilities |
| Pod can't access `/mnt/data` | Ensure pod is scheduled on GPU node with filesystem mount |
| Checkpoint not found | Check if pod is on same node or using shared filesystem |
| OOM during training | Reduce batch size or enable gradient checkpointing |

### Useful Commands

```bash
# Check GPU node status
kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-H100-80GB-HBM3

# Check filesystem mount on node
kubectl debug node/<node-name> -it --image=busybox -- df -h /host/mnt/data

# Check filesystem contents
kubectl debug node/<node-name> -it --image=busybox -- ls -la /host/mnt/data/

# Check GPU utilization
kubectl exec <pod-name> -- nvidia-smi

# Check NCCL connectivity
kubectl exec <pod-name> -- python -c "import torch.distributed as dist; ..."
```
