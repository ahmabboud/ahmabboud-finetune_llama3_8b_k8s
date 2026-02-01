# Infrastructure Quick Start Guide

This guide covers how to deploy, manage, and debug the K8s training cluster on Nebius.

---

## Prerequisites

### Required Tools

| Tool | Installation | Verify |
|------|--------------|--------|
| Nebius CLI | `curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh \| bash` | `nebius version` |
| Terraform | `brew install terraform` | `terraform version` |
| kubectl | `brew install kubectl` | `kubectl version` |
| jq | `brew install jq` | `jq --version` |

### Nebius CLI Configuration

```bash
# Configure CLI (first time only)
nebius init

# Verify authentication
nebius iam whoami
```

---

## Deploy Infrastructure

### 1. Configure Environment Variables

Edit `infra/k8s-installation/environment.sh`:

```bash
NEBIUS_TENANT_ID='your-tenant-id'
NEBIUS_PROJECT_ID='your-project-id'
NEBIUS_REGION='eu-north1'
```

### 2. Configure Cluster Settings

Edit `infra/k8s-installation/terraform.tfvars`:

```hcl
# SSH - use your public key
ssh_public_key = {
  path = "~/.ssh/id_ed25519.pub"
}

# GPU Configuration
gpu_nodes_count_per_group = 2          # Number of GPU nodes
gpu_nodes_platform = "gpu-h100-sxm"    # H100 GPUs
gpu_nodes_preset = "8gpu-128vcpu-1600gb"
infiniband_fabric = "fabric-6"         # Check availability

# Storage
filestore_disk_size = 2 * (1024 * 1024 * 1024 * 1024)  # 2TB
```

### 3. Deploy

```bash
cd infra/k8s-installation

# Initialize environment (creates bucket, service account, etc.)
source ./environment.sh

# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Deploy (takes ~30-40 minutes)
terraform apply
```

### 4. Get Cluster Credentials

```bash
# Get cluster ID from terraform output
cd infra/k8s-installation
CLUSTER_ID=$(terraform output -raw kube_cluster | grep -o 'mk8scluster-[^"]*' | head -1)

# Or get from Nebius directly
CLUSTER_ID=$(nebius mk8s v1 cluster list --parent-id $NEBIUS_PROJECT_ID --format json | jq -r '.items[0].metadata.id')

# Get kubectl credentials
nebius mk8s v1 cluster get-credentials --id $CLUSTER_ID --external

# Verify connection
kubectl get nodes
```

---

## Access Grafana Dashboard

### Start Port-Forward

```bash
kubectl -n o11y port-forward svc/grafana-and-prometheus 8080:80
```

### Login

- **URL:** http://localhost:8080
- **Username:** `admin`
- **Password:** Run `terraform output -raw grafana_password` in `infra/k8s-installation/`

### Key Dashboards

| Dashboard | Purpose |
|-----------|---------|
| Node Exporter | CPU, memory, disk metrics |
| NVIDIA DCGM | GPU utilization, memory, temperature |

---

## Essential kubectl Commands

### Cluster Status

```bash
# Check all nodes
kubectl get nodes -o wide

# Check node details (including GPU info)
kubectl describe nodes | grep -A10 "Allocatable"

# Check GPU availability
kubectl describe nodes | grep -A5 "nvidia.com/gpu"
```

### Pod Management

```bash
# All pods across namespaces
kubectl get pods -A

# Pods in specific namespace
kubectl get pods -n <namespace>

# Pod details
kubectl describe pod <pod-name> -n <namespace>

# Pod logs
kubectl logs <pod-name> -n <namespace>
kubectl logs <pod-name> -n <namespace> --tail=100 -f  # Follow
```

### Resource Monitoring

```bash
# Node resource usage
kubectl top nodes

# Pod resource usage
kubectl top pods -A
```

### Namespaces

```bash
# List namespaces
kubectl get ns

# Default namespaces after deployment:
# - default
# - o11y (monitoring)
# - nvidia-device-plugin
# - network-operator
```

---

## Debugging

### Node Issues

```bash
# Check node conditions
kubectl describe node <node-name> | grep -A20 "Conditions"

# Check node events
kubectl get events --field-selector involvedObject.kind=Node

# SSH to a node (if it has public IP)
ssh ubuntu@<node-external-ip>
```

### GPU Issues

```bash
# Verify NVIDIA device plugin
kubectl get pods -n nvidia-device-plugin
kubectl logs -n nvidia-device-plugin -l app=nvidia-device-plugin

# Check GPU resources on nodes
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'

# Test GPU access (run a test pod)
kubectl run gpu-test --image=nvidia/cuda:12.0-base --rm -it --restart=Never \
  --limits=nvidia.com/gpu=1 -- nvidia-smi
```

### Network/InfiniBand Issues

```bash
# Check network operator
kubectl get pods -n network-operator
kubectl logs -n network-operator -l app=network-operator

# Verify InfiniBand devices on GPU nodes
kubectl exec -it <gpu-pod> -- ibstat
```

### Storage Issues

```bash
# Check PersistentVolumes
kubectl get pv

# Check PersistentVolumeClaims
kubectl get pvc -A

# Verify filestore mount on nodes
ssh ubuntu@<node-ip> "df -h | grep mnt"
```

### Pod Troubleshooting

```bash
# Pod won't start - check events
kubectl describe pod <pod-name> -n <namespace>

# Pod in CrashLoopBackOff - check logs
kubectl logs <pod-name> -n <namespace> --previous

# Exec into running pod
kubectl exec -it <pod-name> -n <namespace> -- /bin/bash
```

---

## Monitoring Stack

### Components

| Component | Namespace | Purpose |
|-----------|-----------|---------|
| Grafana | o11y | Dashboards |
| Prometheus | o11y | Metrics collection |
| Node Exporter | o11y | Host metrics |
| DCGM Exporter | nvidia-device-plugin | GPU metrics |

### Check Monitoring Health

```bash
# All monitoring pods
kubectl get pods -n o11y

# Prometheus targets
kubectl port-forward -n o11y svc/prometheus-server 9090:80
# Open http://localhost:9090/targets
```

---

## Infrastructure Management

### Update Configuration

```bash
cd infra/k8s-installation
source ./environment.sh  # Refresh credentials
terraform plan           # Preview changes
terraform apply          # Apply changes
```

### Scale GPU Nodes

Edit `terraform.tfvars`:
```hcl
gpu_nodes_count_per_group = 4  # Change from 2 to 4
```

Then apply:
```bash
terraform apply
```

### Destroy Infrastructure

```bash
cd infra/k8s-installation
source ./environment.sh
terraform destroy
```

⚠️ **Warning:** This deletes ALL resources including storage!

---

## Common Issues & Solutions

### "Resource Exhausted" when creating GPU nodes

**Cause:** No GPU capacity in the selected fabric.

**Solution:** Try a different fabric in `terraform.tfvars`:
```hcl
infiniband_fabric = "fabric-6"  # Try fabric-2, fabric-3, fabric-4, etc.
```

### kubectl connection refused

**Cause:** Credentials expired or not configured.

**Solution:**
```bash
source ./environment.sh
nebius mk8s v1 cluster get-credentials --id <cluster-id> --external
```

### Grafana port-forward disconnects

**Cause:** Idle timeout or network issues.

**Solution:** Restart the port-forward:
```bash
kubectl -n o11y port-forward svc/grafana-and-prometheus 8080:80
```

### Pods stuck in Pending

**Cause:** Insufficient resources or node not ready.

**Solution:**
```bash
# Check why
kubectl describe pod <pod-name>

# Check node status
kubectl get nodes
kubectl describe node <node-name>
```

---

## Useful Aliases

Add to your `~/.zshrc` or `~/.bashrc`:

```bash
# Kubernetes shortcuts
alias k='kubectl'
alias kgp='kubectl get pods'
alias kgn='kubectl get nodes'
alias kga='kubectl get all -A'
alias kd='kubectl describe'
alias kl='kubectl logs'
alias kex='kubectl exec -it'

# Grafana quick access
alias grafana='kubectl -n o11y port-forward svc/grafana-and-prometheus 8080:80'

# GPU check
alias gpus='kubectl describe nodes | grep -A5 "nvidia.com/gpu"'
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Deploy cluster | `source environment.sh && terraform apply` |
| Get credentials | `nebius mk8s v1 cluster get-credentials --id <id> --external` |
| Check nodes | `kubectl get nodes -o wide` |
| Check GPUs | `kubectl describe nodes \| grep nvidia.com/gpu` |
| Open Grafana | `kubectl -n o11y port-forward svc/grafana-and-prometheus 8080:80` |
| View pod logs | `kubectl logs <pod> -n <ns> -f` |
| SSH to node | `ssh ubuntu@<external-ip>` |
| Destroy cluster | `terraform destroy` |

---

*Last updated: 2026-02-01*
