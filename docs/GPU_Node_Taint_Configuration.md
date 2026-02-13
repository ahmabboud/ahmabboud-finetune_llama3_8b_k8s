# GPU Node Taint Configuration

## Overview
This configuration prevents non-GPU workloads from being scheduled on expensive GPU nodes, ensuring they scale to zero when idle.

## Changes Made

### 1. Terraform Configuration

#### Added Variable ([infra/k8s-installation/variables.tf](../infra/k8s-installation/variables.tf))
```terraform
variable "gpu_node_taints" {
  description = "Taints to apply to GPU nodes to prevent non-GPU workloads from being scheduled."
  type = list(object({
    key    = string
    value  = string
    effect = string
  }))
  default = [
    {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }
  ]
}
```

#### Applied to GPU Nodes ([infra/k8s-installation/main.tf](../infra/k8s-installation/main.tf))
```terraform
taints = length(var.gpu_node_taints) > 0 ? var.gpu_node_taints : null
```

### 2. KubeRay GPU Worker Tolerations

Updated [infra/modules/kuberay/files/ray-values.yaml.tftpl](../infra/modules/kuberay/files/ray-values.yaml.tftpl) to add:
```yaml
tolerations:
  - key: nvidia.com/gpu
    operator: Equal
    value: "present"
    effect: NoSchedule
```

### 3. Existing K8s Jobs

The following jobs already have proper tolerations:
- ✅ `k8s/preflight-check-job.yaml`
- ✅ `k8s/inference-test-job.yaml`
- ✅ `k8s/ray-training-job.yaml` (uses Ray cluster workers)

## How It Works

1. **GPU nodes are tainted** with `nvidia.com/gpu=present:NoSchedule`
2. **Regular pods** (monitoring, Kubeflow, etc.) **cannot schedule** on GPU nodes
3. **GPU workloads** must have matching **tolerations** to use GPU nodes
4. **Autoscaler can scale GPU nodes to zero** when no GPU pods exist

## Deployment

### Apply Changes
```bash
cd infra/k8s-installation

# Initialize/update Terraform
terraform init

# Review changes
terraform plan

# Apply changes (this will taint GPU nodes and update KubeRay)
terraform apply
```

### Verify Taints
```bash
kubectl get nodes -l nebius.com/gpu=true -o custom-columns='NAME:.metadata.name,TAINTS:.spec.taints[*].key'
```

Expected output:
```
NAME                                 TAINTS
computeinstance-xxxxx                nvidia.com/gpu
computeinstance-yyyyy                nvidia.com/gpu
```

### Force Pod Rescheduling (Optional)

If pods are still on GPU nodes after applying taints:
```bash
# Drain each GPU node to move non-GPU pods to CPU nodes
kubectl drain computeinstance-xxxxx --ignore-daemonsets --delete-emptydir-data
kubectl drain computeinstance-yyyyy --ignore-daemonsets --delete-emptydir-data

# Uncordon nodes to allow GPU workloads
kubectl uncordon computeinstance-xxxxx
kubectl uncordon computeinstance-yyyyy
```

## Customization

### Disable Taints
To disable GPU node taints (allow any pod on GPU nodes):
```hcl
# terraform.tfvars
gpu_node_taints = []
```

### Custom Taint
```hcl
# terraform.tfvars
gpu_node_taints = [
  {
    key    = "custom-taint"
    value  = "gpu-required"
    effect = "NO_SCHEDULE"  # or NO_EXECUTE, PREFER_NO_SCHEDULE
  }
]
```

## Adding Tolerations to New Workloads

For any new pods that need GPUs, add this toleration:

```yaml
spec:
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"  # Matches any value
    effect: "NoSchedule"
  
  # Or with exact match:
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Equal"
    value: "present"
    effect: "NoSchedule"
```

## Troubleshooting

### Pods Pending
If GPU workload pods are stuck in `Pending` state:
```bash
kubectl describe pod <pod-name> -n <namespace>
```

Look for:
- `FailedScheduling`: Check if tolerations are missing
- `Insufficient nvidia.com/gpu`: GPU nodes scaled to zero, will auto-scale up

### GPU Nodes Not Scaling Down
```bash
# Check what's running on GPU nodes
kubectl get pods -A -o wide | grep computeinstance-<gpu-node-id>

# Check node conditions
kubectl describe node <gpu-node-name>
```

Common causes:
- Pods without tolerations (shouldn't happen with taints)
- DaemonSets (normal, but don't prevent scale-down)
- Pods with local storage (PodDisruptionBudget may block)
