# GPU Node Autoscaling with Scale-to-Zero

This document describes how to configure the Nebius Kubernetes cluster to automatically scale GPU nodes to zero when idle, reducing costs while maintaining the ability to run training jobs on-demand.

## Overview

The setup involves two key configurations:
1. **Nebius MK8s Node Group Autoscaling** - Allows GPU nodes to scale from 0 to N based on demand
2. **KubeRay CPU Worker Affinity** - Ensures CPU workers don't schedule on GPU nodes, allowing them to scale down

## Problem

By default, Ray CPU worker pods could be scheduled on GPU nodes, which prevented the cluster autoscaler from scaling down GPU nodes even when no GPU workloads were running. This resulted in unnecessary GPU node costs.

## Solution

### 1. Enable GPU Node Group Autoscaling

Configure the GPU node group to use autoscaling instead of a fixed node count.

**File:** `infra/k8s-installation/variables.tf`

```hcl
variable "gpu_autoscaling_enabled" {
  description = "Enable autoscaling for GPU node group (allows scale to zero)"
  type        = bool
  default     = false
}

variable "gpu_min_nodes" {
  description = "Minimum number of GPU nodes (set to 0 for scale-to-zero)"
  type        = number
  default     = 0
}

variable "gpu_max_nodes" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 2
}
```

**File:** `infra/k8s-installation/main.tf` (GPU node group resource)

```hcl
resource "nebius_mk8s_v1_node_group" "gpu-node-group" {
  # ... other configuration ...

  # Use autoscaling OR fixed_node_count (mutually exclusive)
  fixed_node_count = var.gpu_autoscaling_enabled ? null : var.gpu_nodes_count_per_group
  
  autoscaling = var.gpu_autoscaling_enabled ? {
    min_node_count = var.gpu_min_nodes
    max_node_count = var.gpu_max_nodes
  } : null
}
```

**File:** `infra/k8s-installation/terraform.tfvars`

```hcl
# GPU Autoscaling Configuration
gpu_autoscaling_enabled = true
gpu_min_nodes = 0      # Scale to zero when idle
gpu_max_nodes = 2      # Scale up to 2 nodes when needed
```

### 2. Configure KubeRay CPU Workers to Avoid GPU Nodes

Add node affinity to CPU worker pods to prevent them from scheduling on GPU nodes.

**File:** `infra/modules/kuberay/files/ray-values.yaml.tftpl`

```yaml
workerGroupSpecs:
  - groupName: cpu-worker
    replicas: ${min_cpu_replicas}
    minReplicas: ${min_cpu_replicas}
    maxReplicas: ${max_cpu_replicas}
    rayStartParams:
      num-cpus: "${cpu_resources.cpus}"
    template:
      spec:
        restartPolicy: Never
        # Prevent CPU workers from scheduling on GPU nodes
        affinity:
          nodeAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              nodeSelectorTerms:
                - matchExpressions:
                    - key: nebius.com/gpu
                      operator: NotIn
                      values:
                        - "true"
        containers:
          - name: ray-worker
            # ... rest of container spec
```

### 3. Set KubeRay GPU Workers to Scale to Zero

Configure KubeRay to allow GPU workers to scale down to zero replicas.

**File:** `infra/k8s-installation/terraform.tfvars`

```hcl
# KubeRay GPU worker configuration
kuberay_min_gpu_replicas = 0  # Scale to zero when idle
kuberay_max_gpu_replicas = 2  # Max 2 workers (one per GPU node)
```

## How It Works

### Scale Down Flow (Idle → Zero GPU Nodes)

1. Training job completes
2. KubeRay autoscaler scales GPU workers to 0 (after `idleTimeoutSeconds`)
3. GPU worker pods are terminated
4. GPU nodes become idle (only DaemonSet pods remain)
5. Nebius cluster autoscaler detects underutilized GPU nodes
6. GPU nodes are removed (typically within 10-15 minutes)

### Scale Up Flow (Zero → Active GPU Nodes)

1. New RayJob submitted requiring GPU workers
2. KubeRay creates GPU worker pods
3. Pods are in `Pending` state (no GPU nodes available)
4. Nebius cluster autoscaler detects unschedulable pods
5. New GPU nodes are provisioned (typically 2-3 minutes)
6. GPU worker pods are scheduled
7. Training starts

## Node Labels Reference

Nebius MK8s uses the following labels to identify GPU nodes:

| Label | GPU Node Value | CPU Node Value |
|-------|---------------|----------------|
| `nebius.com/gpu` | `"true"` | `<not set>` |
| `nvidia.com/gpu.count` | `"8"` | `<not set>` |
| `nvidia.com/gpu.product` | `"NVIDIA-H100-80GB-HBM3"` | `<not set>` |

The affinity rule uses `nebius.com/gpu NotIn ["true"]` to exclude GPU nodes.

## Verification Commands

### Check Node Status
```bash
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.metadata.labels.nebius\.com/gpu,STATUS:.status.conditions[-1].type'
```

### Check Ray Worker Pods
```bash
kubectl get pods -n ray-cluster -l ray.io/group=cpu-worker -o wide
kubectl get pods -n ray-cluster -l ray.io/group=gpu-worker -o wide
```

### Check Nebius Node Group Autoscaling Config
```bash
nebius mk8s node-group list --parent-id <cluster-id> --format json | \
  jq '.items[] | {name: .metadata.name, fixed: .spec.fixed_node_count, autoscaling: .spec.autoscaling}'
```

### Check Cluster Autoscaler Events
```bash
kubectl get events -A --field-selector reason=ScaleDown
kubectl get events -A --field-selector reason=TriggeredScaleUp
```

## Cost Impact

| Configuration | GPU Nodes When Idle | Monthly Cost (Idle) |
|--------------|---------------------|---------------------|
| Before (fixed) | 2 nodes | ~$50,000+ |
| After (autoscaling) | 0 nodes | $0 |

*Note: Costs are approximate and depend on GPU type (H100-SXM) and region.*

## Troubleshooting

### GPU Nodes Not Scaling Down

1. **Check for pods on GPU nodes:**
   ```bash
   kubectl get pods -A -o wide --field-selector spec.nodeName=<gpu-node-name>
   ```

2. **Check if CPU workers have correct affinity:**
   ```bash
   kubectl get pod -n ray-cluster <cpu-worker-pod> -o jsonpath='{.spec.affinity}'
   ```

3. **Check cluster autoscaler logs:**
   ```bash
   kubectl get events -A | grep -i autoscaler
   ```

### GPU Nodes Not Scaling Up

1. **Check for pending pods:**
   ```bash
   kubectl get pods -A --field-selector status.phase=Pending
   ```

2. **Check pod events:**
   ```bash
   kubectl describe pod -n ray-cluster <pending-pod> | grep -A10 Events
   ```

3. **Verify node group max is not reached:**
   ```bash
   nebius mk8s node-group get --id <node-group-id> --format json | jq '.spec.autoscaling'
   ```

## Related Files

- `infra/k8s-installation/variables.tf` - Terraform variables for autoscaling
- `infra/k8s-installation/main.tf` - GPU node group resource definition
- `infra/k8s-installation/terraform.tfvars` - User configuration values
- `infra/modules/kuberay/files/ray-values.yaml.tftpl` - KubeRay Helm values template
