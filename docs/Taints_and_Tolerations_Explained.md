# Taints and Tolerations - Complete Explanation

## The Confusion: NO_SCHEDULE vs NoSchedule

They are **THE SAME THING** - just different formats:
- **Terraform**: `NO_SCHEDULE` (uppercase with underscore)
- **Kubernetes YAML**: `NoSchedule` (PascalCase)

## How It Actually Works

### 🔒 The Taint (Guard at the Door)

**GPU Node has this taint:**
```yaml
taints:
  - key: nvidia.com/gpu
    value: present
    effect: NoSchedule
```

**What it means:**
> "I am a GPU node. Don't schedule ANY pod on me UNLESS that pod has a matching toleration (key)."

### 🔑 The Toleration (The Key)

**Ray GPU Pod has this toleration:**
```yaml
tolerations:
  - key: nvidia.com/gpu        # ← Matches the taint key
    value: present              # ← Matches the taint value  
    effect: NoSchedule          # ← Matches the taint effect
```

**What it means:**
> "I have permission to be scheduled on nodes with the taint `nvidia.com/gpu=present:NoSchedule`"

## Real Examples

### ❌ Example 1: Monitoring Pod (Gets Blocked)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: grafana-pod
spec:
  containers:
  - name: grafana
    image: grafana/grafana
  # ❌ NO tolerations section at all!
```

**What happens:**
```
1. Scheduler: "Let me schedule grafana-pod"
2. Scheduler: "GPU nodes available... checking taints"
3. GPU Node: "I have taint: nvidia.com/gpu=present:NoSchedule"
4. Scheduler: "Does grafana-pod have matching toleration?"
5. Scheduler: "No toleration found! ❌ BLOCKED"
6. Scheduler: "Let me try CPU nodes instead..."
7. CPU Node: "No taints, come on in! ✅"
```

### ✅ Example 2: Training Pod (Gets Access)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-pod
spec:
  tolerations:           # ✅ HAS tolerations!
  - key: nvidia.com/gpu
    value: present
    effect: NoSchedule
  containers:
  - name: training
    image: ray-gpu:latest
    resources:
      limits:
        nvidia.com/gpu: 8
```

**What happens:**
```
1. Scheduler: "Let me schedule training-pod"
2. Scheduler: "GPU nodes available... checking taints"
3. GPU Node: "I have taint: nvidia.com/gpu=present:NoSchedule"
4. Scheduler: "Does training-pod have matching toleration?"
5. Scheduler: "YES! It tolerates nvidia.com/gpu=present:NoSchedule ✅"
6. Scheduler: "Access granted! Scheduling on GPU node"
```

## The Three Taint Effects

| Effect | Terraform | Kubernetes | What It Does |
|--------|-----------|------------|--------------|
| **NoSchedule** | `NO_SCHEDULE` | `NoSchedule` | Blocks NEW pods without toleration. Existing pods stay. |
| **PreferNoSchedule** | `PREFER_NO_SCHEDULE` | `PreferNoSchedule` | Try to avoid, but allow if necessary. |
| **NoExecute** | `NO_EXECUTE` | `NoExecute` | Blocks NEW pods AND evicts existing pods without toleration. |

We use `NoSchedule` because:
- ✅ Blocks new non-GPU pods
- ✅ Doesn't evict existing DaemonSets (they need to run everywhere)
- ✅ Allows autoscaling to work properly

## Visual Flow

```
                          Kubernetes Scheduler
                                  │
                                  │ "Where should I put this pod?"
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
            ┌───────────────┐          ┌───────────────┐
            │   CPU Node    │          │   GPU Node    │
            │   No Taints   │          │   🔒 TAINTED  │
            └───────────────┘          │  nvidia.com/  │
                    ▲                  │  gpu:NoSchedule│
                    │                  └───────────────┘
                    │                           ▲
                    │                           │
                    │                  ┌────────┴────────┐
                    │                  │                 │
                    │             No Toleration    Has Toleration
                    │                  │                 │
            ┌───────┴────────┐  ┌──────┴──────┐  ┌──────┴──────┐
            │ Grafana        │  │ Grafana     │  │ Training    │
            │ ❌ NO GPU      │  │ ❌ NO GPU   │  │ ✅ GPU:8    │
            │ ❌ NO Toleration│ │ ❌ NO Tol.  │  │ ✅ Toleration│
            └────────────────┘  └─────────────┘  └─────────────┘
                    │                  │                 │
                    ▼                  │                 ▼
              ✅ Scheduled         ────┘           ✅ Scheduled
              on CPU Node         Blocked!         on GPU Node
                                  Redirected
                                  to CPU ➜
```

## Our Configuration

### Terraform (Applies Taint to GPU Nodes)

**File: `infra/k8s-installation/variables.tf`**
```terraform
variable "gpu_node_taints" {
  default = [
    {
      key    = "nvidia.com/gpu"    # ← Taint key
      value  = "present"            # ← Taint value
      effect = "NO_SCHEDULE"        # ← Terraform format
    }
  ]
}
```

**File: `infra/k8s-installation/main.tf`**
```terraform
resource "nebius_mk8s_v1_node_group" "gpu" {
  template = {
    taints = var.gpu_node_taints  # ← Applied to GPU nodes
  }
}
```

### Kubernetes (Ray GPU Pods Get Access)

**File: `infra/modules/kuberay/files/ray-values.yaml.tftpl`**
```yaml
spec:
  tolerations:                      # ← Key to access GPU nodes
    - key: nvidia.com/gpu           # ← Must match taint key
      operator: Equal               # ← Exact match
      value: "present"              # ← Must match taint value
      effect: NoSchedule            # ← Kubernetes format (same as NO_SCHEDULE)
```

## The Result

### Before (No Taints)
```
GPU Nodes: Anyone can schedule here
├─ Grafana ✅
├─ Prometheus ✅
├─ Kubeflow ✅
├─ Ray GPU worker ✅
└─ Result: GPU nodes can't scale to 0 ❌
```

### After (With Taints)
```
GPU Nodes: Only pods with toleration key
├─ Grafana ❌ → Redirected to CPU
├─ Prometheus ❌ → Redirected to CPU
├─ Kubeflow ❌ → Redirected to CPU
├─ Ray GPU worker ✅ → Has toleration
└─ Result: GPU nodes scale to 0 when idle ✅
```

## Quick Reference

| Component | Has Toleration? | Can Use GPU Nodes? |
|-----------|----------------|-------------------|
| Grafana | ❌ No | ❌ No (blocked by taint) |
| Prometheus | ❌ No | ❌ No (blocked by taint) |
| Kubeflow | ❌ No | ❌ No (blocked by taint) |
| Ray CPU Worker | ❌ No | ❌ No (blocked by taint) |
| Ray GPU Worker | ✅ Yes | ✅ Yes (toleration = key) |
| Training Jobs | ✅ Yes | ✅ Yes (toleration = key) |
| GPU DaemonSets | ✅ Usually | ✅ Yes (system manages) |

## Summary

**The taint and toleration BOTH say `NoSchedule` - this is correct!**

- **Taint with NoSchedule** = 🔒 "Lock the door with NoSchedule lock"
- **Toleration with NoSchedule** = 🔑 "I have the key for NoSchedule lock"

The taint doesn't give access - it **blocks** access unless you have the matching toleration (key).

Think of it like a secure building:
- Building has: "Badge required to enter" (taint)
- Employee has: "I have a badge" (toleration)
- Visitor without badge: Blocked (no toleration)
