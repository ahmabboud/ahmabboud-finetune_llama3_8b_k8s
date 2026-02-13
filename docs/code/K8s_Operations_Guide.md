# Kubernetes Operations Guide

Simple, practical commands for daily cluster operations and debugging.

---

## Job Management

**Understanding Jobs vs Pods:**
- **Job** = Controller that creates and manages Pods
- **Pod** = Actual container(s) that run your workload
- Jobs ensure Pods run to completion
- You apply/delete Jobs, but you get logs from Pods

Example:
```
Job: preflight-check            (What you created with kubectl apply)
 ├─ Pod: preflight-check-0      (Created automatically, this runs the work)
 └─ Pod: preflight-check-1      (Second pod for multi-node tests)
```

### Apply a Job

```bash
# Apply from file
kubectl apply -f k8s/preflight-check-job.yaml
```

**Flags explained:**
- `-f` = file path to the YAML manifest

**Watch it get created:**
```bash
kubectl get jobs -n ray-cluster -w
```
- `-n` = namespace (where to look)
- `-w` = watch mode (updates in real-time, Ctrl+C to exit)

---

### Check Job Status

```bash
# Simple status
kubectl get jobs -n ray-cluster

# Output shows:
# NAME              COMPLETIONS   DURATION   AGE
# preflight-check   2/2           8m32s      10m
#                   ^^^
#                   completed/total pods
```

**Detailed job information:**
```bash
kubectl describe job preflight-check -n ray-cluster
```
- `describe` = show detailed information including events
- Last section shows events (Created, Completed, Failed, etc.)

---

### List All Jobs

```bash
# Jobs in specific namespace
kubectl get jobs -n ray-cluster

# Jobs in all namespaces
kubectl get jobs --all-namespaces

# Or shorthand
kubectl get jobs -A
```
- `-A` = shorthand for `--all-namespaces`

**Show more details (wide output):**
```bash
kubectl get jobs -n ray-cluster -o wide
```
- `-o wide` = show additional columns (selector, images, etc.)

---

### Delete a Job

```bash
# Delete specific job
kubectl delete job preflight-check -n ray-cluster

# Delete from file (same as apply)
kubectl delete -f k8s/preflight-check-job.yaml

# Force delete if stuck
kubectl delete job preflight-check -n ray-cluster --force --grace-period=0
```

**Flags explained:**
- `--force` = don't wait for graceful termination
- `--grace-period=0` = terminate immediately (default is 30 seconds)

**Delete multiple jobs:**
```bash
# By name pattern
kubectl delete job -n ray-cluster -l app=preflight-check

# By label selector
kubectl delete job -n ray-cluster -l job-type=training
```
- `-l` = label selector (match by labels)

---

## Pod Management

### List Pods

```bash
# Pods in namespace
kubectl get pods -n ray-cluster

# All pods in cluster
kubectl get pods -A

# Show pod IPs and nodes
kubectl get pods -n ray-cluster -o wide
```

**Filter by status:**
```bash
# Only running pods
kubectl get pods -n ray-cluster --field-selector=status.phase=Running

# Only failed pods
kubectl get pods -n ray-cluster --field-selector=status.phase=Failed
```
- `--field-selector` = filter by field values (phase, node, etc.)

---

### Check Pod Logs

**Important:** Logs come from **Pods**, not Jobs. A Job creates Pods that run the actual containers.

```bash
# First, find the pod name created by the job
kubectl get pods -n ray-cluster

# Output shows:
# NAME                READY   STATUS    RESTARTS   AGE
# preflight-check-0   1/1     Running   0          5m
# preflight-check-1   1/1     Running   0          5m
#
# Use the POD name (preflight-check-0), not the JOB name (preflight-check)

# View logs from a pod
kubectl logs preflight-check-0 -n ray-cluster

# Follow logs in real-time
kubectl logs preflight-check-0 -n ray-cluster -f

# Last 100 lines
kubectl logs preflight-check-0 -n ray-cluster --tail=100

# Logs from previous pod instance (if crashed)
kubectl logs preflight-check-0 -n ray-cluster --previous
```

**Flags explained:**
- `-f` = follow (stream logs in real-time, like `tail -f`)
- `--tail=N` = show last N lines only
- `--previous` = show logs from crashed/restarted container

**Note on naming:** For Indexed Jobs (like preflight-check), pods are named:
- `<job-name>-0` = first pod (rank 0)
- `<job-name>-1` = second pod (rank 1)
- etc.

For regular Jobs, pods are named `<job-name>-<random-hash>` (e.g., `my-job-abc123`)

**Multiple containers in pod:**
```bash
# Specify container name
kubectl logs preflight-check-0 -n ray-cluster -c preflight

# All containers
kubectl logs preflight-check-0 -n ray-cluster --all-containers
```
- `-c` = container name (required if pod has multiple containers)
- `--all-containers` = show logs from all containers

---

### Pod Details

```bash
# Detailed pod information
kubectl describe pod preflight-check-0 -n ray-cluster
```

**What it shows:**
- Pod IP, node assignment
- Resource requests/limits
- Environment variables
- Volume mounts
- **Events** (most useful for debugging!)

**Get pod in YAML format:**
```bash
kubectl get pod preflight-check-0 -n ray-cluster -o yaml
```
- `-o yaml` = output in YAML format (also: `-o json` for JSON)

---

## Node Management

### List Nodes

```bash
# All nodes
kubectl get nodes

# Show more details
kubectl get nodes -o wide

# Only GPU nodes
kubectl get nodes -l nebius.com/gpu=true

# Only CPU nodes
kubectl get nodes -l nebius.com/node-group=cpu-d3
```
- `-l` = label selector (filter by labels)

**Show node resources:**
```bash
kubectl get nodes -o custom-columns=\
NAME:.metadata.name,\
CPU:.status.capacity.cpu,\
MEMORY:.status.capacity.memory,\
GPU:.status.capacity.'nvidia\.com/gpu'
```

---

### List Pods on Specific Node

```bash
# All pods on a node
kubectl get pods --all-namespaces --field-selector spec.nodeName=<node-name>

# Or with grep
kubectl get pods -A -o wide | grep <node-name>

# Count pods on node
kubectl get pods -A --field-selector spec.nodeName=<node-name> --no-headers | wc -l
```

**Flags explained:**
- `--field-selector spec.nodeName=X` = filter pods by node assignment
- `--no-headers` = omit column headers (useful for counting)

**Show what's using resources:**
```bash
# Pods requesting GPUs on a specific node
kubectl get pods -A --field-selector spec.nodeName=<node-name> -o json | \
  jq -r '.items[] | select(.spec.containers[].resources.limits."nvidia.com/gpu" != null) | "\(.metadata.namespace)/\(.metadata.name)"'
```

---

### Node Details

```bash
# Detailed node information
kubectl describe node <node-name>
```

**What it shows:**
- Labels and taints
- Capacity (CPU, memory, GPUs)
- Allocatable resources
- Running pods
- Resource usage
- **Events** (node ready, pressure, etc.)

**Key sections to check:**
```
Taints:              nvidia.com/gpu=present:NoSchedule
Capacity:
  cpu:                128
  memory:             1600Gi
  nvidia.com/gpu:     8
Allocated resources:
  nvidia.com/gpu     8 (100%)
```

---

### Drain a Node

**What is draining?** Safely evict all pods from a node before maintenance.

```bash
# Standard drain
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# Force drain (use carefully)
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data --force
```

**Flags explained:**
- `--ignore-daemonsets` = skip DaemonSet pods (they'll respawn anyway)
- `--delete-emptydir-data` = delete data in emptyDir volumes (temp data)
- `--force` = force delete pods not managed by controllers

**What happens:**
1. Node marked as unschedulable (cordoned)
2. Pods gracefully terminated
3. Pods rescheduled on other nodes
4. Node sits empty but still in cluster

**Undo drain (make node schedulable again):**
```bash
kubectl uncordon <node-name>
```

---

### Delete a Node

```bash
# Delete node from cluster
kubectl delete node <node-name>
```

**When to use:**
- Node is permanently dead
- Replacing faulty hardware
- Scaling down manually

**Note:** This removes node from Kubernetes but doesn't delete the VM. For full cleanup, delete via cloud provider (Terraform, console, etc.)

---

### Restart a Node

**Option 1: Drain and delete (let autoscaler recreate)**
```bash
# 1. Drain the node
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data --force

# 2. Delete the node
kubectl delete node <node-name>

# 3. Autoscaler or node group will create a new node automatically
```

**Option 2: Cordon, delete pods, uncordon**
```bash
# 1. Mark node unschedulable
kubectl cordon <node-name>

# 2. Delete all pods (they'll reschedule elsewhere)
kubectl delete pods --all -n <namespace> --field-selector spec.nodeName=<node-name>

# 3. Wait for pods to move

# 4. Make node schedulable again
kubectl uncordon <node-name>
```

**Flags explained:**
- `cordon` = mark node as unschedulable (no new pods)
- `uncordon` = mark node as schedulable (allow new pods)

---

## Service Management

### List Services

```bash
# Services in namespace
kubectl get svc -n ray-cluster

# All services
kubectl get svc -A

# Show endpoints
kubectl get svc -n ray-cluster -o wide
```

---

### Delete a Service

```bash
kubectl delete svc preflight-check -n ray-cluster

# Or from file
kubectl delete -f k8s/preflight-check-job.yaml
```

---

### Port Forwarding

```bash
# Forward local port to service
kubectl port-forward -n ray-cluster svc/ray-cluster-head-svc 8265:8265

# Forward local port to pod
kubectl port-forward -n ray-cluster preflight-check-0 8080:80

# Forward to random local port
kubectl port-forward -n ray-cluster svc/ray-cluster-head-svc :8265
```

**Flags explained:**
- First number = local port (on your machine)
- Second number = remote port (in cluster)
- `:8265` = let kubectl choose random local port

**Run in background:**
```bash
kubectl port-forward -n ray-cluster svc/ray-cluster-head-svc 8265:8265 &
```
- `&` = run in background (use `fg` to bring back, `jobs` to list)

---

## Namespace Management

### List Namespaces

```bash
kubectl get namespaces

# Or shorthand
kubectl get ns
```

---

### Set Default Namespace

```bash
# Set default namespace for current context
kubectl config set-context --current --namespace=ray-cluster

# Now you can omit -n flag
kubectl get pods  # shows pods in ray-cluster
```

**Undo (set back to default):**
```bash
kubectl config set-context --current --namespace=default
```

---

## Resource Monitoring

### Top Commands

```bash
# Node resource usage
kubectl top nodes

# Pod resource usage
kubectl top pods -n ray-cluster

# Sort by CPU
kubectl top pods -n ray-cluster --sort-by=cpu

# Sort by memory
kubectl top pods -n ray-cluster --sort-by=memory
```

**Note:** Requires metrics-server to be installed.

---

### Events

**Cluster-wide events (last hour):**
```bash
kubectl get events --all-namespaces --sort-by='.lastTimestamp'

# Recent events only
kubectl get events -A --sort-by='.lastTimestamp' | tail -20

# Events in namespace
kubectl get events -n ray-cluster
```

**Filter by type:**
```bash
# Warning events only
kubectl get events -A --field-selector type=Warning

# Events for specific object
kubectl get events -n ray-cluster --field-selector involvedObject.name=preflight-check-0
```

---

## Quick Debugging Checklist

### Job Not Starting?

```bash
# 1. Check job exists
kubectl get jobs -n ray-cluster

# 2. Check pods created
kubectl get pods -n ray-cluster

# 3. Check pod status
kubectl describe pod <pod-name> -n ray-cluster
# Look at Events section at bottom

# 4. Common issues:
# - ImagePullBackOff: wrong image or credentials
# - Pending: no resources available (check nodes)
# - CrashLoopBackOff: container keeps crashing (check logs)
```

---

### Pod Stuck Pending?

```bash
# 1. Describe the pod
kubectl describe pod <pod-name> -n ray-cluster
# Look for: "FailedScheduling" in Events

# 2. Check if nodes available
kubectl get nodes

# 3. Check if resources available
kubectl top nodes

# 4. Check pod requirements vs node capacity
kubectl get pod <pod-name> -n ray-cluster -o yaml | grep -A 5 resources

# Common reasons:
# - No nodes with required labels
# - Insufficient CPU/memory/GPU
# - Node taints without tolerations
# - PVC not bound
```

---

### Pod Keeps Crashing?

```bash
# 1. Check logs
kubectl logs <pod-name> -n ray-cluster

# 2. Check previous logs (from crash)
kubectl logs <pod-name> -n ray-cluster --previous

# 3. Check restart count
kubectl get pods -n ray-cluster
# Look at RESTARTS column

# 4. Describe pod for events
kubectl describe pod <pod-name> -n ray-cluster
```

---

### Node Not Ready?

```bash
# 1. Check node status
kubectl get nodes

# 2. Describe node
kubectl describe node <node-name>
# Look at Conditions section

# 3. Check node pressure
kubectl get node <node-name> -o json | jq '.status.conditions'

# Common issues:
# - DiskPressure: node out of disk space
# - MemoryPressure: node out of memory
# - NetworkUnavailable: network plugin issue
```

---

## Watch Mode (Real-Time Updates)

Any `get` command can be watched in real-time:

```bash
# Watch nodes
kubectl get nodes -w

# Watch pods
kubectl get pods -n ray-cluster -w

# Watch jobs
kubectl get jobs -n ray-cluster -w

# Watch events (most useful!)
kubectl get events -n ray-cluster -w
```

**Tip:** Press `Ctrl+C` to exit watch mode.

---

## Useful Aliases

Add to `~/.zshrc` or `~/.bashrc`:

```bash
# Short aliases
alias k='kubectl'
alias kgp='kubectl get pods'
alias kgn='kubectl get nodes'
alias kgj='kubectl get jobs'
alias kdp='kubectl describe pod'
alias kdn='kubectl describe node'
alias kl='kubectl logs'
alias klf='kubectl logs -f'

# Namespace shortcuts
alias kgpa='kubectl get pods -A'
alias kgpr='kubectl get pods -n ray-cluster'

# Quick describe
alias kdesc='kubectl describe'

# Watch shortcuts
alias kgpw='kubectl get pods -w'
alias kgnw='kubectl get nodes -w'
```

**Usage:**
```bash
k get pods -n ray-cluster
kgp -n ray-cluster
kl preflight-check-0 -n ray-cluster
```

---

## Common Patterns

### Check GPU Nodes and Their Workloads

```bash
# 1. List GPU nodes
kubectl get nodes -l nebius.com/gpu=true

# 2. For each GPU node, check pods
for node in $(kubectl get nodes -l nebius.com/gpu=true -o name | cut -d/ -f2); do
  echo "=== Pods on $node ==="
  kubectl get pods -A --field-selector spec.nodeName=$node
  echo ""
done
```

---

### Clean Up Completed Jobs

```bash
# Delete completed jobs
kubectl delete jobs -n ray-cluster --field-selector status.successful=1

# Delete failed jobs
kubectl delete jobs -n ray-cluster --field-selector status.failed=1
```

---

### Find Pods Requesting GPUs

```bash
kubectl get pods -A -o json | \
  jq -r '.items[] | select(.spec.containers[].resources.requests."nvidia.com/gpu" != null) | "\(.metadata.namespace)/\(.metadata.name)"'
```

---

## Key Takeaways

### Essential Commands to Remember

| Task | Command |
|------|---------|
| **Apply config** | `kubectl apply -f <file>` |
| **List resources** | `kubectl get <type> -n <namespace>` |
| **Details** | `kubectl describe <type> <name> -n <namespace>` |
| **Logs** | `kubectl logs <pod-name> -n <namespace> -f` |
| **Delete** | `kubectl delete <type> <name> -n <namespace>` |
| **Drain node** | `kubectl drain <node> --ignore-daemonsets --delete-emptydir-data` |
| **Watch** | `kubectl get <type> -n <namespace> -w` |
| **Events** | `kubectl get events -n <namespace>` |

### Flag Reference

| Flag | Meaning | Example |
|------|---------|---------|
| `-n` | Namespace | `-n ray-cluster` |
| `-A` | All namespaces | `kubectl get pods -A` |
| `-f` | File path | `kubectl apply -f job.yaml` |
| `-l` | Label selector | `kubectl get nodes -l gpu=true` |
| `-w` | Watch mode | `kubectl get pods -w` |
| `-o` | Output format | `-o yaml`, `-o json`, `-o wide` |
| `--field-selector` | Filter by field | `--field-selector spec.nodeName=node1` |
| `--force` | Force operation | `kubectl delete pod x --force` |
| `--grace-period=0` | Immediate delete | `kubectl delete pod x --grace-period=0` |
| `--ignore-daemonsets` | Skip DaemonSets | `kubectl drain node --ignore-daemonsets` |
| `--previous` | Previous container | `kubectl logs pod --previous` |
| `--tail=N` | Last N lines | `kubectl logs pod --tail=100` |

---

## Pro Tips

1. **Always specify namespace** with `-n` to avoid confusion
2. **Use `-o wide`** to see more details (IPs, nodes, etc.)
3. **Use `describe`** when debugging - Events section is gold
4. **Use `-w`** (watch) to see real-time updates
5. **Use `--previous`** to see logs from crashed containers
6. **Use `-A`** when you don't know which namespace
7. **Check events first** - they show what Kubernetes is doing
8. **DaemonSets are normal** on nodes - they don't prevent scale-down

---

## Need More Help?

```bash
# Get help for any command
kubectl <command> --help

# Examples:
kubectl get --help
kubectl describe --help
kubectl logs --help
```

**Official docs:** https://kubernetes.io/docs/reference/kubectl/
