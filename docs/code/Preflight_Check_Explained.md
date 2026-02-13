# Preflight Check Job - Complete Explanation

## Overview

The preflight check is a comprehensive system validation script that runs **before training** to catch problems early. Think of it as a "health check" before committing hours to distributed training on 16x H100 GPUs.

```
Purpose: Verify cluster is ready for distributed training
Runs: 2 pods (one per GPU node) with 8 GPUs each
Time: ~5-10 minutes
When: Before starting training jobs
```

## Architecture

```yaml
Job Type: Indexed Job (parallelism: 2, completions: 2)
├─ Pod 0 (Rank 0) - First GPU node
│  └─ 8x H100 GPUs, 128GB RAM, 32 CPUs
│
└─ Pod 1 (Rank 1) - Second GPU node
   └─ 8x H100 GPUs, 128GB RAM, 32 CPUs

Headless Service: preflight-check.ray-cluster.svc
└─ Allows pods to discover each other via DNS
```

## Key Components

### Headless Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: preflight-check
spec:
  clusterIP: None  # Headless - returns pod IPs directly
  publishNotReadyAddresses: true  # Include pods before ready
```

**Why headless?** Both pods need to discover each other's IPs for NCCL initialization. A headless service returns all pod IPs when queried via DNS.

### Indexed Job
```yaml
apiVersion: batch/v1
kind: Job
spec:
  parallelism: 2
  completions: 2
  completionMode: Indexed  # Each pod gets JOB_COMPLETION_INDEX (0 or 1)
```

**Why Indexed?** Each pod needs a unique rank (0 or 1) for distributed communication. The job automatically assigns `JOB_COMPLETION_INDEX`.

### Pod Anti-Affinity
```yaml
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            app: preflight-check
        topologyKey: "kubernetes.io/hostname"
```

**Why?** Ensures the 2 pods run on **different physical nodes** to test inter-node networking.

### GPU Toleration
```yaml
tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
```

**Why?** GPU nodes have taints to prevent non-GPU workloads. This toleration grants access.

---

## Test Conditions & Thresholds

This section documents all the pass/fail criteria, thresholds, timeouts, and testing conditions used throughout the preflight check.

### Hardware Requirements
| Component | Requirement | Fail Condition |
|-----------|-------------|----------------|
| **GPU Count** | Exactly 8 GPUs per node | < 8 or > 8 GPUs detected |
| **GPU Model** | NVIDIA H100 80GB HBM3 | Any other model |
| **GPU Memory** | 80GB (81559 MiB) per GPU | Less than expected |
| **GPU Temperature** | < 85°C per GPU | ≥ 85°C (thermal throttling risk) |
| **NVLink Pairs** | 28 out of 28 pairs connected | < 28 pairs (degraded performance) |

### Software Requirements
| Component | Requirement | Validation |
|-----------|-------------|------------|
| **nvidia-smi** | Must be available | Command execution check |
| **PyTorch CUDA** | torch.cuda.is_available() = True | CUDA runtime accessible |
| **PyTorch GPU Count** | torch.cuda.device_count() = 8 | Must match physical GPUs |
| **NCCL Backend** | torch.distributed.is_nccl_available() | Required for distributed training |
| **CUDA Tensor Ops** | Matrix multiply on each GPU | Basic compute validation |

### Network & Communication
| Component | Requirement | Timeout/Retry |
|-----------|-------------|---------------|
| **Peer Discovery** | Find ≥ 2 pods via DNS | 120 attempts × 5s = 10 minutes max |
| **DNS Query Interval** | Every 5 seconds | Progress log every 60s (12 attempts) |
| **Synchronization Barrier** | 10 second sleep | Ensures all ranks ready before NCCL init |
| **NCCL Init Timeout** | 300 seconds (5 minutes) | Set in torch.distributed.init_process_group |
| **Master Port** | 29500 | Fixed for deterministic connection |
| **Master Election** | Smallest IP becomes master | Deterministic across all pods |
| **Rank Assignment** | Based on sorted IP order | More reliable than JOB_COMPLETION_INDEX |

### NCCL Test Parameters
| Test | Size | Iterations | Warmup | Expected (InfiniBand) | Expected (TCP) |
|------|------|------------|--------|----------------------|----------------|
| **AllReduce Validation** | 1MB (1024×1024 float32) | 1 | None | Sum correctness | Sum correctness |
| **Bandwidth Test 1MB** | 1MB | 10 | 3 | 100-150 GB/s | 1-3 GB/s |
| **Bandwidth Test 10MB** | 10MB | 10 | 3 | 120-180 GB/s | 2-5 GB/s |
| **Bandwidth Test 100MB** | 100MB | 10 | 3 | 120-180 GB/s | 2-5 GB/s |

### Storage Requirements
| Check | Requirement | Validation |
|-------|-------------|------------|
| **Mount Point** | /mnt/data exists | Directory check |
| **Free Space** | > 100GB available | df -BG /mnt/data |
| **Checkpoints Dir** | /mnt/data/checkpoints (optional) | Count existing checkpoints |

### Ray Cluster
| Check | Requirement | Validation |
|-------|-------------|------------|
| **Ray Head** | ray://ray-cluster-head-svc:10001 | Connection test |
| **Cluster Resources** | Report nodes, CPUs, GPUs | ray.cluster_resources() |
| **Expected GPUs** | 16 GPUs total (2 nodes × 8) | Matches physical hardware |

### Training Pipeline (Smoke Test)
| Check | Condition | Details |
|-------|-----------|---------|
| **Execution Rank** | Only Rank 0 | Avoids duplicate work |
| **GPU Mode** | Single GPU (CUDA_VISIBLE_DEVICES=0) | Lightweight test |
| **Env Isolation** | Unset WORLD_SIZE, RANK, MASTER_ADDR | Forces single-GPU mode |
| **Test Model** | GPT-2 (124M params) | Fast to load vs Llama-3 (8B) |
| **Training Steps** | 2 steps only | Just validates pipeline works |
| **LoRA Config** | r=8, target_modules=["c_attn"] | Minimal adapter size |
| **Dataset** | 40 samples | Tiny dataset for speed |

### Environment Variables
| Variable | Value | Purpose |
|----------|-------|---------|
| **NCCL_DEBUG** | INFO | Enable detailed NCCL logs |
| **NCCL_DEBUG_SUBSYS** | INIT,NET | Focus on initialization and networking |
| **NCCL_IB_DISABLE** | 0 | Enable InfiniBand (don't disable) |
| **NCCL_NET_GDR_LEVEL** | 5 | Maximum GPUDirect RDMA level |
| **WORLD_SIZE** | 2 | Total number of processes (2 nodes) |
| **MASTER_PORT** | 29500 | Fixed port for master process |

### Timeouts & Retries
| Operation | Timeout | Retry Logic |
|-----------|---------|-------------|
| **Job TTL** | 300s after completion | Auto-cleanup finished job |
| **Peer Discovery** | 10 minutes (120 × 5s) | Retry every 5s until 2 peers found |
| **NCCL Init** | 5 minutes | Built into torch.distributed |
| **Sync Barrier** | 10 seconds | Simple sleep for coordination |

### Pass/Fail Criteria Summary
✅ **PASS:** All checks return true, bandwidth ≥ 120 GB/s (InfiniBand) or ≥ 2 GB/s (TCP)

❌ **FAIL:** Any check fails, including:
- GPU count ≠ 8
- Temperature ≥ 85°C
- NVLink pairs < 28
- NCCL init fails
- Storage < 100GB free
- Ray connection fails
- Training pipeline errors
- Bandwidth < 2 GB/s

---

## The 9 Validation Steps

### **Step 1: Install Dependencies** ⚙️

**What it does:**
```bash
pip install -q torch transformers datasets peft accelerate bitsandbytes wandb deepspeed trl
```

**Validates:**
- ✅ All packages install without conflicts
- ✅ Correct versions are available
- ✅ Package registry is accessible

**Why it matters:** Catches dependency issues before training starts (missing packages, version conflicts).

---

### **Step 2: GPU Availability** 🎮

**What it does:**
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
```

**Validates:**
- ✅ All 8 GPUs are visible to nvidia-smi
- ✅ Each GPU has correct memory (80GB for H100)
- ✅ GPU driver is loaded properly

**Sample output:**
```
index, name, memory.total [MiB]
0, NVIDIA H100 80GB HBM3, 81559 MiB
1, NVIDIA H100 80GB HBM3, 81559 MiB
...
7, NVIDIA H100 80GB HBM3, 81559 MiB
```

**Why it matters:** Detects missing GPUs, driver issues, or wrong hardware configuration.

---

### **Step 3: GPU Health** 🌡️

**What it does:**
```bash
nvidia-smi --query-gpu=index,temperature.gpu --format=csv
# Check each GPU temperature < 85°C
```

**Validates:**
- ✅ GPU temperatures are within safe range
- ✅ Cooling system is working
- ✅ No thermal throttling

**Warning threshold:** 85°C (thermal throttling starts around 82-85°C)

**Why it matters:** Overheating GPUs throttle performance and can cause training to slow down or crash.

---

### **Step 4: CUDA & PyTorch** 🔥

**What it does:**
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Quick tensor test on each GPU
for i in range(gpu_count):
    x = torch.randn(1000, 1000, device=f'cuda:{i}')
    y = torch.matmul(x, x)  # Matrix multiply
```

**Validates:**
- ✅ PyTorch is installed correctly
- ✅ CUDA runtime is accessible
- ✅ PyTorch sees all 8 GPUs
- ✅ Basic compute operations work on each GPU

**Why it matters:** Ensures PyTorch can actually use the GPUs (driver issues can make GPUs invisible to PyTorch).

---

### **Step 5: Intra-Node NVLink** 🔗

**What it does:**
```python
# Test GPU-to-GPU communication within same node
for i in range(world_size):
    for j in range(i+1, world_size):
        if torch.cuda.can_device_access_peer(i, j):
            nvlink_pairs += 1
```

**Validates:**
- ✅ NVLink P2P (peer-to-peer) is enabled
- ✅ GPUs can directly access each other's memory
- ✅ All 28 GPU pairs are connected (8 choose 2)

**NVLink bandwidth:** 900 GB/s between GPUs (vs 16 GB/s over PCIe)

**Why it matters:** Training uses all 8 GPUs simultaneously. NVLink allows fast gradient synchronization. Without it, training would be 50x slower!

**Expected output:**
```
✓ PASS  NVLink P2P connectivity    28/28 pairs connected
```

---

### **Step 6: Inter-Node NCCL** 🌐 (The Most Critical Test!)

This is the **most complex and important test**. It validates that the two GPU nodes can communicate at high speed.

#### Architecture
```
Node 0 (10.0.26.195)                Node 1 (10.0.26.200)
├─ GPU 0                            ├─ GPU 0
├─ GPU 1                            ├─ GPU 1
...                                 ...
└─ GPU 7                            └─ GPU 7
     ↓                                   ↓
  InfiniBand fabric (400 Gb/s)
     ↑                                   ↑
```

#### Phase 1: Peer Discovery

**Goal:** Find the other pod's IP address via DNS

**The Challenge:** When pods start, they don't know each other's IPs. Kubernetes assigns IPs dynamically, and pods may start at different times.

**The Solution:** Headless service DNS lookup with retry logic

```bash
SERVICE_DNS="preflight-check.ray-cluster.svc.cluster.local"

# Retry up to 120 times (10 minutes total)
for attempt in $(seq 1 120); do
  # Query DNS for all pod IPs
  ALL_IPS=$(getent hosts $SERVICE_DNS | awk '{print $1}' | sort)
  IP_COUNT=$(echo "$ALL_IPS" | grep -c . || echo 0)
  
  # Need at least 2 pods
  if [ "$IP_COUNT" -ge 2 ]; then
    echo "Found $IP_COUNT peers: $ALL_IPS"
    break
  fi
  
  # Progress logging every 60 seconds (12 attempts)
  if [ $((attempt % 12)) -eq 0 ]; then
    echo "Waiting for peers... ($attempt/120, found $IP_COUNT so far)"
  fi
  
  sleep 5  # Wait 5 seconds between retries
done

# Fail if peers not found after 10 minutes
if [ "$IP_COUNT" -lt 2 ]; then
  echo "ERROR: Could not discover 2 peers after 10 minutes"
  exit 1
fi
```

**Test Conditions:**
- **Success:** IP_COUNT ≥ 2 within 10 minutes
- **Fail:** Less than 2 IPs after 120 attempts
- **Retry:** Every 5 seconds
- **Timeout:** 10 minutes (600 seconds)
- **Progress:** Log every 60 seconds

**Why DNS?** Pods don't know each other's IPs ahead of time. The headless service provides dynamic discovery.

**Example Output:**
```
Discovering peers via headless service DNS...
Waiting for peers... (12/120, found 1 so far)
Waiting for peers... (24/120, found 1 so far)
Found 2 peers: 10.0.26.195 10.0.26.200
```

#### Phase 2: Master Election
```bash
# Use smallest IP as master (deterministic)
MASTER_ADDR=$(echo "$ALL_IPS" | head -1)
# Result: 10.0.26.195
```

**Why deterministic?** Both pods must agree on who's the master without coordination.

#### Phase 3: Rank Assignment
```bash
# Pods determine their rank by IP ordering
SORTED_IPS=$(echo "$ALL_IPS" | sort)
ACTUAL_RANK=0
for ip in $SORTED_IPS; do
  if [ "$ip" = "$POD_IP" ]; then
    break
  fi
  ACTUAL_RANK=$((ACTUAL_RANK + 1))
done
```

**Why?** `JOB_COMPLETION_INDEX` can be unreliable across nodes. IP-based ranking is deterministic.

#### Phase 3b: Synchronization Barrier

**Before starting NCCL, both pods must be ready simultaneously.**

```bash
# Simple 10-second sleep barrier
echo "Waiting 10s for all ranks to be ready..."
sleep 10
```

**Why needed?** If one pod tries to initialize NCCL before the other is ready, connection will timeout. The 10-second sleep gives both pods time to:
- Complete peer discovery
- Compute their ranks
- Export environment variables

**Test Condition:** Fixed 10-second wait (not configurable)

#### Phase 4: NCCL Initialization
```python
import torch.distributed as dist

# Set environment variables (from previous steps)
os.environ['MASTER_ADDR'] = master_addr  # 10.0.26.195
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = str(rank)           # 0 or 1
os.environ['WORLD_SIZE'] = '2'

# Initialize NCCL with 5-minute timeout
dist.init_process_group(
    backend="nccl",
    init_method="env://",
    timeout=timedelta(seconds=300)  # 5 minutes
)
```

**What happens:**
1. Both pods connect to master at `10.0.26.195:29500`
2. NCCL negotiates communication channels
3. Pods synchronize and establish connections

**Test Conditions:**
- **Success:** Process group initialized within 300 seconds
- **Fail:** Timeout after 5 minutes (network issues, firewall, wrong configuration)
- **Logs:** Check NCCL_DEBUG=INFO output for transport details

**NCCL will try (in order):**
1. First: InfiniBand/RoCE (if available) → 120-180 GB/s ✅
2. Fallback: TCP over Ethernet → 1-5 GB/s ⚠️

**Critical logs to check:**
```
# Good - InfiniBand detected:
NCCL INFO NET/IB/0: Using InfiniBand device mlx5_0:1
NCCL INFO NET/IB : Using 4 interfaces

# Fallback - TCP only:
NCCL INFO NET/Socket : Using [0]eth0:10.0.26.195<0>
```

#### Phase 5: AllReduce Test

**What is AllReduce?**

AllReduce is a collective communication operation that:
1. **Reduces** (combines) tensors from all ranks using an operation (SUM, MAX, etc.)
2. **Distributes** the result back to all ranks

```
BEFORE AllReduce:          AFTER AllReduce (SUM):
Node 0: [1, 1, 1, 1]       Node 0: [3, 3, 3, 3]
Node 1: [2, 2, 2, 2]       Node 1: [3, 3, 3, 3]
         ↓                          ↑
    AllReduce SUM          Result = 1+2 = 3
```

**How it works (Ring-AllReduce for 2 nodes):**
```
Step 1: Scatter-Reduce
Node 0 ──[sends half]──→ Node 1
Node 1 ──[sends half]──→ Node 0
Both reduce their received data

Step 2: AllGather
Node 0 ──[sends result]──→ Node 1
Node 1 ──[sends result]──→ Node 0
Both now have complete result
```

**The Test:**
```python
# Each rank creates a tensor with its rank+1
tensor = torch.ones(1024, 1024, device='cuda:0') * (rank + 1)
# Rank 0: tensor = [[1, 1, ...], [1, 1, ...]]
# Rank 1: tensor = [[2, 2, ...], [2, 2, ...]]

# AllReduce sums tensors across all ranks
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# Result on both ranks: tensor = [[3, 3, ...], [3, 3, ...]]
expected = 1 + 2 = 3
```

**What it validates:**
- ✅ Data transfers correctly between nodes
- ✅ GPU-to-GPU communication works
- ✅ No data corruption
- ✅ NCCL collective operations work

#### Phase 6: Bandwidth Test

**Purpose:** Measure network throughput for different message sizes

```python
sizes = [1*1024*1024, 10*1024*1024, 100*1024*1024]  # 1MB, 10MB, 100MB

for size in sizes:
    tensor = torch.randn(size // 4, device='cuda:0')
    
    # Warmup (3 iterations to stabilize)
    for _ in range(3):
        dist.all_reduce(tensor)
    
    # Timed runs (10 iterations for accuracy)
    start = time.time()
    for _ in range(10):
        dist.all_reduce(tensor)
    elapsed = time.time() - start
    
    bandwidth_gbps = (size * 10 * 2) / elapsed / 1e9
    print(f"AllReduce {size/1024/1024:.0f}MB: {bandwidth_gbps:.1f} GB/s")
```

**Understanding the Bandwidth Formula:**

```
bandwidth_gbps = (size * 10 * 2) / elapsed / 1e9
```

Let's break down each component:

**`size`** = Bytes in one tensor (e.g., 100MB = 100 * 1024 * 1024 bytes)

**`10`** = Number of AllReduce operations in the timed loop

**`2`** = **The bidirectional factor** - Here's why:

For a 2-node AllReduce, the network traffic is:
```
Node 0 ────[size bytes]───→ Node 1
Node 0 ←───[size bytes]──── Node 1

Total network traffic = 2 × size bytes (bidirectional)
```

Each AllReduce operation involves:
- Node 0 sends its data to Node 1
- Node 1 sends its data to Node 0
- Both reduce and exchange results

**Total data transferred:** `size * 10 * 2` bytes
- `size` = bytes per tensor
- `10` = number of iterations
- `2` = bidirectional (send + receive)

**`elapsed`** = Total time for 10 iterations (seconds)

**`/ 1e9`** = Convert bytes/second to GB/s (gigabytes per second)

**Example calculation:**
```
- Tensor size: 100 MB = 104,857,600 bytes
- Iterations: 10
- Time: 0.5 seconds
- Bandwidth = (104,857,600 * 10 * 2) / 0.5 / 1e9
           = 2,097,152,000 / 0.5 / 1e9
           = 4.194 GB/s or ~4.2 GB/s
```

**Expected results:**
- **InfiniBand (400 Gb/s):** 120-180 GB/s ✅
- **TCP:** 1-5 GB/s ⚠️

**Why different message sizes?**
- **Small (1MB):** Tests latency-sensitive operations
- **Medium (10MB):** Typical gradient chunk size
- **Large (100MB):** Maximum throughput test

**Why it matters:** 
During training with 16 GPUs, gradients are synchronized using AllReduce after every training step. With model size of ~16GB and batch processing, fast AllReduce is critical:
- **120+ GB/s (InfiniBand):** Gradient sync takes ~100-200ms ✅
- **2-5 GB/s (TCP):** Gradient sync takes 3-8 seconds ❌ (bottleneck!)

Slow networking = most of training time spent waiting for communication!

**Sample output:**
```
  ✓ PASS  NCCL process group initialized
  ✓ PASS  NCCL communicator initialized
  ✓ PASS  Distributed barrier           Both nodes synchronized
  ✓ PASS  Inter-node AllReduce          Expected sum=3, got 3.0
  ✓ PASS  AllReduce 1MB                 125.3 GB/s
  ✓ PASS  AllReduce 10MB                132.1 GB/s
  ✓ PASS  AllReduce 100MB               128.7 GB/s

NOTE: Check NCCL_INFO logs above for network transport used:
  - 'NET/Socket' = TCP (1-5 GB/s typical)
  - 'NET/IB' = InfiniBand 400 Gb/s (120-180 GB/s typical)
```

---

### **Step 7: Storage** 💾

**What it does:**
```bash
# Check mount
ls -la /mnt/data

# Check free space
df -BG /mnt/data

# Check for existing checkpoints
find /mnt/data/checkpoints -name "checkpoint-*"
```

**Validates:**
- ✅ Shared filesystem is mounted at `/mnt/data`
- ✅ At least 100GB free space
- ✅ Checkpoint directory exists (if training was run before)

**Why it matters:** 
- Training saves checkpoints every few steps
- Models are large (16GB+ for Llama-3 8B)
- Running out of space mid-training = lost progress

---

### **Step 8: Ray Cluster** ☁️

**What it does:**
```python
import ray
ray.init(address='ray://ray-cluster-head-svc.ray-cluster.svc.cluster.local:10001')

resources = ray.cluster_resources()
cpus = resources.get('CPU', 0)
gpus = resources.get('GPU', 0)
nodes = len(ray.nodes())

print(f"Ray cluster: {nodes} nodes, {cpus} CPUs, {gpus} GPUs")
```

**Validates:**
- ✅ Ray head is accessible
- ✅ Ray cluster has GPU workers
- ✅ Ray sees all cluster resources

**Why it matters:** Training jobs submit through Ray. If Ray is down, jobs can't start.

---

### **Step 9: Training Pipeline Smoke Test** 🧪

**Execution Condition:** This test runs **only on Rank 0** to avoid duplicate work.

```bash
if [ "$RANK" = "0" ]; then
  echo "Running training pipeline smoke test..."
  
  # Isolate environment for single-GPU test
  unset WORLD_SIZE LOCAL_RANK RANK MASTER_ADDR MASTER_PORT
  export CUDA_VISIBLE_DEVICES=0
  
  # Run smoke test
  python3 /scripts/smoke_test.py
else
  echo "Skipping training pipeline test (Rank $RANK)"
  echo "Test only runs on Rank 0 to avoid duplicate work"
fi
```

**Why single-GPU mode?**
- Distributed training already validated in NCCL test (step 6)
- Single-GPU test is faster and simpler
- Validates basic training pipeline without distributed complexity

**Environment Isolation:**
| Variable | Action | Reason |
|----------|--------|--------|
| `WORLD_SIZE` | Unset | Force single-process mode |
| `LOCAL_RANK` | Unset | Disable distributed launcher |
| `RANK` | Unset | Single process, no rank needed |
| `MASTER_ADDR` | Unset | No master needed for single GPU |
| `MASTER_PORT` | Unset | No inter-process communication |
| `CUDA_VISIBLE_DEVICES` | Set to `0` | Use only first GPU (out of 8) |

**What it does:**
```python
# 1. Load small model (GPT-2 for speed)
model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cuda:0")

# 2. Wrap with LoRA adapters
lora_config = LoraConfig(r=8, target_modules=["c_attn"])
peft_model = get_peft_model(model, lora_config)

# 3. Create tiny dataset (40 samples)
dataset = Dataset.from_dict({"text": ["Hello!"] * 40})

# 4. Create SFTTrainer
trainer = SFTTrainer(
    model=peft_model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        max_steps=2,              # Only 2 training steps
        output_dir="/tmp/test",   # Temporary output
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
    )
)

# 5. Run training (2 steps only)
result = trainer.train()
print(f"Training loss: {result.training_loss:.4f}")
```

**Test Conditions:**
- **Model:** GPT-2 (124M params) - fast to load
- **Dataset:** 40 samples - just enough for 2 batches
- **Training Steps:** 2 - validates forward/backward pass
- **Batch Size:** 4 samples per step
- **GPU:** Single GPU (cuda:0) - simpler than distributed

**Why GPT-2?**
- Fast to download (124M parameters vs 8B for Llama-3)
- Uses same training APIs (transformers, peft, trl)
- Validates the training stack works

**What it validates:**
- ✅ `transformers` can load models
- ✅ `peft` can wrap models with LoRA
- ✅ `trl.SFTTrainer` initializes correctly
- ✅ Forward pass works
- ✅ Backward pass (gradient computation) works
- ✅ Optimizer step works
- ✅ Training loop completes

**Additional checks:**
- Llama-3 tokenizer (if cached in `/mnt/data/huggingface`)
- Training dataset (if cached)

**Why it matters:** This catches library issues, API changes, or configuration problems before you try training Llama-3 for hours.

---

## How to Run

### Submit the job:
```bash
kubectl apply -f k8s/preflight-check-job.yaml
```

### Monitor progress:
```bash
# Follow logs from both pods
kubectl logs -n ray-cluster -l app=preflight-check -f --all-containers

# Or watch specific pod
kubectl logs -n ray-cluster preflight-check-0 -f
```

### Check final status:
```bash
kubectl get jobs -n ray-cluster preflight-check

# Success:
# NAME              COMPLETIONS   DURATION   AGE
# preflight-check   2/2           8m32s      10m

# Failed:
# NAME              COMPLETIONS   DURATION   AGE
# preflight-check   0/2           8m32s      10m
```

### View results:
```bash
# Get logs from rank 0 (includes smoke test)
kubectl logs -n ray-cluster preflight-check-0

# Get logs from rank 1
kubectl logs -n ray-cluster preflight-check-1
```

### Cleanup:
```bash
# Job auto-deletes after 300 seconds (ttlSecondsAfterFinished: 300)
# Or manually delete:
kubectl delete job -n ray-cluster preflight-check
```

---

## Reading the Output

### Success Case ✅
```bash
==================================================
 PRE-FLIGHT CHECKS FOR MULTI-NODE TRAINING
 Pod: preflight-check-0 | Rank: 0 | IP: 10.0.26.195
==================================================

1. INSTALLING DEPENDENCIES
Dependencies installed.

2. GPU AVAILABILITY
  ✓ PASS  nvidia-smi available
  ✓ PASS  GPU count = 8              Found 8 H100 GPUs

index, name, memory.total [MiB]
0, NVIDIA H100 80GB HBM3, 81559 MiB
...

3. GPU HEALTH
  ✓ PASS  GPU 0 temperature          62°C
  ✓ PASS  GPU 1 temperature          64°C
  ...

4. CUDA & PyTorch
  ✓ PASS  PyTorch installed          Version 2.5.1
  ✓ PASS  CUDA available
  ✓ PASS  CUDA version                12.1
  ✓ PASS  PyTorch sees 8 GPUs
  ✓ PASS  CUDA tensor ops on all GPUs

5. INTRA-NODE NVLink P2P
  ✓ PASS  NVLink P2P connectivity    28/28 pairs connected
  ✓ PASS  NCCL backend available

6. INTER-NODE NCCL COMMUNICATION
  Rank: 0
  World Size: 2
  My IP: 10.0.26.195
  Discovering peers via headless service DNS...
  Found 2 peers: 10.0.26.195 10.0.26.200
  Master address (smallest IP): 10.0.26.195
  Computed rank from IP ordering: 0
  Starting NCCL test with RANK=0 MASTER_ADDR=10.0.26.195
  
  ✓ PASS  NCCL process group initialized
  ✓ PASS  NCCL communicator initialized
  ✓ PASS  Distributed barrier           Both nodes synchronized
  ✓ PASS  Inter-node AllReduce          Expected sum=3, got 3.0
  ✓ PASS  AllReduce 1MB                 125.3 GB/s
  ✓ PASS  AllReduce 10MB                132.1 GB/s
  ✓ PASS  AllReduce 100MB               128.7 GB/s

7. STORAGE
  ✓ PASS  Storage mounted at /mnt/data
  ✓ PASS  Free space > 100GB           450GB available
  ✓ PASS  Checkpoints directory        Found 0 checkpoint(s)

8. RAY CLUSTER CONNECTION
  ✓ PASS  Ray cluster connection       3 node(s), 64 CPUs, 16 GPUs

9. TRAINING PIPELINE (Rank 0 only)
  Testing training imports...
  ✓ PASS  trl                          v0.11.4
  ✓ PASS  peft                         v0.13.2
  ✓ PASS  accelerate                   v1.1.1
  ✓ PASS  datasets                     v3.2.0
  ✓ PASS  SFTTrainer/SFTConfig
  ✓ PASS  LoraConfig/get_peft_model
  
  Testing SFTConfig...
  ✓ PASS  SFTConfig                    max_length=128
  
  Testing LoraConfig...
  ✓ PASS  LoraConfig                   r=8, alpha=16
  
  Testing PEFT model wrapping...
  ✓ PASS  PEFT model                   trainable: 294,912 (0.24%)
  
  Testing dataset...
  ✓ PASS  Dataset                      40 samples
  
  Testing training step (single GPU)...
  ✓ PASS  Training step                loss=2.3456
  
  Testing Llama-3 tokenizer...
  ✓ PASS  Llama-3 tokenizer            vocab_size=128256
  
  Testing training dataset...
  ✓ PASS  Training dataset             Not cached (skip)
  
  ----------------------------------------
  ✓ Training pipeline: All 8 checks passed!

==================================================
 PREFLIGHT CHECK COMPLETE
==================================================
  Node: computeinstance-e00e3jy870jnn5cd4k
  Rank: 0
  All checks finished.
```

---

## Common Failures & How to Fix

### ❌ GPU Count Mismatch
```
✗ FAIL  GPU count = 8    Found 6 GPUs (expected 8)
```

**Cause:** Wrong GPU configuration or hardware failure

**Fix:**
```bash
# Check node GPU configuration
kubectl describe node <gpu-node-name> | grep -A 10 "Capacity:"

# Check terraform config
cat infra/k8s-installation/terraform.tfvars | grep gpu_nodes_preset
# Should be: 8gpu-128vcpu-1600gb
```

### ❌ High GPU Temperature
```
✗ FAIL  GPU 0 temperature    87°C (too hot!)
```

**Cause:** Cooling system issue or GPU under load

**Fix:**
```bash
# Check current GPU status
kubectl exec -n ray-cluster <pod-name> -- nvidia-smi

# Check datacenter cooling
# Contact infrastructure team if sustained high temps
```

### ❌ NVLink Not Connected
```
✗ FAIL  NVLink P2P connectivity    0/28 pairs connected
```

**Cause:** NVLink disabled in BIOS or firmware issue

**Fix:**
```bash
# Check NVLink status
kubectl exec -n ray-cluster <pod-name> -- nvidia-smi nvlink --status

# May require:
# - BIOS update
# - NVLink bridge installation
# - GPU firmware update
```

### ❌ NCCL Initialization Timeout
```
✗ FAIL  NCCL process group initialized    
         Connection timeout after 300s
```

**Cause:** Network connectivity issues

**Fix:**
```bash
# Check pod IPs
kubectl get pods -n ray-cluster -l app=preflight-check -o wide

# Check if pods can ping each other
kubectl exec -n ray-cluster preflight-check-0 -- ping -c 3 <pod-1-ip>

# Check firewall/network policies
kubectl get networkpolicies -A

# Check InfiniBand status (if available)
kubectl exec -n ray-cluster <pod-name> -- ibstatus
```

### ❌ Low Bandwidth (TCP Fallback)
```
✓ PASS  AllReduce 100MB    2.3 GB/s
NOTE: NET/Socket is being used (TCP)
```

**Cause:** InfiniBand not available, NCCL falling back to TCP

**Fix:**
```bash
# Check NCCL logs for transport details
kubectl logs -n ray-cluster <pod-name> | grep "NCCL INFO"

# Look for:
# NCCL INFO NET/IB : Using InfiniBand  ← Good! ✅
# NCCL INFO NET/Socket : Using TCP     ← Fallback ⚠️

# Check InfiniBand drivers
kubectl exec -n ray-cluster <pod-name> -- ibv_devices

# May need to install/configure:
# - NVIDIA Network Operator
# - InfiniBand drivers
# - RDMA devices
```

### ❌ Storage Not Mounted
```
✗ FAIL  Storage mounted    /mnt/data not found
```

**Cause:** Filestore not configured or mount failed

**Fix:**
```bash
# Check terraform filestore config
cat infra/k8s-installation/terraform.tfvars | grep enable_filestore
# Should be: enable_filestore = true

# Check if filesystem exists
kubectl get filesystem -A

# Re-apply terraform if needed
cd infra/k8s-installation
terraform apply
```

### ❌ Ray Connection Failed
```
✗ FAIL  Ray cluster connection    Connection refused
```

**Cause:** Ray cluster not running or head service unreachable

**Fix:**
```bash
# Check Ray cluster status
kubectl get pods -n ray-cluster

# Check Ray head service
kubectl get svc -n ray-cluster ray-cluster-head-svc

# Restart Ray cluster if needed
kubectl delete pod -n ray-cluster -l ray.io/node-type=head
```

### ❌ Training Step Failed
```
✗ FAIL  Training step    ModuleNotFoundError: No module named 'peft'
```

**Cause:** Package installation failed

**Fix:**
```bash
# Check installation logs in pod output
kubectl logs -n ray-cluster <pod-name> | grep -A 20 "INSTALLING DEPENDENCIES"

# May need to:
# - Update package versions
# - Fix package conflicts
# - Check PyPI/package registry connectivity
```

---

## Integration with Training Workflow

### Before Training
```bash
# 1. Run preflight check
kubectl apply -f k8s/preflight-check-job.yaml

# 2. Wait for completion (5-10 min)
kubectl wait --for=condition=complete --timeout=15m job/preflight-check -n ray-cluster

# 3. Check if passed
kubectl get jobs -n ray-cluster preflight-check
# COMPLETIONS should be 2/2

# 4. If passed, proceed with training
kubectl apply -f k8s/ray-training-job.yaml
```

### After Infrastructure Changes
Run preflight check after:
- Terraform changes to GPU nodes
- Kubernetes upgrades
- Network configuration changes
- Driver/firmware updates
- Long cluster downtime

---

## Performance Benchmarks

### Expected Timings

| Step | Duration |
|------|----------|
| Dependency installation | 1-2 min |
| GPU checks | 10-30 sec |
| CUDA/PyTorch tests | 10-20 sec |
| NVLink tests | 5-10 sec |
| NCCL initialization | 30-60 sec |
| Bandwidth tests | 20-30 sec |
| Storage checks | 5 sec |
| Ray connection | 5-10 sec |
| Training smoke test | 2-3 min |
| **Total** | **5-8 min** |

### Bandwidth Expectations

| Network | 1MB | 10MB | 100MB |
|---------|-----|------|-------|
| **InfiniBand 400 Gb/s (Good)** | 100-150 GB/s | 120-180 GB/s | 120-180 GB/s |
| **TCP (Acceptable)** | 1-3 GB/s | 2-5 GB/s | 2-5 GB/s |
| **TCP (Slow)** | < 1 GB/s | < 1 GB/s | < 1 GB/s |

---

## Key Takeaways

1. **Run before every training session** - Catches issues early
2. **Check NCCL logs** - Verify InfiniBand is being used
3. **Monitor bandwidth** - Should be 120+ GB/s for good performance (400 Gb/s InfiniBand)
4. **Don't skip it** - 10 minutes of checking saves hours of debugging
5. **Keep logs** - Useful for troubleshooting training issues later

The preflight check is your **first line of defense** against infrastructure problems. A successful preflight check means your cluster is ready for production training! 🚀
