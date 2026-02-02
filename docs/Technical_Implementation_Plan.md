# **Multi-Node LLM Fine-Tuning (16x H100)**

## **Technical Requirements Summary**

Before proceeding with the implementation, the following requirements must be met based on the PoC allocation and the "Function Calling" task objectives:

### **Hardware Requirements**

* **Compute:** 16x NVIDIA H100 GPU (80GB VRAM each), provisioned as 2 nodes of 8 GPUs.
* **Interconnect:** High-bandwidth InfiniBand (fabric-3) for low-latency multi-node collective communication (NCCL).
* **Node Storage:** ~1TB SSD per GPU node for OS, container images, and local caching.
* **Shared Storage:** 2TB SSD Shared Filesystem (mounted at `/mnt/filestore`) for datasets, model weights, and checkpoints.

### **Software Requirements**

* **Orchestration:** Managed Kubernetes (MK8s) with GPU Operator and Network Operator.
* **Base Image:** nvcr.io/nvidia/pytorch:24.07-py3 (or newer) with H100 Hopper architecture support.
* **Deep Learning Framework:** PyTorch with native DistributedDataParallel (DDP) or DeepSpeed.
* **Model:** Llama-3-8B-Instruct (Open-source base).
* **Optimization Libraries:** flash-attn (v2.x), peft (for LoRA), and deepspeed (ZeRO-3).

### **Operational Requirements**

* **Strict High Utilization:** Training must maintain >85% GPU VRAM saturation to demonstrate hardware efficiency.
* **Monitoring:** Real-time visibility into GPU metrics via DCGM Exporter and Prometheus/Grafana.
* **Reproducibility:** Fully automated environment setup via Terraform and Kubernetes manifests.

---

## **Technical Implementation Task List**

### **1. Infrastructure Provisioning (Terraform)** ⏳ **IN PROGRESS**

**Purpose:** Automate the creation of the 16x H100 cluster and necessary storage.

**Location:** `/infra/k8s-installation/`

- [ ] **Configure Environment Variables:**
  - [x] Set `NEBIUS_TENANT_ID`, `NEBIUS_PROJECT_ID`, `NEBIUS_REGION` in `environment.sh`
  - [ ] Run `source ./environment.sh` to initialize backend and credentials

- [ ] **Configure Terraform Variables:** (`terraform.tfvars`)
  - [x] SSH key configuration (`~/.ssh/id_ed25519.pub`)
  - [x] GPU platform: `gpu-h100-sxm` with `8gpu-128vcpu-1600gb` preset
  - [x] GPU nodes: 2 nodes × 8 GPUs = 16 H100s
  - [x] InfiniBand fabric: `fabric-3`
  - [x] Shared storage: 2TB Filestore
  - [x] CPU nodes: 2 nodes for cluster services

- [ ] **Deploy Infrastructure:**
  - [ ] `terraform init`
  - [ ] `terraform plan`
  - [ ] `terraform apply`

- [ ] **Verify Deployment:**
  - [ ] Get kubeconfig: `nebius mk8s v1 cluster get-credentials --id <cluster-id> --external`
  - [ ] Verify nodes: `kubectl get nodes`
  - [ ] Verify GPUs: `kubectl describe nodes | grep nvidia.com/gpu`

**Storage Allocation (4TB Total Quota):**
| Type | Size | Purpose |
|------|------|---------|
| Shared Filestore | 2TB | Datasets, model weights, checkpoints (`/mnt/filestore`) |
| GPU Node Disks | ~1TB × 2 | OS, containers, local cache |
| CPU Node Disks | ~128GB × 2 | Cluster services |

---

### **2. Environment & Container Setup**

**Purpose:** Prepare the training environment with all required dependencies.

- [ ] **Create Training Container Image:**
  - [ ] Base: `nvcr.io/nvidia/pytorch:24.07-py3`
  - [ ] Install: `lightning`, `deepspeed`, `transformers`, `peft`, `flash-attn`, `datasets`
  - [ ] Push to container registry

- [ ] **Create Kubernetes Resources:**
  - [ ] Namespace for training workloads
  - [ ] PersistentVolume and PVC for shared filestore
  - [ ] ConfigMaps for training configuration

---

### **3. Dataset & Model Preparation**

**Purpose:** Stage the data and model weights on shared storage.

- [ ] **Model Staging:**
  - [ ] Download `meta-llama/Meta-Llama-3-8B-Instruct` to `/mnt/filestore/models/`
  - [ ] Verify model integrity

- [ ] **Dataset Processing:**
  - [ ] Download `Salesforce/xLAM-v0.1-r` dataset
  - [ ] Tokenize and preprocess for function calling format
  - [ ] Store processed data at `/mnt/filestore/datasets/`

---

### **4. Distributed Training Job**

**Purpose:** Implement multi-node fine-tuning optimized for >85% VRAM utilization.

- [ ] **Training Script:**
  - [ ] PyTorch DDP or DeepSpeed ZeRO-3 configuration
  - [ ] FlashAttention-2 enabled
  - [ ] LoRA (PEFT) for parameter-efficient fine-tuning
  - [ ] Gradient checkpointing for memory optimization

- [ ] **Kubernetes Job Manifest:**
  - [ ] Multi-node PyTorchJob or native Job with hostNetwork for NCCL
  - [ ] Resource requests: 8 GPUs per pod
  - [ ] Node affinity for GPU nodes
  - [ ] Shared volume mounts

---

### **5. Monitoring & Observability**

**Purpose:** Provide proof of high GPU utilization.

- [ ] **Verify DCGM Exporter:** (installed by Terraform)
  - [ ] Check GPU metrics: `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_MEM_COPY_UTIL`

- [ ] **Access Grafana Dashboard:**
  - [ ] Port-forward: `kubectl -n o11y port-forward svc/grafana-and-prometheus 8080:80`
  - [ ] View GPU utilization dashboards

- [ ] **Experiment Tracking:**
  - [ ] Integrate Weights & Biases for loss curves and metrics

---

### **6. Validation & Evaluation**

**Purpose:** Verify the success of the "Function Calling" fine-tuning task.

- [ ] **Merge LoRA Weights:**
  - [ ] Combine adapter with base model

- [ ] **Inference Testing:**
  - [ ] Test against tool-use prompts
  - [ ] Verify JSON tool call formatting per xLAM schema

---

## **Current Status**

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Infrastructure | ⏳ In Progress | Terraform config ready, pending deployment |
| 2. Environment | ⬜ Not Started | Waiting for cluster |
| 3. Data Prep | ⬜ Not Started | |
| 4. Training | ⬜ Not Started | |
| 5. Monitoring | ⬜ Not Started | Deployed with infra |
| 6. Validation | ⬜ Not Started | |

---

## **Next Steps**

1. Run `source ./environment.sh` in `/infra/k8s-installation/`
2. Run `terraform init && terraform plan`
3. Review plan and run `terraform apply`
4. Wait for cluster provisioning (~30-40 minutes)
5. Configure kubectl and verify cluster health