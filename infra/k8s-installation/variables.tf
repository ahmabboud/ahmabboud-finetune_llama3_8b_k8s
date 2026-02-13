# Global
variable "tenant_id" {
  description = "Tenant ID (organization identifier in Nebius IAM)."
  type        = string
}

variable "parent_id" {
  description = "Project ID (parent resource for all infrastructure)."
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID where K8s nodes will be deployed."
  type        = string
}

variable "region" {
  description = "Nebius region (e.g., eu-north1, eu-west1, us-central1)."
  type        = string
}

# K8s cluster 
variable "k8s_version" {
  description = "Kubernetes version (e.g., 1.31). Leave null to use Nebius backend default (recommended)."
  type        = string
  default     = null
}

variable "etcd_cluster_size" {
  description = "Number of etcd replicas for K8s control plane (3 = HA, 1 = single node testing)."
  type        = number
  default     = 3
}

variable "enable_egress_gateway" {
  description = "Enable Cilium Egress Gateway for controlled external traffic routing."
  type        = bool
  default     = false
}

# K8s filestore
variable "enable_filestore" {
  description = "Enable Nebius Filestore (managed NFS) for shared storage across nodes."
  type        = bool
  default     = false
}

variable "filestore_disk_type" {
  description = "Filestore disk type (NETWORK_SSD default, SSD-backed NFS)."
  type        = string
  default     = "NETWORK_SSD"
}

variable "filestore_disk_size" {
  description = "Filestore capacity in bytes (e.g., 2TB = 2 * 1024^4)."
  type        = number
  default     = 1 * 1024 * 1024 * 1024 # 1 GiB
}

variable "filestore_block_size" {
  description = "Filestore block size in bytes (4KB default, affects small file I/O performance)."
  type        = number
  default     = 4096
}

# K8s access
variable "ssh_user_name" {
  description = "SSH username."
  type        = string
  default     = "ubuntu"
}

variable "ssh_public_key" {
  description = "SSH Public Key to access the cluster nodes"
  type = object({
    key  = optional(string),
    path = optional(string, "~/.ssh/id_rsa.pub")
  })
  default = {}
  validation {
    condition     = var.ssh_public_key.key != null || fileexists(var.ssh_public_key.path)
    error_message = "SSH Public Key must be set by `key` or file `path` ${var.ssh_public_key.path}"
  }
}

# K8s CPU node group
variable "cpu_nodes_count" {
  description = "Number of CPU-only nodes (for control plane workloads, monitoring, Ray head)."
  type        = number
  default     = 3
}

variable "cpu_nodes_platform" {
  description = "CPU platform (e.g., cpu-d3). Leave null to use region default."
  type        = string
  default     = null
}

variable "cpu_nodes_preset" {
  description = "CPU preset defining vCPUs and RAM (e.g., 4vcpu-16gb, 16vcpu-64gb). Leave null for region default."
  type        = string
  default     = null
}

variable "cpu_disk_type" {
  description = "Boot disk type for CPU nodes (NETWORK_SSD = balanced, NETWORK_SSD_IO_M3 = high IOPS)."
  type        = string
  default     = "NETWORK_SSD"
}

variable "cpu_disk_size" {
  description = "Boot disk size in GB for CPU nodes (OS + local storage)."
  type        = string
  default     = "128"
}

# K8s GPU node group
variable "gpu_nodes_count_per_group" {
  description = "Number of nodes in the GPU node group (used when gpu_autoscaling_enabled=false)."
  type        = number
  default     = 2
}

variable "gpu_autoscaling_enabled" {
  description = "Enable autoscaling for GPU node group (allows scale to zero)."
  type        = bool
  default     = false
}

variable "gpu_min_nodes" {
  description = "Minimum number of GPU nodes when autoscaling is enabled (can be 0 for scale-to-zero)."
  type        = number
  default     = 0
}

variable "gpu_max_nodes" {
  description = "Maximum number of GPU nodes when autoscaling is enabled."
  type        = number
  default     = 2
}

variable "gpu_node_groups" {
  description = "Number of GPU node groups."
  type        = number
  default     = 1
}

variable "gpu_nodes_platform" {
  description = "Platform for nodes in the GPU node group."
  type        = string
  default     = null
}

variable "gpu_nodes_driverfull_image" {
  description = "Use node images with pre-installed NVIDIA drivers (true = device-plugin, false = gpu-operator manages drivers)."
  type        = bool
  default     = false
}

variable "gpu_nodes_preset" {
  description = "GPU preset defining GPU count, vCPUs, RAM (e.g., 8gpu-128vcpu-1600gb for 8xH100). Leave null for region default."
  type        = string
  default     = null
}

variable "gpu_disk_type" {
  description = "Boot disk type for GPU nodes (NETWORK_SSD = balanced, NETWORK_SSD_NON_REPLICATED = higher performance but no redundancy)."
  type        = string
  default     = "NETWORK_SSD" # NETWORK_SSD NETWORK_SSD_NON_REPLICATED NETWORK_SSD_IO_M3
}

variable "gpu_disk_size" {
  description = "Boot disk size in GB for GPU nodes (1TB default for model cache, containers)."
  type        = string
  default     = "1023"
}

variable "enable_gpu_cluster" {
  description = "Enable InfiniBand GPU cluster fabric for high-speed inter-node GPU communication (required for multi-node training)."
  type        = bool
  default     = true
}

variable "infiniband_fabric" {
  description = "InfiniBand fabric name (region-specific, e.g., fabric-2, fabric-3). Leave null to use region default."
  type        = string
  default     = null
}

variable "gpu_nodes_assign_public_ip" {
  description = "Assign public IPs to GPU nodes (false recommended, access via bastion for security)."
  type        = bool
  default     = false
}

variable "gpu_node_taints" {
  description = "Taints applied to GPU nodes to prevent non-GPU workloads from scheduling (nvidia.com/gpu:NoSchedule default)."
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

variable "enable_k8s_node_group_sa" {
  description = "Enable service account for K8s node groups to access Nebius cloud resources (storage, IAM, etc.)."
  type        = bool
  default     = true
}

variable "mig_parted_config" {
  description = "MIG partitioning config (e.g., all-1g.10gb, all-7g.80gb). Leave null to disable MIG. See locals.tf for valid configs per platform."
  type        = string
  default     = null

  validation {
    condition     = var.mig_parted_config == null || contains(local.valid_mig_parted_configs[local.gpu_nodes_platform], coalesce(var.mig_parted_config, "null"))
    error_message = "Invalid MIG config '${coalesce(var.mig_parted_config, "null")}' for the selected GPU platform '${local.gpu_nodes_platform}'. Must be one of ${join(", ", local.valid_mig_parted_configs[local.gpu_nodes_platform])} or left unset."
  }
}

# Observability
variable "enable_loki" {
  description = "Enable Loki for centralized log aggregation (collects logs from all pods)."
  type        = bool
  default     = true
}

variable "enable_prometheus" {
  description = "Enable Prometheus for metrics collection (GPU utilization, pod metrics, cluster health)."
  type        = bool
  default     = true
}

variable "loki_access_key_id" {
  description = "Loki S3 access key ID for remote storage (optional, leave null for local storage)."
  type    = string
  default = null
}

variable "loki_secret_key" {
  description = "Loki S3 secret key for remote storage (optional)."
  type    = string
  default = null
}

variable "loki_custom_replication_factor" {
  description = "Loki replica count (default: 1 replica per 20 nodes). Set manually to override auto-calculation."
  type        = number
  default     = null
}

# Helm
variable "iam_token" {
  description = "IAM token for Helm provider authentication (source from environment.sh, not committed to git)."
  type        = string
}

variable "test_mode" {
  description = "Enable test mode (deploys NCCL bandwidth tests, disables production optimizations)."
  type        = bool
  default     = false
}

variable "enable_kuberay" {
  description = "Enable KubeRay operator for Ray cluster management (distributed ML workloads)."
  type        = bool
  default     = false
}

variable "kuberay_cpu_worker_image" {
  description = "Docker image for Ray CPU worker pods (leave null for default Nebius-provided image)."
  default     = null
}

variable "kuberay_min_cpu_replicas" {
  description = "Minimum Ray CPU worker pods (for job submission, lightweight coordination tasks)."
  type        = number
  default     = 0
}

variable "kuberay_max_cpu_replicas" {
  description = "Maximum Ray CPU worker pods (autoscale between min and max based on workload)."
  type        = number
  default     = 0
}

variable "kuberay_cpu_resources" {
  description = "Resources allocated to each Ray CPU worker pod (CPUs and memory in GB)."
  type = object({
    cpus   = number
    memory = number
  })
  default = null
}

# GPU worker pod setup
variable "kuberay_gpu_worker_image" {
  description = "Docker image for Ray GPU worker pods (must include GPU drivers, InfiniBand, NCCL, Ray)."
  default     = null
}

variable "kuberay_min_gpu_replicas" {
  description = "Minimum Ray GPU worker pods (0 = scale to zero when idle for cost savings)."
  type        = number
  default     = 0
}

variable "kuberay_max_gpu_replicas" {
  description = "Maximum Ray GPU worker pods (each pod = 1 K8s node with 8 GPUs by default)."
  type        = number
  default     = 0
}

variable "kuberay_gpu_resources" {
  description = "Resources per Ray GPU worker pod: gpus (number of GPUs), cpus (vCPUs), memory (GB)."
  type = object({
    cpus   = number
    gpus   = number
    memory = number
  })
  default = null
}

variable "mig_strategy" {
  description = "MIG strategy for GPU operator (single = all GPUs same MIG config, mixed = different configs, none = MIG disabled)."
  type        = string
  default     = null
}

variable "cpu_nodes_preemptible" {
  description = "Use preemptible (spot) VMs for CPU nodes (lower cost, can be terminated, not recommended for production)."
  type        = bool
  default     = false
}

variable "gpu_nodes_preemptible" {
  description = "Use preemptible (spot) VMs for GPU nodes (strongly discouraged for training jobs due to interruptions)."
  type        = bool
  default     = false
}

variable "gpu_health_cheker" {
  description = "Enable GPU health monitoring via Node Problem Detector (detects thermal, memory, NVLink failures)."
  type        = bool
  default     = true
}
