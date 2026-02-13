# SSH config
ssh_user_name = "ubuntu" # Username you want to use to connect to the nodes
ssh_public_key = {
  # key = ""
  path = "~/.ssh/id_ed25519.pub"
}

# K8s nodes
cpu_nodes_count           = 2 # Number of CPU nodes

# GPU nodes - autoscaling configuration (allows scale to zero)
gpu_autoscaling_enabled   = true  # Enable autoscaling (scale to zero when idle)
gpu_min_nodes             = 0     # Minimum nodes (0 = scale to zero)
gpu_max_nodes             = 2     # Maximum nodes when needed
gpu_nodes_count_per_group = 2     # Only used if gpu_autoscaling_enabled=false (fixed node count per group)
gpu_node_groups           = 1     # Number of separate GPU node groups (use multiple groups if scaling >100 nodes, Nebius limit per node group)
# CPU platform and presets: https://docs.nebius.com/compute/virtual-machines/types#cpu-configurations
cpu_nodes_platform = "cpu-d3"     # CPU nodes platform
cpu_nodes_preset   = "4vcpu-16gb" # CPU nodes preset
# GPU platform and preset: https://docs.nebius.com/compute/virtual-machines/types#gpu-configurations
gpu_nodes_platform = "gpu-h100-sxm"        # GPU nodes platform: gpu-h100-sxm, gpu-h200-sxm, gpu-b200-sxm
gpu_nodes_preset   = "8gpu-128vcpu-1600gb" # GPU nodes preset: 8gpu-128vcpu-1600gb, 8gpu-128vcpu-1600gb, 8gpu-160vcpu-1792gb
# Infiniband fabrics: https://docs.nebius.com/compute/clusters/gpu#fabrics
infiniband_fabric = "fabric-2" # Infiniband fabric name (switched from fabric-6)

# Node configuration (critical for InfiniBand + reliability)
gpu_nodes_driverfull_image = true  # TRUE = required for InfiniBand (privileged containers need pre-installed drivers)
enable_k8s_node_group_sa   = true  # TRUE = required for nodes to access Nebius Filestore, Container Registry
enable_egress_gateway      = false # FALSE = simplifies networking (no external API calls in training)
cpu_nodes_preemptible      = false # FALSE = control plane stability (Ray head, monitoring) worth +$15/day
gpu_nodes_preemptible      = false # FALSE = training reliability over 60% cost savings (avoid mid-run interruptions)

# MIG (Multi Instance GPU) configuration (disabled for large model training)
# mig_strategy =        # COMMENTED = Full H100 needed (20GB memory per worker, can't fit in MIG slices)
# mig_parted_config =   # COMMENTED = Use MIG only for multi-tenant inference or small models (<3B params)

# Observability
enable_prometheus = true  # Enable or disable Prometheus and Grafana deployment with true or false
enable_loki       = true # Enable or disable Loki deployment with true or false

# Storage
enable_filestore     = true                            # Enable or disable Filestore integration with true or false
filestore_disk_size  = 2 * (1024 * 1024 * 1024 * 1024) # Set Filestore disk size in bytes. 2TB shared filesystem
filestore_block_size = 4096                            # Filestore block size in bytes (4KB default, affects I/O performance for small files)


# KubeRay
# for GPU isolation to work with kuberay, gpu_nodes_driverfull_image must be set 
# to false.  This is because we enable acess to infiniband via securityContext.privileged
enable_kuberay = true # Enable KubeRay for distributed ML workloads

#kuberay CPU worker setup
# if you have no CPU only nodes, set these to zero
# kuberay_cpu_worker_image = ""  # Ray container image for CPU workers (uses default if commented)
kuberay_min_cpu_replicas = 1      # Minimum CPU worker pods (for job submission, coordination tasks)
kuberay_max_cpu_replicas = 2      # Maximum CPU worker pods (autoscale up to this limit)
# kuberay_cpu_resources = {
#   cpus = 2
#   memory = 4  # memory allocation in gigabytes
# }

#kuberay GPU worker pod setup
kuberay_gpu_worker_image = "cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/ray-gpu-infiniband:2.46.0-py310" # Ray container with GPU drivers, InfiniBand, Python 3.10
kuberay_min_gpu_replicas = 0  # Scale to zero when idle
kuberay_max_gpu_replicas = 2  # Max 2 nodes (each with 8 GPUs)
kuberay_gpu_resources = {
  cpus   = 120   # CPUs per Ray worker pod (leave ~8 for system, dataloaders use 8-16 cores)
  gpus   = 8     # All 8 H100s per node
  memory = 1400  # ~1400 GB per node
}

# NPD (Node Problem Detector) with GPU health checker
gpu_health_cheker = false # FALSE = demo simplicity (manual monitoring sufficient for 2 nodes, enable for 512 GPU production)
