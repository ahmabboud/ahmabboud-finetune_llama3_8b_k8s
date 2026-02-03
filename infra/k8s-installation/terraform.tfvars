# SSH config
ssh_user_name = "ubuntu" # Username you want to use to connect to the nodes
ssh_public_key = {
  # key = ""
  path = "~/.ssh/id_ed25519.pub"
}

# K8s nodes
cpu_nodes_count           = 2 # Number of CPU nodes
gpu_nodes_count_per_group = 2 # Number of GPU nodes per group (temporarily 0 to switch fabric)
gpu_node_groups           = 1 # In case you need more then 100 nodes in cluster you have to put multiple node groups
# CPU platform and presets: https://docs.nebius.com/compute/virtual-machines/types#cpu-configurations
cpu_nodes_platform = "cpu-d3"     # CPU nodes platform
cpu_nodes_preset   = "4vcpu-16gb" # CPU nodes preset
# GPU platform and preset: https://docs.nebius.com/compute/virtual-machines/types#gpu-configurations
gpu_nodes_platform = "gpu-h100-sxm"        # GPU nodes platform: gpu-h100-sxm, gpu-h200-sxm, gpu-b200-sxm
gpu_nodes_preset   = "8gpu-128vcpu-1600gb" # GPU nodes preset: 8gpu-128vcpu-1600gb, 8gpu-128vcpu-1600gb, 8gpu-160vcpu-1792gb
# Infiniband fabrics: https://docs.nebius.com/compute/clusters/gpu#fabrics
infiniband_fabric = "fabric-2" # Infiniband fabric name (switched from fabric-6)

gpu_nodes_driverfull_image = true
enable_k8s_node_group_sa   = true
enable_egress_gateway      = false
cpu_nodes_preemptible      = false
gpu_nodes_preemptible      = false

# MIG configuration
# mig_strategy =        # If set, possible values include 'single', 'mixed', 'none'
# mig_parted_config =   # If set, value will be checked against allowed for the selected 'gpu_nodes_platform'

# Observability
enable_prometheus = true  # Enable or disable Prometheus and Grafana deployment with true or false
enable_loki       = true # Enable or disable Loki deployment with true or false

# Storage
enable_filestore     = true                            # Enable or disable Filestore integration with true or false
filestore_disk_size  = 2 * (1024 * 1024 * 1024 * 1024) # Set Filestore disk size in bytes. 2TB shared filesystem
filestore_block_size = 4096                            # Set Filestore block size in bytes


# KubeRay
# for GPU isolation to work with kuberay, gpu_nodes_driverfull_image must be set 
# to false.  This is because we enable acess to infiniband via securityContext.privileged
enable_kuberay = true # Enable KubeRay for distributed ML workloads

#kuberay CPU worker setup
# if you have no CPU only nodes, set these to zero
# kuberay_cpu_worker_image = ""  # set default CPU worker can leave it commented out in most cases
kuberay_min_cpu_replicas = 1
kuberay_max_cpu_replicas = 2
# kuberay_cpu_resources = {
#   cpus = 2
#   memory = 4  # memory allocation in gigabytes
# }

#kuberay GPU worker pod setup
kuberay_gpu_worker_image = "cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/ray-gpu-infiniband:2.46.0-py310"
kuberay_min_gpu_replicas = 2
kuberay_max_gpu_replicas = 2  # Max 2 nodes (each with 8 GPUs)
kuberay_gpu_resources = {
  cpus   = 120
  gpus   = 8     # All 8 H100s per node
  memory = 1400  # ~1400 GB per node
}

# NPD nebius-gpu-health-checker helm install
gpu_health_cheker = false
