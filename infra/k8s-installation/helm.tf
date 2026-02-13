# NVIDIA Network Operator - manages InfiniBand networking for GPU nodes
# Deploys: RDMA device plugin, NCCL plugin, host network CNI
module "network-operator" {
  depends_on = [
    nebius_mk8s_v1_node_group.cpu-only,
    nebius_mk8s_v1_node_group.gpu,
  ]
  source     = "../modules/network-operator"
  parent_id  = var.parent_id
  cluster_id = nebius_mk8s_v1_cluster.k8s-cluster.id
}

# NVIDIA GPU Operator - manages GPU drivers and device plugins
# Only used when gpu_nodes_driverfull_image = false (drivers managed by operator)
module "gpu-operator" {
  count = var.gpu_nodes_driverfull_image ? 0 : 1

  depends_on = [
    module.network-operator
  ]
  source       = "../modules/gpu-operator"
  parent_id    = var.parent_id
  cluster_id   = nebius_mk8s_v1_cluster.k8s-cluster.id
  mig_strategy = var.mig_strategy  # MIG partitioning strategy (single/mixed/none)
}

# NVIDIA Device Plugin - exposes GPUs to Kubernetes scheduler
# Only used when gpu_nodes_driverfull_image = true (drivers pre-installed in node image)
module "device-plugin" {
  count = var.gpu_nodes_driverfull_image ? 1 : 0

  source     = "../modules/device-plugin"
  parent_id  = var.parent_id
  cluster_id = nebius_mk8s_v1_cluster.k8s-cluster.id
}

# Observability stack: Prometheus, Grafana, Loki
# Deploys monitoring and logging infrastructure
module "o11y" {
  source          = "../modules/o11y"
  parent_id       = var.parent_id
  tenant_id       = var.tenant_id
  cluster_id      = nebius_mk8s_v1_cluster.k8s-cluster.id
  cpu_nodes_count = var.cpu_nodes_count
  gpu_nodes_count = var.gpu_autoscaling_enabled ? var.gpu_max_nodes * var.gpu_node_groups : var.gpu_nodes_count_per_group * var.gpu_node_groups

  o11y = {
    loki = {
      enabled            = var.enable_loki             # Log aggregation
      replication_factor = var.loki_custom_replication_factor
      region             = var.region
    }
    prometheus = {
      enabled = var.enable_prometheus  # Metrics collection
      pv_size = "25Gi"                 # Persistent volume size for metrics storage
    }
  }
  test_mode = var.test_mode
}

# NCCL bandwidth test - validates GPU-to-GPU communication
# Only deployed in test_mode for performance validation
module "nccl-test" {
  count = var.test_mode ? 1 : 0
  depends_on = [
    module.gpu-operator,
  ]
  source          = "../modules/nccl-test"
  number_of_hosts = var.gpu_autoscaling_enabled ? var.gpu_max_nodes : nebius_mk8s_v1_node_group.gpu[0].fixed_node_count
}

# Nebius GPU Health Checker - monitors GPU hardware health
# Detects: thermal issues, memory errors, NVLink failures
resource "helm_release" "nebius_gpu_health_checker" {
  count = var.gpu_health_cheker ? 1 : 0

  depends_on = [
    nebius_mk8s_v1_node_group.gpu,
  ]

  name      = "nebius-gpu-health-checker"
  chart     = "${path.module}/npd-helm/nebius-npd-0.2.0.tgz"
  namespace = "default"

  set {
    name  = "hardware.profile"
    value = local.platform_preset_to_hardware_profile[local.hardware_profile_key]  # e.g., "8xH100"
  }
}
