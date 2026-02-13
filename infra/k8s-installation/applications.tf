# KubeRay Operator - Ray cluster management on Kubernetes
# Deploys: Ray head node, Ray worker pods (CPU + GPU), autoscaling, monitoring
module "kuberay" {
  source = "../modules/kuberay"
  count  = var.enable_kuberay ? 1 : 0

  depends_on = [
    nebius_mk8s_v1_node_group.cpu-only,
    nebius_mk8s_v1_node_group.gpu,
    module.network-operator,
    module.gpu-operator,
  ]

  parent_id  = var.parent_id
  cluster_id = nebius_mk8s_v1_cluster.k8s-cluster.id
  
  # CPU worker configuration
  cpu_platform     = local.cpu_nodes_platform
  cpu_worker_image = var.kuberay_cpu_worker_image  # Ray container image for CPU pods
  min_cpu_replicas = var.kuberay_min_cpu_replicas  # Minimum Ray CPU worker pods
  max_cpu_replicas = var.kuberay_max_cpu_replicas  # Maximum Ray CPU worker pods (autoscale)
  cpu_resources    = var.kuberay_cpu_resources     # CPUs and memory per CPU worker pod
  
  # GPU worker configuration
  gpu_platform     = local.gpu_nodes_platform
  gpu_worker_image = var.kuberay_gpu_worker_image  # Ray container with GPU drivers, IB, NCCL
  min_gpu_replicas = var.kuberay_min_gpu_replicas  # Min GPU worker pods (0 = scale to zero)
  max_gpu_replicas = var.kuberay_max_gpu_replicas  # Max GPU worker pods
  gpu_resources    = var.kuberay_gpu_resources     # GPUs, CPUs, memory per GPU worker pod

}
