# KubeRay Helm chart deployment via Nebius Application Catalog
# Deploys: Ray head node, CPU/GPU worker pools, autoscaler, monitoring integration
resource "nebius_applications_v1alpha1_k8s_release" "this" {
  parent_id        = var.parent_id
  cluster_id       = var.cluster_id
  application_name = var.name          # Default: "ray-cluster"
  namespace        = var.namespace      # Default: "ray-cluster"
  product_slug     = "nebius/ray-cluster"  # Nebius-managed KubeRay chart
  
  # Render Helm values from template with user-provided configuration
  values = templatefile("${path.module}/files/ray-values.yaml.tftpl", {
    cpu_platform     = var.cpu_platform      # CPU node platform for worker affinity
    cpu_worker_image = var.cpu_worker_image  # Ray container for CPU pods
    min_cpu_replicas = var.min_cpu_replicas  # Min CPU worker pods
    max_cpu_replicas = var.max_cpu_replicas  # Max CPU worker pods (autoscale)
    gpu_platform     = var.gpu_platform      # GPU node platform for worker affinity
    cpu_resources    = var.cpu_resources     # CPUs and memory per CPU pod
    gpu_worker_image = var.gpu_worker_image  # Ray container with GPU drivers
    min_gpu_replicas = var.min_gpu_replicas  # Min GPU worker pods (0 = scale to zero)
    max_gpu_replicas = var.max_gpu_replicas  # Max GPU worker pods
    gpu_resources    = var.gpu_resources     # GPUs, CPUs, memory per GPU pod
  })
}
