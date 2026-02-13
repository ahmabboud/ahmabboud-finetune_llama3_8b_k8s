# Kubernetes cluster connection details
# Use these outputs to configure kubectl access and retrieve cluster info
output "kube_cluster" {
  description = "Kubernetes cluster info."
  value = {
    id        = try(nebius_mk8s_v1_cluster.k8s-cluster.id, null)
    name      = try(nebius_mk8s_v1_cluster.k8s-cluster.name, null)
    endpoints = nebius_mk8s_v1_cluster.k8s-cluster.status.control_plane.endpoints  # K8s API endpoint
  }
}

# Grafana admin password for monitoring dashboard access
# Retrieve with: terraform output -raw grafana_password
output "grafana_password" {
  sensitive = true  # Hidden from console output
  value     = module.o11y.grafana_password
}