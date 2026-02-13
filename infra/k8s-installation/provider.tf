# Terraform provider configuration
# Defines required providers and their authentication settings
terraform {
  required_providers {
    nebius = {
      source = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"  # Nebius cloud provider
    }
    kubernetes = {
      source = "hashicorp/kubernetes"  # K8s resource management
    }

    helm = {
      source  = "hashicorp/helm"       # Helm chart deployments
      version = "<3.0.0"
    }
  }
}

# Nebius cloud provider for infrastructure resources
provider "nebius" {
  domain = "api.eu.nebius.cloud:443"  # Nebius API endpoint (EU region)
}

# Helm provider for deploying charts (KubeRay, monitoring, operators)
# Authenticates to K8s cluster using IAM token
provider "helm" {
  kubernetes {
    host                   = nebius_mk8s_v1_cluster.k8s-cluster.status.control_plane.endpoints.public_endpoint
    cluster_ca_certificate = nebius_mk8s_v1_cluster.k8s-cluster.status.control_plane.auth.cluster_ca_certificate
    token                  = var.iam_token  # IAM token from environment.sh
  }
}

# Kubernetes provider for managing K8s resources (namespaces, secrets, configmaps)
# Authenticates to K8s cluster using IAM token
provider "kubernetes" {
  host                   = nebius_mk8s_v1_cluster.k8s-cluster.status.control_plane.endpoints.public_endpoint
  cluster_ca_certificate = nebius_mk8s_v1_cluster.k8s-cluster.status.control_plane.auth.cluster_ca_certificate
  token                  = var.iam_token  # IAM token from environment.sh
}
