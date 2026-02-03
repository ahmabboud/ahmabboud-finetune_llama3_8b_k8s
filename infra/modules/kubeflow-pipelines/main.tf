###############################################################################
# Kubeflow Pipelines Module
#
# Deploys standalone Kubeflow Pipelines (KFP) for ML workflow orchestration.
# This is the lightweight version without the full Kubeflow stack.
#
# Architecture:
# - Kubeflow Pipelines API server
# - Metadata store (MySQL)
# - Artifact storage (MinIO)
# - Pipeline persistence
# - UI for pipeline visualization
###############################################################################

terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.0"
    }
  }
}

###############################################################################
# Namespace
###############################################################################

resource "kubernetes_namespace" "kubeflow" {
  metadata {
    name = var.namespace
    labels = {
      "app.kubernetes.io/name"    = "kubeflow-pipelines"
      "app.kubernetes.io/part-of" = "kubeflow"
    }
  }
}

###############################################################################
# Kubeflow Pipelines via Helm
###############################################################################

resource "helm_release" "kubeflow_pipelines" {
  name       = "kubeflow-pipelines"
  repository = "https://deployments.kubeflow.org/kfp-helm-release"
  chart      = "kubeflow-pipelines"
  namespace  = kubernetes_namespace.kubeflow.metadata[0].name
  version    = var.kfp_version

  # Create namespace if it doesn't exist
  create_namespace = false # Already created above

  # Wait for all resources to be ready
  wait          = true
  wait_for_jobs = true
  timeout       = 600 # 10 minutes

  # Basic configuration
  values = [
    yamlencode({
      # MySQL configuration (metadata store)
      mysql = {
        enabled = true
        mysqlRootPassword = var.mysql_root_password
        mysqlPassword     = var.mysql_password
        persistence = {
          enabled      = true
          storageClass = var.storage_class
          size         = var.mysql_storage_size
        }
      }

      # MinIO configuration (artifact storage)
      minio = {
        enabled = true
        accessKey = var.minio_access_key
        secretKey = var.minio_secret_key
        persistence = {
          enabled      = true
          storageClass = var.storage_class
          size         = var.minio_storage_size
        }
      }

      # Pipeline API server
      apiServer = {
        image = {
          repository = "gcr.io/ml-pipeline/api-server"
          tag        = var.kfp_version
        }
        resources = {
          requests = {
            cpu    = "250m"
            memory = "512Mi"
          }
          limits = {
            cpu    = "1000m"
            memory = "2Gi"
          }
        }
      }

      # Pipeline persistence agent
      persistenceAgent = {
        enabled = true
        image = {
          repository = "gcr.io/ml-pipeline/persistenceagent"
          tag        = var.kfp_version
        }
      }

      # Pipeline scheduled workflow controller
      scheduledWorkflow = {
        enabled = true
        image = {
          repository = "gcr.io/ml-pipeline/scheduledworkflow"
          tag        = var.kfp_version
        }
      }

      # UI
      ui = {
        enabled = true
        image = {
          repository = "gcr.io/ml-pipeline/frontend"
          tag        = var.kfp_version
        }
        serviceType = "ClusterIP"
      }

      # Viewer CRD controller
      viewerCrdController = {
        enabled = true
        image = {
          repository = "gcr.io/ml-pipeline/viewer-crd-controller"
          tag        = var.kfp_version
        }
      }

      # Argo Workflow controller (backend for KFP)
      argo = {
        enabled = true
        controller = {
          image = {
            repository = "quay.io/argoproj/workflow-controller"
            tag        = "v3.4.11"
          }
          resources = {
            requests = {
              cpu    = "100m"
              memory = "256Mi"
            }
            limits = {
              cpu    = "500m"
              memory = "1Gi"
            }
          }
        }
      }
    })
  ]

  depends_on = [kubernetes_namespace.kubeflow]
}

###############################################################################
# Service for accessing KFP UI
###############################################################################

resource "kubernetes_service" "kfp_ui" {
  metadata {
    name      = "ml-pipeline-ui"
    namespace = kubernetes_namespace.kubeflow.metadata[0].name
    labels = {
      app = "ml-pipeline-ui"
    }
  }

  spec {
    type = "ClusterIP"

    selector = {
      app = "ml-pipeline-ui"
    }

    port {
      name        = "http"
      port        = 80
      target_port = 3000
      protocol    = "TCP"
    }
  }

  depends_on = [helm_release.kubeflow_pipelines]
}

###############################################################################
# ConfigMap for shared pipeline storage
###############################################################################

resource "kubernetes_config_map" "pipeline_config" {
  metadata {
    name      = "pipeline-install-config"
    namespace = kubernetes_namespace.kubeflow.metadata[0].name
  }

  data = {
    # Configure pipeline defaults
    "defaultPipelineRunnerServiceAccount" = var.pipeline_runner_sa
    "cacheEnabled"                        = "true"
    "autoUpdatePipelineDefault"           = "true"
  }

  depends_on = [kubernetes_namespace.kubeflow]
}

###############################################################################
# ServiceAccount for pipeline execution
###############################################################################

resource "kubernetes_service_account" "pipeline_runner" {
  metadata {
    name      = var.pipeline_runner_sa
    namespace = kubernetes_namespace.kubeflow.metadata[0].name
  }

  depends_on = [kubernetes_namespace.kubeflow]
}

resource "kubernetes_cluster_role_binding" "pipeline_runner" {
  metadata {
    name = "pipeline-runner-binding"
  }

  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "ClusterRole"
    name      = "cluster-admin" # Note: In production, use more restrictive permissions
  }

  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account.pipeline_runner.metadata[0].name
    namespace = kubernetes_namespace.kubeflow.metadata[0].name
  }
}

###############################################################################
# Output instructions
###############################################################################

resource "null_resource" "print_instructions" {
  provisioner "local-exec" {
    command = <<-EOT
      echo "=============================================="
      echo "Kubeflow Pipelines Installed Successfully!"
      echo "=============================================="
      echo ""
      echo "Access the UI with:"
      echo "  kubectl port-forward -n ${kubernetes_namespace.kubeflow.metadata[0].name} svc/ml-pipeline-ui 8080:80"
      echo "  Then open: http://localhost:8080"
      echo ""
      echo "Install KFP SDK:"
      echo "  pip install kfp==2.5.0"
      echo ""
      echo "Namespace: ${kubernetes_namespace.kubeflow.metadata[0].name}"
      echo "=============================================="
    EOT
  }

  depends_on = [
    helm_release.kubeflow_pipelines,
    kubernetes_service.kfp_ui
  ]
}
