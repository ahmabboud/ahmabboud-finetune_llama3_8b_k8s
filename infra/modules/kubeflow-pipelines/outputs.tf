output "namespace" {
  description = "Kubeflow Pipelines namespace"
  value       = kubernetes_namespace.kubeflow.metadata[0].name
}

output "ui_service_name" {
  description = "KFP UI service name"
  value       = kubernetes_service.kfp_ui.metadata[0].name
}

output "pipeline_runner_sa" {
  description = "Pipeline runner service account"
  value       = kubernetes_service_account.pipeline_runner.metadata[0].name
}

output "access_instructions" {
  description = "Instructions to access Kubeflow Pipelines UI"
  value       = <<-EOT
    Access Kubeflow Pipelines UI:
      kubectl port-forward -n ${kubernetes_namespace.kubeflow.metadata[0].name} svc/ml-pipeline-ui 8080:80
      Then open: http://localhost:8080

    Install KFP SDK:
      pip install kfp==2.5.0
  EOT
}
