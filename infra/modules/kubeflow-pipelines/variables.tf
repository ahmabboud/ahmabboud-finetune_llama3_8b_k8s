variable "namespace" {
  description = "Kubernetes namespace for Kubeflow Pipelines"
  type        = string
  default     = "kubeflow"
}

variable "kfp_version" {
  description = "Kubeflow Pipelines version"
  type        = string
  default     = "2.0.5"
}

variable "storage_class" {
  description = "Storage class for persistent volumes"
  type        = string
  default     = "standard"
}

variable "mysql_root_password" {
  description = "MySQL root password"
  type        = string
  sensitive   = true
  default     = "rootpassword"
}

variable "mysql_password" {
  description = "MySQL user password"
  type        = string
  sensitive   = true
  default     = "password"
}

variable "mysql_storage_size" {
  description = "MySQL persistent storage size"
  type        = string
  default     = "20Gi"
}

variable "minio_access_key" {
  description = "MinIO access key"
  type        = string
  sensitive   = true
  default     = "minio"
}

variable "minio_secret_key" {
  description = "MinIO secret key"
  type        = string
  sensitive   = true
  default     = "minio123"
}

variable "minio_storage_size" {
  description = "MinIO persistent storage size"
  type        = string
  default     = "20Gi"
}

variable "pipeline_runner_sa" {
  description = "Service account name for pipeline execution"
  type        = string
  default     = "pipeline-runner"
}
