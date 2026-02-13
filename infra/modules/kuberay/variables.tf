variable "parent_id" {
  description = "Nebius project ID (parent resource for KubeRay deployment)."
  type        = string
}

variable "cluster_id" {
  description = "Kubernetes cluster ID where KubeRay will be deployed."
  type        = string
}

variable "name" {
  description = "Application name for KubeRay deployment (used for Helm release)."
  type        = string
  default     = "ray-cluster"
}

variable "namespace" {
  description = "Kubernetes namespace for KubeRay resources (ray-cluster default)."
  type        = string
  default     = "ray-cluster"
}

variable "cpu_platform" {
  description = "CPU node platform for worker affinity (e.g., cpu-d3)."
  type        = string
}

variable "cpu_worker_image" {
  description = "Docker image for Ray CPU workers (default: official Ray image with Python 3.10)."
  type        = string
  default     = "rayproject/ray:2.46.0-py310"
  nullable    = false
}

variable "min_cpu_replicas" {
  description = "Minimum Ray CPU worker pods (for job submission, lightweight tasks)."
  type        = number
  default     = 0
  nullable    = false
}

variable "max_cpu_replicas" {
  description = "Maximum Ray CPU worker pods (autoscales between min and max)."
  type        = number
}

variable "cpu_resources" {
  description = "CPUs and memory per CPU worker pod (memory in GiB)."
  type = object({
    cpus   = number
    memory = number
  })
  default = {
    cpus   = 2
    memory = 4 # in GiB
  }
  nullable = false
}

variable "gpu_platform" {
  description = "GPU node platform for worker affinity (e.g., gpu-h100-sxm)."
  type        = string
}

variable "gpu_worker_image" {
  description = "Docker image for Ray GPU workers (must include GPU drivers, NCCL, InfiniBand support)."
  type        = string
  default     = "rayproject/ray:2.46.0-py310-gpu"
  nullable    = false
}

variable "min_gpu_replicas" {
  description = "Minimum Ray GPU worker pods (0 = scale to zero when idle for cost savings)."
  type        = number
  default     = 0
  nullable    = false
}

variable "max_gpu_replicas" {
  description = "Maximum Ray GPU worker pods (each pod typically maps to one GPU node with 8 GPUs)."
  type        = number
  default     = 1
}

variable "gpu_resources" {
  description = "GPUs, CPUs, and memory per GPU worker pod (memory in GiB). Typical: {gpus=8, cpus=120, memory=1400}."
  type = object({
    cpus   = number
    memory = number
    gpus   = number
  })
  default = {
    cpus   = 15
    memory = 150 # in GiB
    gpus   = 1
  }
  nullable = false
}
