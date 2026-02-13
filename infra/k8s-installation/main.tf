# Main Kubernetes cluster definition
# Managed K8s control plane with etcd, API server, scheduler, controller manager
resource "nebius_mk8s_v1_cluster" "k8s-cluster" {
  parent_id = var.parent_id
  name      = join("-", ["k8s-training", local.release-suffix])
  control_plane = {
    endpoints = {
      public_endpoint = {}  # Expose K8s API publicly for kubectl access
    }
    etcd_cluster_size = var.etcd_cluster_size  # Number of etcd replicas (3 = HA)
    subnet_id         = var.subnet_id
    version           = var.k8s_version        # K8s version (null = use backend default)
  }
}

# Cilium Egress Gateway for controlled external traffic routing
# Optional: only deployed if enable_egress_gateway = true
module "cilium-egress-gateway" {
  count  = var.enable_egress_gateway ? 1 : 0
  source = "../modules/cilium-egress-gateway"

  mk8s_cluster_id = resource.nebius_mk8s_v1_cluster.k8s-cluster.id
  mk8s_version    = var.k8s_version
  project_id      = var.parent_id
  ssh_user_name   = var.ssh_user_name
  ssh_public_key  = local.ssh_public_key
  subnet_id       = var.subnet_id

  depends_on = [
    resource.nebius_mk8s_v1_node_group.cpu-only
  ]
}

# Service Account for K8s node groups to access cloud resources
# Grants nodes permissions to access Nebius services (storage, logs, etc.)
data "nebius_iam_v1_group" "editors" {
  count     = var.enable_k8s_node_group_sa ? 1 : 0
  name      = "editors"       # IAM group with editor permissions
  parent_id = var.tenant_id
}

resource "nebius_iam_v1_service_account" "k8s_node_group_sa" {
  count     = var.enable_k8s_node_group_sa ? 1 : 0
  parent_id = var.parent_id
  name      = join("-", ["k8s_node_group_sa", local.release-suffix])
}

# Add service account to editors group for cloud resource access
resource "nebius_iam_v1_group_membership" "k8s_node_group_sa-admin" {
  count     = var.enable_k8s_node_group_sa ? 1 : 0
  parent_id = data.nebius_iam_v1_group.editors[0].id
  member_id = nebius_iam_v1_service_account.k8s_node_group_sa[count.index].id
}
################
# CPU NODE GROUP
################
# CPU-only nodes for control plane workloads, monitoring, KubeRay head
resource "nebius_mk8s_v1_node_group" "cpu-only" {
  fixed_node_count = var.cpu_nodes_count  # Fixed count (no autoscaling for CPU nodes)
  parent_id        = nebius_mk8s_v1_cluster.k8s-cluster.id
  name             = join("-", ["k8s-ng-cpu", local.release-suffix])
  labels = {
    "library-solution" : "k8s-training",
  }
  version = var.k8s_version
  template = {
    boot_disk = {
      size_gibibytes = var.cpu_disk_size  # OS and local storage
      type           = var.cpu_disk_type   # NETWORK_SSD default
    }

    service_account_id = var.enable_k8s_node_group_sa ? nebius_iam_v1_service_account.k8s_node_group_sa[0].id : null

    network_interfaces = [
      {
        public_ip_address = {}  # Assign public IP for external access
        subnet_id         = var.subnet_id
      }
    ]
    resources = {
      platform = local.cpu_nodes_platform  # cpu-d3 default
      preset   = local.cpu_nodes_preset    # 4vcpu-16gb or higher
    }
    preemptible = var.cpu_nodes_preemptible ? {
      on_preemption = "STOP"   # Stop (not delete) when preempted
      priority      = 1
    } : null
    filesystems = var.enable_filestore ? [
      {
        attach_mode         = "READ_WRITE"  # Mount NFS filesystem
        mount_tag           = "data"        # Mount at /mnt/data
        existing_filesystem = nebius_compute_v1_filesystem.shared-filesystem[0]
      }
    ] : null
    underlay_required = false  # Don't require direct network access (overlay OK)
    preemptible = var.cpu_nodes_preemptible ? {
      on_preemption = "STOP"
      priority      = 3
    } : null
    cloud_init_user_data = templatefile("${path.module}/../modules/cloud-init/k8s-cloud-init.tftpl", {
      enable_filestore = var.enable_filestore ? "true" : "false",
      ssh_user_name    = var.ssh_user_name,
      ssh_public_key   = local.ssh_public_key
    })
  }
}
#################
# GPU NODE GROUPS
#################
# GPU nodes for distributed training (H100/H200/B200 with InfiniBand)
resource "nebius_mk8s_v1_node_group" "gpu" {
  count            = var.gpu_node_groups  # Supports multiple groups for >100 nodes
  
  # Use either fixed count or autoscaling (mutually exclusive)
  fixed_node_count = var.gpu_autoscaling_enabled ? null : var.gpu_nodes_count_per_group
  
  # Autoscaling configuration (allows scale to zero for cost savings)
  autoscaling = var.gpu_autoscaling_enabled ? {
    min_node_count = var.gpu_min_nodes  # 0 = scale to zero when idle
    max_node_count = var.gpu_max_nodes  # Maximum nodes under load
  } : null
  
  parent_id        = nebius_mk8s_v1_cluster.k8s-cluster.id
  name             = join("-", ["k8s-ng-gpu", local.release-suffix, count.index])
  labels = {
    "library-solution" : "k8s-training",
  }
  version = var.k8s_version
  template = {
    metadata = {
      labels = var.mig_parted_config != null ? {
        "nvidia.com/mig.config" = var.mig_parted_config  # MIG partitioning config
      } : {}
    }

    boot_disk = {
      size_gibibytes = var.gpu_disk_size  # 1TB default for GPU nodes
      type           = var.gpu_disk_type   # NETWORK_SSD recommended
    }

    service_account_id = var.enable_k8s_node_group_sa ? nebius_iam_v1_service_account.k8s_node_group_sa[0].id : null

    network_interfaces = [
      {
        subnet_id         = var.subnet_id
        public_ip_address = var.gpu_nodes_assign_public_ip ? {} : null  # Optional public IP
      }
    ]
    resources = {
      platform = local.gpu_nodes_platform  # gpu-h100-sxm, gpu-h200-sxm, etc.
      preset   = local.gpu_nodes_preset    # 8gpu-128vcpu-1600gb default
    }
    preemptible = var.gpu_nodes_preemptible ? {
      on_preemption = "STOP"  # Warning: preemptible not recommended for long training
      priority      = 1
    } : null
    filesystems = var.enable_filestore ? [
      {
        attach_mode         = "READ_WRITE"
        mount_tag           = "data"
        existing_filesystem = nebius_compute_v1_filesystem.shared-filesystem[0]
      }
    ] : null
    gpu_cluster  = var.enable_gpu_cluster ? nebius_compute_v1_gpu_cluster.fabric_2[0] : null  # InfiniBand fabric
    gpu_settings = var.gpu_nodes_driverfull_image ? { drivers_preset = local.device_preset } : null  # NVIDIA drivers
    preemptible = var.gpu_nodes_preemptible ? {
      on_preemption = "STOP"
      priority      = 3
    } : null

    taints = length(var.gpu_node_taints) > 0 ? var.gpu_node_taints : null  # Prevent non-GPU workloads

    underlay_required = false
    cloud_init_user_data = templatefile("${path.module}/../modules/cloud-init/k8s-cloud-init.tftpl", {
      enable_filestore = var.enable_filestore ? "true" : "false",
      ssh_user_name    = var.ssh_user_name,
      ssh_public_key   = local.ssh_public_key
    })
  }
}
