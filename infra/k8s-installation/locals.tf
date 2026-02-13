locals {
  # Random suffix for unique resource naming (tied to project ID)
  release-suffix = random_string.random.result
  
  # SSH key resolution: use inline key if provided, otherwise read from file path
  ssh_public_key = var.ssh_public_key.key != null ? var.ssh_public_key.key : (
  fileexists(var.ssh_public_key.path) ? file(var.ssh_public_key.path) : null)

  # Default configurations per Nebius region
  # Each region has optimal CPU/GPU platform, preset, and InfiniBand fabric
  regions_default = {
    eu-west1 = {
      cpu_nodes_platform = "cpu-d3"
      cpu_nodes_preset   = "16vcpu-64gb"
      gpu_nodes_platform = "gpu-h200-sxm"         # H200 80GB SXM5
      gpu_nodes_preset   = "8gpu-128vcpu-1600gb"
      infiniband_fabric  = "fabric-5"             # Region-specific IB fabric
    }
    eu-north1 = {
      cpu_nodes_platform = "cpu-d3"
      cpu_nodes_preset   = "16vcpu-64gb"
      gpu_nodes_platform = "gpu-h100-sxm"         # H100 80GB SXM5
      gpu_nodes_preset   = "8gpu-128vcpu-1600gb"
      infiniband_fabric  = "fabric-3"             # Region-specific IB fabric
    }
    eu-north2 = {
      cpu_nodes_platform = "cpu-d3"
      cpu_nodes_preset   = "16vcpu-64gb"
      gpu_nodes_platform = "gpu-h200-sxm"
      gpu_nodes_preset   = "8gpu-128vcpu-1600gb"
      infiniband_fabric  = "eu-north2-a"
    }
    us-central1 = {
      cpu_nodes_platform = "cpu-d3"
      cpu_nodes_preset   = "16vcpu-64gb"
      gpu_nodes_platform = "gpu-h200-sxm"
      gpu_nodes_preset   = "8gpu-128vcpu-1600gb"
      infiniband_fabric  = "us-central1-a"
    }
    me-west1 = {
      cpu_nodes_platform = "cpu-d3"
      cpu_nodes_preset   = "16vcpu-64gb"
      gpu_nodes_platform = "gpu-b200-sxm-a"       # B200 (Blackwell architecture)
      gpu_nodes_preset   = "8gpu-160vcpu-1792gb"  # Higher memory for B200
      infiniband_fabric  = "ramon"                # Specific fabric name
    }
  }

  # Resolve current region's defaults
  current_region_defaults = local.regions_default[var.region]

  # Use user-provided values, or fall back to region defaults
  cpu_nodes_preset   = coalesce(var.cpu_nodes_preset, local.current_region_defaults.cpu_nodes_preset)
  cpu_nodes_platform = coalesce(var.cpu_nodes_platform, local.current_region_defaults.cpu_nodes_platform)
  gpu_nodes_platform = coalesce(var.gpu_nodes_platform, local.current_region_defaults.gpu_nodes_platform)
  gpu_nodes_preset   = coalesce(var.gpu_nodes_preset, local.current_region_defaults.gpu_nodes_preset)
  infiniband_fabric  = coalesce(var.infiniband_fabric, local.current_region_defaults.infiniband_fabric)

  # Map GPU platform to CUDA version (for specialized platforms)
  platform_to_cuda = {
    gpu-b200-sxm-a = "cuda12.8"  # B200 requires CUDA 12.8+
  }
  device_preset = lookup(local.platform_to_cuda, local.gpu_nodes_platform, "cuda12")  # Default to cuda12

  # Valid MIG (Multi-Instance GPU) partitioning configs per platform
  # MIG allows slicing a single GPU into multiple isolated instances
  valid_mig_parted_configs = {
    "gpu-h100-sxm" = ["all-disabled", "all-enabled", "all-balanced", "all-1g.10gb", "all-1g.10gb.me", "all-1g.20gb", "all-2g.20gb", "all-3g.40gb", "all-4g.40gb", "all-7g.80gb"]
    "gpu-h200-sxm" = ["all-disabled", "all-enabled", "all-balanced", "all-1g.18gb", "all-1g.18gb.me", "all-1g.35gb", "all-2g.35gb", "all-3g.71gb", "all-4g.71gb", "all-7g.141gb"]
    "gpu-b200-sxm" = ["all-disabled", "all-enabled", "all-balanced", "all-1g.23gb", "all-1g.23gb.me", "all-1g.45gb", "all-2g.45gb", "all-3g.90gb", "all-4g.90gb", "all-7g.180gb"]
    "gpu-b200-sxm-a" = ["all-disabled", "all-enabled", "all-balanced", "all-1g.23gb", "all-1g.23gb.me", "all-1g.45gb", "all-2g.45gb", "all-3g.90gb", "all-4g.90gb", "all-7g.180gb"]
  }

  # Hardware profile mapping for Nebius GPU Health Checker
  # Maps platform+preset to hardware profile string used by monitoring
  platform_preset_to_hardware_profile = {
    # H100 configurations
    "gpu-h100-sxm-1gpu-16vcpu-200gb"   = "1xH100"
    "gpu-h100-sxm-8gpu-128vcpu-1600gb" = "8xH100"

    # H200 configurations
    "gpu-h200-sxm-1gpu-16vcpu-200gb"   = "1xH200"
    "gpu-h200-sxm-8gpu-128vcpu-1600gb" = "8xH200"

    # B200 configurations
    "gpu-b200-sxm-1gpu-20vcpu-224gb"   = "1xB200"
    "gpu-b200-sxm-8gpu-160vcpu-1792gb" = "8xB200"
    "gpu-b200-sxm-a-8gpu-160vcpu-1792gb" = "8xB200"

    # L40 configurations
    # TODO add support for L40s
  }

  # Create key for hardware profile lookup (combines platform + preset)
  hardware_profile_key = "${local.gpu_nodes_platform}-${local.gpu_nodes_preset}"
}

# Random string generator for unique resource names
resource "random_string" "random" {
  keepers = {
    ami_id = "${var.parent_id}"  # Regenerate if project changes
  }
  length  = 6
  upper   = true
  lower   = true
  numeric = true
  special = false
}
