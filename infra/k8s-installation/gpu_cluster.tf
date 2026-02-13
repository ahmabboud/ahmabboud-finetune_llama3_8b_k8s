# InfiniBand GPU cluster fabric for high-speed inter-GPU communication
# Provides: 400 Gb/s NDR InfiniBand between GPU nodes, NCCL optimization
# Required for: multi-node distributed training with minimal latency
resource "nebius_compute_v1_gpu_cluster" "fabric_2" {
  count = var.enable_gpu_cluster ? 1 : 0

  infiniband_fabric = local.infiniband_fabric  # Region-specific fabric (e.g., fabric-2, fabric-3)
  parent_id         = var.parent_id
  name              = join("-", [local.infiniband_fabric, local.release-suffix])
}
