# Nebius Filestore (NFS) for shared storage across GPU/CPU nodes
# Used for: model cache, checkpoints, datasets, training artifacts
resource "nebius_compute_v1_filesystem" "shared-filesystem" {
  count            = var.enable_filestore ? 1 : 0
  parent_id        = var.parent_id
  name             = join("-", ["filesystem-tf", local.release-suffix])
  type             = var.filestore_disk_type     # NETWORK_SSD (default)
  size_bytes       = var.filestore_disk_size     # Total capacity in bytes
  block_size_bytes = var.filestore_block_size    # 4KB default, affects I/O for small files
}