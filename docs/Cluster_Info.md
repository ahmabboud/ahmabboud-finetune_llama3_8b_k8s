# K8s Cluster Information

## Retrieve Cluster Details

```bash
# Get cluster info
nebius mk8s v1 cluster list --parent-id $NEBIUS_PROJECT_ID --format yaml

# Get cluster ID
CLUSTER_ID=$(nebius mk8s v1 cluster list --parent-id $NEBIUS_PROJECT_ID --format json | jq -r '.items[0].metadata.id')

# Get cluster name
CLUSTER_NAME=$(nebius mk8s v1 cluster list --parent-id $NEBIUS_PROJECT_ID --format json | jq -r '.items[0].metadata.name')

# Get endpoints
nebius mk8s v1 cluster get --id $CLUSTER_ID --format yaml | grep -A2 "endpoints"
```

## Nebius Project

Set these in `infra/k8s-installation/environment.sh`:

```bash
NEBIUS_TENANT_ID='<your-tenant-id>'    # Get from: nebius iam tenant list
NEBIUS_PROJECT_ID='<your-project-id>'  # Get from: nebius iam project list
NEBIUS_REGION='eu-north1'
```

## Node Groups

### CPU Nodes
| Property | Value |
|----------|-------|
| Count | 2 |
| Platform | `cpu-d3` |
| Preset | `4vcpu-16gb` |

### GPU Nodes
| Property | Value |
|----------|-------|
| Count | 2 |
| Platform | `gpu-h100-sxm` |
| Preset | `8gpu-128vcpu-1600gb` |
| GPUs per Node | 8 |
| **Total GPUs** | **16 x H100** |
| InfiniBand Fabric | Configured in `terraform.tfvars` |

## Storage

| Property | Value |
|----------|-------|
| Shared Filestore | 2TB |
| Mount Path | `/mnt/data` (on all nodes) |
| Type | NETWORK_SSD |

## Grafana Access

### Get Credentials

```bash
cd infra/k8s-installation

# Username is always 'admin'
# Get password:
terraform output -raw grafana_password
```

### Connect to Grafana

```bash
kubectl -n o11y port-forward svc/grafana-and-prometheus 8080:80
```

Then open: http://localhost:8080

## Quick Commands

### Get kubectl credentials
```bash
# First, get your cluster ID
CLUSTER_ID=$(nebius mk8s v1 cluster list --parent-id $NEBIUS_PROJECT_ID --format json | jq -r '.items[0].metadata.id')

# Then get credentials
nebius mk8s v1 cluster get-credentials --id $CLUSTER_ID --external
```

### Check nodes
```bash
kubectl get nodes -o wide
```

### Check GPUs
```bash
kubectl describe nodes | grep -A5 "nvidia.com/gpu"
```

### Check all pods
```bash
kubectl get pods -A
```

### SSH to nodes
```bash
# Get node external IPs
kubectl get nodes -o wide

# SSH using your configured key
ssh ubuntu@<node-external-ip>
```

## Terraform Management

### Re-initialize environment (if session expired)
```bash
cd infra/k8s-installation
source ./environment.sh
```

### Check state
```bash
terraform state list
```

### Get all outputs
```bash
terraform output
```

### Destroy cluster
```bash
terraform destroy
```

---

*Note: Store deployment-specific values in `.env` (gitignored) after deployment.*
