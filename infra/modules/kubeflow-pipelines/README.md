# Kubeflow Pipelines Module

This Terraform module deploys **Kubeflow Pipelines (KFP)** standalone on Kubernetes.

## What is Kubeflow Pipelines?

Kubeflow Pipelines is a platform for building and deploying portable, scalable machine learning workflows based on Docker containers.

## Components Installed

1. **KFP API Server** - REST API for pipelines
2. **MySQL** - Metadata storage
3. **MinIO** - Artifact storage (models, data, logs)
4. **Argo Workflows** - Pipeline execution engine
5. **UI** - Web interface for managing pipelines
6. **Persistence Agent** - Tracks workflow execution
7. **Scheduled Workflow** - Cron-based pipeline scheduling

## Usage

### In your Terraform configuration:

```hcl
module "kubeflow_pipelines" {
  source = "./modules/kubeflow-pipelines"

  namespace          = "kubeflow"
  kfp_version       = "2.0.5"
  storage_class     = "standard"

  # Credentials (use secrets in production)
  mysql_root_password = var.mysql_root_password
  mysql_password      = var.mysql_password
  minio_access_key    = var.minio_access_key
  minio_secret_key    = var.minio_secret_key
}
```

### Apply Terraform:

```bash
terraform init
terraform plan
terraform apply
```

### Access the UI:

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
# Open http://localhost:8080
```

## Install KFP SDK

```bash
pip install kfp==2.5.0
```

## Example: Simple Pipeline

```python
from kfp import dsl
from kfp import compiler
from kfp.client import Client

@dsl.component(base_image='python:3.10')
def say_hello(name: str) -> str:
    message = f"Hello, {name}!"
    print(message)
    return message

@dsl.pipeline(name='hello-pipeline')
def hello_pipeline(name: str = 'World'):
    hello_task = say_hello(name=name)

# Compile pipeline
compiler.Compiler().compile(hello_pipeline, 'hello_pipeline.yaml')

# Submit to KFP
client = Client(host='http://localhost:8080')
run = client.create_run_from_pipeline_func(
    hello_pipeline,
    arguments={'name': 'Kubeflow'}
)
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Kubeflow Pipelines (kubeflow ns)            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │   UI     │  │ API Server│  │  MySQL   │              │
│  │ (Port 80)│◄─┤  (8888)  │◄─┤(Metadata)│              │
│  └──────────┘  └──────────┘  └──────────┘              │
│                      │                                   │
│                      ▼                                   │
│              ┌──────────────┐                           │
│              │     Argo     │                           │
│              │  Workflows   │                           │
│              └──────────────┘                           │
│                      │                                   │
│                      ▼                                   │
│              ┌──────────────┐                           │
│              │    MinIO     │                           │
│              │  (Artifacts) │                           │
│              └──────────────┘                           │
└──────────────────────────────────────────────────────────┘
```

## Storage

- **MySQL**: Stores pipeline metadata, runs, experiments
- **MinIO**: Stores artifacts (input/output data, models, logs)

Both use persistent volumes with the configured storage class.

## Security Notes

**For Production:**
- Change default passwords
- Use Kubernetes secrets instead of plain text
- Restrict ServiceAccount permissions (currently using cluster-admin)
- Enable TLS for UI and API
- Configure authentication (Dex, OIDC)

## Troubleshooting

### Check pod status:
```bash
kubectl get pods -n kubeflow
```

### View API server logs:
```bash
kubectl logs -n kubeflow -l app=ml-pipeline
```

### Check storage:
```bash
kubectl get pvc -n kubeflow
```

### Port-forward not working?
```bash
# Try the API server directly
kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888
```

## Integration with Your Training

Once installed, you can convert your manual training workflow into an automated pipeline:

```python
@dsl.pipeline(name='llama3-training-pipeline')
def llama3_pipeline():
    data_prep = data_prep_component()
    train = training_component().after(data_prep)
    eval = evaluation_component().after(train)
```

See the main documentation for full examples.

## Resources

- [Kubeflow Pipelines Docs](https://www.kubeflow.org/docs/components/pipelines/)
- [KFP SDK Reference](https://kubeflow-pipelines.readthedocs.io/)
- [Argo Workflows](https://argoproj.github.io/argo-workflows/)
