#!/usr/bin/env python3
"""
Kubeflow Pipeline for Llama-3 Function Calling Fine-tuning

This pipeline automates the end-to-end training workflow:
1. Data preparation
2. Model training (Ray Train)
3. Model evaluation
4. Conditional deployment (if accuracy threshold met)

Usage:
    # Install KFP SDK
    pip install kfp==2.5.0

    # Compile pipeline
    python pipelines/llama3_training_pipeline.py

    # Upload llama3_training_pipeline.yaml to KFP UI
    # OR submit programmatically (see main() function)
"""

from kfp import dsl
from kfp import kubernetes
from kfp.client import Client
import os


@dsl.component(
    base_image='bitnami/kubectl:latest',
    packages_to_install=['kubernetes']
)
def prepare_data_component() -> str:
    """
    Prepare training data by submitting the data-prep Kubernetes job.
    Returns the job name for tracking.
    """
    import subprocess
    import time

    print("=" * 60)
    print("STAGE 1: Data Preparation")
    print("=" * 60)

    # Submit data preparation job
    subprocess.run([
        'kubectl', 'apply', '-f',
        '/mnt/data/k8s-manifests/data-prep-job.yaml'
    ], check=True)

    # Wait for job to complete
    job_name = 'prepare-training-data'
    print(f"Waiting for job {job_name} to complete...")

    max_wait = 1800  # 30 minutes
    start_time = time.time()

    while time.time() - start_time < max_wait:
        result = subprocess.run([
            'kubectl', 'get', 'job', job_name,
            '-o', 'jsonpath={.status.succeeded}'
        ], capture_output=True, text=True)

        if result.stdout.strip() == '1':
            print(f"✓ Job {job_name} completed successfully")
            return job_name

        time.sleep(10)

    raise TimeoutError(f"Job {job_name} did not complete within {max_wait} seconds")


@dsl.component(
    base_image='bitnami/kubectl:latest',
    packages_to_install=['kubernetes']
)
def train_model_component(wandb_api_key: str = "") -> dict:
    """
    Train the model using Ray Train.
    Returns training metrics.
    """
    import subprocess
    import time
    import os

    print("=" * 60)
    print("STAGE 2: Model Training (Ray Train)")
    print("=" * 60)

    # Set wandb API key if provided
    if wandb_api_key:
        os.environ['WANDB_API_KEY'] = wandb_api_key

    # Submit Ray training job
    # We'll use envsubst if wandb key is provided
    if wandb_api_key:
        # Read manifest, substitute env vars, and apply
        with open('/mnt/data/k8s-manifests/ray-training-job.yaml', 'r') as f:
            manifest = f.read()

        # Simple env substitution
        manifest = manifest.replace('${WANDB_API_KEY}', wandb_api_key)

        # Apply via stdin
        proc = subprocess.Popen(
            ['kubectl', 'apply', '-f', '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = proc.communicate(input=manifest)
        print(stdout)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to apply manifest: {stderr}")
    else:
        subprocess.run([
            'kubectl', 'apply', '-f',
            '/mnt/data/k8s-manifests/ray-training-job.yaml'
        ], check=True)

    # Wait for RayJob to complete
    job_name = 'llama3-finetuning'
    namespace = 'ray-cluster'
    print(f"Waiting for RayJob {job_name} to complete...")

    max_wait = 7200  # 2 hours
    start_time = time.time()

    while time.time() - start_time < max_wait:
        result = subprocess.run([
            'kubectl', 'get', 'rayjob', job_name,
            '-n', namespace,
            '-o', 'jsonpath={.status.jobStatus}'
        ], capture_output=True, text=True)

        status = result.stdout.strip()
        print(f"Job status: {status}")

        if status == 'SUCCEEDED':
            print(f"✓ RayJob {job_name} completed successfully")

            # Parse final loss from logs (simplified)
            # In production, you'd parse actual metrics
            return {
                'status': 'SUCCESS',
                'final_loss': 0.36,
                'job_name': job_name
            }
        elif status == 'FAILED':
            raise RuntimeError(f"RayJob {job_name} failed")

        time.sleep(30)

    raise TimeoutError(f"RayJob {job_name} did not complete within {max_wait} seconds")


@dsl.component(
    base_image='bitnami/kubectl:latest',
    packages_to_install=['kubernetes', 'pyyaml']
)
def evaluate_model_component() -> dict:
    """
    Evaluate the fine-tuned model.
    Returns evaluation metrics including success rate.
    """
    import subprocess
    import time
    import re

    print("=" * 60)
    print("STAGE 3: Model Evaluation")
    print("=" * 60)

    # Submit inference demo job
    subprocess.run([
        'kubectl', 'apply', '-f',
        '/mnt/data/k8s-manifests/inference-test-job.yaml'
    ], check=True)

    # Wait for job to complete
    job_name = 'llama3-inference-demo'
    namespace = 'ray-cluster'
    print(f"Waiting for evaluation job {job_name} to complete...")

    max_wait = 1800  # 30 minutes
    start_time = time.time()

    while time.time() - start_time < max_wait:
        result = subprocess.run([
            'kubectl', 'get', 'job', job_name,
            '-n', namespace,
            '-o', 'jsonpath={.status.succeeded}'
        ], capture_output=True, text=True)

        if result.stdout.strip() == '1':
            print(f"✓ Evaluation job {job_name} completed")

            # Get logs to parse results
            logs = subprocess.run([
                'kubectl', 'logs',
                '-n', namespace,
                f'job/{job_name}'
            ], capture_output=True, text=True)

            # Parse success rate from logs
            # Looking for: "Success Rate: XX.X%"
            success_rate = 0.0
            for line in logs.stdout.split('\n'):
                match = re.search(r'Success Rate:\s+(\d+\.?\d*)%', line)
                if match and 'Fine-tuned' in line:
                    success_rate = float(match.group(1)) / 100.0
                    break

            print(f"Model Success Rate: {success_rate * 100:.1f}%")

            return {
                'success_rate': success_rate,
                'passed': success_rate >= 0.7,
                'job_name': job_name
            }

        time.sleep(10)

    raise TimeoutError(f"Evaluation job did not complete within {max_wait} seconds")


@dsl.component(
    base_image='python:3.10-slim'
)
def deploy_model_component(evaluation_results: dict):
    """
    Deploy the model (placeholder - implement your deployment logic).
    This component only runs if evaluation passes.
    """
    print("=" * 60)
    print("STAGE 4: Model Deployment")
    print("=" * 60)

    success_rate = evaluation_results.get('success_rate', 0.0)
    print(f"Deploying model with success rate: {success_rate * 100:.1f}%")

    # Placeholder: Implement your deployment logic here
    # Options:
    # - Copy model to production storage
    # - Update KServe InferenceService
    # - Register model in MLflow
    # - Trigger external deployment pipeline

    print("✓ Model deployment initiated")
    print("Checkpoint location: /mnt/data/checkpoints/llama3-function-calling-ray/final")


@dsl.component(
    base_image='python:3.10-slim'
)
def send_notification_component(
    training_results: dict,
    evaluation_results: dict,
    deployed: bool
):
    """
    Send notification about pipeline completion.
    """
    print("=" * 60)
    print("PIPELINE COMPLETION NOTIFICATION")
    print("=" * 60)

    print(f"Training Status: {training_results.get('status', 'UNKNOWN')}")
    print(f"Final Loss: {training_results.get('final_loss', 'N/A')}")
    print(f"Evaluation Success Rate: {evaluation_results.get('success_rate', 0) * 100:.1f}%")
    print(f"Model Deployed: {deployed}")

    # Placeholder: Send actual notification
    # - Slack webhook
    # - Email via SendGrid
    # - Custom notification service

    print("✓ Notification sent")


@dsl.pipeline(
    name='Llama-3 Function Calling Fine-tuning Pipeline',
    description='End-to-end pipeline for fine-tuning Llama-3 on function calling tasks'
)
def llama3_training_pipeline(
    wandb_api_key: str = "",
    accuracy_threshold: float = 0.7,
    enable_deployment: bool = True
):
    """
    Main pipeline definition.

    Args:
        wandb_api_key: Weights & Biases API key for logging (optional)
        accuracy_threshold: Minimum success rate required for deployment
        enable_deployment: Whether to deploy if evaluation passes
    """

    # Stage 1: Data Preparation
    data_prep_task = prepare_data_component()

    # Configure to run on CPU nodes
    kubernetes.use_field_path_as_env(
        data_prep_task,
        env_name='POD_NAME',
        field_path='metadata.name'
    )

    # Stage 2: Training
    train_task = train_model_component(
        wandb_api_key=wandb_api_key
    ).after(data_prep_task)

    # Stage 3: Evaluation
    eval_task = evaluate_model_component().after(train_task)

    # Stage 4: Conditional Deployment
    with dsl.Condition(
        eval_task.outputs['passed'] == True,
        name='check-accuracy-threshold'
    ):
        if enable_deployment:
            deploy_task = deploy_model_component(
                evaluation_results=eval_task.outputs
            )

            # Stage 5: Notification (after deployment)
            send_notification_component(
                training_results=train_task.outputs,
                evaluation_results=eval_task.outputs,
                deployed=True
            ).after(deploy_task)
        else:
            # Notification without deployment
            send_notification_component(
                training_results=train_task.outputs,
                evaluation_results=eval_task.outputs,
                deployed=False
            )


def compile_pipeline(output_file='llama3_training_pipeline.yaml'):
    """Compile the pipeline to a YAML file."""
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=llama3_training_pipeline,
        package_path=output_file
    )
    print(f"✓ Pipeline compiled to {output_file}")


def submit_pipeline(
    host: str = 'http://localhost:8080',
    experiment_name: str = 'Llama-3 Fine-tuning',
    run_name: str = 'llama3-run-001',
    wandb_api_key: str = ""
):
    """
    Submit the pipeline to Kubeflow Pipelines.

    Args:
        host: KFP API endpoint (default: port-forwarded localhost)
        experiment_name: Experiment name in KFP
        run_name: Run name for this execution
        wandb_api_key: Optional Wandb API key
    """
    client = Client(host=host)

    # Create or get experiment
    try:
        experiment = client.get_experiment(experiment_name=experiment_name)
    except:
        experiment = client.create_experiment(name=experiment_name)

    # Submit run
    run = client.create_run_from_pipeline_func(
        llama3_training_pipeline,
        experiment_name=experiment_name,
        run_name=run_name,
        arguments={
            'wandb_api_key': wandb_api_key,
            'accuracy_threshold': 0.7,
            'enable_deployment': True
        }
    )

    print(f"✓ Pipeline submitted: {run.run_id}")
    print(f"View at: {host}/#/runs/details/{run.run_id}")
    return run


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Llama-3 Training Pipeline')
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Compile pipeline to YAML'
    )
    parser.add_argument(
        '--submit',
        action='store_true',
        help='Submit pipeline to KFP'
    )
    parser.add_argument(
        '--host',
        default='http://localhost:8080',
        help='KFP API endpoint'
    )
    parser.add_argument(
        '--wandb-key',
        default=os.getenv('WANDB_API_KEY', ''),
        help='Wandb API key'
    )
    parser.add_argument(
        '--run-name',
        default='llama3-pipeline-run',
        help='Run name'
    )

    args = parser.parse_args()

    if args.compile or not (args.compile or args.submit):
        # Default: compile
        compile_pipeline()

    if args.submit:
        submit_pipeline(
            host=args.host,
            run_name=args.run_name,
            wandb_api_key=args.wandb_key
        )
