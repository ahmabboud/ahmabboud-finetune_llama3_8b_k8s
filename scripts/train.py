#!/usr/bin/env python3
"""
Multi-node distributed training script for Llama-3 function calling fine-tuning.

Usage:
    # Single node (8 GPUs)
    torchrun --nproc_per_node=8 scripts/train.py --config configs/train_config.yaml
    
    # Multi-node (2 nodes, 16 GPUs total)
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        scripts/train.py --config configs/train_config.yaml
"""
import argparse
import os
from pathlib import Path

import torch
import yaml
import wandb
from transformers import AutoTokenizer

from src.data import FunctionCallingDataset
from src.data.dataset import load_local_dataset
from src.training import DistributedTrainer


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Llama-3 for function calling")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get distributed info
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    is_main_process = rank == 0
    
    if is_main_process:
        print("=" * 60)
        print("Llama-3 Function Calling Fine-tuning")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Config: {args.config}")
        print()
    
    # Initialize wandb on main process
    if is_main_process and config.get("wandb", {}).get("enabled", True):
        wandb.init(
            project=config["wandb"].get("project", "llama3-function-calling"),
            name=config["wandb"].get("run_name", "training-run"),
            config=config,
        )
    
    # Load tokenizer
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    data_config = config["data"]
    train_data = load_local_dataset(data_config["train_path"])
    val_data = load_local_dataset(data_config["val_path"])
    
    train_dataset = FunctionCallingDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=data_config.get("max_length", 2048),
    )
    
    val_dataset = FunctionCallingDataset(
        data=val_data,
        tokenizer=tokenizer,
        max_length=data_config.get("max_length", 2048),
    )
    
    if is_main_process:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    
    # Get loss configuration
    loss_config = config.get("loss", {})
    completion_only = loss_config.get("completion_only", True)
    
    if is_main_process:
        print(f"Completion-only loss: {completion_only}")
    
    # Initialize trainer
    trainer = DistributedTrainer(
        model_name=model_name,
        output_dir=config["training"]["output_dir"],
        use_lora=config["model"].get("use_lora", True),
        lora_config=config.get("lora", {}),
        use_flash_attention=config["model"].get("use_flash_attention", True),
        torch_dtype=config["model"].get("torch_dtype", "bfloat16"),
        completion_only_loss=completion_only,
    )
    trainer.setup()
    
    # Get training arguments
    training_args = trainer.get_training_args(
        **config["training"],
        deepspeed_config=config.get("deepspeed", {}).get("config_path"),
    )
    
    # Train
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        training_args=training_args,
        resume_from_checkpoint=args.resume,
    )
    
    # Merge and save final model (main process only)
    if is_main_process and config["model"].get("use_lora", True):
        merge_path = Path(config["training"]["output_dir"]) / "merged"
        trainer.merge_and_save(str(merge_path))
    
    if is_main_process:
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        wandb.finish()


if __name__ == "__main__":
    main()
