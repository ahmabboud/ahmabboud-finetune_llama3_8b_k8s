"""
Distributed trainer for multi-node LLM fine-tuning.
Supports DeepSpeed ZeRO and PyTorch FSDP.
"""
import os
from typing import Optional, Dict, Any, Callable
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from peft import get_peft_model, prepare_model_for_kbit_training

from .lora_config import get_lora_config
from .data_collator import get_data_collator
from .evaluation import compute_perplexity


class DistributedTrainer:
    """
    Wrapper for distributed training with DeepSpeed/FSDP support.
    Optimized for multi-node GPU clusters.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        output_dir: str = "./output",
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        use_flash_attention: bool = True,
        torch_dtype: str = "bfloat16",
        completion_only_loss: bool = True,  # Only compute loss on assistant responses
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.use_lora = use_lora
        self.lora_config = lora_config or {}
        self.use_flash_attention = use_flash_attention
        self.torch_dtype = getattr(torch, torch_dtype)
        self.completion_only_loss = completion_only_loss
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def setup(self):
        """Initialize model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Model loading kwargs
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": True,
        }
        
        # Enable Flash Attention 2 if available
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        
        # Apply LoRA if enabled
        if self.use_lora:
            print("Applying LoRA configuration")
            lora_config = get_lora_config(**self.lora_config)
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        return self
    
    def get_training_args(
        self,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.03,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
        deepspeed_config: Optional[str] = None,
        **kwargs,
    ) -> TrainingArguments:
        """
        Create training arguments optimized for distributed training.
        
        Args:
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per GPU
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Peak learning rate
            warmup_ratio: Warmup ratio
            deepspeed_config: Path to DeepSpeed config file
            
        Returns:
            TrainingArguments instance
        """
        args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_total_limit=3,
            save_safetensors=True,  # Use safetensors format (safer, faster)
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",  # Select best by eval loss
            greater_is_better=False,  # Lower loss is better
            # Distributed training settings
            bf16=True,
            tf32=True,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            group_by_length=True,
            # Logging
            logging_dir=str(self.output_dir / "logs"),
            report_to=["tensorboard", "wandb"],
            run_name=f"llama3-function-calling",
            # DeepSpeed
            deepspeed=deepspeed_config,
            # Memory optimization
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            **kwargs,
        )
        
        return args
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        training_args: Optional[TrainingArguments] = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Run distributed training.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            training_args: Training arguments
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        if self.model is None:
            self.setup()
        
        if training_args is None:
            training_args = self.get_training_args()
        
        # Data collator with completion-only loss masking
        data_collator = get_data_collator(
            tokenizer=self.tokenizer,
            completion_only=self.completion_only_loss,
            model_type="llama3",
        )
        
        # Compute metrics function
        def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
            """Compute evaluation metrics."""
            logits, labels = eval_pred
            
            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Compute loss only on non-masked tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(
                torch.tensor(shift_logits), 
                torch.tensor(shift_labels)
            )
            
            # Mask out padding/ignored tokens
            mask = shift_labels != -100
            if mask.sum() > 0:
                loss = losses[mask].mean().item()
            else:
                loss = 0.0
            
            return {
                "eval_loss": loss,
                "perplexity": compute_perplexity(loss),
            }
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            # Note: compute_metrics disabled by default for efficiency
            # Enable with preprocess_logits_for_metrics for full eval
        )
        
        # Train
        print("Starting training...")
        print(f"  Completion-only loss: {self.completion_only_loss}")
        print(f"  Data collator: {type(data_collator).__name__}")
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        self.save()
        
        return self.trainer
    
    def save(self, path: Optional[str] = None):
        """Save model and tokenizer."""
        save_path = Path(path) if path else self.output_dir / "final"
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving model to {save_path}")
        
        if self.use_lora:
            # Save only LoRA weights
            self.model.save_pretrained(save_path)
        else:
            # Save full model
            self.trainer.save_model(save_path)
        
        self.tokenizer.save_pretrained(save_path)
    
    def merge_and_save(self, path: str):
        """Merge LoRA weights with base model and save."""
        if not self.use_lora:
            print("Not using LoRA, skipping merge")
            return
        
        print(f"Merging LoRA weights and saving to {path}")
        
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
