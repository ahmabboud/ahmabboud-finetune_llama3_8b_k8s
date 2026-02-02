"""
FSDP + LoRA trainer using TRL SFTTrainer.
This is the recommended approach for multi-node training with PEFT.

Based on official HuggingFace PEFT FSDP example:
https://huggingface.co/docs/peft/accelerate/fsdp
"""
import os
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig


class FSDPTrainer:
    """
    FSDP + LoRA trainer using TRL SFTTrainer.
    Designed for multi-node distributed training.
    
    Usage with accelerate:
        accelerate launch --config_file configs/fsdp_config.yaml \\
            scripts/train_fsdp.py --config configs/train_config.yaml
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        output_dir: str = "./output",
        lora_config: Optional[Dict[str, Any]] = None,
        use_sdpa: bool = True,  # Use PyTorch SDPA (safer than flash-attn)
        torch_dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.lora_config = lora_config or {}
        self.use_sdpa = use_sdpa
        self.torch_dtype = getattr(torch, torch_dtype)
        
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        self.trainer = None
        
        # Distributed info
        self.world_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
    def _log(self, msg: str):
        """Log message with rank prefix."""
        print(f"[Rank {self.world_rank}/{self.world_size}] {msg}")
    
    def setup_tokenizer(self) -> AutoTokenizer:
        """Initialize tokenizer."""
        self._log(f"Loading tokenizer: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        return self.tokenizer
    
    def setup_model(self) -> AutoModelForCausalLM:
        """Initialize model for FSDP training."""
        self._log(f"Loading model: {self.model_name}")
        
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": True,
            "use_cache": False,  # Required for gradient checkpointing
        }
        
        # Use SDPA (PyTorch native) instead of flash-attn
        # More compatible across versions
        if self.use_sdpa:
            model_kwargs["attn_implementation"] = "sdpa"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        
        return self.model
    
    def setup_peft_config(self) -> LoraConfig:
        """Create PEFT LoRA configuration."""
        target_modules = self.lora_config.get(
            "target_modules", 
            ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        if isinstance(target_modules, str):
            target_modules = [m.strip() for m in target_modules.split(",")]
        
        self.peft_config = LoraConfig(
            r=self.lora_config.get("r", 64),
            lora_alpha=self.lora_config.get("lora_alpha", 128),
            lora_dropout=self.lora_config.get("lora_dropout", 0.05),
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        
        self._log(f"LoRA config: r={self.peft_config.r}, alpha={self.peft_config.lora_alpha}")
        return self.peft_config
    
    def get_sft_config(
        self,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 1.5e-4,
        lr_scheduler_type: str = "cosine",
        warmup_steps: int = 200,
        weight_decay: float = 0.01,
        logging_steps: int = 10,
        save_steps: int = 200,
        eval_steps: int = 200,
        save_total_limit: int = 3,
        max_length: int = 2048,
        gradient_checkpointing: bool = True,
        report_to: str = "wandb",
        run_name: str = "fsdp-lora",
        **kwargs,
    ) -> SFTConfig:
        """Create SFTConfig for training."""
        
        # Determine if we have eval data
        has_eval = kwargs.pop("has_eval", True)
        
        config = SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_strategy="steps" if has_eval else "no",
            eval_steps=eval_steps if has_eval else None,
            save_total_limit=save_total_limit,
            bf16=True,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=max_length,
            dataset_text_field="text",
            report_to=[report_to] if report_to and os.getenv("WANDB_API_KEY") else ["none"],
            run_name=run_name,
            ddp_find_unused_parameters=False,
        )
        
        return config
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        sft_config: Optional[SFTConfig] = None,
        **training_kwargs,
    ):
        """
        Run FSDP + LoRA training with TRL SFTTrainer.
        
        Args:
            train_dataset: Training dataset (must have 'text' field)
            eval_dataset: Optional evaluation dataset
            sft_config: Optional SFTConfig (created if not provided)
            **training_kwargs: Additional args for get_sft_config
        """
        # Setup components if not already done
        if self.tokenizer is None:
            self.setup_tokenizer()
        if self.model is None:
            self.setup_model()
        if self.peft_config is None:
            self.setup_peft_config()
        
        # Create SFT config if not provided
        if sft_config is None:
            training_kwargs["has_eval"] = eval_dataset is not None
            sft_config = self.get_sft_config(**training_kwargs)
        
        self._log(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            self._log(f"Eval samples: {len(eval_dataset)}")
        
        # Create trainer - SFTTrainer handles PEFT+FSDP integration
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=self.peft_config,
        )
        
        # Train
        self._log("Starting FSDP + LoRA training...")
        self.trainer.train()
        
        # Save final model (LoRA adapters)
        if self.world_rank == 0:
            final_path = self.output_dir / "final"
            final_path.mkdir(parents=True, exist_ok=True)
            self.trainer.save_model(str(final_path))
            self.tokenizer.save_pretrained(str(final_path))
            self._log(f"Saved final model to {final_path}")
        
        return self.trainer
    
    def merge_and_save(self, path: str):
        """Merge LoRA weights with base model and save."""
        if self.world_rank != 0:
            return
        
        self._log(f"Merging LoRA weights and saving to {path}")
        
        # Get the PEFT model from trainer
        peft_model = self.trainer.model
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def load_preprocessed_dataset(path: str) -> Dataset:
    """Load a preprocessed dataset from disk."""
    return load_from_disk(path)
