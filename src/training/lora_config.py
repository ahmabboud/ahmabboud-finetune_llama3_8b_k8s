"""
LoRA and quantization configuration for efficient fine-tuning.
"""
from typing import Optional, List

from peft import LoraConfig, TaskType


def get_lora_config(
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> LoraConfig:
    """
    Get LoRA configuration for Llama-3 fine-tuning.
    
    Args:
        r: LoRA rank (higher = more parameters, better quality)
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: Dropout probability
        target_modules: Which modules to apply LoRA to
        task_type: Task type (CAUSAL_LM for text generation)
        
    Returns:
        LoraConfig instance
    """
    if target_modules is None:
        # Default targets for Llama-3
        target_modules = [
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=task_type,
        bias="none",
        inference_mode=False,
    )
    
    return config


def get_bnb_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "bfloat16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
):
    """
    Get BitsAndBytes quantization config.
    Note: For H100 with 80GB VRAM, we typically don't need quantization
    and can train in full bf16 for better quality.
    
    Args:
        load_in_4bit: Enable 4-bit quantization
        bnb_4bit_compute_dtype: Compute dtype
        bnb_4bit_quant_type: Quantization type (nf4 or fp4)
        bnb_4bit_use_double_quant: Use double quantization
        
    Returns:
        BitsAndBytesConfig instance
    """
    import torch
    from transformers import BitsAndBytesConfig
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    
    config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=dtype_map.get(bnb_4bit_compute_dtype, torch.bfloat16),
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )
    
    return config


# Recommended configurations for different scenarios

LORA_CONFIG_SMALL = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
}

LORA_CONFIG_MEDIUM = {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
}

LORA_CONFIG_LARGE = {
    "r": 128,
    "lora_alpha": 256,
    "lora_dropout": 0.05,
}

# For 16x H100 (80GB each), we can use larger configs
LORA_CONFIG_H100_16X = {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}
