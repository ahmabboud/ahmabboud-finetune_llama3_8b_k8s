"""
Dataset loading and processing for function calling fine-tuning.
"""
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class FunctionCallingDataset(Dataset):
    """
    PyTorch Dataset for function calling training data.
    Handles tokenization and formatting for Llama-3 instruction format.
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        include_labels: bool = True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_labels = include_labels
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Format as chat messages
        messages = item.get("messages", [])
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        if self.include_labels:
            # For causal LM, labels are the same as input_ids
            # Mask padding tokens with -100
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            result["labels"] = labels
        
        return result


def load_xlam_dataset(
    dataset_name: str = "Salesforce/xlam-function-calling-60k",
    split: str = "train",
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load the xLAM function calling dataset from HuggingFace.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load
        cache_dir: Optional cache directory for downloaded data
        
    Returns:
        List of processed examples
    """
    print(f"Loading dataset: {dataset_name}")
    
    dataset = load_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir,
    )
    
    print(f"Loaded {len(dataset)} examples")
    return list(dataset)


def load_glaive_dataset(
    dataset_name: str = "glaiveai/glaive-function-calling-v2",
    split: str = "train",
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load the Glaive function calling dataset from HuggingFace.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load (can include slice like "train[:10000]")
        cache_dir: Optional cache directory for downloaded data
        max_samples: Maximum number of samples to load
        
    Returns:
        List of raw examples (with 'system' and 'chat' fields)
    """
    print(f"Loading dataset: {dataset_name}")
    
    # Add slice if max_samples specified and not already in split
    if max_samples and "[" not in split:
        split = f"{split}[:{max_samples}]"
    
    dataset = load_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir,
    )
    
    print(f"Loaded {len(dataset)} examples")
    return list(dataset)


def load_local_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from local JSON/JSONL file.
    
    Args:
        path: Path to JSON or JSONL file
        
    Returns:
        List of examples
    """
    path = Path(path)
    
    if path.suffix == ".jsonl":
        data = []
        with open(path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(path, "r") as f:
            data = json.load(f)
    
    print(f"Loaded {len(data)} examples from {path}")
    return data


def save_dataset(data: List[Dict[str, Any]], path: str) -> None:
    """
    Save dataset to local JSONL file.
    
    Args:
        data: List of examples
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Saved {len(data)} examples to {path}")
