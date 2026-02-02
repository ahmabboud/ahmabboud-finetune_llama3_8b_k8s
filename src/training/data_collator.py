"""
Custom data collators for function calling training.
Implements response-only loss masking.
"""
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from transformers import PreTrainedTokenizer


@dataclass
class DataCollatorForCompletionOnly:
    """
    Data collator that masks loss for all tokens except the completion (assistant response).
    This ensures the model only learns to generate responses, not to repeat prompts.
    
    For Llama-3 chat format:
    - System/User messages: masked (label = -100)
    - Assistant messages: included in loss
    """
    
    tokenizer: PreTrainedTokenizer
    response_template: str = "<|start_header_id|>assistant<|end_header_id|>"
    instruction_template: str = "<|start_header_id|>user<|end_header_id|>"
    mlm: bool = False
    
    def __post_init__(self):
        # Tokenize the templates
        self.response_token_ids = self.tokenizer.encode(
            self.response_template, add_special_tokens=False
        )
        self.instruction_token_ids = self.tokenizer.encode(
            self.instruction_template, add_special_tokens=False
        )
    
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {
            key: torch.stack([example[key] for example in examples])
            for key in examples[0].keys()
        }
        
        # Create labels with masking
        labels = batch["input_ids"].clone()
        
        # Mask all tokens by default
        labels[:] = -100
        
        # For each sequence, find assistant response regions and unmask them
        for i in range(len(labels)):
            input_ids = batch["input_ids"][i].tolist()
            
            # Find all occurrences of response template
            response_starts = self._find_all_occurrences(
                input_ids, self.response_token_ids
            )
            
            # Find all occurrences of instruction template (marks end of assistant)
            instruction_starts = self._find_all_occurrences(
                input_ids, self.instruction_token_ids
            )
            
            # Also consider end of sequence as potential end
            instruction_starts.append(len(input_ids))
            
            # Unmask assistant responses
            for resp_start in response_starts:
                # Start after the response template
                content_start = resp_start + len(self.response_token_ids)
                
                # Find the next instruction (or end of sequence)
                content_end = len(input_ids)
                for inst_start in instruction_starts:
                    if inst_start > resp_start:
                        content_end = inst_start
                        break
                
                # Unmask this region (include the response for learning)
                labels[i, content_start:content_end] = batch["input_ids"][i, content_start:content_end]
        
        batch["labels"] = labels
        return batch
    
    def _find_all_occurrences(
        self, sequence: List[int], pattern: List[int]
    ) -> List[int]:
        """Find all starting positions of pattern in sequence."""
        positions = []
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i + len(pattern)] == pattern:
                positions.append(i)
        return positions


@dataclass  
class DataCollatorForCausalLMWithPadding:
    """
    Simple data collator for causal LM that handles padding properly.
    Masks padding tokens in labels.
    """
    
    tokenizer: PreTrainedTokenizer
    mlm: bool = False
    
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {
            key: torch.stack([example[key] for example in examples])
            for key in examples[0].keys()
        }
        
        # Create labels from input_ids
        labels = batch["input_ids"].clone()
        
        # Mask padding tokens
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        
        batch["labels"] = labels
        return batch


def get_data_collator(
    tokenizer: PreTrainedTokenizer,
    completion_only: bool = True,
    model_type: str = "llama3",
) -> Any:
    """
    Get appropriate data collator based on configuration.
    
    Args:
        tokenizer: The tokenizer
        completion_only: If True, only compute loss on assistant responses
        model_type: Model type for template detection
        
    Returns:
        Data collator instance
    """
    if completion_only:
        # Templates for different model types
        templates = {
            "llama3": {
                "response": "<|start_header_id|>assistant<|end_header_id|>",
                "instruction": "<|start_header_id|>user<|end_header_id|>",
            },
            "llama2": {
                "response": "[/INST]",
                "instruction": "[INST]",
            },
            "chatml": {
                "response": "<|im_start|>assistant",
                "instruction": "<|im_start|>user",
            },
        }
        
        template = templates.get(model_type, templates["llama3"])
        
        return DataCollatorForCompletionOnly(
            tokenizer=tokenizer,
            response_template=template["response"],
            instruction_template=template["instruction"],
        )
    else:
        return DataCollatorForCausalLMWithPadding(
            tokenizer=tokenizer,
        )
