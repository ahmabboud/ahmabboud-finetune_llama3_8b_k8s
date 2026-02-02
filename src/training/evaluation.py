"""
Evaluation metrics for function calling fine-tuning.
Includes metrics specific to tool/function calling tasks.
"""
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np


def extract_function_call(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract function call from model output.
    Supports formats:
    - <functioncall> {"name": "...", "arguments": {...}}
    - {"tool": "...", "arguments": {...}}
    - {"name": "...", "arguments": {...}}
    """
    # Try to find functioncall tag format
    match = re.search(r'<functioncall>\s*(\{.*?\})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON object
    # Look for JSON that contains "name" or "tool" key
    json_pattern = r'\{[^{}]*(?:"name"|"tool")[^{}]*\}'
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try to parse entire text as JSON
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict) and ('name' in parsed or 'tool' in parsed):
            return parsed
    except json.JSONDecodeError:
        pass
    
    return None


def is_valid_json(text: str) -> bool:
    """Check if text contains valid JSON."""
    try:
        # Try to find any JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json.loads(match.group(0))
            return True
    except json.JSONDecodeError:
        pass
    return False


def compute_function_calling_metrics(
    predictions: List[str],
    references: List[str],
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compute function calling specific metrics.
    
    Args:
        predictions: List of model outputs
        references: List of expected outputs
        verbose: Print detailed analysis
        
    Returns:
        Dictionary of metrics
    """
    metrics = defaultdict(float)
    total = len(predictions)
    
    if total == 0:
        return dict(metrics)
    
    exact_match = 0
    function_name_match = 0
    json_valid = 0
    has_function_call = 0
    argument_match = 0
    
    for pred, ref in zip(predictions, references):
        # Check JSON validity
        if is_valid_json(pred):
            json_valid += 1
        
        # Extract function calls
        pred_call = extract_function_call(pred)
        ref_call = extract_function_call(ref)
        
        if pred_call is not None:
            has_function_call += 1
        
        if pred_call and ref_call:
            # Check function name match
            pred_name = pred_call.get('name') or pred_call.get('tool')
            ref_name = ref_call.get('name') or ref_call.get('tool')
            
            if pred_name == ref_name:
                function_name_match += 1
                
                # Check arguments match
                pred_args = pred_call.get('arguments', {})
                ref_args = ref_call.get('arguments', {})
                
                # Handle string arguments (need to parse)
                if isinstance(pred_args, str):
                    try:
                        pred_args = json.loads(pred_args)
                    except:
                        pred_args = {}
                if isinstance(ref_args, str):
                    try:
                        ref_args = json.loads(ref_args)
                    except:
                        ref_args = {}
                
                if pred_args == ref_args:
                    argument_match += 1
        
        # Exact match (normalized)
        pred_norm = pred.strip().lower()
        ref_norm = ref.strip().lower()
        if pred_norm == ref_norm:
            exact_match += 1
    
    metrics['exact_match'] = exact_match / total
    metrics['function_name_accuracy'] = function_name_match / total
    metrics['json_validity'] = json_valid / total
    metrics['function_call_rate'] = has_function_call / total
    metrics['argument_accuracy'] = argument_match / total if function_name_match > 0 else 0.0
    
    if verbose:
        print(f"\n=== Function Calling Metrics ===")
        print(f"Total samples: {total}")
        print(f"Exact match: {exact_match}/{total} ({metrics['exact_match']:.2%})")
        print(f"Function name accuracy: {function_name_match}/{total} ({metrics['function_name_accuracy']:.2%})")
        print(f"JSON validity: {json_valid}/{total} ({metrics['json_validity']:.2%})")
        print(f"Function call rate: {has_function_call}/{total} ({metrics['function_call_rate']:.2%})")
        print(f"Argument accuracy: {metrics['argument_accuracy']:.2%}")
    
    return dict(metrics)


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    return np.exp(min(loss, 100))  # Cap to avoid overflow


class FunctionCallingEvaluator:
    """
    Evaluator for function calling models.
    Generates predictions and computes metrics.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.0,  # Greedy for evaluation
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
    
    def generate_prediction(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate model prediction for a conversation.
        
        Args:
            messages: List of messages (without final assistant response)
            
        Returns:
            Generated text
        """
        # Apply chat template without the last assistant message
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only new tokens
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        
        return generated.strip()
    
    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        num_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset: List of examples with 'messages' key
            num_samples: Number of samples to evaluate (None for all)
            verbose: Print progress
            
        Returns:
            Dictionary of metrics
        """
        import torch
        
        if num_samples:
            dataset = dataset[:num_samples]
        
        predictions = []
        references = []
        
        self.model.eval()
        
        for i, example in enumerate(dataset):
            messages = example['messages']
            
            # Find last assistant message as reference
            assistant_msgs = [m for m in messages if m['role'] == 'assistant']
            if not assistant_msgs:
                continue
            
            reference = assistant_msgs[-1]['content']
            
            # Create prompt (all messages except last assistant)
            prompt_messages = []
            for msg in messages:
                prompt_messages.append(msg)
                if msg['role'] == 'assistant' and msg['content'] == reference:
                    prompt_messages.pop()  # Remove the reference we're predicting
                    break
            
            # Generate prediction
            try:
                prediction = self.generate_prediction(prompt_messages)
                predictions.append(prediction)
                references.append(reference)
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"Evaluated {i + 1}/{len(dataset)} samples")
                    
            except Exception as e:
                if verbose:
                    print(f"Error on sample {i}: {e}")
        
        # Compute metrics
        metrics = compute_function_calling_metrics(
            predictions, references, verbose=verbose
        )
        
        return metrics


# Import torch for the evaluator
try:
    import torch
except ImportError:
    pass
