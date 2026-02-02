#!/usr/bin/env python3
"""
Comprehensive evaluation script for function calling fine-tuned models.

Usage:
    python scripts/evaluate.py --model /mnt/data/checkpoints/llama3-function-calling/merged --test-data data/val.jsonl
    python scripts/evaluate.py --model /mnt/data/checkpoints/llama3-function-calling/final --test-data data/val.jsonl --adapter
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.training.evaluation import (
    compute_function_calling_metrics,
    extract_function_call,
    compute_perplexity,
)


def load_model(model_path: str, adapter_path: str = None):
    """Load model and tokenizer."""
    print(f"Loading model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if adapter_path:
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    return model, tokenizer


def load_test_data(path: str) -> list:
    """Load test data from JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_tool_call(response: str) -> Dict[str, Any]:
    """Extract tool call from response."""
    try:
        # Try to parse as JSON
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON in response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass
    return None


def compare_tool_calls(predicted: Dict, expected: str) -> Dict[str, bool]:
    """Compare predicted tool call with expected."""
    try:
        expected_parsed = json.loads(expected) if isinstance(expected, str) else expected
        
        # Handle list format
        if isinstance(expected_parsed, list):
            expected_parsed = expected_parsed[0] if expected_parsed else {}
        
        if predicted is None:
            return {"tool_match": False, "args_match": False}
        
        # Compare tool name
        pred_tool = predicted.get("tool") or predicted.get("name")
        exp_tool = expected_parsed.get("tool") or expected_parsed.get("name")
        tool_match = pred_tool == exp_tool
        
        # Compare arguments (flexible matching)
        pred_args = predicted.get("arguments") or predicted.get("parameters", {})
        exp_args = expected_parsed.get("arguments") or expected_parsed.get("parameters", {})
        args_match = pred_args == exp_args
        
        return {"tool_match": tool_match, "args_match": args_match}
    except Exception:
        return {"tool_match": False, "args_match": False}


def generate(model, tokenizer, messages: list, max_new_tokens: int = 256) -> str:
    """Generate response."""
    prompt = tokenizer.apply_chat_template(
        messages[:-1],  # Exclude assistant message
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for evaluation
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def evaluate(model, tokenizer, test_data: list, num_samples: int = None) -> Dict[str, float]:
    """Run comprehensive evaluation."""
    if num_samples:
        test_data = test_data[:num_samples]
    
    predictions = []
    references = []
    
    results = {
        "total": len(test_data),
        "tool_correct": 0,
        "args_correct": 0,
        "exact_match": 0,
    }
    
    print(f"\nEvaluating on {len(test_data)} samples...")
    
    for example in tqdm(test_data):
        messages = example.get("messages", [])
        
        if len(messages) < 3:
            continue
        
        # Generate prediction
        predicted_response = generate(model, tokenizer, messages)
        predictions.append(predicted_response)
        
        # Get expected response (last assistant message)
        expected_response = ""
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                expected_response = msg["content"]
                break
        references.append(expected_response)
        
        # Legacy comparison for backwards compatibility
        predicted_call = extract_function_call(predicted_response)
        comparison = compare_tool_calls(predicted_call, expected_response)
        
        if comparison["tool_match"]:
            results["tool_correct"] += 1
        if comparison["args_match"]:
            results["args_correct"] += 1
        if comparison["tool_match"] and comparison["args_match"]:
            results["exact_match"] += 1
    
    # Calculate percentages (legacy)
    total = results["total"]
    results["tool_accuracy"] = round(results["tool_correct"] / total * 100, 2)
    results["args_accuracy"] = round(results["args_correct"] / total * 100, 2)
    results["exact_match_accuracy"] = round(results["exact_match"] / total * 100, 2)
    
    # Add comprehensive metrics from evaluation module
    fc_metrics = compute_function_calling_metrics(predictions, references, verbose=True)
    results.update({
        "function_name_accuracy_pct": round(fc_metrics["function_name_accuracy"] * 100, 2),
        "json_validity_pct": round(fc_metrics["json_validity"] * 100, 2),
        "function_call_rate_pct": round(fc_metrics["function_call_rate"] * 100, 2),
        "argument_accuracy_pct": round(fc_metrics["argument_accuracy"] * 100, 2),
    })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate function calling model")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--test-data", type=str, required=True, help="Test data path")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model, args.adapter)
    test_data = load_test_data(args.test_data)
    
    print("\n" + "="*60)
    print(" Function Calling Evaluation")
    print("="*60)
    
    results = evaluate(model, tokenizer, test_data, args.num_samples)
    
    print("\n" + "="*60)
    print(" Results")
    print("="*60)
    print(f"  Total samples:           {results['total']}")
    print(f"  Tool name accuracy:      {results['tool_accuracy']}%")
    print(f"  Arguments accuracy:      {results['args_accuracy']}%")
    print(f"  Exact match:             {results['exact_match_accuracy']}%")
    print(f"  JSON validity:           {results['json_validity_pct']}%")
    print(f"  Function call rate:      {results['function_call_rate_pct']}%")
    print("="*60)
    
    # Save results
    output_path = Path(args.model) / "eval_results.json"
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    except Exception:
        print(f"\nResults: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
