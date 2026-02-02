#!/usr/bin/env python3
"""
Simple inference script to test the fine-tuned model.

Usage:
    python scripts/inference.py --model /mnt/data/checkpoints/llama3-function-calling/merged
    python scripts/inference.py --model meta-llama/Meta-Llama-3-8B-Instruct --adapter /mnt/data/checkpoints/llama3-function-calling/final
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_path: str, adapter_path: str = None):
    """Load model and tokenizer."""
    print(f"Loading model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    
    # Load LoRA adapter if provided
    if adapter_path:
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer


def create_prompt(query: str, tools: list) -> list:
    """Create chat messages with tools."""
    tools_str = json.dumps(tools, indent=2)
    
    system_prompt = f"""You are a helpful assistant with access to tools. Use them when appropriate.

Available tools:
{tools_str}

When using a tool, respond with JSON: {{"tool": "tool_name", "arguments": {{"arg": "value"}}}}"""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]


def generate(model, tokenizer, messages: list, max_new_tokens: int = 256) -> str:
    """Generate response from model."""
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# Example tools for testing
EXAMPLE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body"}
            },
            "required": ["to", "subject", "body"]
        }
    }
]

# Test queries
TEST_QUERIES = [
    "What's the weather like in Paris?",
    "Search for the latest news about AI",
    "Send an email to john@example.com about the meeting tomorrow",
    "Tell me a joke",  # Should not use tools
]


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned model")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF model ID")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model, args.adapter)
    
    print("\n" + "="*60)
    print(" Function Calling Inference Test")
    print("="*60)
    
    if args.interactive:
        # Interactive mode
        print("\nEnter your queries (type 'quit' to exit):\n")
        while True:
            query = input("You: ").strip()
            if query.lower() == "quit":
                break
            
            messages = create_prompt(query, EXAMPLE_TOOLS)
            response = generate(model, tokenizer, messages)
            print(f"Assistant: {response}\n")
    else:
        # Run test queries
        for query in TEST_QUERIES:
            print(f"\n{'─'*60}")
            print(f"Query: {query}")
            print("─"*60)
            
            messages = create_prompt(query, EXAMPLE_TOOLS)
            response = generate(model, tokenizer, messages)
            
            print(f"Response: {response}")
            
            # Try to parse as JSON tool call
            try:
                tool_call = json.loads(response)
                print(f"  → Tool: {tool_call.get('tool')}")
                print(f"  → Args: {tool_call.get('arguments')}")
            except json.JSONDecodeError:
                print("  → (Free-form response, no tool call)")
    
    print("\n" + "="*60)
    print(" Inference test complete!")
    print("="*60)


if __name__ == "__main__":
    main()
