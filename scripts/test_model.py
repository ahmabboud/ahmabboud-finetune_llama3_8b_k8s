#!/usr/bin/env python3
"""Test the fine-tuned Llama-3 function calling model."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("=" * 60)
print("LLAMA-3 FUNCTION CALLING MODEL TEST")
print("=" * 60)

BASE_MODEL = "/mnt/data/models/llama3-8b-instruct"
LORA_ADAPTER = "/mnt/data/checkpoints/llama3-function-calling-ray/final"

print(f"\nLoading from: {BASE_MODEL}")
print(f"LoRA adapter: {LORA_ADAPTER}")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model (this takes ~30 seconds)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_ADAPTER)
print("Merging adapter into base model...")
model = model.merge_and_unload()
model.eval()

print("\n" + "=" * 60)
print("MODEL LOADED SUCCESSFULLY!")
print("=" * 60)

def generate(system_prompt, user_prompt):
    """Generate a response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# Test cases - using format similar to training data
tests = [
    {
        "name": "Weather Query",
        "system": """You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "get_weather",
    "description": "Get the current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name"
            },
            "unit": {
                "type": "string",
                "description": "Temperature unit (celsius or fahrenheit)"
            }
        },
        "required": ["location"]
    }
}""",
        "user": "What's the weather like in Tokyo?"
    },
    {
        "name": "Calculator",
        "system": """You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "calculate",
    "description": "Evaluate a mathematical expression",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
}""",
        "user": "What is 15% of 250?"
    },
    {
        "name": "Generate Password",
        "system": """You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "generate_random_password",
    "description": "Generate a random password with specified criteria",
    "parameters": {
        "type": "object",
        "properties": {
            "length": {
                "type": "integer",
                "description": "The length of the password"
            },
            "include_uppercase": {
                "type": "boolean",
                "description": "Whether to include uppercase letters"
            },
            "include_numbers": {
                "type": "boolean",
                "description": "Whether to include numbers"
            }
        },
        "required": ["length"]
    }
}""",
        "user": "Generate a secure password with 16 characters including uppercase and numbers"
    },
]

print("\n" + "=" * 60)
print("RUNNING FUNCTION CALLING TESTS")
print("=" * 60)

for i, test in enumerate(tests, 1):
    print(f"\n--- Test {i}: {test['name']} ---")
    print(f"User: {test['user']}")
    
    response = generate(test["system"], test["user"])
    print(f"Model: {response}")
    
    if "{" in response and ("functioncall" in response.lower() or "name" in response.lower()):
        print("✅ Contains function call!")
    else:
        print("⚠️  May not be a proper function call")

print("\n" + "=" * 60)
print("TEST COMPLETE!")
print("=" * 60)
