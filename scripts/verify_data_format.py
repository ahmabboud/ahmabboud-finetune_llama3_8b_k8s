#!/usr/bin/env python3
"""Test script to verify training data format."""
import json

# Load samples from training data
with open('/mnt/data/datasets/train.jsonl') as f:
    examples = [json.loads(f.readline()) for _ in range(3)]

for i, example in enumerate(examples):
    print(f"\n{'='*60}")
    print(f"EXAMPLE {i+1}")
    print('='*60)
    
    print(f"\nKeys in example: {list(example.keys())}")
    
    # Check if it has 'messages' key
    if 'messages' in example:
        print(f"Has 'messages' key: YES ({len(example['messages'])} messages)")
        for j, msg in enumerate(example['messages']):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            content_preview = content[:80] + '...' if len(content) > 80 else content
            print(f"  [{j}] {role}: {content_preview}")
    else:
        print("Has 'messages' key: NO")
    
    # What OLD format function would produce
    print(f"\n--- OLD FORMAT FUNCTION OUTPUT ---")
    system = example.get('system', 'You are a helpful assistant.')
    user = example.get('user', example.get('instruction', ''))
    assistant = example.get('assistant', example.get('output', ''))
    print(f"system: '{system[:60]}...' (len={len(system)})" if len(system) > 60 else f"system: '{system}'")
    print(f"user: '{user[:60]}...' (len={len(user)})" if len(user) > 60 else f"user: '{user}' (EMPTY!)" if not user else f"user: '{user}'")
    print(f"assistant: '{assistant[:60]}...' (len={len(assistant)})" if len(assistant) > 60 else f"assistant: '{assistant}' (EMPTY!)" if not assistant else f"assistant: '{assistant}'")

print("\n" + "="*60)
print("DIAGNOSIS: The OLD format function was getting:")
print("  - Default 'You are a helpful assistant.' for system")
print("  - EMPTY strings for user and assistant")
print("  - This caused the model to learn garbage patterns!")
print("="*60)
