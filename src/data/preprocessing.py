"""
Data preprocessing utilities for function calling fine-tuning.
Supports multiple dataset formats:
- xLAM dataset format (query/tools/answers)
- Glaive format (system/chat)
"""
import json
import re
from typing import Dict, List, Any, Optional


def parse_glaive_format(system: str, chat: str) -> List[Dict[str, str]]:
    """
    Parse Glaive function calling format into chat messages.
    
    Glaive format:
    - system: "SYSTEM: You are a helpful assistant with access to the following functions..."
    - chat: "USER: ...\n\n\nASSISTANT: ...\n\n\nFUNCTION RESPONSE: ...\n\n\nASSISTANT: ..."
    
    Returns:
        List of chat messages in standard format
    """
    messages = []
    
    # Parse system message - remove "SYSTEM: " prefix
    system_content = system
    if system_content.startswith("SYSTEM: "):
        system_content = system_content[8:]
    
    messages.append({
        "role": "system",
        "content": system_content.strip(),
    })
    
    # Parse chat - split by conversation turns
    # The chat format uses triple newlines as separators
    # Markers: USER:, ASSISTANT:, FUNCTION RESPONSE:
    
    # Split by markers while preserving them
    parts = re.split(r'\n\n+(?=USER:|ASSISTANT:|FUNCTION RESPONSE:)', chat)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if part.startswith("USER:"):
            content = part[5:].strip()
            # Remove <|endoftext|> tokens
            content = content.replace("<|endoftext|>", "").strip()
            if content:
                messages.append({
                    "role": "user",
                    "content": content,
                })
        elif part.startswith("ASSISTANT:"):
            content = part[10:].strip()
            # Remove <|endoftext|> tokens
            content = content.replace("<|endoftext|>", "").strip()
            if content:
                messages.append({
                    "role": "assistant",
                    "content": content,
                })
        elif part.startswith("FUNCTION RESPONSE:"):
            # This represents tool output - we can add as a special message
            # or merge with context. For training, we'll add as tool response
            content = part[18:].strip()
            content = content.replace("<|endoftext|>", "").strip()
            if content:
                messages.append({
                    "role": "tool",
                    "content": content,
                })
    
    return messages


def preprocess_glaive_dataset(
    examples: List[Dict[str, Any]],
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Preprocess Glaive function calling dataset for training.
    
    Args:
        examples: Raw dataset examples with 'system' and 'chat' fields
        verbose: Print progress
        
    Returns:
        List of processed examples with chat format
    """
    processed = []
    skipped = 0
    
    for i, example in enumerate(examples):
        try:
            system = example.get("system", "")
            chat = example.get("chat", "")
            
            if not system or not chat:
                skipped += 1
                continue
            
            messages = parse_glaive_format(system, chat)
            
            # Need at least system + user + assistant
            if len(messages) < 3:
                skipped += 1
                continue
            
            processed.append({
                "messages": messages,
                "id": i,
            })
            
        except Exception as e:
            if verbose and i < 10:  # Only print first few errors
                print(f"Error processing example {i}: {e}")
            skipped += 1
    
    if verbose:
        print(f"Processed {len(processed)} examples, skipped {skipped}")
    
    return processed


def create_chat_format(
    query: str,
    tools: List[Dict[str, Any]],
    response: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Create Llama-3 chat format from function calling data.
    
    Args:
        query: User query/instruction
        tools: List of available tool definitions
        response: Expected model response (tool call)
        system_prompt: Optional system prompt
        
    Returns:
        List of chat messages
    """
    messages = []
    
    # System message with tool definitions
    if system_prompt is None:
        system_prompt = create_system_prompt(tools)
    
    messages.append({
        "role": "system",
        "content": system_prompt,
    })
    
    # User message
    messages.append({
        "role": "user",
        "content": query,
    })
    
    # Assistant response (the tool call)
    messages.append({
        "role": "assistant",
        "content": response,
    })
    
    return messages


def create_system_prompt(tools: List[Dict[str, Any]]) -> str:
    """
    Create a system prompt that includes tool definitions.
    
    Args:
        tools: List of tool definitions
        
    Returns:
        System prompt string
    """
    tools_str = json.dumps(tools, indent=2)
    
    return f"""You are a helpful assistant with access to the following tools. Use them when appropriate to help the user.

Available tools:
{tools_str}

When you need to use a tool, respond with a JSON object in this format:
{{"tool": "tool_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}

If no tool is needed, respond normally."""


def preprocess_for_training(
    examples: List[Dict[str, Any]],
    max_tools: int = 10,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Preprocess xLAM dataset examples for training.
    
    Args:
        examples: Raw dataset examples
        max_tools: Maximum number of tools to include in context
        verbose: Print progress
        
    Returns:
        List of processed examples with chat format
    """
    processed = []
    skipped = 0
    
    for i, example in enumerate(examples):
        try:
            # Extract fields from xLAM format
            query = example.get("query", example.get("instruction", ""))
            tools = example.get("tools", example.get("functions", []))
            answers = example.get("answers", example.get("response", ""))
            
            if not query or not answers:
                skipped += 1
                continue
            
            # Limit tools if too many
            if len(tools) > max_tools:
                tools = tools[:max_tools]
            
            # Format the response
            if isinstance(answers, list):
                response = json.dumps(answers, indent=2)
            elif isinstance(answers, str):
                response = answers
            else:
                response = json.dumps(answers)
            
            # Create chat format
            messages = create_chat_format(
                query=query,
                tools=tools,
                response=response,
            )
            
            processed.append({
                "messages": messages,
                "id": example.get("id", i),
            })
            
        except Exception as e:
            if verbose:
                print(f"Error processing example {i}: {e}")
            skipped += 1
    
    if verbose:
        print(f"Processed {len(processed)} examples, skipped {skipped}")
    
    return processed


def split_dataset(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple:
    """
    Split dataset into train and validation sets.
    
    Args:
        data: Full dataset
        train_ratio: Fraction for training
        seed: Random seed
        
    Returns:
        (train_data, val_data)
    """
    import random
    random.seed(seed)
    
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_data = shuffled[:split_idx]
    val_data = shuffled[split_idx:]
    
    print(f"Split: {len(train_data)} train, {len(val_data)} validation")
    
    return train_data, val_data
