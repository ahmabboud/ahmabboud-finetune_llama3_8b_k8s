#!/usr/bin/env python3
"""
Download Llama-3 model weights to shared storage.

Usage:
    python scripts/download_model.py --model meta-llama/Meta-Llama-3-8B-Instruct --output-dir /mnt/data/models
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download model from HuggingFace")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model identifier on HuggingFace",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Output directory",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()
    
    token = args.token or os.environ.get("HF_TOKEN")
    
    if token is None:
        print("Warning: No HuggingFace token provided. Some models require authentication.")
        print("Set HF_TOKEN environment variable or use --token argument.")
    
    output_dir = Path(args.output_dir) / args.model.replace("/", "--")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {args.model} to {output_dir}")
    
    snapshot_download(
        repo_id=args.model,
        local_dir=str(output_dir),
        token=token,
        ignore_patterns=["*.md", "*.txt"],  # Skip non-essential files
    )
    
    print(f"\nModel downloaded to: {output_dir}")
    print("Contents:")
    for f in output_dir.iterdir():
        size = f.stat().st_size / (1024 * 1024 * 1024)  # GB
        print(f"  {f.name}: {size:.2f} GB")


if __name__ == "__main__":
    main()
