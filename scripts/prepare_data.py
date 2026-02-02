#!/usr/bin/env python3
"""
Script to download and preprocess function calling datasets for fine-tuning.
Supports both xLAM and Glaive dataset formats.

Usage:
    python scripts/prepare_data.py --output-dir /mnt/data/datasets
    python scripts/prepare_data.py --dataset glaiveai/glaive-function-calling-v2 --format glaive
"""
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_xlam_dataset, preprocess_for_training
from src.data.dataset import load_glaive_dataset, save_dataset
from src.data.preprocessing import split_dataset, preprocess_glaive_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument(
        "--dataset",
        type=str,
        default="glaiveai/glaive-function-calling-v2",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["xlam", "glaive"],
        default="glaive",
        help="Dataset format (xlam or glaive)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for downloaded data",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.95,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to load (for testing)",
    )
    parser.add_argument(
        "--max-tools",
        type=int,
        default=10,
        help="Maximum number of tools per example (xlam format only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("Data Preparation Pipeline")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Format: {args.format}")
    
    # Step 1: Load raw dataset
    print("\n[1/4] Loading dataset...")
    if args.format == "glaive":
        raw_data = load_glaive_dataset(
            dataset_name=args.dataset,
            cache_dir=args.cache_dir,
            max_samples=args.max_samples,
        )
    else:
        raw_data = load_xlam_dataset(
            dataset_name=args.dataset,
            cache_dir=args.cache_dir,
        )
        if args.max_samples:
            raw_data = raw_data[:args.max_samples]
    
    # Step 2: Preprocess
    print("\n[2/4] Preprocessing data...")
    if args.format == "glaive":
        processed_data = preprocess_glaive_dataset(
            raw_data,
            verbose=True,
        )
    else:
        processed_data = preprocess_for_training(
            raw_data,
            max_tools=args.max_tools,
            verbose=True,
        )
    
    # Step 3: Split
    print("\n[3/4] Splitting dataset...")
    train_data, val_data = split_dataset(
        processed_data,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    
    # Step 4: Save
    print("\n[4/4] Saving processed data...")
    save_dataset(train_data, output_dir / "train.jsonl")
    save_dataset(val_data, output_dir / "val.jsonl")
    
    # Summary
    print("\n" + "=" * 50)
    print("Data preparation complete!")
    print("=" * 50)
    print(f"Total examples: {len(processed_data)}")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - {output_dir / 'train.jsonl'}")
    print(f"  - {output_dir / 'val.jsonl'}")


if __name__ == "__main__":
    main()
