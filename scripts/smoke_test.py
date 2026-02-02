#!/usr/bin/env python3
"""
Smoke test for FSDP + LoRA training.
Validates all components work before running full training.

Usage:
    # Local test (CPU, no GPU required)
    python scripts/smoke_test.py --cpu
    
    # GPU test (single GPU)
    python scripts/smoke_test.py
    
    # Multi-GPU test
    accelerate launch --num_processes=2 scripts/smoke_test.py
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test all required imports work."""
    print("\n" + "=" * 60)
    print("1. Testing imports...")
    print("=" * 60)
    
    errors = []
    
    # Core packages
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("trl", "trl"),
        ("peft", "peft"),
        ("accelerate", "accelerate"),
        ("datasets", "datasets"),
    ]
    
    for name, pkg in packages:
        try:
            mod = __import__(pkg)
            print(f"  ✓ {name}: {mod.__version__}")
        except ImportError as e:
            print(f"  ✗ {name}: FAILED - {e}")
            errors.append(name)
    
    # TRL specific imports
    try:
        from trl import SFTTrainer, SFTConfig
        print(f"  ✓ SFTTrainer, SFTConfig imported")
    except ImportError as e:
        print(f"  ✗ SFTTrainer/SFTConfig: FAILED - {e}")
        errors.append("SFTTrainer")
    
    # PEFT specific imports
    try:
        from peft import LoraConfig, TaskType
        print(f"  ✓ LoraConfig, TaskType imported")
    except ImportError as e:
        print(f"  ✗ LoraConfig/TaskType: FAILED - {e}")
        errors.append("LoraConfig")
    
    # Our trainer
    try:
        from src.training.fsdp_trainer import FSDPTrainer
        print(f"  ✓ FSDPTrainer imported")
    except ImportError as e:
        print(f"  ✗ FSDPTrainer: FAILED - {e}")
        errors.append("FSDPTrainer")
    
    if errors:
        print(f"\n  ✗ {len(errors)} import(s) failed")
        return False
    
    print(f"\n  ✓ All imports successful")
    return True


def test_sft_config():
    """Test SFTConfig creation with our parameters."""
    print("\n" + "=" * 60)
    print("2. Testing SFTConfig...")
    print("=" * 60)
    
    try:
        from trl import SFTConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SFTConfig(
                output_dir=tmpdir,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=1e-4,
                lr_scheduler_type="cosine",
                warmup_steps=10,
                weight_decay=0.01,
                logging_steps=1,
                save_steps=100,
                bf16=False,  # CPU doesn't support bf16
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                max_length=128,  # Small for test
                dataset_text_field="text",
                report_to=["none"],
            )
            
            print(f"  ✓ SFTConfig created")
            print(f"    - max_length: {config.max_length}")
            print(f"    - gradient_checkpointing: {config.gradient_checkpointing}")
            return True
            
    except Exception as e:
        print(f"  ✗ SFTConfig failed: {e}")
        return False


def test_lora_config():
    """Test LoRA configuration."""
    print("\n" + "=" * 60)
    print("3. Testing LoraConfig...")
    print("=" * 60)
    
    try:
        from peft import LoraConfig, TaskType
        
        config = LoraConfig(
            r=8,  # Small for test
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        
        print(f"  ✓ LoraConfig created")
        print(f"    - r: {config.r}, alpha: {config.lora_alpha}")
        print(f"    - target_modules: {config.target_modules}")
        return True
        
    except Exception as e:
        print(f"  ✗ LoraConfig failed: {e}")
        return False


def test_dummy_dataset():
    """Test dataset creation."""
    print("\n" + "=" * 60)
    print("4. Testing Dataset creation...")
    print("=" * 60)
    
    try:
        from datasets import Dataset
        
        # Create dummy data
        data = {
            "text": [
                "Hello, how are you?",
                "I am fine, thank you!",
                "What is the weather today?",
                "It is sunny outside.",
            ] * 10  # 40 samples
        }
        
        dataset = Dataset.from_dict(data)
        print(f"  ✓ Dataset created: {len(dataset)} samples")
        print(f"    - columns: {dataset.column_names}")
        return dataset
        
    except Exception as e:
        print(f"  ✗ Dataset failed: {e}")
        return None


def test_tokenizer(model_name: str = "gpt2"):
    """Test tokenizer loading (use GPT-2 for speed)."""
    print("\n" + "=" * 60)
    print("5. Testing Tokenizer...")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test tokenization
        text = "Hello, this is a test."
        tokens = tokenizer(text, return_tensors="pt")
        
        print(f"  ✓ Tokenizer loaded: {model_name}")
        print(f"    - vocab_size: {tokenizer.vocab_size}")
        print(f"    - test tokens: {tokens['input_ids'].shape}")
        return tokenizer
        
    except Exception as e:
        print(f"  ✗ Tokenizer failed: {e}")
        return None


def test_model_loading(model_name: str = "gpt2", use_cpu: bool = True):
    """Test model loading (use GPT-2 for speed)."""
    print("\n" + "=" * 60)
    print("6. Testing Model loading...")
    print("=" * 60)
    
    try:
        import torch
        from transformers import AutoModelForCausalLM
        
        device = "cpu" if use_cpu else "cuda"
        dtype = torch.float32 if use_cpu else torch.bfloat16
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model loaded: {model_name}")
        print(f"    - parameters: {param_count:,}")
        print(f"    - dtype: {dtype}")
        return model
        
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return None


def test_peft_model(model, use_cpu: bool = True):
    """Test PEFT model wrapping."""
    print("\n" + "=" * 60)
    print("7. Testing PEFT model wrapping...")
    print("=" * 60)
    
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["c_attn"],  # GPT-2 uses c_attn
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        
        peft_model = get_peft_model(model, lora_config)
        
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        
        print(f"  ✓ PEFT model created")
        print(f"    - trainable: {trainable:,} ({100*trainable/total:.2f}%)")
        print(f"    - total: {total:,}")
        return peft_model
        
    except Exception as e:
        print(f"  ✗ PEFT model failed: {e}")
        return None


def test_sft_trainer(tokenizer, dataset, use_cpu: bool = True):
    """Test SFTTrainer creation (without actually training)."""
    print("\n" + "=" * 60)
    print("8. Testing SFTTrainer creation...")
    print("=" * 60)
    
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, TaskType
        from trl import SFTTrainer, SFTConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Load small model
            dtype = torch.float32 if use_cpu else torch.bfloat16
            model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                torch_dtype=dtype,
            )
            
            # LoRA config
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["c_attn"],
                task_type=TaskType.CAUSAL_LM,
            )
            
            # SFT config
            sft_config = SFTConfig(
                output_dir=tmpdir,
                num_train_epochs=1,
                max_steps=2,  # Just 2 steps for smoke test
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                learning_rate=1e-4,
                warmup_steps=1,
                logging_steps=1,
                bf16=False,
                gradient_checkpointing=False,  # Disable for small test
                max_length=32,
                dataset_text_field="text",
                report_to=["none"],
            )
            
            # Create trainer
            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                args=sft_config,
                train_dataset=dataset,
                peft_config=peft_config,
            )
            
            print(f"  ✓ SFTTrainer created successfully")
            return trainer
            
    except Exception as e:
        print(f"  ✗ SFTTrainer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_training_step(trainer):
    """Test a single training step."""
    print("\n" + "=" * 60)
    print("9. Testing training step...")
    print("=" * 60)
    
    try:
        # Run training (just 2 steps)
        trainer.train()
        
        print(f"  ✓ Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nccl_communication(use_cpu: bool = True):
    """Test NCCL multi-GPU communication."""
    print("\n" + "=" * 60)
    print("10. Testing NCCL Communication...")
    print("=" * 60)
    
    if use_cpu:
        print("  ⊘ Skipped (CPU mode)")
        return None  # None means skipped
    
    try:
        import torch
        import torch.distributed as dist
        
        if not torch.cuda.is_available():
            print("  ⊘ Skipped (no CUDA)")
            return None
        
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            print(f"  ⊘ Skipped (only {num_gpus} GPU, need 2+ for NCCL test)")
            return None
        
        # Check NCCL version
        nccl_version = torch.cuda.nccl.version()
        print(f"  ✓ NCCL version: {nccl_version}")
        
        # Check if distributed is already initialized (e.g., by accelerate)
        if dist.is_initialized():
            # Test AllReduce
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            device = torch.device(f"cuda:{rank % num_gpus}")
            
            # Create tensor with rank value
            tensor = torch.tensor([rank * 1.0], device=device)
            
            # AllReduce (sum)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            
            expected = sum(range(world_size))  # 0 + 1 + ... + (world_size-1)
            
            if abs(tensor.item() - expected) < 0.001:
                print(f"  ✓ AllReduce: {world_size} ranks, sum={tensor.item()}")
                return True
            else:
                print(f"  ✗ AllReduce mismatch: got {tensor.item()}, expected {expected}")
                return False
        else:
            # Not in distributed mode, just verify NCCL is available
            print(f"  ✓ NCCL available (not in distributed mode)")
            print(f"    - GPUs: {num_gpus}")
            print(f"    - To test AllReduce, run with: accelerate launch --num_processes=2")
            return True
            
    except Exception as e:
        print(f"  ✗ NCCL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_infiniband():
    """Test InfiniBand availability (for multi-node)."""
    print("\n" + "=" * 60)
    print("11. Testing InfiniBand...")
    print("=" * 60)
    
    try:
        import subprocess
        
        # Check for ibv_devinfo
        result = subprocess.run(
            ["ibv_devinfo"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            # Count devices
            lines = result.stdout.split('\n')
            devices = [l for l in lines if 'hca_id' in l]
            active = result.stdout.count('PORT_ACTIVE')
            
            print(f"  ✓ InfiniBand available")
            print(f"    - HCA devices: {len(devices)}")
            print(f"    - Active ports: {active}")
            return True
        else:
            print(f"  ⊘ InfiniBand not available (ibv_devinfo failed)")
            return None
            
    except FileNotFoundError:
        print(f"  ⊘ InfiniBand tools not installed (ibv_devinfo not found)")
        return None
    except subprocess.TimeoutExpired:
        print(f"  ⊘ InfiniBand check timed out")
        return None
    except Exception as e:
        print(f"  ✗ InfiniBand test failed: {e}")
        return False


def test_fsdp_trainer_class():
    """Test our FSDPTrainer class."""
    print("\n" + "=" * 60)
    print("12. Testing FSDPTrainer class...")
    print("=" * 60)
    
    try:
        from src.training.fsdp_trainer import FSDPTrainer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FSDPTrainer(
                model_name="gpt2",  # Use GPT-2 for testing
                output_dir=tmpdir,
                lora_config={
                    "r": 8,
                    "lora_alpha": 16,
                    "target_modules": ["c_attn"],
                },
                use_sdpa=False,  # GPT-2 doesn't support SDPA
            )
            
            # Setup components
            trainer.setup_tokenizer()
            trainer.setup_peft_config()
            
            print(f"  ✓ FSDPTrainer initialized")
            print(f"    - model: {trainer.model_name}")
            print(f"    - output: {trainer.output_dir}")
            return True
            
    except Exception as e:
        print(f"  ✗ FSDPTrainer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Smoke test for FSDP + LoRA training")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU only")
    parser.add_argument("--full", action="store_true", help="Run full test with training step")
    args = parser.parse_args()
    
    print("=" * 60)
    print("FSDP + LoRA Training Smoke Test")
    print("=" * 60)
    print(f"Mode: {'CPU' if args.cpu else 'GPU'}")
    print(f"Full test: {args.full}")
    
    results = []
    
    # 1. Test imports
    results.append(("Imports", test_imports()))
    
    # 2. Test SFTConfig
    results.append(("SFTConfig", test_sft_config()))
    
    # 3. Test LoraConfig
    results.append(("LoraConfig", test_lora_config()))
    
    # 4. Test dataset
    dataset = test_dummy_dataset()
    results.append(("Dataset", dataset is not None))
    
    # 5. Test tokenizer
    tokenizer = test_tokenizer("gpt2")
    results.append(("Tokenizer", tokenizer is not None))
    
    # 6. Test model loading
    model = test_model_loading("gpt2", args.cpu)
    results.append(("Model", model is not None))
    
    # 7. Test PEFT wrapping
    if model:
        peft_model = test_peft_model(model, args.cpu)
        results.append(("PEFT", peft_model is not None))
    
    # 8. Test SFTTrainer creation
    if tokenizer and dataset:
        trainer = test_sft_trainer(tokenizer, dataset, args.cpu)
        results.append(("SFTTrainer", trainer is not None))
        
        # 9. Test training step (optional)
        if args.full and trainer:
            results.append(("Training", test_training_step(trainer)))
    
    # 10. Test NCCL communication (multi-GPU only)
    nccl_result = test_nccl_communication(args.cpu)
    if nccl_result is not None:
        results.append(("NCCL", nccl_result))
    
    # 11. Test InfiniBand (if available)
    ib_result = test_infiniband()
    if ib_result is not None:
        results.append(("InfiniBand", ib_result))
    
    # 12. Test our FSDPTrainer class
    results.append(("FSDPTrainer", test_fsdp_trainer_class()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n  Total: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\n✗ Some tests failed!")
        sys.exit(1)
    else:
        print("\n✓ All tests passed! Ready for training.")
        sys.exit(0)


if __name__ == "__main__":
    main()
