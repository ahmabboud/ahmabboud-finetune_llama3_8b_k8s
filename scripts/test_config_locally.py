#!/usr/bin/env python3
"""
Local test to verify TRL SFTConfig parameters work before deploying to K8s.
This runs on CPU with minimal memory to validate configuration.
"""
import sys

def test_sftconfig_parameters():
    """Test that SFTConfig accepts our parameters"""
    print("Testing TRL SFTConfig parameters...")
    
    try:
        from trl import SFTConfig
        print(f"✓ TRL imported successfully")
        
        # Test the exact parameters we'll use
        config = SFTConfig(
            output_dir="/tmp/test",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=200,  # Use warmup_steps instead of deprecated warmup_ratio
            weight_decay=0.01,
            logging_steps=10,
            save_steps=100,
            bf16=False,  # CPU doesn't support bf16
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=2048,  # This is the key parameter to test
            dataset_text_field="text",
            report_to=["none"],
        )
        print(f"✓ SFTConfig created successfully with max_length=2048")
        print(f"  - max_length: {config.max_length}")
        return True
        
    except TypeError as e:
        print(f"✗ SFTConfig parameter error: {e}")
        return False
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_peft_lora():
    """Test PEFT LoRA config"""
    print("\nTesting PEFT LoraConfig...")
    
    try:
        from peft import LoraConfig, TaskType
        
        config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        print(f"✓ LoraConfig created successfully")
        print(f"  - r: {config.r}, alpha: {config.lora_alpha}")
        return True
        
    except Exception as e:
        print(f"✗ LoraConfig error: {e}")
        return False

def test_fsdp_wrap_policy():
    """Test PEFT FSDP auto wrap policy"""
    print("\nTesting PEFT FSDP auto wrap policy...")
    
    try:
        from peft.utils.other import fsdp_auto_wrap_policy
        print(f"✓ fsdp_auto_wrap_policy imported successfully")
        return True
    except ImportError as e:
        print(f"✗ fsdp_auto_wrap_policy not available: {e}")
        print("  This is OK - accelerate handles wrapping automatically with TRANSFORMER_BASED_WRAP")
        return True  # Not critical

def print_versions():
    """Print package versions"""
    print("Package versions:")
    packages = ["torch", "transformers", "trl", "peft", "accelerate", "datasets"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            print(f"  {pkg}: {mod.__version__}")
        except ImportError:
            print(f"  {pkg}: NOT INSTALLED")
        except AttributeError:
            print(f"  {pkg}: (version unknown)")

if __name__ == "__main__":
    print("=" * 60)
    print("TRL/PEFT Configuration Validator")
    print("=" * 60)
    
    print_versions()
    print()
    
    results = []
    results.append(("SFTConfig", test_sftconfig_parameters()))
    results.append(("LoraConfig", test_peft_lora()))
    results.append(("FSDP Policy", test_fsdp_wrap_policy()))
    
    print("\n" + "=" * 60)
    print("Summary:")
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed! Configuration should work on K8s.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Fix issues before deploying.")
        sys.exit(1)
