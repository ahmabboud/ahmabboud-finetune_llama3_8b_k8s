#!/usr/bin/env python3
"""
Pre-flight checks for multi-node GPU training.
Run this before starting training to verify cluster health.

Usage:
    python scripts/preflight_check.py
    torchrun --nproc_per_node=8 scripts/preflight_check.py  # Multi-GPU
"""
import os
import sys
import socket
import subprocess
from pathlib import Path

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def check(name: str, passed: bool, details: str = ""):
    """Print check result."""
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"  {status}  {name}")
    if details and not passed:
        print(f"         {YELLOW}{details}{RESET}")
    return passed


def check_cuda():
    """Check CUDA availability and GPU count."""
    print("\n" + "="*50)
    print(" 1. GPU AVAILABILITY")
    print("="*50)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        check("CUDA available", cuda_available)
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            check("GPU count >= 1", gpu_count >= 1, f"Found {gpu_count} GPUs")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / 1e9
                check(
                    f"GPU {i}: {props.name}",
                    mem_gb >= 40,  # At least 40GB
                    f"{mem_gb:.1f} GB memory"
                )
            return True
        return False
    except ImportError:
        check("PyTorch installed", False, "pip install torch")
        return False


def check_gpu_health():
    """Check for GPU errors using nvidia-smi."""
    print("\n" + "="*50)
    print(" 2. GPU HEALTH")
    print("="*50)
    
    try:
        # Check ECC errors
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            all_healthy = True
            for i, line in enumerate(lines):
                parts = line.split(", ")
                corrected = parts[0].strip()
                uncorrected = parts[1].strip() if len(parts) > 1 else "0"
                
                # "N/A" means ECC not supported (which is fine)
                if corrected == "N/A":
                    check(f"GPU {i} ECC status", True, "ECC not available")
                else:
                    has_errors = int(uncorrected) > 0
                    check(f"GPU {i} ECC errors", not has_errors, f"Uncorrected: {uncorrected}")
                    if has_errors:
                        all_healthy = False
            return all_healthy
        else:
            check("nvidia-smi available", False)
            return False
    except FileNotFoundError:
        check("nvidia-smi available", False, "nvidia-smi not found in PATH")
        return False


def check_nccl():
    """Check NCCL communication between GPUs."""
    print("\n" + "="*50)
    print(" 3. MULTI-GPU COMMUNICATION (NCCL)")
    print("="*50)
    
    try:
        import torch
        import torch.distributed as dist
        
        if not torch.cuda.is_available():
            check("CUDA required", False)
            return False
        
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            check("Multi-GPU test", True, "Single GPU, skipping NCCL test")
            return True
        
        # Check if already in distributed mode
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            check(f"Distributed initialized", True, f"Rank {rank}/{world_size}")
            
            # Test all-reduce
            tensor = torch.ones(1000, device=f"cuda:{rank % gpu_count}")
            dist.all_reduce(tensor)
            expected = world_size
            passed = tensor[0].item() == expected
            check("AllReduce test", passed, f"Expected {expected}, got {tensor[0].item()}")
            return passed
        else:
            check("Distributed mode", True, "Not initialized (single process)")
            return True
            
    except Exception as e:
        check("NCCL test", False, str(e))
        return False


def check_storage():
    """Check storage access."""
    print("\n" + "="*50)
    print(" 4. STORAGE")
    print("="*50)
    
    # Check common mount points
    mount_points = ["/mnt/data", "/data", "./data"]
    found_storage = False
    
    for mount in mount_points:
        path = Path(mount)
        if path.exists():
            # Test read
            can_read = os.access(path, os.R_OK)
            # Test write
            try:
                test_file = path / ".preflight_test"
                test_file.touch()
                test_file.unlink()
                can_write = True
            except:
                can_write = False
            
            check(f"Storage at {mount}", can_read and can_write, 
                  f"Read: {can_read}, Write: {can_write}")
            found_storage = True
    
    if not found_storage:
        check("Storage mount", False, "No storage found at /mnt/data, /data, or ./data")
    
    return found_storage


def check_dependencies():
    """Check required Python packages."""
    print("\n" + "="*50)
    print(" 5. DEPENDENCIES")
    print("="*50)
    
    required = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("peft", "PEFT (LoRA)"),
        ("deepspeed", "DeepSpeed"),
        ("wandb", "Weights & Biases"),
    ]
    
    all_present = True
    for package, name in required:
        try:
            __import__(package)
            check(name, True)
        except ImportError:
            check(name, False, f"pip install {package}")
            all_present = False
    
    # Check flash-attn separately (optional but recommended)
    try:
        import flash_attn
        check("Flash Attention", True, "(recommended)")
    except ImportError:
        check("Flash Attention", False, "Optional: pip install flash-attn")
    
    return all_present


def check_hf_access():
    """Check HuggingFace token and model access."""
    print("\n" + "="*50)
    print(" 6. HUGGINGFACE ACCESS")
    print("="*50)
    
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    check("HF_TOKEN set", hf_token is not None, "Set HF_TOKEN environment variable")
    
    if hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            user = api.whoami(token=hf_token)
            check("HF token valid", True, f"User: {user.get('name', 'unknown')}")
            return True
        except Exception as e:
            check("HF token valid", False, str(e))
    
    return False


def check_network():
    """Check network connectivity for distributed training."""
    print("\n" + "="*50)
    print(" 7. NETWORK")
    print("="*50)
    
    hostname = socket.gethostname()
    check(f"Hostname: {hostname}", True)
    
    # Check master address if set
    master_addr = os.environ.get("MASTER_ADDR")
    if master_addr:
        try:
            socket.gethostbyname(master_addr)
            check(f"MASTER_ADDR reachable", True, master_addr)
        except socket.gaierror:
            check(f"MASTER_ADDR reachable", False, f"Cannot resolve {master_addr}")
    else:
        check("MASTER_ADDR", True, "Not set (single-node mode)")
    
    return True


def main():
    print("\n" + "="*50)
    print(" PRE-FLIGHT CHECKS FOR MULTI-NODE TRAINING")
    print("="*50)
    
    results = []
    
    results.append(("GPU Availability", check_cuda()))
    results.append(("GPU Health", check_gpu_health()))
    results.append(("NCCL Communication", check_nccl()))
    results.append(("Storage", check_storage()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("HuggingFace Access", check_hf_access()))
    results.append(("Network", check_network()))
    
    # Summary
    print("\n" + "="*50)
    print(" SUMMARY")
    print("="*50)
    
    all_passed = True
    for name, passed in results:
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print(f" {GREEN}All checks passed! Ready for training.{RESET}")
    else:
        print(f" {RED}Some checks failed. Please fix issues above.{RESET}")
    print("="*50 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
