#!/usr/bin/env python3
"""
GPU benchmark script to verify cluster performance before training.
Measures GPU compute, memory bandwidth, and multi-GPU communication.

Usage:
    python scripts/benchmark.py
    torchrun --nproc_per_node=8 scripts/benchmark.py  # Multi-GPU
"""
import os
import time
import torch
import torch.distributed as dist


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def benchmark_matmul(size: int = 8192, iterations: int = 100) -> dict:
    """Benchmark matrix multiplication (TFLOPS)."""
    device = get_device()
    
    a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    
    # Warmup
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Calculate TFLOPS (2 * N^3 operations for matmul)
    flops = 2 * size**3 * iterations
    tflops = flops / elapsed / 1e12
    
    return {
        "operation": "MatMul",
        "size": f"{size}x{size}",
        "tflops": round(tflops, 2),
        "time_per_iter_ms": round(elapsed / iterations * 1000, 2),
    }


def benchmark_memory_bandwidth(size_gb: float = 1.0, iterations: int = 50) -> dict:
    """Benchmark GPU memory bandwidth."""
    device = get_device()
    
    num_elements = int(size_gb * 1e9 / 4)  # float32
    a = torch.randn(num_elements, device=device, dtype=torch.float32)
    b = torch.empty_like(a)
    
    # Warmup
    for _ in range(5):
        b.copy_(a)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        b.copy_(a)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Calculate bandwidth (read + write)
    bytes_transferred = 2 * a.nbytes * iterations
    bandwidth_gbps = bytes_transferred / elapsed / 1e9
    
    return {
        "operation": "Memory Copy",
        "size_gb": size_gb,
        "bandwidth_gbps": round(bandwidth_gbps, 2),
    }


def benchmark_allreduce(size_mb: float = 100.0, iterations: int = 50) -> dict:
    """Benchmark multi-GPU all-reduce communication."""
    if not dist.is_initialized():
        return {"operation": "AllReduce", "status": "skipped (single GPU)"}
    
    device = get_device()
    world_size = dist.get_world_size()
    
    num_elements = int(size_mb * 1e6 / 4)
    tensor = torch.randn(num_elements, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(5):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Calculate effective bandwidth
    bytes_per_iter = tensor.nbytes * 2 * (world_size - 1) / world_size
    bandwidth_gbps = bytes_per_iter * iterations / elapsed / 1e9
    
    return {
        "operation": "AllReduce",
        "size_mb": size_mb,
        "world_size": world_size,
        "bandwidth_gbps": round(bandwidth_gbps, 2),
        "time_per_iter_ms": round(elapsed / iterations * 1000, 2),
    }


def get_gpu_info() -> dict:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"status": "No GPU available"}
    
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(props.total_memory / 1e9, 2),
        "num_gpus": torch.cuda.device_count(),
    }


def print_results(title: str, results: dict):
    """Pretty print results."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print('='*50)
    for key, value in results.items():
        print(f"  {key}: {value}")


def main():
    # Initialize distributed if available
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        rank = 0
    
    is_main = rank == 0
    
    if is_main:
        print("\n" + "="*50)
        print(" GPU CLUSTER BENCHMARK")
        print("="*50)
        
        # GPU Info
        print_results("GPU Information", get_gpu_info())
    
    # Run benchmarks
    results = []
    
    # MatMul
    matmul_result = benchmark_matmul()
    results.append(matmul_result)
    if is_main:
        print_results("Compute (MatMul)", matmul_result)
    
    # Memory
    mem_result = benchmark_memory_bandwidth()
    results.append(mem_result)
    if is_main:
        print_results("Memory Bandwidth", mem_result)
    
    # AllReduce (multi-GPU only)
    allreduce_result = benchmark_allreduce()
    results.append(allreduce_result)
    if is_main:
        print_results("Multi-GPU Communication", allreduce_result)
    
    if is_main:
        print("\n" + "="*50)
        print(" BENCHMARK COMPLETE")
        print("="*50)
        
        # Summary for H100
        print("\nExpected H100 Performance:")
        print("  - MatMul (BF16): ~900-1000 TFLOPS")
        print("  - Memory Bandwidth: ~3000 GB/s")
        print("  - AllReduce (InfiniBand): ~300-400 GB/s")
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
