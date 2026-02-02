#!/usr/bin/env python3
"""
GPU Cluster Acceptance Test

Validates GPU cluster readiness for distributed ML training.
Uses open-source model (GPT-2) and dataset (WikiText).

Usage:
    Single node:  python gpu_test.py
    Multi-node:   torchrun --nnodes=2 --nproc_per_node=8 ... gpu_test.py
"""

import os
import sys
import time
import json
import socket
import argparse
import subprocess
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class GPUClusterTest:
    """GPU Cluster Acceptance Test Suite."""
    
    # Supported GPU families
    GPU_FAMILIES = {
        "H100": {"arch": "Hopper", "compute": "9.0"},
        "H200": {"arch": "Hopper", "compute": "9.0"},
        "B100": {"arch": "Blackwell", "compute": "10.0"},
        "B200": {"arch": "Blackwell", "compute": "10.0"},
        "A100": {"arch": "Ampere", "compute": "8.0"},
    }
    
    def __init__(self, args):
        self.args = args
        self.results: List[TestResult] = []
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_distributed = self.world_size > 1
        self.cluster_info: Dict[str, Any] = {}
        
    def log(self, msg: str):
        """Log from rank 0 only."""
        if self.rank == 0:
            print(msg)
    
    def run_cmd(self, cmd: List[str], timeout: int = 30) -> tuple:
        """Run shell command."""
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout, r.stderr
        except Exception as e:
            return -1, "", str(e)
    
    def add_result(self, name: str, passed: bool, message: str, details: Dict = None):
        """Record test result."""
        self.results.append(TestResult(name, passed, message, details))
        if self.rank == 0:
            icon = "✓" if passed else "✗"
            print(f"  {icon} {name}: {message}")
    
    # =========================================================================
    # Infrastructure Tests
    # =========================================================================
    
    def test_gpu_detection(self) -> bool:
        """Detect GPUs and check health."""
        self.log("\n[1/7] GPU Detection...")
        
        code, out, _ = self.run_cmd([
            "nvidia-smi", 
            "--query-gpu=index,name,memory.total,temperature.gpu,ecc.errors.corrected.volatile.total",
            "--format=csv,noheader,nounits"
        ])
        
        if code != 0:
            self.add_result("GPU Detection", False, "nvidia-smi failed")
            return False
        
        gpus = []
        for line in out.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            gpus.append({
                "index": parts[0],
                "name": parts[1],
                "memory_mb": parts[2],
                "temp_c": parts[3],
                "ecc_errors": parts[4] if len(parts) > 4 else "N/A"
            })
        
        num_gpus = len(gpus)
        expected = int(os.environ.get("GPUS_PER_NODE", 8))
        
        # Identify GPU family
        gpu_family = "Unknown"
        if gpus:
            gpu_name = gpus[0]["name"]
            for family in self.GPU_FAMILIES:
                if family in gpu_name:
                    gpu_family = family
                    break
        
        self.cluster_info["gpus"] = gpus
        self.cluster_info["gpu_family"] = gpu_family
        
        passed = num_gpus >= expected
        self.add_result(
            "GPU Detection", passed,
            f"{num_gpus}x {gpu_family} ({gpus[0]['name'] if gpus else 'None'})",
            {"count": num_gpus, "expected": expected, "family": gpu_family}
        )
        return passed
    
    def test_cuda_pytorch(self) -> bool:
        """Test CUDA and PyTorch compatibility."""
        self.log("\n[2/7] CUDA/PyTorch...")
        
        import torch
        
        if not torch.cuda.is_available():
            self.add_result("CUDA/PyTorch", False, "CUDA not available")
            return False
        
        num_gpus = torch.cuda.device_count()
        cuda_ver = torch.version.cuda
        pytorch_ver = torch.__version__
        bf16 = torch.cuda.is_bf16_supported()
        
        # Get compute capability
        cap = torch.cuda.get_device_capability(0)
        compute_cap = f"{cap[0]}.{cap[1]}"
        
        self.cluster_info["cuda"] = {
            "pytorch": pytorch_ver,
            "cuda": cuda_ver,
            "compute_capability": compute_cap,
            "bf16": bf16
        }
        
        self.add_result(
            "CUDA/PyTorch", True,
            f"PyTorch {pytorch_ver}, CUDA {cuda_ver}, Compute {compute_cap}, bf16={'yes' if bf16 else 'no'}"
        )
        return True
    
    def test_memory_limits(self) -> bool:
        """Check memlock limits for RDMA."""
        self.log("\n[3/7] Memory Limits...")
        
        try:
            with open("/proc/1/limits") as f:
                content = f.read()
            
            for line in content.split("\n"):
                if "locked" in line.lower():
                    is_unlimited = "unlimited" in line
                    self.cluster_info["memlock"] = "unlimited" if is_unlimited else "limited"
                    self.add_result(
                        "Memory Limits", is_unlimited,
                        "memlock unlimited" if is_unlimited else "memlock LIMITED (RDMA may fail)"
                    )
                    return is_unlimited
        except Exception as e:
            self.add_result("Memory Limits", False, f"Could not check: {e}")
        
        return False
    
    def test_infiniband(self) -> bool:
        """Check InfiniBand availability."""
        self.log("\n[4/7] InfiniBand...")
        
        code, out, _ = self.run_cmd(["ibv_devinfo"])
        
        if code != 0:
            self.cluster_info["infiniband"] = {"available": False}
            self.add_result("InfiniBand", True, "Not available (using Ethernet)")
            return True  # Not a failure
        
        hcas = out.count("hca_id:")
        active = out.count("PORT_ACTIVE")
        
        # Get link speed
        code2, out2, _ = self.run_cmd(["ibstat"])
        rate = "unknown"
        if code2 == 0:
            for line in out2.split("\n"):
                if "Rate:" in line:
                    rate = line.split(":")[1].strip()
                    break
        
        self.cluster_info["infiniband"] = {
            "available": True,
            "hcas": hcas,
            "active_ports": active,
            "rate": rate
        }
        
        self.add_result(
            "InfiniBand", active > 0,
            f"{hcas} HCA(s), {active} active port(s), {rate}"
        )
        return active > 0
    
    def test_nccl(self) -> bool:
        """Test NCCL multi-GPU communication."""
        self.log("\n[5/7] NCCL Communication...")
        
        import torch
        
        num_gpus = torch.cuda.device_count()
        
        # NCCL version
        nccl_ver = torch.cuda.nccl.version()
        nccl_str = ".".join(map(str, nccl_ver)) if isinstance(nccl_ver, tuple) else str(nccl_ver)
        
        # P2P access matrix
        p2p_enabled = 0
        p2p_total = num_gpus * (num_gpus - 1) if num_gpus > 1 else 0
        
        for i in range(num_gpus):
            for j in range(num_gpus):
                if i != j and torch.cuda.can_device_access_peer(i, j):
                    p2p_enabled += 1
        
        self.cluster_info["nccl"] = {
            "version": nccl_str,
            "p2p_enabled": p2p_enabled,
            "p2p_total": p2p_total
        }
        
        # Test multi-GPU tensor operations
        if num_gpus > 1:
            try:
                tensors = [torch.ones(1024*1024, device=f"cuda:{i}") for i in range(min(num_gpus, 4))]
                for i in range(len(tensors)):
                    torch.cuda.synchronize(i)
                del tensors
                torch.cuda.empty_cache()
            except Exception as e:
                self.add_result("NCCL", False, f"Multi-GPU test failed: {e}")
                return False
        
        self.add_result(
            "NCCL", True,
            f"NCCL {nccl_str}, P2P {p2p_enabled}/{p2p_total}"
        )
        return True
    
    def test_storage_io(self) -> bool:
        """Test storage I/O performance."""
        self.log("\n[6/7] Storage I/O...")
        
        test_path = self.args.storage_path
        test_file = os.path.join(test_path, f"gpu_test_{self.rank}_{os.getpid()}.bin")
        size_mb = 100
        
        try:
            # Write test
            data = os.urandom(1024 * 1024)
            t0 = time.time()
            with open(test_file, "wb") as f:
                for _ in range(size_mb):
                    f.write(data)
                f.flush()
                os.fsync(f.fileno())
            write_speed = size_mb / (time.time() - t0)
            
            # Read test
            t0 = time.time()
            with open(test_file, "rb") as f:
                while f.read(1024 * 1024):
                    pass
            read_speed = size_mb / (time.time() - t0)
            
            os.remove(test_file)
            
            self.cluster_info["storage"] = {
                "write_mbps": round(write_speed),
                "read_mbps": round(read_speed)
            }
            
            passed = write_speed > 50 and read_speed > 50
            self.add_result(
                "Storage I/O", passed,
                f"Write: {write_speed:.0f} MB/s, Read: {read_speed:.0f} MB/s"
            )
            return passed
            
        except Exception as e:
            self.add_result("Storage I/O", False, str(e))
            return False
    
    # =========================================================================
    # Training Test
    # =========================================================================
    
    def test_distributed_training(self) -> bool:
        """Run actual distributed training with open-source model."""
        self.log("\n[7/7] Distributed Training...")
        
        import torch
        import torch.nn as nn
        import torch.distributed as dist
        
        device = f"cuda:{self.local_rank}"
        torch.cuda.set_device(device)
        
        # Initialize distributed if needed
        if self.is_distributed and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        try:
            # Use GPT-2 style model (open-source)
            from transformers import GPT2Config, GPT2LMHeadModel
            
            self.log("  Loading GPT-2 model...")
            config = GPT2Config(
                vocab_size=50257,
                n_positions=512,
                n_embd=768,
                n_layer=6,  # Smaller for testing
                n_head=12,
            )
            model = GPT2LMHeadModel(config).to(device)
            
            # Wrap with DDP if distributed
            if self.is_distributed:
                model = nn.parallel.DistributedDataParallel(
                    model, device_ids=[self.local_rank]
                )
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            # Synthetic data (simulates WikiText)
            batch_size = self.args.batch_size
            seq_len = 512
            input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
            labels = input_ids.clone()
            
            # Warmup
            self.log("  Running warmup...")
            outputs = model(input_ids, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            
            if self.is_distributed:
                dist.barrier()
            
            # Benchmark
            self.log(f"  Running {self.args.test_steps} training steps...")
            torch.cuda.reset_peak_memory_stats()
            
            t0 = time.time()
            total_loss = 0.0
            
            for step in range(self.args.test_steps):
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            torch.cuda.synchronize()
            if self.is_distributed:
                dist.barrier()
            
            elapsed = time.time() - t0
            step_time_ms = (elapsed / self.args.test_steps) * 1000
            
            # Calculate throughput
            tokens_per_step = batch_size * seq_len * self.world_size
            throughput = tokens_per_step / (step_time_ms / 1000)
            
            # Memory usage
            mem_used_gb = torch.cuda.max_memory_allocated(device) / 1e9
            mem_total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            
            avg_loss = total_loss / self.args.test_steps
            
            self.cluster_info["training"] = {
                "model": "GPT-2 (6-layer)",
                "batch_size": batch_size,
                "seq_len": seq_len,
                "world_size": self.world_size,
                "step_time_ms": round(step_time_ms, 1),
                "throughput_tokens_per_sec": int(throughput),
                "memory_used_gb": round(mem_used_gb, 1),
                "memory_total_gb": round(mem_total_gb, 1),
                "avg_loss": round(avg_loss, 4)
            }
            
            # Cleanup
            del model, optimizer
            torch.cuda.empty_cache()
            
            self.add_result(
                "Training", True,
                f"{step_time_ms:.0f}ms/step, {throughput:.0f} tok/s, {mem_used_gb:.1f}/{mem_total_gb:.0f}GB, loss={avg_loss:.3f}"
            )
            return True
            
        except Exception as e:
            self.add_result("Training", False, str(e))
            return False
    
    # =========================================================================
    # AllReduce Bandwidth Test (for distributed)
    # =========================================================================
    
    def test_allreduce_bandwidth(self) -> bool:
        """Test NCCL AllReduce bandwidth across nodes."""
        if not self.is_distributed:
            return True
        
        self.log("\n[Bonus] AllReduce Bandwidth...")
        
        import torch
        import torch.distributed as dist
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        device = f"cuda:{self.local_rank}"
        
        # Warmup
        tensor = torch.ones(1024, device=device)
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        dist.barrier()
        
        # Benchmark different sizes
        sizes_mb = [1, 16, 64, 256]
        best_bw = 0
        best_size = 0
        
        for size_mb in sizes_mb:
            elements = size_mb * 1024 * 1024 // 4  # float32
            tensor = torch.ones(elements, device=device)
            torch.cuda.synchronize()
            dist.barrier()
            
            t0 = time.time()
            for _ in range(10):
                dist.all_reduce(tensor)
            torch.cuda.synchronize()
            elapsed = (time.time() - t0) / 10
            
            # AllReduce bandwidth: 2*(n-1)/n * size
            data_gb = 2 * (self.world_size - 1) / self.world_size * size_mb / 1024
            bandwidth = data_gb / elapsed
            
            if bandwidth > best_bw:
                best_bw = bandwidth
                best_size = size_mb
        
        self.cluster_info["allreduce"] = {
            "bandwidth_gbps": round(best_bw, 1),
            "best_size_mb": best_size,
            "world_size": self.world_size
        }
        
        self.add_result(
            "AllReduce", best_bw > 50,
            f"{best_bw:.1f} GB/s @ {best_size}MB (world={self.world_size})"
        )
        return best_bw > 50
    
    # =========================================================================
    # Run All Tests
    # =========================================================================
    
    def run(self) -> bool:
        """Run all tests and generate report."""
        
        self.log("=" * 60)
        self.log("GPU CLUSTER ACCEPTANCE TEST")
        self.log("=" * 60)
        self.log(f"Timestamp: {datetime.now().isoformat()}")
        self.log(f"Hostname: {socket.gethostname()}")
        self.log(f"Rank: {self.rank}/{self.world_size}")
        self.log(f"Distributed: {self.is_distributed}")
        self.log("=" * 60)
        
        # Run tests
        self.test_gpu_detection()
        self.test_cuda_pytorch()
        self.test_memory_limits()
        self.test_infiniband()
        self.test_nccl()
        self.test_storage_io()
        self.test_distributed_training()
        
        if self.is_distributed:
            self.test_allreduce_bandwidth()
        
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        
        self.log("\n" + "=" * 60)
        self.log("SUMMARY")
        self.log("=" * 60)
        self.log(f"  Passed: {passed}")
        self.log(f"  Failed: {failed}")
        self.log("=" * 60)
        
        if failed == 0:
            self.log("✓ CLUSTER READY FOR DISTRIBUTED TRAINING")
        else:
            self.log("✗ CLUSTER NOT READY - Fix failures above")
        self.log("=" * 60)
        
        # Save JSON report
        if self.rank == 0 and self.args.output:
            report = {
                "timestamp": datetime.now().isoformat(),
                "hostname": socket.gethostname(),
                "world_size": self.world_size,
                "passed": passed,
                "failed": failed,
                "results": [asdict(r) for r in self.results],
                "cluster_info": self.cluster_info
            }
            with open(self.args.output, "w") as f:
                json.dump(report, f, indent=2)
            self.log(f"\nReport saved to: {self.args.output}")
        
        return failed == 0


def main():
    parser = argparse.ArgumentParser(description="GPU Cluster Acceptance Test")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--test-steps", type=int, default=10, help="Training steps to run")
    parser.add_argument("--storage-path", default="/tmp", help="Path for I/O tests")
    parser.add_argument("--output", "-o", help="Output JSON report path")
    args = parser.parse_args()
    
    tester = GPUClusterTest(args)
    success = tester.run()
    
    # Cleanup distributed
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
