#!/usr/bin/env python3
"""
PyTorch CUDA Test - Test PyTorch installation and CUDA availability on Jetson Orin Nano.
This script verifies that PyTorch is properly installed and can access GPU acceleration.
"""

import sys
import os
import time

def test_pytorch_import():
    """Test PyTorch import and basic functionality."""
    print("🧪 Testing PyTorch Import...")
    try:
        import torch
        print(f"✓ PyTorch successfully imported")
        print(f"  Version: {torch.__version__}")
        return torch
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        print("  Install PyTorch with: pip install torch torchvision torchaudio")
        return None

def test_cuda_availability(torch):
    """Test CUDA availability and functionality."""
    print("\n🔧 Testing CUDA Availability...")
    
    if torch.cuda.is_available():
        print("✓ CUDA is available")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory // (1024**2)} MB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
        
        return True
    else:
        print("✗ CUDA is not available")
        print("  Running on CPU only")
        return False

def test_tensor_operations(torch, use_cuda):
    """Test basic tensor operations."""
    print("\n🔢 Testing Tensor Operations...")
    
    try:
        # Create test tensors
        device = torch.device("cuda" if use_cuda else "cpu")
        print(f"  Using device: {device}")
        
        # Basic tensor creation
        a = torch.randn(3, 3, device=device)
        b = torch.randn(3, 3, device=device)
        
        print("✓ Tensor creation successful")
        print(f"  Tensor shape: {a.shape}")
        print(f"  Tensor device: {a.device}")
        print(f"  Tensor dtype: {a.dtype}")
        
        # Basic operations
        c = torch.mm(a, b)  # Matrix multiplication
        d = a + b           # Element-wise addition
        e = torch.sum(a)    # Reduction
        
        print("✓ Basic tensor operations successful")
        print(f"  Matrix multiplication result shape: {c.shape}")
        print(f"  Sum result: {e.item():.4f}")
        
        # Move tensors between devices if CUDA is available
        if use_cuda:
            cpu_tensor = a.cpu()
            gpu_tensor = cpu_tensor.cuda()
            print("✓ Device transfer successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Tensor operations failed: {e}")
        return False

def test_neural_network(torch, use_cuda):
    """Test a simple neural network."""
    print("\n🧠 Testing Neural Network...")
    
    try:
        import torch.nn as nn
        import torch.optim as optim
        
        device = torch.device("cuda" if use_cuda else "cpu")
        
        # Define a simple neural network
        class SimpleNet(nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc1 = nn.Linear(10, 50)
                self.fc2 = nn.Linear(50, 20)
                self.fc3 = nn.Linear(20, 1)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        # Create network and move to device
        net = SimpleNet().to(device)
        print(f"✓ Neural network created and moved to {device}")
        
        # Create sample data
        x = torch.randn(32, 10, device=device)  # Batch of 32, 10 features
        y = torch.randn(32, 1, device=device)   # Target values
        
        # Forward pass
        output = net(x)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Compute loss and backward pass
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Backward pass and optimization successful")
        print(f"  Loss: {loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Neural network test failed: {e}")
        return False

def benchmark_performance(torch, use_cuda):
    """Simple performance benchmark."""
    print("\n⚡ Performance Benchmark...")
    
    try:
        device = torch.device("cuda" if use_cuda else "cpu")
        
        # Large matrix multiplication benchmark
        size = 1000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warm up
        for _ in range(5):
            _ = torch.mm(a, b)
        
        if use_cuda:
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        num_ops = 10
        
        for _ in range(num_ops):
            result = torch.mm(a, b)
        
        if use_cuda:
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_ops
        
        print(f"✓ Matrix multiplication benchmark completed")
        print(f"  Matrix size: {size}x{size}")
        print(f"  Average time per operation: {avg_time*1000:.2f} ms")
        print(f"  Operations per second: {1/avg_time:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return False

def main():
    """Main function to run PyTorch tests."""
    print("🚀 PyTorch CUDA Test for Jetson Orin Nano")
    print("=" * 50)
    
    # Test PyTorch import
    torch = test_pytorch_import()
    if not torch:
        print("\nCannot proceed without PyTorch. Install it first.")
        input("\nPress Enter to return to navigator...")
        return
    
    # Test CUDA
    cuda_available = test_cuda_availability(torch)
    
    # Test tensor operations
    tensor_success = test_tensor_operations(torch, cuda_available)
    
    # Test neural network
    if tensor_success:
        nn_success = test_neural_network(torch, cuda_available)
    
    # Performance benchmark
    if tensor_success:
        benchmark_performance(torch, cuda_available)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {'Yes' if cuda_available else 'No'}")
    print(f"Device: {'GPU' if cuda_available else 'CPU'}")
    
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        memory_allocated = torch.cuda.memory_allocated() / (1024**2)
        memory_cached = torch.cuda.memory_reserved() / (1024**2)
        print(f"GPU Memory Used: {memory_allocated:.1f} MB")
        print(f"GPU Memory Cached: {memory_cached:.1f} MB")
    
    print("\n🎉 PyTorch testing completed!")
    
    if cuda_available:
        print("Your Jetson Orin Nano is ready for GPU-accelerated AI/ML workloads!")
    else:
        print("Consider installing CUDA-enabled PyTorch for better performance.")
    
    input("\nPress Enter to return to navigator...")

if __name__ == "__main__":
    main()
