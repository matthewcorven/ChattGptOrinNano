#!/bin/bash
# Setup script for ChattGptOrinNano Navigator
# This script configures automatic startup for the navigation menu

echo "ChattGptOrinNano Navigator Setup"
echo "================================="
echo ""

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_START_SCRIPT="$PROJECT_DIR/auto_start.sh"

echo "Project directory: $PROJECT_DIR"
echo ""

# Function to add auto-start to a shell config file
add_to_shell_config() {
    local config_file="$1"
    local config_name="$2"
    
    if [[ -f "$config_file" ]]; then
        # Check if already configured
        if grep -q "# ChattGptOrinNano Navigator Auto-Start" "$config_file"; then
            echo "✓ $config_name already configured"
            return 0
        fi
        
        echo "Configuring $config_name..."
        
        # Add the auto-start configuration
        cat >> "$config_file" << EOF

# ChattGptOrinNano Navigator Auto-Start
# Added by setup script on $(date)
if [[ -f "$AUTO_START_SCRIPT" ]]; then
    source "$AUTO_START_SCRIPT"
fi
EOF
        echo "✓ $config_name configured"
        return 0
    else
        echo "✗ $config_name not found"
        return 1
    fi
}

echo "Configuring automatic startup..."
echo ""

# Configure for different shells
add_to_shell_config "$HOME/.bashrc" ".bashrc"
add_to_shell_config "$HOME/.bash_profile" ".bash_profile"
add_to_shell_config "$HOME/.profile" ".profile"

echo ""
echo "Setup complete!"
echo ""
echo "The ChattGptOrinNano Navigator will now automatically start when you"
echo "log in to a console session. You can:"
echo ""
echo "• Press ESC during the 10-second countdown to skip auto-launch"
echo "• Run manually with: python3 $PROJECT_DIR/navigator.py"
echo "• Disable auto-start by commenting out the lines in your shell config"
echo ""
echo "To test the setup, open a new terminal or run:"
echo "  source ~/.bashrc"
echo ""

# Create some example scripts to demonstrate the system
echo "Creating example scripts..."
echo ""

# Create example script directories
mkdir -p "$PROJECT_DIR/scripts/hello_jetson"
mkdir -p "$PROJECT_DIR/scripts/system_info"
mkdir -p "$PROJECT_DIR/scripts/pytorch_test"

# Mark all Python scripts in the scripts directory as executable
find "$PROJECT_DIR/scripts" -name '*.py' -exec chmod +x {} \;
# Hello Jetson example
cat > "$PROJECT_DIR/scripts/hello_jetson/main.py" << 'EOF'
#!/usr/bin/env python3
"""
Hello Jetson - A simple welcome script for the Jetson Orin Nano.
This script demonstrates basic system interaction and provides a welcome message.
"""

import os
import sys
import time
import platform

def print_banner():
    """Print a welcome banner."""
    print("=" * 60)
    print("   HELLO FROM JETSON ORIN NANO 8GB DEVELOPER KIT!")
    print("=" * 60)
    print()

def show_system_info():
    """Display basic system information."""
    print("🖥️  System Information:")
    print(f"   Platform: {platform.platform()}")
    print(f"   Python Version: {sys.version.split()[0]}")
    print(f"   Architecture: {platform.architecture()[0]}")
    print()
    
    # Try to get some Jetson-specific info
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
            print(f"   Device Model: {model}")
    except:
        print("   Device Model: Unknown")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   Memory: {memory.total // (1024**3)} GB total, {memory.available // (1024**3)} GB available")
    except ImportError:
        print("   Memory: Install psutil for memory info")
    
    print()

def main():
    """Main function."""
    print_banner()
    show_system_info()
    
    print("🚀 Welcome to the ChattGptOrinNano project!")
    print()
    print("This is an example script that demonstrates:")
    print("• Basic Python script structure")
    print("• System information gathering")
    print("• Integration with the navigation menu")
    print()
    print("You can create your own AI/ML scripts in the scripts/ directory.")
    print("Each script should be in its own subdirectory with a main.py file.")
    print()
    
    # Simple interactive demo
    try:
        name = input("What's your name? ").strip()
        if name:
            print(f"\nHello, {name}! Welcome to Jetson AI development! 🤖")
        else:
            print("\nHello, anonymous developer! Welcome to Jetson AI development! 🤖")
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
    except EOFError:
        print("\nGoodbye! 👋")
    
    print("\nScript completed. Returning to navigator...")
    time.sleep(2)

if __name__ == "__main__":
    main()
EOF

# System Info example
cat > "$PROJECT_DIR/scripts/system_info/main.py" << 'EOF'
#!/usr/bin/env python3
"""
System Information Display - Comprehensive system information for Jetson Orin Nano.
This script provides detailed hardware and software information useful for AI development.
"""

import sys
import os
import subprocess
import platform
import time

# Add parent directories to path for lib imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from lib.jetson_info import jetson_info
except ImportError:
    jetson_info = None

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def get_gpu_info():
    """Get detailed GPU information."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,power.draw', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return "GPU information not available"

def get_cpu_info():
    """Get CPU information."""
    info = {}
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('model name'):
                    info['model'] = line.split(':')[1].strip()
                    break
        
        info['cores'] = os.cpu_count()
        info['architecture'] = platform.architecture()[0]
        
        # Get CPU frequency
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'cpu MHz' in line:
                        info['frequency'] = line.split(':')[1].strip() + " MHz"
                        break
        except:
            pass
            
    except:
        pass
    
    return info

def get_memory_info():
    """Get memory information."""
    try:
        with open('/proc/meminfo', 'r') as f:
            mem_info = {}
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    mem_info[key.strip()] = value.strip()
            
            total = int(mem_info.get('MemTotal', '0').split()[0]) // 1024
            free = int(mem_info.get('MemFree', '0').split()[0]) // 1024
            available = int(mem_info.get('MemAvailable', '0').split()[0]) // 1024
            used = total - free
            
            return {
                'total': total,
                'used': used,
                'free': free,
                'available': available
            }
    except:
        return {}

def get_storage_info():
    """Get storage information."""
    try:
        result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                return {
                    'filesystem': parts[0],
                    'size': parts[1],
                    'used': parts[2],
                    'available': parts[3],
                    'use_percent': parts[4]
                }
    except:
        pass
    return {}

def main():
    """Main function to display comprehensive system information."""
    print("🖥️  JETSON ORIN NANO SYSTEM INFORMATION")
    print("Generated on:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    # Platform Information
    print_section_header("PLATFORM INFORMATION")
    print(f"System: {platform.system()}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
            print(f"Device Model: {model}")
    except:
        print("Device Model: Unknown")
    
    # CPU Information
    print_section_header("CPU INFORMATION")
    cpu_info = get_cpu_info()
    print(f"Model: {cpu_info.get('model', 'Unknown')}")
    print(f"Cores: {cpu_info.get('cores', 'Unknown')}")
    print(f"Architecture: {cpu_info.get('architecture', 'Unknown')}")
    if 'frequency' in cpu_info:
        print(f"Frequency: {cpu_info['frequency']}")
    
    # Memory Information
    print_section_header("MEMORY INFORMATION")
    mem_info = get_memory_info()
    if mem_info:
        print(f"Total Memory: {mem_info['total']} MB ({mem_info['total']/1024:.1f} GB)")
        print(f"Used Memory: {mem_info['used']} MB ({mem_info['used']/1024:.1f} GB)")
        print(f"Free Memory: {mem_info['free']} MB ({mem_info['free']/1024:.1f} GB)")
        print(f"Available Memory: {mem_info['available']} MB ({mem_info['available']/1024:.1f} GB)")
        print(f"Usage: {(mem_info['used']/mem_info['total']*100):.1f}%")
    else:
        print("Memory information not available")
    
    # Storage Information
    print_section_header("STORAGE INFORMATION")
    storage_info = get_storage_info()
    if storage_info:
        print(f"Filesystem: {storage_info['filesystem']}")
        print(f"Total Size: {storage_info['size']}")
        print(f"Used: {storage_info['used']}")
        print(f"Available: {storage_info['available']}")
        print(f"Usage: {storage_info['use_percent']}")
    else:
        print("Storage information not available")
    
    # GPU Information
    print_section_header("GPU INFORMATION")
    gpu_info = get_gpu_info()
    if "not available" not in gpu_info:
        print("GPU Details:")
        print(gpu_info)
    else:
        print(gpu_info)
    
    # AI/ML Framework Information
    print_section_header("AI/ML FRAMEWORK INFORMATION")
    if jetson_info:
        info = jetson_info.get_system_info()
        print(f"CUDA Version: {info.get('cuda_version', 'Unknown')}")
        print(f"PyTorch Version: {info.get('pytorch_version', 'Unknown')}")
        print(f"JetPack Version: {info.get('jetpack_version', 'Unknown')}")
    else:
        print("AI/ML framework information not available")
    
    # Python Environment
    print_section_header("PYTHON ENVIRONMENT")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Path: {sys.prefix}")
    
    # Try to show some important AI/ML packages
    packages = ['numpy', 'tensorflow', 'torch', 'cv2', 'sklearn', 'pandas']
    print("\nInstalled AI/ML Packages:")
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown version')
            print(f"  ✓ {package}: {version}")
        except ImportError:
            print(f"  ✗ {package}: Not installed")
    
    print("\n" + "=" * 60)
    print("System information scan complete!")
    print("=" * 60)
    
    input("\nPress Enter to return to navigator...")

if __name__ == "__main__":
    main()
EOF

# PyTorch Test example
cat > "$PROJECT_DIR/scripts/pytorch_test/main.py" << 'EOF'
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
EOF

echo "✓ Created example scripts:"
echo "  - hello_jetson: Basic welcome and system info"
echo "  - system_info: Comprehensive system information"
echo "  - pytorch_test: PyTorch and CUDA functionality test"
echo ""
echo "You can now run the navigator to test these scripts!"
echo "Run: python3 $PROJECT_DIR/navigator.py"
