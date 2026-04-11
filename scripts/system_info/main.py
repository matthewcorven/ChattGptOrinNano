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
