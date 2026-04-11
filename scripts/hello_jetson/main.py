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
