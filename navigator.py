#!/usr/bin/env python3
"""
ChattGptOrinNano - Console Menu Navigation System

This is the main navigation menu for the ChattGptOrinNano project.
It provides a console-driven interface to execute AI/ML scripts and applications
on the Jetson Orin Nano development kit.

Features:
- Auto-launch on console login (with 10-second escape option)
- Browse and execute scripts from the scripts/ directory
- Set up to 5 favorites accessible via F1-F5 keys
- System information display
- Process management to ensure only one script runs at a time
"""

import os
import sys
import time
import json
import subprocess
import threading
import termios
import tty
import select
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import signal

# Add lib directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

try:
    from lib.jetson_info import jetson_info
except ImportError:
    # Fallback if lib import fails
    jetson_info = None


class Colors:
    """ANSI color codes for console output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'


class ScriptInfo:
    """Information about a script in the scripts directory."""
    
    def __init__(self, name: str, path: str, description: str = ""):
        self.name = name
        self.path = path
        self.description = description
        self.is_favorite = False


class MenuNavigator:
    """Main navigation menu class."""
    
    def __init__(self):
        self.scripts_dir = Path(__file__).parent / "scripts"
        self.config_file = Path(__file__).parent / ".menu_config.json"
        self.scripts: List[ScriptInfo] = []
        self.favorites: Dict[int, Optional[ScriptInfo]] = {i: None for i in range(1, 6)}
        self.current_process: Optional[subprocess.Popen] = None
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.load_config()
        self.scan_scripts()
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n{Colors.YELLOW}Received signal {signum}. Shutting down gracefully...{Colors.RESET}")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up running processes."""
        if self.current_process and self.current_process.poll() is None:
            print(f"{Colors.YELLOW}Terminating running script...{Colors.RESET}")
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
    
    def load_config(self):
        """Load menu configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Load favorites mapping
                    if 'favorites' in config:
                        for key, script_name in config['favorites'].items():
                            if script_name:
                                # Find the corresponding ScriptInfo object
                                script_info = next((s for s in self.scripts if s.name == script_name), None)
                                if script_info:
                                    self.favorites[int(key)] = script_info
                                else:
                                    print(f"{Colors.YELLOW}Warning: Script '{script_name}' not found in scripts directory.{Colors.RESET}")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"{Colors.YELLOW}Warning: Could not load config file: {e}{Colors.RESET}")
    
    def save_config(self):
        """Save menu configuration to file."""
        try:
            config = {
                'favorites': {
                    str(k): v.name if v else None 
                    for k, v in self.favorites.items()
                }
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"{Colors.RED}Error saving config: {e}{Colors.RESET}")
    
    def scan_scripts(self):
        """Scan the scripts directory for available scripts."""
        self.scripts.clear()
        
        if not self.scripts_dir.exists():
            self.scripts_dir.mkdir(exist_ok=True)
            return
        
        for item in self.scripts_dir.iterdir():
            if item.is_dir():
                # Look for Python scripts in subdirectories
                python_files = list(item.glob("*.py"))
                if python_files:
                    main_script = None
                    # Look for main.py first, then any .py file
                    for py_file in python_files:
                        if py_file.name == "main.py":
                            main_script = py_file
                            break
                    if not main_script:
                        main_script = python_files[0]
                    
                    # Try to read description from script docstring or README
                    description = self.get_script_description(item, main_script)
                    
                    script_info = ScriptInfo(
                        name=item.name,
                        path=str(main_script),
                        description=description
                    )
                    
                    # Check if it's a favorite
                    for fav_key, fav_script in self.favorites.items():
                        if fav_script and fav_script == script_info.name:
                            script_info.is_favorite = True
                            self.favorites[fav_key] = script_info
                            break
                    
                    self.scripts.append(script_info)
        
        # Sort scripts alphabetically
        self.scripts.sort(key=lambda x: x.name.lower())
    
    def get_script_description(self, script_dir: Path, script_file: Path) -> str:
        """Extract description from script file or README."""
        # Try to read docstring from Python file
        try:
            with open(script_file, 'r') as f:
                content = f.read()
                # Look for module docstring
                lines = content.split('\n')
                in_docstring = False
                docstring_lines = []
                for line in lines:
                    stripped = line.strip()
                    if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                        in_docstring = True
                        if len(stripped) > 3:
                            docstring_lines.append(stripped[3:])
                    elif in_docstring:
                        if stripped.endswith('"""') or stripped.endswith("'''"):
                            if len(stripped) > 3:
                                docstring_lines.append(stripped[:-3])
                            break
                        else:
                            docstring_lines.append(line)
                
                if docstring_lines:
                    return ' '.join(docstring_lines).strip()[:100]
        except Exception:
            pass
        
        # Try to read README in script directory
        for readme_name in ['README.md', 'README.txt', 'readme.md', 'readme.txt']:
            readme_path = script_dir / readme_name
            if readme_path.exists():
                try:
                    with open(readme_path, 'r') as f:
                        content = f.read().strip()
                        if content:
                            # Get first non-empty line that's not a title
                            lines = [line.strip() for line in content.split('\n') if line.strip()]
                            for line in lines:
                                if not line.startswith('#') and len(line) > 10:
                                    return line[:100]
                except Exception:
                    pass
        
        return "No description available"
    
    def clear_screen(self):
        """Clear the console screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_header(self):
        """Print the application header."""
        print(f"{Colors.CYAN}{Colors.BOLD}")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "JETSON ORIN NANO AI SCRIPT NAVIGATOR" + " " * 21 + "║")
        print("║" + " " * 25 + "ChattGptOrinNano Project" + " " * 29 + "║")
        print("╚" + "═" * 78 + "╝")
        print(f"{Colors.RESET}")
    
    def print_system_info_brief(self):
        """Print brief system information."""
        if jetson_info:
            info = jetson_info.get_system_info()
            print(f"{Colors.GRAY}CUDA: {info.get('cuda_version', 'Unknown')} | "
                  f"PyTorch: {info.get('pytorch_version', 'Unknown')} | "
                  f"Python: {info.get('python_version', 'Unknown')}{Colors.RESET}")
        print()
    
    def print_favorites(self):
        """Print the favorites section."""
        print(f"{Colors.GREEN}{Colors.BOLD}FAVORITES (F1-F5):{Colors.RESET}")
        for i in range(1, 6):
            fav = self.favorites[i]
            if fav:
                status = f"{Colors.GREEN}[F{i}] {fav.name}{Colors.RESET}"
            else:
                status = f"{Colors.GRAY}[F{i}] (empty){Colors.RESET}"
            print(f"  {status}")
        print()
    
    def print_scripts_list(self, selected_index: int = -1):
        """Print the list of available scripts."""
        print(f"{Colors.BLUE}{Colors.BOLD}AVAILABLE SCRIPTS:{Colors.RESET}")
        if not self.scripts:
            print(f"  {Colors.GRAY}No scripts found in scripts/ directory{Colors.RESET}")
            print(f"  {Colors.GRAY}Create subdirectories with Python scripts to get started{Colors.RESET}")
        else:
            for i, script in enumerate(self.scripts):
                prefix = f"{Colors.WHITE}►{Colors.RESET}" if i == selected_index else " "
                fav_marker = f"{Colors.YELLOW}★{Colors.RESET}" if script.is_favorite else " "
                print(f"  {prefix} {fav_marker} {Colors.WHITE}{i+1:2d}.{Colors.RESET} {Colors.CYAN}{script.name}{Colors.RESET}")
                if script.description:
                    desc = script.description[:70] + "..." if len(script.description) > 70 else script.description
                    print(f"       {Colors.GRAY}{desc}{Colors.RESET}")
        print()
    
    def print_menu_options(self):
        """Print menu options."""
        print(f"{Colors.YELLOW}{Colors.BOLD}MENU OPTIONS:{Colors.RESET}")
        print(f"  {Colors.WHITE}1-9{Colors.RESET}     : Select and run script")
        print(f"  {Colors.WHITE}F1-F5{Colors.RESET}   : Run favorite scripts")
        print(f"  {Colors.WHITE}s{Colors.RESET}       : Set/remove favorites")
        print(f"  {Colors.WHITE}r{Colors.RESET}       : Refresh script list")
        print(f"  {Colors.WHITE}i{Colors.RESET}       : Show detailed system information")
        print(f"  {Colors.WHITE}h{Colors.RESET}       : Show help")
        print(f"  {Colors.WHITE}q{Colors.RESET}       : Quit")
        if self.current_process and self.current_process.poll() is None:
            print(f"  {Colors.RED}k{Colors.RESET}       : Kill running script")
        print()
    
    def get_key(self) -> str:
        """Get a single keypress from the user."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            
            # Handle escape sequences (function keys)
            if ch == '\x1b':
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    return f"ESC[{ch3}"
                elif ch2 == 'O':
                    ch3 = sys.stdin.read(1)
                    return f"ESCO{ch3}"
            
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def run_script(self, script: ScriptInfo):
        """Run a selected script."""
        if self.current_process and self.current_process.poll() is None:
            print(f"{Colors.RED}A script is already running. Kill it first (press 'k') or wait for it to finish.{Colors.RESET}")
            input("Press Enter to continue...")
            return
        
        print(f"{Colors.GREEN}Running script: {script.name}{Colors.RESET}")
        print(f"{Colors.GRAY}Path: {script.path}{Colors.RESET}")
        print(f"{Colors.YELLOW}Press Ctrl+C to interrupt the script{Colors.RESET}")
        print("-" * 80)
        
        try:
            # Change to script directory
            script_dir = os.path.dirname(script.path)
            
            # Check if there's a virtual environment
            venv_paths = [
                os.path.join(script_dir, 'venv', 'bin', 'python'),
                os.path.join(script_dir, '.venv', 'bin', 'python'),
                os.path.join(script_dir, 'env', 'bin', 'python'),
            ]
            
            python_cmd = 'python3'
            for venv_path in venv_paths:
                if os.path.exists(venv_path):
                    python_cmd = venv_path
                    print(f"{Colors.GREEN}Using virtual environment: {venv_path}{Colors.RESET}")
                    break
            
            # Run the script
            self.current_process = subprocess.Popen(
                [python_cmd, os.path.basename(script.path)],
                cwd=script_dir,
                stdout=sys.stdout,
                stderr=sys.stderr,
                stdin=sys.stdin
            )
            
            # Wait for completion
            returncode = self.current_process.wait()
            self.current_process = None
            
            print("-" * 80)
            if returncode == 0:
                print(f"{Colors.GREEN}Script completed successfully{Colors.RESET}")
            else:
                print(f"{Colors.RED}Script exited with code {returncode}{Colors.RESET}")
                
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Script interrupted by user{Colors.RESET}")
            if self.current_process:
                self.current_process.terminate()
                self.current_process = None
        except Exception as e:
            print(f"{Colors.RED}Error running script: {e}{Colors.RESET}")
            self.current_process = None
        
        input(f"\n{Colors.CYAN}Press Enter to return to menu...{Colors.RESET}")
    
    def manage_favorites(self):
        """Manage favorite scripts."""
        while True:
            self.clear_screen()
            self.print_header()
            print(f"{Colors.GREEN}{Colors.BOLD}MANAGE FAVORITES{Colors.RESET}\n")
            
            self.print_favorites()
            self.print_scripts_list()
            
            print(f"{Colors.YELLOW}Options:{Colors.RESET}")
            print("  1-5 : Set favorite slot")
            print("  c   : Clear a favorite slot")
            print("  b   : Back to main menu")
            print()
            
            choice = input(f"{Colors.CYAN}Enter choice: {Colors.RESET}").strip().lower()
            
            if choice == 'b':
                break
            elif choice == 'c':
                slot = input(f"{Colors.CYAN}Clear which favorite slot (1-5)? {Colors.RESET}").strip()
                try:
                    slot_num = int(slot)
                    if 1 <= slot_num <= 5:
                        if self.favorites[slot_num]:
                            self.favorites[slot_num].is_favorite = False
                        self.favorites[slot_num] = None
                        self.save_config()
                        print(f"{Colors.GREEN}Favorite slot F{slot_num} cleared{Colors.RESET}")
                        time.sleep(1)
                except ValueError:
                    print(f"{Colors.RED}Invalid slot number{Colors.RESET}")
                    time.sleep(1)
            elif choice in '12345':
                slot_num = int(choice)
                if not self.scripts:
                    print(f"{Colors.RED}No scripts available{Colors.RESET}")
                    time.sleep(1)
                    continue
                
                print(f"{Colors.CYAN}Select script for F{slot_num}:{Colors.RESET}")
                for i, script in enumerate(self.scripts):
                    print(f"  {i+1}. {script.name}")
                
                script_choice = input(f"{Colors.CYAN}Enter script number: {Colors.RESET}").strip()
                try:
                    script_index = int(script_choice) - 1
                    if 0 <= script_index < len(self.scripts):
                        # Clear old favorite
                        if self.favorites[slot_num]:
                            self.favorites[slot_num].is_favorite = False
                        
                        # Set new favorite
                        selected_script = self.scripts[script_index]
                        self.favorites[slot_num] = selected_script
                        selected_script.is_favorite = True
                        self.save_config()
                        
                        print(f"{Colors.GREEN}Set {selected_script.name} as favorite F{slot_num}{Colors.RESET}")
                        time.sleep(1)
                except (ValueError, IndexError):
                    print(f"{Colors.RED}Invalid script number{Colors.RESET}")
                    time.sleep(1)
    
    def show_help(self):
        """Show detailed help information."""
        self.clear_screen()
        self.print_header()
        
        print(f"{Colors.GREEN}{Colors.BOLD}HELP - ChattGptOrinNano Navigator{Colors.RESET}\n")
        
        print(f"{Colors.BLUE}{Colors.BOLD}ABOUT:{Colors.RESET}")
        print("This menu system provides easy access to AI/ML demonstration scripts")
        print("designed for the Jetson Orin Nano 8GB development kit.\n")
        
        print(f"{Colors.BLUE}{Colors.BOLD}DIRECTORY STRUCTURE:{Colors.RESET}")
        print("scripts/")
        print("├── script1/")
        print("│   ├── main.py          # Main script file")
        print("│   ├── requirements.txt # Dependencies")
        print("│   ├── venv/           # Virtual environment (optional)")
        print("│   └── README.md       # Script description")
        print("└── script2/")
        print("    └── ...\n")
        
        print(f"{Colors.BLUE}{Colors.BOLD}FEATURES:{Colors.RESET}")
        print("• Auto-discovery of Python scripts in subdirectories")
        print("• Virtual environment support (venv/, .venv/, env/)")
        print("• Favorite scripts accessible via F1-F5 keys")
        print("• Process management (only one script runs at a time)")
        print("• System information display (CUDA, PyTorch versions)")
        print("• Automatic startup with escape option\n")
        
        print(f"{Colors.BLUE}{Colors.BOLD}NAVIGATION:{Colors.RESET}")
        print("• Use number keys (1-9) to select and run scripts")
        print("• Use F1-F5 to quickly run favorite scripts")
        print("• Press 's' to manage favorites")
        print("• Press 'r' to refresh the script list")
        print("• Press 'i' for detailed system information")
        print("• Press 'q' to quit the navigator\n")
        
        print(f"{Colors.BLUE}{Colors.BOLD}VIRTUAL ENVIRONMENTS:{Colors.RESET}")
        print("Scripts can use virtual environments for dependency isolation.")
        print("Create venv/, .venv/, or env/ in your script directory.")
        print("The navigator will automatically detect and use them.\n")
        
        input(f"{Colors.CYAN}Press Enter to return to menu...{Colors.RESET}")
    
    def show_detailed_system_info(self):
        """Show detailed system information."""
        self.clear_screen()
        self.print_header()
        
        if jetson_info:
            jetson_info.print_system_info()
        else:
            print(f"{Colors.RED}System information module not available{Colors.RESET}")
        
        # Additional system info
        print(f"{Colors.BLUE}{Colors.BOLD}ADDITIONAL SYSTEM INFO:{Colors.RESET}")
        try:
            # Memory info
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        mem_total = line.split()[1]
                        print(f"Total Memory:     {int(mem_total)//1024} MB")
                        break
        except:
            pass
        
        try:
            # CPU info
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read()
                cpu_count = content.count('processor')
                print(f"CPU Cores:        {cpu_count}")
        except:
            pass
        
        print()
        input(f"{Colors.CYAN}Press Enter to return to menu...{Colors.RESET}")
    
    def startup_countdown(self) -> bool:
        """Show startup countdown with escape option."""
        print(f"{Colors.CYAN}{Colors.BOLD}")
        print("╔" + "═" * 60 + "╗")
        print("║" + " " * 10 + "ChattGptOrinNano AI Script Navigator" + " " * 13 + "║")
        print("║" + " " * 60 + "║")
        print("║" + " " * 15 + "Press ESC to skip auto-launch" + " " * 16 + "║")
        print("╚" + "═" * 60 + "╝")
        print(f"{Colors.RESET}")
        
        for i in range(10, 0, -1):
            print(f"\r{Colors.YELLOW}Auto-launching in {i} seconds... {Colors.RESET}", end='', flush=True)
            
            # Check for escape key
            ready, _, _ = select.select([sys.stdin], [], [], 1)
            if ready:
                char = sys.stdin.read(1)
                if char == '\x1b':  # ESC key
                    print(f"\n{Colors.GREEN}Auto-launch cancelled by user{Colors.RESET}")
                    return False
        
        print(f"\n{Colors.GREEN}Launching navigator...{Colors.RESET}")
        time.sleep(1)
        return True
    
    def run(self, show_countdown: bool = True):
        """Main menu loop."""
        # Show countdown if requested
        if show_countdown and not self.startup_countdown():
            return
        
        while self.running:
            try:
                self.clear_screen()
                self.print_header()
                self.print_system_info_brief()
                self.print_favorites()
                self.print_scripts_list()
                self.print_menu_options()
                
                print(f"{Colors.CYAN}Enter choice: {Colors.RESET}", end='', flush=True)
                
                choice = self.get_key().lower()
                
                if choice == 'q':
                    self.running = False
                    print(f"\n{Colors.GREEN}Goodbye!{Colors.RESET}")
                    
                elif choice == 'r':
                    print(f"\n{Colors.YELLOW}Refreshing script list...{Colors.RESET}")
                    self.scan_scripts()
                    time.sleep(1)
                    
                elif choice == 's':
                    self.manage_favorites()
                    
                elif choice == 'h':
                    self.show_help()
                    
                elif choice == 'i':
                    self.show_detailed_system_info()
                    
                elif choice == 'k':
                    if self.current_process and self.current_process.poll() is None:
                        print(f"\n{Colors.YELLOW}Killing running script...{Colors.RESET}")
                        self.current_process.terminate()
                        try:
                            self.current_process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self.current_process.kill()
                        self.current_process = None
                        print(f"{Colors.GREEN}Script terminated{Colors.RESET}")
                        time.sleep(1)
                    
                elif choice.startswith('esco'):  # Function keys F1-F5
                    func_key = choice[4]
                    if func_key in 'PQRST':  # F1-F5
                        fav_num = ord(func_key) - ord('P') + 1
                        if 1 <= fav_num <= 5 and self.favorites[fav_num]:
                            self.run_script(self.favorites[fav_num])
                        else:
                            print(f"\n{Colors.YELLOW}No script assigned to F{fav_num}{Colors.RESET}")
                            time.sleep(1)
                            
                elif choice.isdigit():
                    script_num = int(choice)
                    if 1 <= script_num <= len(self.scripts):
                        self.run_script(self.scripts[script_num - 1])
                    else:
                        print(f"\n{Colors.RED}Invalid script number{Colors.RESET}")
                        time.sleep(1)
                        
                else:
                    print(f"\n{Colors.RED}Invalid choice. Press 'h' for help.{Colors.RESET}")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Use 'q' to quit or 'k' to kill running script{Colors.RESET}")
                time.sleep(1)
            except Exception as e:
                print(f"\n{Colors.RED}Unexpected error: {e}{Colors.RESET}")
                time.sleep(2)
        
        self.cleanup()


def main():
    """Main entry point."""
    try:
        navigator = MenuNavigator()
        
        # Check if this is an auto-launch
        auto_launch = len(sys.argv) == 1 or '--auto' in sys.argv
        
        navigator.run(show_countdown=auto_launch)
        
    except Exception as e:
        print(f"{Colors.RED}Fatal error: {e}{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
