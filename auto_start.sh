#!/bin/bash
# Auto-startup script for ChattGptOrinNano Navigator
# This script should be called from .bashrc or .profile for automatic startup

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if we're in an interactive terminal (not SSH, not script execution)
if [[ $- == *i* ]] && [[ -z "$SSH_CLIENT" ]] && [[ -z "$SSH_TTY" ]] && [[ "$TERM" != "dumb" ]]; then
    # Check if navigator should auto-start (not if it's already running)
    if [[ -z "$CHATGPT_NAVIGATOR_RUNNING" ]]; then
        export CHATGPT_NAVIGATOR_RUNNING=1
        
        echo "ChattGptOrinNano - Jetson Orin Nano AI Script Navigator"
        echo "======================================================"
        
        # Run the navigator with auto-launch countdown
        cd "$SCRIPT_DIR"
        python3 navigator.py --auto
        
        unset CHATGPT_NAVIGATOR_RUNNING
    fi
fi
