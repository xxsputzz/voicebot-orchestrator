"""
Debug Sprint 3 CLI subprocess issues
"""
import subprocess
import sys
from pathlib import Path

def debug_command(cmd_args):
    """Debug a specific command."""
    print(f"Testing: {' '.join(cmd_args)}")
    try:
        result = subprocess.run(
            cmd_args, 
            capture_output=True, 
            text=True, 
            cwd=Path(__file__).parent,
            timeout=30
        )
        print(f"Return code: {result.returncode}")
        print(f"STDOUT length: {len(result.stdout)}")
        print(f"STDERR length: {len(result.stderr)}")
        
        if result.stdout:
            print("STDOUT preview:")
            print(result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print("-" * 50)
        return result.returncode == 0
    except Exception as e:
        print(f"Exception: {e}")
        print("-" * 50)
        return False

# Test commands
commands = [
    [sys.executable, '-m', 'voicebot_orchestrator.sprint3_cli', '--help'],
    [sys.executable, '-m', 'voicebot_orchestrator.sprint3_cli', 'orchestrator-log', '--metrics']
]

for cmd in commands:
    debug_command(cmd)
