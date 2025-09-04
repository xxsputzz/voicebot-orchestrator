#!/usr/bin/env python3
"""
Debug STT Service Launcher
=========================
Launch STT service exactly like Enhanced Service Manager but show logs
"""
import subprocess
import sys
from pathlib import Path
import time

# Get the project root and virtual environment Python
project_root = Path(__file__).parent
venv_python = project_root / ".venv" / "Scripts" / "python.exe"

if venv_python.exists():
    python_exe = str(venv_python)
else:
    python_exe = sys.executable

# Service script path  
script_path = Path(__file__).parent / "aws_microservices" / "stt_whisper_service.py"

print(f"🔧 Debug STT Service Launcher")
print(f"Python: {python_exe}")
print(f"Script: {script_path}")
print(f"CWD: {project_root}")
print("-" * 50)

# Start the service process EXACTLY like Enhanced Service Manager
# but WITHOUT capturing stdout/stderr so we can see the logs
process = subprocess.Popen([
    python_exe, str(script_path), "--direct"
], cwd=project_root)  # NO stdout/stderr capture!

print(f"🚀 STT Service started with PID: {process.pid}")
print("📋 Service will be available at: http://localhost:8003")
print("❌ Press Ctrl+C to stop")
print("-" * 50)

try:
    # Wait for the process
    process.wait()
except KeyboardInterrupt:
    print("\n🛑 Stopping service...")
    process.terminate()
    process.wait()
    print("✅ Service stopped")
