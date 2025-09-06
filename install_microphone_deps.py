#!/usr/bin/env python3
"""
Install dependencies for microphone testing suite
"""

import subprocess
import sys

def install_microphone_dependencies():
    """Install required packages for microphone testing"""
    dependencies = [
        "pyaudio",
        "wave", 
        "websockets"
    ]
    
    print("🎤 Installing Microphone Testing Dependencies")
    print("=" * 50)
    
    for dep in dependencies:
        try:
            print(f"📦 Installing {dep}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], capture_output=True, text=True, check=True)
            
            if result.returncode == 0:
                print(f"✅ {dep} installed successfully")
            else:
                print(f"⚠️ {dep} installation had warnings: {result.stderr}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            print(f"   Error output: {e.stderr}")
    
    print("\n🎤 Dependency installation complete!")
    print("💡 Note: If PyAudio installation fails on Windows, you may need to:")
    print("   1. Install Visual Studio Build Tools")
    print("   2. Or download a pre-compiled wheel from:")
    print("      https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")

if __name__ == "__main__":
    install_microphone_dependencies()
