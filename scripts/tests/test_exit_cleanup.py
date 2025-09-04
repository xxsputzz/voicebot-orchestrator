#!/usr/bin/env python3
"""
Test script to demonstrate exit cleanup functionality
"""
import subprocess
import time

def test_exit_cleanup():
    """Test the enhanced CLI exit cleanup feature"""
    print("🧪 Testing Enhanced CLI Exit Cleanup")
    print("=" * 50)
    
    print("Starting CLI and initializing services...")
    
    # Create a script to simulate user input
    test_script = """
init-stt
init-llm mistral
init-tts-kokoro
status
exit
y
"""
    
    try:
        # Run the CLI with simulated input
        process = subprocess.Popen(
            ['.venv/Scripts/python.exe', 'voicebot_orchestrator/enhanced_cli.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='C:/Users/miken/Desktop/Orkestra'
        )
        
        stdout, stderr = process.communicate(input=test_script, timeout=30)
        
        print("CLI Output:")
        print("-" * 30)
        print(stdout)
        
        if stderr:
            print("Errors:")
            print(stderr)
        
        print(f"\nExit code: {process.returncode}")
        
    except subprocess.TimeoutExpired:
        print("⚠️ Test timed out")
        process.kill()
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_exit_cleanup()
