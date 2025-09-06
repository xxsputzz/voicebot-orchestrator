#!/usr/bin/env python3
"""
Start orchestrator with enhanced stability
"""
import subprocess
import time
import requests
import signal
import sys

def start_orchestrator():
    """Start the orchestrator service"""
    print("🎛️ Starting WebSocket Orchestrator...")
    
    try:
        # Start with process group isolation
        if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP'):
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            creation_flags = 0
            
        process = subprocess.Popen(
            ["python", "ws_orchestrator_service.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creation_flags,
            cwd="."
        )
        
        print(f"🚀 Orchestrator started (PID: {process.pid})")
        
        # Wait for it to be ready
        print("⏱️ Waiting for orchestrator to be ready...")
        for i in range(10):
            try:
                response = requests.get("http://localhost:8080/services", timeout=2)
                if response.status_code == 200:
                    print("✅ Orchestrator is ready!")
                    return process
            except:
                pass
            time.sleep(1)
        
        print("⚠️ Orchestrator may not be fully ready, but continuing...")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start orchestrator: {e}")
        return None

def main():
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down orchestrator...")
        if 'orchestrator_process' in globals() and orchestrator_process:
            try:
                orchestrator_process.terminate()
                time.sleep(2)
                if orchestrator_process.poll() is None:
                    orchestrator_process.kill()
                print("✅ Orchestrator stopped")
            except:
                pass
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start orchestrator
    orchestrator_process = start_orchestrator()
    
    if orchestrator_process:
        print("\n📊 Orchestrator Status:")
        print("- HTTP API: http://localhost:8080")
        print("- WebSocket: ws://localhost:9001") 
        print("- Registry: http://localhost:8080/services")
        print("\nPress Ctrl+C to stop the orchestrator")
        
        # Keep running
        try:
            while True:
                if orchestrator_process.poll() is not None:
                    print("❌ Orchestrator process ended unexpectedly")
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            pass
    else:
        print("❌ Failed to start orchestrator")

if __name__ == "__main__":
    main()
