"""
GPU and Memory Cleanup Script

Properly frees up GPU memory and system resources without breaking your desktop.
"""

import subprocess
import time
import sys
import os
import psutil

def run_command(cmd, description=""):
    """Run command safely and show results."""
    print(f"{'🔧 ' + description if description else ''}Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            if result.stdout.strip():
                print(f"✅ Success: {result.stdout.strip()}")
            else:
                print("✅ Command completed successfully")
        else:
            if result.stderr.strip():
                print(f"⚠️ Warning: {result.stderr.strip()}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_memory_usage():
    """Check current memory usage."""
    memory = psutil.virtual_memory()
    print(f"\n💾 System Memory:")
    print(f"   Total: {memory.total / (1024**3):.1f} GB")
    print(f"   Available: {memory.available / (1024**3):.1f} GB")
    print(f"   Used: {memory.percent:.1f}%")
    
def check_gpu_usage():
    """Check GPU usage if nvidia-smi is available."""
    print(f"\n🎮 Checking GPU...")
    if run_command("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits", "GPU Memory Check"):
        pass
    else:
        print("   No NVIDIA GPU detected or nvidia-smi not available")

def cleanup_ollama():
    """Clean up Ollama processes and models."""
    print(f"\n🧹 Cleaning up Ollama...")
    
    # Stop Ollama service gracefully
    run_command("ollama stop", "Stopping Ollama models")
    time.sleep(2)
    
    # Check for remaining Ollama processes
    ollama_pids = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if 'ollama' in proc.info['name'].lower():
                memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                print(f"   Found Ollama process: PID {proc.info['pid']} using {memory_mb:.0f} MB")
                ollama_pids.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # If we found large Ollama processes, offer to terminate them
    if ollama_pids:
        large_processes = [pid for pid in ollama_pids 
                          if psutil.Process(pid).memory_info().rss > 100 * 1024 * 1024]  # > 100MB
        
        if large_processes:
            print(f"   Found {len(large_processes)} large Ollama processes")
            choice = input("   Terminate large Ollama processes? (y/N): ").lower()
            if choice == 'y':
                for pid in large_processes:
                    try:
                        proc = psutil.Process(pid)
                        proc.terminate()
                        print(f"   ✅ Terminated Ollama process {pid}")
                        time.sleep(1)
                    except Exception as e:
                        print(f"   ⚠️ Could not terminate {pid}: {e}")

def cleanup_python_services():
    """Clean up Python service processes."""
    print(f"\n🐍 Cleaning up Python services...")
    
    service_keywords = ['stt_service', 'tts_service', 'llm_service', 'orchestrator']
    python_services = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            if proc.info['name'].lower().startswith('python') and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline']).lower()
                if any(keyword in cmdline for keyword in service_keywords):
                    memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                    python_services.append({
                        'pid': proc.info['pid'],
                        'cmdline': ' '.join(proc.info['cmdline']),
                        'memory_mb': memory_mb
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            pass
    
    if python_services:
        print(f"   Found {len(python_services)} Python service processes:")
        for service in python_services:
            print(f"   PID {service['pid']}: {service['memory_mb']:.0f} MB - {service['cmdline'][:80]}...")
        
        choice = input("   Stop these Python services? (y/N): ").lower()
        if choice == 'y':
            for service in python_services:
                try:
                    proc = psutil.Process(service['pid'])
                    proc.terminate()
                    print(f"   ✅ Stopped service {service['pid']}")
                except Exception as e:
                    print(f"   ⚠️ Could not stop {service['pid']}: {e}")

def cleanup_gpu_memory():
    """Try to clear GPU memory."""
    print(f"\n🎮 Attempting GPU memory cleanup...")
    
    # Try to clear PyTorch cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   ✅ Cleared PyTorch CUDA cache")
    except ImportError:
        print("   PyTorch not available")
    except Exception as e:
        print(f"   ⚠️ Could not clear PyTorch cache: {e}")
    
    # Force garbage collection
    import gc
    gc.collect()
    print("   ✅ Forced garbage collection")

def main():
    """Main cleanup routine."""
    print("=" * 60)
    print("🧹 GPU & Memory Cleanup Tool")
    print("=" * 60)
    
    print("\n📊 Before Cleanup:")
    check_memory_usage()
    check_gpu_usage()
    
    # Cleanup steps
    cleanup_ollama()
    cleanup_python_services()
    cleanup_gpu_memory()
    
    print("\n⏳ Waiting for system to stabilize...")
    time.sleep(3)
    
    print("\n📊 After Cleanup:")
    check_memory_usage()
    check_gpu_usage()
    
    print("\n✨ Cleanup completed!")
    print("\nTo prevent future memory issues:")
    print("• Use 'ollama stop' before closing services")
    print("• Monitor GPU memory with 'nvidia-smi'")
    print("• Consider using smaller models for development")

if __name__ == "__main__":
    main()
