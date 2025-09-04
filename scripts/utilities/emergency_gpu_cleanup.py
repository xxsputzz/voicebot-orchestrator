#!/usr/bin/env python3
"""
Emergency GPU Cleanup Utility for Tortoise TTS
Use this script to manually clean up GPU memory when processes are stuck
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("🚨 Emergency GPU Cleanup Utility for Tortoise TTS")
    print("=" * 50)
    
    try:
        # Import GPU management
        from tortoise_gpu_manager import emergency_gpu_cleanup, TortoiseProcessManager
        
        print("1. Killing GPU processes...")
        killed_pids = TortoiseProcessManager.force_kill_gpu_processes()
        if killed_pids:
            print(f"   ✅ Killed {len(killed_pids)} GPU processes: {killed_pids}")
        else:
            print("   ℹ️ No GPU processes found to kill")
        
        print("2. Killing Python processes...")
        killed_python = TortoiseProcessManager.kill_python_processes(exclude_current=True)
        if killed_python:
            print(f"   ✅ Killed {len(killed_python)} Python processes: {killed_python}")
        else:
            print("   ℹ️ No Python processes found to kill")
        
        print("3. Emergency GPU cleanup...")
        emergency_gpu_cleanup()
        print("   ✅ Emergency cleanup completed")
        
        # Check final GPU status
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**2)
                reserved = torch.cuda.memory_reserved(0) / (1024**2)
                print(f"4. Final GPU status:")
                print(f"   Allocated: {allocated:.1f}MB")
                print(f"   Reserved: {reserved:.1f}MB")
                
                if allocated < 100 and reserved < 500:
                    print("   ✅ GPU memory appears to be cleaned up successfully")
                else:
                    print("   ⚠️ GPU memory still appears high - may need system restart")
            else:
                print("   ℹ️ CUDA not available")
        except Exception as e:
            print(f"   ⚠️ Could not check GPU status: {e}")
    
    except ImportError as e:
        print(f"❌ Could not import GPU manager: {e}")
        print("Falling back to basic cleanup...")
        
        # Basic cleanup without GPU manager
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("✅ Basic GPU cache cleared")
            
            # Kill Python processes using psutil
            import psutil
            current_pid = os.getpid()
            killed = []
            
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if proc.info['name'].lower().startswith('python') and proc.info['pid'] != current_pid:
                        proc.terminate()
                        killed.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if killed:
                print(f"✅ Killed Python processes: {killed}")
            else:
                print("ℹ️ No Python processes to kill")
                
        except Exception as e:
            print(f"❌ Basic cleanup failed: {e}")
    
    except Exception as e:
        print(f"❌ Emergency cleanup failed: {e}")
    
    print("\n🎯 Cleanup completed!")
    print("You can now check GPU status with: nvidia-smi")

if __name__ == "__main__":
    main()
