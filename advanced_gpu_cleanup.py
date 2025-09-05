"""
Advanced GPU Memory Cleanup

More aggressive GPU memory cleanup options.
"""

import subprocess
import time
import os

def run_powershell(cmd):
    """Run PowerShell command."""
    try:
        result = subprocess.run(
            ["powershell", "-Command", cmd], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        return result.stdout.strip(), result.returncode == 0
    except Exception as e:
        return str(e), False

def cleanup_gpu_processes():
    """Find and optionally clean up GPU-using processes."""
    print("🎮 Advanced GPU Cleanup")
    print("=" * 40)
    
    # Check current GPU usage
    print("\n📊 Current GPU Memory Usage:")
    os.system("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits")
    
    # List processes that might be using GPU
    print("\n🔍 Checking for GPU-using processes...")
    
    gpu_keywords = [
        'python', 'torch', 'tensorflow', 'ollama', 'whisper', 
        'cuda', 'nvidia', 'ml', 'ai', 'pytorch'
    ]
    
    cmd = f"""
    Get-Process | Where-Object {{
        $_.ProcessName -match '({"|".join(gpu_keywords)})' -or
        $_.WorkingSet -gt 100MB
    }} | Select-Object ProcessName, Id, @{{Name="MemoryMB";Expression={{[math]::Round($_.WorkingSet/1MB,0)}}}} | 
    Sort-Object MemoryMB -Descending | Format-Table -AutoSize
    """
    
    output, success = run_powershell(cmd)
    if success:
        print(output)
    
    print("\n🧹 Cleanup Options:")
    print("1. Restart GPU driver (requires admin)")
    print("2. Kill all Python processes (aggressive)")
    print("3. Manual process termination")
    print("4. Just clear caches")
    print("0. Exit")
    
    choice = input("\nChoose option (0-4): ").strip()
    
    if choice == "1":
        restart_gpu_driver()
    elif choice == "2":
        kill_python_processes()
    elif choice == "3":
        manual_process_kill()
    elif choice == "4":
        clear_caches_only()
    else:
        print("Exiting...")

def restart_gpu_driver():
    """Attempt to restart GPU driver (requires admin)."""
    print("\n🔄 Attempting to restart GPU driver...")
    print("⚠️  This requires administrator privileges")
    
    # Try to restart display driver
    cmd = "pnputil /restart-device 'PCI\\VEN_10DE*'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if "successfully" in result.stdout.lower():
        print("✅ GPU driver restart successful")
        time.sleep(5)
        os.system("nvidia-smi")
    else:
        print("❌ Could not restart GPU driver. Run as administrator or try other options.")
        print(f"Output: {result.stdout}")
        print(f"Error: {result.stderr}")

def kill_python_processes():
    """Kill all Python processes (aggressive)."""
    print("\n⚠️  This will terminate ALL Python processes!")
    confirm = input("Type 'KILL' to confirm: ").strip()
    
    if confirm == "KILL":
        print("🔪 Terminating Python processes...")
        os.system("taskkill /f /im python.exe")
        os.system("taskkill /f /im pythonw.exe")
        time.sleep(2)
        print("✅ Python processes terminated")
        
        # Clear GPU memory
        clear_caches_only()
    else:
        print("❌ Cancelled")

def manual_process_kill():
    """Manual process termination."""
    print("\n🎯 Manual Process Termination")
    pid = input("Enter Process ID to terminate (or 'cancel'): ").strip()
    
    if pid.lower() == 'cancel':
        return
    
    try:
        pid_num = int(pid)
        result = subprocess.run(f"taskkill /f /pid {pid_num}", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Process {pid_num} terminated")
        else:
            print(f"❌ Could not terminate process {pid_num}: {result.stderr}")
    except ValueError:
        print("❌ Invalid process ID")

def clear_caches_only():
    """Clear various caches."""
    print("\n🧹 Clearing caches...")
    
    # Try PyTorch cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✅ Cleared PyTorch CUDA cache")
    except:
        print("⚠️  PyTorch not available")
    
    # Try TensorFlow cache
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
        print("✅ Cleared TensorFlow session")
    except:
        print("⚠️  TensorFlow not available")
    
    # Force garbage collection
    import gc
    gc.collect()
    print("✅ Forced garbage collection")
    
    # Check result
    print("\n📊 GPU Memory After Cleanup:")
    os.system("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits")

if __name__ == "__main__":
    cleanup_gpu_processes()
