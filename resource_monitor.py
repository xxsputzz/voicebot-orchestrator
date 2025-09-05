"""
Quick Memory & GPU Monitor

Simple tool to check your system resources at a glance.
"""

import subprocess
import psutil
import time

def get_gpu_usage():
    """Get GPU memory usage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            used, total = map(int, result.stdout.strip().split(', '))
            return used, total, (used/total)*100
    except:
        pass
    return None, None, None

def check_resources():
    """Check system resources."""
    print("=" * 50)
    print("🖥️  SYSTEM RESOURCE MONITOR")
    print("=" * 50)
    
    # System memory
    memory = psutil.virtual_memory()
    print(f"\n💾 System Memory:")
    print(f"   Total: {memory.total / (1024**3):.1f} GB")
    print(f"   Available: {memory.available / (1024**3):.1f} GB")
    print(f"   Used: {memory.percent:.1f}%")
    
    if memory.percent > 80:
        print("   ⚠️  HIGH MEMORY USAGE")
    elif memory.percent > 60:
        print("   🟡 MODERATE MEMORY USAGE")
    else:
        print("   ✅ GOOD MEMORY USAGE")
    
    # GPU memory
    gpu_used, gpu_total, gpu_percent = get_gpu_usage()
    if gpu_used is not None:
        print(f"\n🎮 GPU Memory:")
        print(f"   Used: {gpu_used} MB")
        print(f"   Total: {gpu_total} MB")
        print(f"   Percentage: {gpu_percent:.1f}%")
        
        if gpu_percent > 90:
            print("   ⚠️  HIGH GPU USAGE")
        elif gpu_percent > 70:
            print("   🟡 MODERATE GPU USAGE")  
        else:
            print("   ✅ GOOD GPU USAGE")
    
    # Top memory consumers
    print(f"\n🔝 Top Memory Consumers:")
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
            if memory_mb > 50:  # Only show processes using more than 50MB
                processes.append({
                    'name': proc.info['name'],
                    'pid': proc.info['pid'],
                    'memory_mb': memory_mb
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Show top 10 memory users
    processes.sort(key=lambda x: x['memory_mb'], reverse=True)
    for i, proc in enumerate(processes[:10]):
        print(f"   {i+1:2}. {proc['name']:15} (PID {proc['pid']:5}): {proc['memory_mb']:6.0f} MB")
    
    # Ollama specific check
    ollama_processes = [p for p in processes if 'ollama' in p['name'].lower()]
    if ollama_processes:
        total_ollama_mb = sum(p['memory_mb'] for p in ollama_processes)
        print(f"\n🦙 Ollama Total Usage: {total_ollama_mb:.0f} MB")
        if total_ollama_mb > 1000:
            print("   ⚠️  Consider running 'ollama stop' to free memory")

def main():
    """Main monitoring function."""
    while True:
        check_resources()
        
        print("\n" + "="*50)
        choice = input("Press Enter to refresh, 'q' to quit: ").strip().lower()
        if choice == 'q':
            break
        
        print("\n" * 2)  # Clear some space

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
