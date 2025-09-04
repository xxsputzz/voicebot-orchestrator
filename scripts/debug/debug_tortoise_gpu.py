#!/usr/bin/env python3
"""
Debug GPU usage in Tortoise TTS
"""
import torch
import sys
import os
import psutil

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

def check_process_gpu_usage():
    """Check if the current process is using GPU"""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            
            current_pid = os.getpid()
            for proc in procs:
                if proc.pid == current_pid:
                    print(f"✅ Process {current_pid} IS using GPU {i}")
                    print(f"   GPU Memory Used: {proc.usedGpuMemory / 1024**2:.1f} MB")
                    return True
            
        print(f"❌ Process {os.getpid()} is NOT using GPU")
        return False
        
    except ImportError:
        print("ℹ️ pynvml not available - install with: pip install nvidia-ml-py3")
        return None

def test_tortoise_gpu():
    """Test if Tortoise TTS is actually using GPU"""
    print("🔍 Testing Tortoise TTS GPU Usage")
    print("=" * 50)
    
    # Test basic CUDA
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
        
        # Check initial GPU memory
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Initial GPU Memory: {initial_memory:.1f} MB")
    
    # Import and test Tortoise
    try:
        print("\n🔄 Importing Tortoise TTS...")
        from tortoise.api import TextToSpeech
        
        print("🔄 Initializing Tortoise TTS...")
        # Explicitly force CUDA device
        tts = TextToSpeech(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Tortoise Device: {tts.device}")
        
        if torch.cuda.is_available():
            after_init_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"GPU Memory After Init: {after_init_memory:.1f} MB")
            print(f"Memory Increase: {after_init_memory - initial_memory:.1f} MB")
        
        # Check if models are on GPU
        print("\n🔍 Checking model device placement...")
        if hasattr(tts, 'autoregressive'):
            ar_device = next(tts.autoregressive.parameters()).device
            print(f"Autoregressive model device: {ar_device}")
        
        if hasattr(tts, 'diffusion'):
            diff_device = next(tts.diffusion.parameters()).device  
            print(f"Diffusion model device: {diff_device}")
            
        if hasattr(tts, 'vocoder'):
            voc_device = next(tts.vocoder.parameters()).device
            print(f"Vocoder model device: {voc_device}")
        
        # Test synthesis with GPU monitoring
        print("\n🔄 Testing short synthesis...")
        
        if torch.cuda.is_available():
            before_synth_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"GPU Memory Before Synthesis: {before_synth_memory:.1f} MB")
        
        # Monitor CPU usage
        cpu_before = psutil.cpu_percent(interval=None)
        
        # Very short synthesis test
        with torch.no_grad():
            gen = tts.tts_with_preset("Hello", preset='ultra_fast')
        
        cpu_after = psutil.cpu_percent(interval=1)
        
        if torch.cuda.is_available():
            after_synth_memory = torch.cuda.memory_allocated() / 1024**2
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"GPU Memory After Synthesis: {after_synth_memory:.1f} MB")
            print(f"Peak GPU Memory: {peak_memory:.1f} MB")
            print(f"Synthesis Memory Usage: {peak_memory - before_synth_memory:.1f} MB")
        
        print(f"CPU Usage: {cpu_before:.1f}% -> {cpu_after:.1f}%")
        
        # Check GPU process usage
        check_process_gpu_usage()
        
        print(f"\n✅ Synthesis completed successfully!")
        print(f"Audio shape: {gen.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tortoise_gpu()
