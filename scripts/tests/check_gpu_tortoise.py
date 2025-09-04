#!/usr/bin/env python3
"""
GPU Detection and Optimization for Tortoise TTS
"""
import torch
import sys
import os

def check_gpu_availability():
    """Check GPU availability and configuration"""
    print("🔍 GPU Detection for Tortoise TTS")
    print("=" * 50)
    
    # Check CUDA
    if torch.cuda.is_available():
        print("✅ CUDA Available!")
        device_count = torch.cuda.device_count()
        print(f"📊 GPU Count: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            
            print(f"\n🎮 GPU {i}: {device_name}")
            print(f"   Memory: {memory_gb:.1f} GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            print(f"   Multi-Processors: {props.multi_processor_count}")
            
            # Check memory availability
            if memory_gb >= 4:
                print(f"   ✅ Suitable for Tortoise TTS")
            else:
                print(f"   ⚠️ May be limited for large models")
        
        # Test GPU with simple tensor
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.matmul(test_tensor, test_tensor)
            print(f"\n✅ GPU Test Passed - Ready for neural synthesis!")
            del test_tensor, result
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n❌ GPU Test Failed: {e}")
            
    elif torch.backends.mps.is_available():
        print("✅ Apple Metal Performance Shaders (MPS) Available!")
        print("🍎 Optimized for Apple Silicon (M1/M2/M3)")
        
        try:
            test_tensor = torch.randn(1000, 1000).to('mps')
            result = torch.matmul(test_tensor, test_tensor)
            print("✅ MPS Test Passed - Ready for neural synthesis!")
            del test_tensor, result
        except Exception as e:
            print(f"❌ MPS Test Failed: {e}")
            
    else:
        print("❌ No GPU acceleration available")
        print("💻 Will use CPU - synthesis will be slower")
        
    # Check PyTorch version
    print(f"\n📦 PyTorch Version: {torch.__version__}")
    
    # Performance recommendations
    print(f"\n🚀 Performance Recommendations:")
    if torch.cuda.is_available():
        print("   • Use 'ultra_fast' preset for real-time synthesis")
        print("   • GPU acceleration will provide 5-10x speedup") 
        print("   • Larger batch sizes possible with more GPU memory")
    else:
        print("   • Stick to 'ultra_fast' preset for best performance")
        print("   • Consider shorter text segments")
        print("   • CPU synthesis may take 2-5 minutes per clip")

if __name__ == "__main__":
    check_gpu_availability()
