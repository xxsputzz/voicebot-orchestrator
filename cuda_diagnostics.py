#!/usr/bin/env python3
"""
CUDA Diagnostics for Hira Dia TTS
"""

import sys
import os

def check_cuda_availability():
    """Comprehensive CUDA detection"""
    print("🔍 CUDA Diagnostics Report")
    print("=" * 50)
    
    # 1. Check PyTorch installation
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch not available: {e}")
        return False
    
    # 2. Check CUDA availability in PyTorch
    cuda_available = torch.cuda.is_available()
    print(f"🎯 torch.cuda.is_available(): {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA not detected by PyTorch")
        
        # Check possible reasons
        print("\n🔍 Checking possible issues:")
        
        # Check CUDA installation
        try:
            import subprocess
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ NVCC found - CUDA toolkit installed")
                print(f"   Output: {result.stdout.split(chr(10))[3] if len(result.stdout.split(chr(10))) > 3 else result.stdout}")
            else:
                print("❌ NVCC not found - CUDA toolkit may not be installed")
        except Exception as e:
            print(f"⚠️ Could not check NVCC: {e}")
        
        # Check PyTorch CUDA build
        print(f"🔧 PyTorch CUDA build version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'None'}")
        
        return False
    
    # 3. Get CUDA device info
    device_count = torch.cuda.device_count()
    print(f"🎮 CUDA devices found: {device_count}")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_props = torch.cuda.get_device_properties(i)
        memory_gb = device_props.total_memory / 1024**3
        
        print(f"   Device {i}: {device_name}")
        print(f"   Memory: {memory_gb:.1f} GB")
        print(f"   Compute Capability: {device_props.major}.{device_props.minor}")
    
    # 4. Test CUDA operations
    try:
        print(f"\n🧪 Testing CUDA operations...")
        device = torch.device("cuda:0")
        test_tensor = torch.randn(10, 10).to(device)
        result = torch.matmul(test_tensor, test_tensor)
        print("✅ CUDA tensor operations successful")
        
        # Check memory
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
        print(f"   Allocated: {memory_allocated:.1f} MB")
        print(f"   Reserved: {memory_reserved:.1f} MB")
        
    except Exception as e:
        print(f"❌ CUDA operations failed: {e}")
        return False
    
    # 5. Check environment variables
    print(f"\n🌍 Environment Variables:")
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES']
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")
    
    return True

def test_dia_requirements():
    """Test specific requirements for Dia model"""
    print(f"\n🎭 Testing Dia-specific requirements...")
    
    try:
        import torch
        # Test if we can import the dia module
        from dia.model import Dia
        print("✅ Dia module import successful")
        
        # Test model loading (this might take time)
        print("⏳ Testing model initialization (this may take a moment)...")
        device = torch.device("cuda")
        
        # Try to load the model
        model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
        print("✅ Dia model loaded successfully")
        
        # Test generation
        test_audio = model.generate(
            text="[S1] Test initialization.",
            max_tokens=64,
            cfg_scale=2.0,
            temperature=0.7
        )
        print("✅ Dia generation test successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Dia import failed: {e}")
        print("   Try: pip install dia-model")
        return False
    except Exception as e:
        print(f"❌ Dia test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting CUDA diagnostics for Hira Dia TTS...\n")
    
    cuda_ok = check_cuda_availability()
    
    if cuda_ok:
        print(f"\n{'='*50}")
        dia_ok = test_dia_requirements()
        
        if dia_ok:
            print(f"\n🎉 All checks passed! Hira Dia TTS should work.")
        else:
            print(f"\n⚠️ CUDA works but Dia model has issues.")
    else:
        print(f"\n❌ CUDA issues detected. Hira Dia TTS will not work.")
        print("\nPossible solutions:")
        print("1. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads")
        print("2. Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("3. Check NVIDIA drivers are up to date")
