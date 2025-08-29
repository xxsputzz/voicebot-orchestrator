#!/usr/bin/env python3
"""
Simple Nari Dia CPU Test
========================

Test if Nari Dia can work on CPU at all.
"""

import asyncio
import sys
import os
import torch

# Add parent directory to path  
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

async def test_nari_cpu():
    """Test Nari Dia on CPU."""
    print("🧪 Testing Nari Dia-1.6B on CPU...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        print("\n🔄 Attempting to load Nari Dia model...")
        from dia.model import Dia
        
        # Try to load with basic settings
        print("📡 Loading from Hugging Face Hub...")
        model = Dia.from_pretrained("nari-labs/Dia-1.6B")
        
        print("✅ Model loaded successfully!")
        
        # Test simple generation
        test_text = "[S1] Hello world"
        print(f"\n🎵 Testing generation: '{test_text}'")
        
        with torch.no_grad():  # Save memory
            audio = model.generate(test_text)
            
        print(f"✅ Generation successful!")
        print(f"📊 Audio shape: {audio.shape if hasattr(audio, 'shape') else 'unknown'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Possible issues:")
        print("  - Model too large for available RAM")
        print("  - CPU inference not fully supported yet")
        print("  - Network issues downloading model")
        return False

if __name__ == "__main__":
    asyncio.run(test_nari_cpu())
