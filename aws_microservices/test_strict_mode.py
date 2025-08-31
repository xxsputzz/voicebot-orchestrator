#!/usr/bin/env python3
"""
Test script to demonstrate the new STRICT MODE behavior for Dia 4-bit TTS
This shows how the service now fails completely instead of falling back to Full Dia
"""

import subprocess
import sys
import time
from pathlib import Path

def run_test(description, command, expected_behavior):
    """Run a test and show results"""
    print(f"\n{'='*60}")
    print(f"🧪 TEST: {description}")
    print(f"{'='*60}")
    print(f"📝 Command: {' '.join(command)}")
    print(f"🎯 Expected: {expected_behavior}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            timeout=120,  # 2 minute timeout
            cwd=Path(__file__).parent
        )
        
        print(f"📊 Exit Code: {result.returncode}")
        
        if result.stdout:
            print("\n📤 STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\n📤 STDERR:")
            print(result.stderr)
            
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out (2 minutes)")
        return -1
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return -1

def main():
    print("🎯 STRICT MODE TESTING FOR DIA 4-BIT TTS")
    print("=" * 80)
    print("Testing the new behavior where Dia 4-bit service fails completely")
    print("instead of falling back to Full Dia when 4-bit engine can't load.")
    print("=" * 80)

    # Test 1: Dia 4-bit strict mode (should fail)
    exit_code_4bit = run_test(
        "Dia 4-bit Strict Mode (Should Fail)",
        ["python", "tts_hira_dia_service.py", "--engine", "4bit"],
        "❌ Complete failure with error about torch_dtype issue"
    )
    
    # Test 2: Full Dia mode (should work)
    exit_code_full = run_test(
        "Full Dia Mode (Should Work)", 
        ["python", "tts_hira_dia_service.py", "--engine", "full"],
        "✅ Success - starts in Full Dia mode"
    )
    
    # Test 3: Auto mode (should fall back to Full Dia)
    exit_code_auto = run_test(
        "Auto Mode (Should Fall Back)",
        ["python", "tts_hira_dia_service.py", "--engine", "auto"],
        "✅ Success - falls back to Full Dia"
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")
    print(f"🔴 Dia 4-bit Strict Mode: {'FAILED as expected' if exit_code_4bit != 0 else 'UNEXPECTED SUCCESS'} (exit code: {exit_code_4bit})")
    print(f"🟢 Full Dia Mode: {'SUCCESS' if exit_code_full == 0 else 'FAILED'} (exit code: {exit_code_full})")
    print(f"🟡 Auto Mode: {'SUCCESS' if exit_code_auto == 0 else 'FAILED'} (exit code: {exit_code_auto})")
    
    print(f"\n🎯 STRICT MODE VERIFICATION:")
    if exit_code_4bit != 0:
        print("✅ SUCCESS: Dia 4-bit mode correctly refuses to fall back and fails completely")
        print("   💡 This forces us to fix the underlying torch_dtype issue")
        print("   🔧 No more silent fallbacks that mask the real problem")
    else:
        print("❌ PROBLEM: Dia 4-bit mode should have failed but didn't")
        print("   🔧 The strict mode implementation needs to be checked")
    
    print(f"\n🔍 ROOT CAUSE:")
    print("   The underlying issue is in the TTS engine:")
    print("   [ERROR] Dia 4-bit failed: Dia.from_pretrained() got an unexpected keyword argument 'torch_dtype'")
    print("   📝 This needs to be fixed in the EnhancedTTSManager/Dia engine loading code")
    
    print(f"\n🎉 BENEFIT OF STRICT MODE:")
    print("   ❌ Old behavior: Silently fell back to Full Dia, masking the issue")
    print("   ✅ New behavior: Fails completely, forcing us to address the root cause")

if __name__ == "__main__":
    main()
