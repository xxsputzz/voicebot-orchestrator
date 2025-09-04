#!/usr/bin/env python3
"""
Microphone Device Detection
===========================

Check available audio input devices and test microphone access.
"""

import pyaudio
import numpy as np
import time

def list_audio_devices():
    """List all available audio input devices"""
    print("🎙️ AUDIO DEVICE DETECTION")
    print("=" * 50)
    
    audio = pyaudio.PyAudio()
    
    try:
        device_count = audio.get_device_count()
        print(f"📊 Found {device_count} audio devices:")
        print()
        
        input_devices = []
        
        for i in range(device_count):
            try:
                device_info = audio.get_device_info_by_index(i)
                
                device_name = device_info.get('name', 'Unknown')
                max_input_channels = device_info.get('maxInputChannels', 0)
                max_output_channels = device_info.get('maxOutputChannels', 0)
                default_sample_rate = device_info.get('defaultSampleRate', 0)
                
                device_type = []
                if max_input_channels > 0:
                    device_type.append("INPUT")
                    input_devices.append(i)
                if max_output_channels > 0:
                    device_type.append("OUTPUT")
                
                type_str = "/".join(device_type) if device_type else "UNKNOWN"
                
                print(f"  📱 Device {i}: {device_name}")
                print(f"     Type: {type_str}")
                print(f"     Input Channels: {max_input_channels}")
                print(f"     Output Channels: {max_output_channels}")
                print(f"     Sample Rate: {default_sample_rate:.0f} Hz")
                print()
                
            except Exception as e:
                print(f"  ❌ Device {i}: Error reading device info - {e}")
                print()
        
        print(f"🎤 Input devices found: {len(input_devices)}")
        if input_devices:
            print(f"   Device IDs: {input_devices}")
        else:
            print("   ❌ No input devices detected!")
        
        return input_devices
    
    finally:
        audio.terminate()

def test_microphone_access(device_id=None):
    """Test microphone access and basic recording"""
    print(f"\n🧪 MICROPHONE ACCESS TEST")
    print("=" * 50)
    
    audio = pyaudio.PyAudio()
    
    try:
        # Test parameters
        settings = {
            'format': pyaudio.paInt16,
            'channels': 1,
            'rate': 16000,
            'chunk': 1024,
            'input_device_index': device_id
        }
        
        print(f"🔧 Test settings:")
        print(f"   Format: 16-bit")
        print(f"   Channels: 1 (mono)")
        print(f"   Sample Rate: 16,000 Hz")
        print(f"   Chunk Size: 1,024")
        print(f"   Device ID: {device_id if device_id is not None else 'Default'}")
        
        # Try to open microphone stream
        print(f"\n🎤 Testing microphone access...")
        
        try:
            stream = audio.open(
                format=settings['format'],
                channels=settings['channels'],
                rate=settings['rate'],
                input=True,
                input_device_index=settings['input_device_index'],
                frames_per_buffer=settings['chunk']
            )
            
            print("✅ Microphone access successful!")
            
            # Test recording a few chunks
            print("🔄 Testing 3-second recording...")
            
            chunks = []
            max_value = 0
            min_value = 0
            
            for i in range(int(settings['rate'] / settings['chunk'] * 3)):
                try:
                    data = stream.read(settings['chunk'], exception_on_overflow=False)
                    chunks.append(data)
                    
                    # Convert to numpy to check audio levels
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    chunk_max = np.max(audio_array)
                    chunk_min = np.min(audio_array)
                    
                    if chunk_max > max_value:
                        max_value = chunk_max
                    if chunk_min < min_value:
                        min_value = chunk_min
                        
                except Exception as e:
                    print(f"❌ Read error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            
            print(f"✅ Recording test completed!")
            print(f"📊 Audio level analysis:")
            print(f"   Max value: {max_value}")
            print(f"   Min value: {min_value}")
            print(f"   Range: {max_value - min_value}")
            
            if max_value == 0 and min_value == 0:
                print("⚠️  WARNING: All audio values are zero (silence detected)")
                print("   This indicates:")
                print("   - Microphone might be muted")
                print("   - No audio input")
                print("   - Permission issues")
            elif abs(max_value - min_value) < 100:
                print("⚠️  WARNING: Very low audio variation")
                print("   This might indicate:")
                print("   - Very quiet environment")
                print("   - Low microphone sensitivity")
                print("   - Background noise only")
            else:
                print("✅ Good audio variation detected!")
                print("   Microphone is capturing audio content")
            
            return True
            
        except Exception as e:
            print(f"❌ Microphone access failed: {e}")
            return False
            
    finally:
        audio.terminate()

def main():
    """Main test function"""
    # List all devices
    input_devices = list_audio_devices()
    
    # Test default microphone
    print("\\n" + "="*50)
    success = test_microphone_access()
    
    if not success and input_devices:
        print(f"\\n💡 Trying specific input devices...")
        for device_id in input_devices[:3]:  # Test first 3 input devices
            print(f"\\n🔄 Testing device {device_id}...")
            if test_microphone_access(device_id):
                print(f"✅ Device {device_id} works!")
                break
            else:
                print(f"❌ Device {device_id} failed")
    
    print(f"\\n🏁 MICROPHONE DIAGNOSIS COMPLETE")
    print("=" * 50)
    
    if success:
        print("✅ Microphone is working and accessible")
        print("💡 The STT issue might be in the transcription service")
    else:
        print("❌ Microphone access issues detected")
        print("💡 Possible solutions:")
        print("   - Check microphone permissions")
        print("   - Try a different microphone")
        print("   - Check Windows audio settings")
        print("   - Restart audio services")

if __name__ == "__main__":
    main()
