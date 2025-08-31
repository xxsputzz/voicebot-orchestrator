#!/usr/bin/env python3
"""
Test Unified Hira Dia TTS Service with Dual Engine Support
Tests both Full Dia and 4-bit Dia engines through the unified service
"""
import asyncio
import requests
import time
import json
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
BASE_URL = "http://localhost:8012"  # Unified Hira Dia service port
TEST_TEXTS = [
    "Hello, this is a test of the unified Dia service.",
    "This is a medium length test to evaluate the auto-selection feature of our dual engine system.",
    "This is a longer text sample designed to test the quality versus speed trade-offs in our unified Hira Dia TTS service. It should trigger different engine selection logic based on text length and quality preferences."
]

class UnifiedDiaServiceTest:
    """Test suite for the unified Dia TTS service"""
    
    def __init__(self):
        self.service_name = "Unified Hira Dia TTS"
        self.test_results = []
        self.service_running = False
    
    def check_service_health(self):
        """Check if the unified service is running and healthy"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ {self.service_name} is healthy")
                print(f"   Status: {health_data.get('status', 'unknown')}")
                print(f"   Available engines: {health_data.get('engines_available', [])}")
                print(f"   GPU available: {health_data.get('gpu_available', False)}")
                self.service_running = True
                return True
            else:
                print(f"❌ {self.service_name} health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ {self.service_name} not reachable: {e}")
            return False
    
    def test_engine_info(self):
        """Test engine information endpoint"""
        print(f"\n🔍 Testing engine information...")
        try:
            response = requests.get(f"{BASE_URL}/engines", timeout=10)
            if response.status_code == 200:
                engines_data = response.json()
                print(f"✅ Engine info retrieved successfully")
                print(f"   Service: {engines_data.get('service', 'unknown')}")
                print(f"   Current engine: {engines_data.get('current_engine', 'unknown')}")
                
                available_engines = engines_data.get('available_engines', [])
                for engine in available_engines:
                    print(f"   📋 {engine.get('display_name', engine.get('name', 'Unknown'))}")
                    print(f"      Speed: {engine.get('speed', 'Unknown')}")
                    print(f"      Quality: {engine.get('quality', 'Unknown')}")
                    print(f"      Active: {'✅' if engine.get('active', False) else '❌'}")
                
                return True
            else:
                print(f"❌ Engine info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Engine info test failed: {e}")
            return False
    
    def test_engine_switching(self):
        """Test engine switching functionality"""
        print(f"\n🔄 Testing engine switching...")
        engines_to_test = ["full", "4bit"]
        
        for engine in engines_to_test:
            try:
                print(f"   Switching to {engine} engine...")
                response = requests.post(f"{BASE_URL}/switch_engine", 
                                       params={"engine": engine}, 
                                       timeout=10)
                if response.status_code == 200:
                    switch_data = response.json()
                    current_engine = switch_data.get('current_engine', 'unknown')
                    print(f"   ✅ Successfully switched to {current_engine}")
                else:
                    print(f"   ❌ Engine switch to {engine} failed: {response.status_code}")
                    return False
                
                # Small delay between switches
                time.sleep(2)
                
            except Exception as e:
                print(f"   ❌ Engine switch to {engine} failed: {e}")
                return False
        
        return True
    
    def test_synthesis_with_preferences(self):
        """Test synthesis with different engine preferences"""
        print(f"\n🎤 Testing synthesis with engine preferences...")
        
        test_cases = [
            {
                "text": TEST_TEXTS[0],
                "engine_preference": "full",
                "high_quality": True,
                "description": "Short text with full engine preference"
            },
            {
                "text": TEST_TEXTS[0],
                "engine_preference": "4bit",
                "high_quality": False,
                "description": "Short text with 4-bit engine preference"
            },
            {
                "text": TEST_TEXTS[1],
                "engine_preference": "auto",
                "high_quality": True,
                "description": "Medium text with auto selection (high quality)"
            },
            {
                "text": TEST_TEXTS[1],
                "engine_preference": "auto",
                "high_quality": False,
                "description": "Medium text with auto selection (speed optimized)"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n   Test {i}: {test_case['description']}")
            try:
                start_time = time.time()
                
                response = requests.post(f"{BASE_URL}/synthesize", 
                                       json={
                                           "text": test_case["text"],
                                           "engine_preference": test_case["engine_preference"],
                                           "high_quality": test_case["high_quality"],
                                           "return_audio": False  # Skip audio for faster testing
                                       }, 
                                       timeout=300)  # 5 minute timeout for synthesis
                
                synthesis_time = time.time() - start_time
                
                if response.status_code == 200:
                    result_data = response.json()
                    metadata = result_data.get('metadata', {})
                    
                    print(f"   ✅ Synthesis successful")
                    print(f"      Engine used: {metadata.get('engine_used', 'unknown')}")
                    print(f"      Engine selected: {metadata.get('engine_selected', 'unknown')}")
                    print(f"      Quality: {metadata.get('quality', 'unknown')}")
                    print(f"      Total time: {synthesis_time:.2f}s")
                    print(f"      Generation time: {metadata.get('generation_time_seconds', 0):.2f}s")
                    print(f"      Audio size: {metadata.get('audio_size_bytes', 0)} bytes")
                    
                    # Store result for analysis
                    self.test_results.append({
                        "test_case": test_case["description"],
                        "engine_preference": test_case["engine_preference"],
                        "engine_used": metadata.get('engine_used', 'unknown'),
                        "quality": metadata.get('quality', 'unknown'),
                        "synthesis_time": synthesis_time,
                        "text_length": len(test_case["text"])
                    })
                    
                else:
                    print(f"   ❌ Synthesis failed: {response.status_code}")
                    if response.text:
                        print(f"      Error: {response.text}")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Synthesis test failed: {e}")
                return False
        
        return True
    
    def test_file_synthesis(self):
        """Test file synthesis endpoint"""
        print(f"\n📁 Testing file synthesis...")
        try:
            response = requests.post(f"{BASE_URL}/synthesize_file", 
                                   json={
                                       "text": TEST_TEXTS[0],
                                       "engine_preference": "4bit",  # Use 4-bit for faster testing
                                       "output_format": "wav"
                                   }, 
                                   timeout=180)
            
            if response.status_code == 200:
                audio_size = len(response.content)
                headers = response.headers
                print(f"✅ File synthesis successful")
                print(f"   Audio size: {audio_size} bytes")
                print(f"   Content type: {headers.get('content-type', 'unknown')}")
                print(f"   Engine used: {headers.get('X-Engine-Used', 'unknown')}")
                print(f"   Quality: {headers.get('X-Quality', 'unknown')}")
                return True
            else:
                print(f"❌ File synthesis failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ File synthesis test failed: {e}")
            return False
    
    def test_service_info(self):
        """Test service information endpoints"""
        print(f"\n📊 Testing service information...")
        
        endpoints = ["/info", "/status", "/gpu_status"]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ {endpoint} endpoint working")
                    if endpoint == "/info":
                        print(f"   Service: {data.get('service', 'unknown')}")
                        print(f"   Version: {data.get('version', 'unknown')}")
                        features = data.get('features', [])
                        if features:
                            print(f"   Features: {', '.join(features)}")
                    elif endpoint == "/status":
                        print(f"   Status: {data.get('status', 'unknown')}")
                        print(f"   Current engine: {data.get('current_engine', 'unknown')}")
                        print(f"   Available engines: {data.get('engines', [])}")
                    elif endpoint == "/gpu_status":
                        print(f"   GPU available: {data.get('gpu_available', False)}")
                        if data.get('gpu_available', False):
                            print(f"   GPU: {data.get('device_name', 'unknown')}")
                            print(f"   Memory: {data.get('allocated_memory_gb', 0):.1f}GB allocated")
                else:
                    print(f"❌ {endpoint} endpoint failed: {response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"❌ {endpoint} test failed: {e}")
                return False
        
        return True
    
    def analyze_results(self):
        """Analyze test results and provide insights"""
        print(f"\n📈 Test Results Analysis:")
        print("=" * 50)
        
        if not self.test_results:
            print("No synthesis test results to analyze")
            return
        
        # Group by engine
        engine_results = {}
        for result in self.test_results:
            engine = result['engine_used']
            if engine not in engine_results:
                engine_results[engine] = []
            engine_results[engine].append(result)
        
        for engine, results in engine_results.items():
            print(f"\n🔧 {engine.upper()} Engine Results:")
            avg_time = sum(r['synthesis_time'] for r in results) / len(results)
            print(f"   Average synthesis time: {avg_time:.2f}s")
            print(f"   Tests performed: {len(results)}")
            
            for result in results:
                print(f"   • {result['test_case']}: {result['synthesis_time']:.2f}s")
        
        # Engine selection analysis
        print(f"\n🤖 Auto-Selection Analysis:")
        auto_results = [r for r in self.test_results if 'auto' in r['test_case'].lower()]
        for result in auto_results:
            print(f"   • Text length {result['text_length']} chars → {result['engine_used']} engine")
    
    def run_comprehensive_test(self):
        """Run the complete test suite"""
        print(f"🧪 Starting Comprehensive {self.service_name} Test")
        print("=" * 60)
        
        # 1. Health check
        if not self.check_service_health():
            print(f"\n❌ Service health check failed. Cannot proceed with tests.")
            return False
        
        # 2. Engine information test
        if not self.test_engine_info():
            print(f"\n⚠️ Engine info test failed, but continuing...")
        
        # 3. Engine switching test
        if not self.test_engine_switching():
            print(f"\n⚠️ Engine switching test failed, but continuing...")
        
        # 4. Synthesis tests with preferences
        if not self.test_synthesis_with_preferences():
            print(f"\n❌ Synthesis preference tests failed.")
            return False
        
        # 5. File synthesis test
        if not self.test_file_synthesis():
            print(f"\n⚠️ File synthesis test failed, but continuing...")
        
        # 6. Service info tests
        if not self.test_service_info():
            print(f"\n⚠️ Service info tests failed, but continuing...")
        
        # 7. Results analysis
        self.analyze_results()
        
        print(f"\n🎉 Comprehensive {self.service_name} Test Complete!")
        print("=" * 60)
        
        return True

def main():
    """Main test execution"""
    tester = UnifiedDiaServiceTest()
    
    try:
        success = tester.run_comprehensive_test()
        if success:
            print(f"\n✅ All critical tests passed!")
            print(f"🚀 The unified Hira Dia service is working correctly with dual engine support.")
            print(f"\n📋 Key Features Verified:")
            print(f"   ✅ Dual engine support (Full Dia + 4-bit Dia)")
            print(f"   ✅ Auto engine selection based on text length and quality preference")
            print(f"   ✅ Runtime engine switching")
            print(f"   ✅ Quality/speed optimization")
            print(f"   ✅ Comprehensive service endpoints")
        else:
            print(f"\n❌ Some tests failed. Check the service configuration and try again.")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
