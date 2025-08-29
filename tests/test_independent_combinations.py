#!/usr/bin/env python3
"""
Independent Microservices TTS/LLM Combination Tester

Tests the independent microservices for TTS/LLM combinations:
- Kokoro TTS + Mistral LLM  
- Kokoro TTS + GPT LLM
- Hira Dia TTS + Mistral LLM
- Hira Dia TTS + GPT LLM

Generates labeled audio files for each combination.
"""

import asyncio
import sys
import time
import json
import base64
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class IndependentServicesCombinationTester:
    """Test TTS/LLM combinations using independent microservices"""
    
    def __init__(self):
        """Initialize the tester"""
        self.base_dir = Path(__file__).parent
        self.audio_dir = self.base_dir / "audio_samples" / "independent_combinations"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Service endpoints
        self.services = {
            "stt": "http://localhost:8001",
            "whisper_stt": "http://localhost:8002",
            "kokoro_tts": "http://localhost:8011", 
            "hira_dia_tts": "http://localhost:8012",
            "mistral_llm": "http://localhost:8021",
            "gpt_llm": "http://localhost:8022"
        }
        
        # Test combinations - including Whisper STT variants
        self.combinations = [
            {"tts": "kokoro_tts", "llm": "mistral_llm", "stt": "whisper_stt", "name": "Whisper + Kokoro + Mistral"},
            {"tts": "kokoro_tts", "llm": "gpt_llm", "stt": "whisper_stt", "name": "Whisper + Kokoro + GPT"},
            {"tts": "hira_dia_tts", "llm": "mistral_llm", "stt": "whisper_stt", "name": "Whisper + Hira Dia + Mistral"},
            {"tts": "hira_dia_tts", "llm": "gpt_llm", "stt": "whisper_stt", "name": "Whisper + Hira Dia + GPT"},
            {"tts": "kokoro_tts", "llm": "mistral_llm", "stt": "stt", "name": "Original STT + Kokoro + Mistral"},
            {"tts": "kokoro_tts", "llm": "gpt_llm", "stt": "stt", "name": "Original STT + Kokoro + GPT"}
        ]
        
        # Banking test prompts
        self.test_prompts = [
            "What is my current account balance?",
            "How can I apply for a personal loan?", 
            "What are your current mortgage interest rates?",
            "I need help with a disputed transaction on my account.",
            "Can you explain the different types of credit cards you offer?"
        ]
        
        self.results = []
    
    def get_timestamp(self) -> str:
        """Get current timestamp for file naming"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_audio_filename(self, combination: str, prompt_index: int) -> str:
        """Generate filename for audio output"""
        timestamp = self.get_timestamp()
        safe_combo = combination.lower().replace(" + ", "_").replace(" ", "_")
        return f"{safe_combo}_prompt{prompt_index + 1}_{timestamp}.wav"
    
    def check_service_health(self, service_name: str) -> bool:
        """Check if a service is available"""
        try:
            url = self.services[service_name]
            response = requests.get(f"{url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def check_all_services(self) -> Dict[str, bool]:
        """Check availability of all required services"""
        print("🔍 Checking service availability...")
        
        availability = {}
        for service_name, url in self.services.items():
            is_available = self.check_service_health(service_name)
            availability[service_name] = is_available
            status_icon = "✅" if is_available else "❌"
            print(f"  {status_icon} {service_name}: {url}")
        
        return availability
    
    async def test_llm_service(self, service_name: str, prompt: str) -> Tuple[bool, Optional[str], Dict]:
        """Test LLM service"""
        try:
            print(f"    🧠 Testing {service_name}")
            
            url = self.services[service_name]
            payload = {
                "text": prompt,
                "use_cache": True,
                "max_tokens": 150
            }
            
            start_time = time.time()
            response = requests.post(f"{url}/generate", json=payload, timeout=30)
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "")
                
                if response_text:
                    metadata = {
                        "service": service_name,
                        "processing_time": processing_time,
                        "model_used": data.get("model_used", "unknown"),
                        "cache_hit": data.get("cache_hit", False),
                        "tokens": len(response_text.split())
                    }
                    return True, response_text, metadata
                else:
                    return False, None, {"error": "Empty response"}
            else:
                return False, None, {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return False, None, {"error": str(e)}
    
    async def test_tts_service(self, service_name: str, text: str) -> Tuple[bool, Optional[bytes], Dict]:
        """Test TTS service"""
        try:
            print(f"    🎙️ Testing {service_name}")
            
            url = self.services[service_name]
            payload = {
                "text": text,
                "return_audio": True
            }
            
            # Add voice parameter for Kokoro
            if service_name == "kokoro_tts":
                payload["voice"] = "af_bella"
            
            start_time = time.time()
            response = requests.post(f"{url}/synthesize", json=payload, timeout=60)
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                audio_base64 = data.get("audio_base64")
                
                if audio_base64:
                    audio_data = base64.b64decode(audio_base64)
                    
                    metadata = {
                        "service": service_name,
                        "processing_time": processing_time,
                        "engine_used": data.get("metadata", {}).get("engine_used", "unknown"),
                        "voice_used": data.get("metadata", {}).get("voice_used", "unknown"),
                        "audio_size": len(audio_data)
                    }
                    return True, audio_data, metadata
                else:
                    return False, None, {"error": "No audio data returned"}
            else:
                return False, None, {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return False, None, {"error": str(e)}
    
    async def test_combination(self, combination: Dict, prompt: str, prompt_index: int) -> Dict:
        """Test a specific TTS/LLM combination"""
        
        combo_name = combination["name"]
        tts_service = combination["tts"]
        llm_service = combination["llm"]
        
        print(f"\n🧪 Testing {combo_name}")
        print(f"Prompt: {prompt}")
        print("-" * 60)
        
        result = {
            "combination": combo_name,
            "tts_service": tts_service,
            "llm_service": llm_service,
            "prompt": prompt,
            "prompt_index": prompt_index,
            "timestamp": self.get_timestamp(),
            "success": False,
            "llm_result": None,
            "tts_result": None,
            "audio_file": None,
            "total_time": 0,
            "errors": []
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Check service availability
            if not self.check_service_health(llm_service):
                result["errors"].append(f"{llm_service} service not available")
                return result
            
            if not self.check_service_health(tts_service):
                result["errors"].append(f"{tts_service} service not available")
                return result
            
            # Step 2: Generate LLM response
            llm_success, llm_response, llm_metadata = await self.test_llm_service(llm_service, prompt)
            
            if not llm_success:
                result["errors"].append(f"LLM generation failed: {llm_metadata.get('error', 'Unknown error')}")
                return result
            
            result["llm_result"] = {
                "success": True,
                "response": llm_response,
                "metadata": llm_metadata
            }
            
            print(f"    ✅ LLM response: {llm_response[:100]}...")
            
            # Step 3: Generate TTS audio
            tts_success, audio_data, tts_metadata = await self.test_tts_service(tts_service, llm_response)
            
            if not tts_success:
                result["errors"].append(f"TTS generation failed: {tts_metadata.get('error', 'Unknown error')}")
                return result
            
            result["tts_result"] = {
                "success": True,
                "metadata": tts_metadata
            }
            
            # Step 4: Save audio file
            if audio_data:
                filename = self.get_audio_filename(combo_name, prompt_index)
                file_path = self.audio_dir / filename
                
                with open(file_path, "wb") as f:
                    f.write(audio_data)
                
                result["audio_file"] = str(file_path)
                print(f"    💾 Audio saved: {filename}")
                print(f"    📊 Audio size: {len(audio_data)} bytes")
            
            result["success"] = True
            result["total_time"] = time.time() - start_time
            
            print(f"    🎉 Combination successful! Total time: {result['total_time']:.2f}s")
            
        except Exception as e:
            result["errors"].append(f"Combination test failed: {str(e)}")
            result["total_time"] = time.time() - start_time
            print(f"    ❌ Combination failed: {e}")
        
        return result
    
    async def run_all_combinations(self):
        """Run all TTS/LLM combinations"""
        
        print("🎭 Independent Services TTS/LLM Combination Testing")
        print("=" * 70)
        print(f"Audio output directory: {self.audio_dir}")
        
        # Check service availability first
        availability = self.check_all_services()
        
        available_services = [name for name, available in availability.items() if available]
        unavailable_services = [name for name, available in availability.items() if not available]
        
        if unavailable_services:
            print(f"\n⚠️ Warning: {len(unavailable_services)} services unavailable: {', '.join(unavailable_services)}")
            print("Some combinations may fail. Start missing services with:")
            print("  cd aws_microservices")
            print("  python enhanced_service_manager.py")
        
        print(f"\n✅ Available services: {', '.join(available_services)}")
        
        # Filter combinations based on available services
        testable_combinations = []
        for combo in self.combinations:
            if availability.get(combo["tts"]) and availability.get(combo["llm"]):
                testable_combinations.append(combo)
            else:
                print(f"❌ Skipping {combo['name']} - services not available")
        
        if not testable_combinations:
            print("\n❌ No testable combinations found. Please start the required services.")
            return
        
        print(f"\n🧪 Testing {len(testable_combinations)} combinations × {len(self.test_prompts)} prompts")
        print(f"Total tests: {len(testable_combinations) * len(self.test_prompts)}")
        
        # Run tests
        for prompt_index, prompt in enumerate(self.test_prompts):
            for combination in testable_combinations:
                result = await self.test_combination(combination, prompt, prompt_index)
                self.results.append(result)
                
                # Small delay between tests
                await asyncio.sleep(2)
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate a summary report of all tests"""
        
        print("\n📊 TEST SUMMARY REPORT")
        print("=" * 70)
        
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r["success"]])
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success Rate: {(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%")
        
        # Breakdown by combination
        print(f"\n📋 Results by Combination:")
        combination_stats = {}
        
        for result in self.results:
            combo = result["combination"]
            if combo not in combination_stats:
                combination_stats[combo] = {"total": 0, "success": 0, "files": []}
            
            combination_stats[combo]["total"] += 1
            if result["success"]:
                combination_stats[combo]["success"] += 1
                if result["audio_file"]:
                    combination_stats[combo]["files"].append(Path(result["audio_file"]).name)
        
        for combo, stats in combination_stats.items():
            success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
            status_icon = "✅" if success_rate == 100 else "⚠️" if success_rate > 0 else "❌"
            print(f"\n  {status_icon} {combo}:")
            print(f"    Tests: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
            print(f"    Audio files: {len(stats['files'])}")
            
            # Show audio files for this combination
            for audio_file in stats["files"][:3]:  # Show first 3 files
                print(f"      📄 {audio_file}")
            if len(stats["files"]) > 3:
                print(f"      ... and {len(stats['files']) - 3} more")
        
        # Performance summary
        successful_results = [r for r in self.results if r["success"]]
        if successful_results:
            total_times = [r["total_time"] for r in successful_results]
            avg_time = sum(total_times) / len(total_times)
            min_time = min(total_times)
            max_time = max(total_times)
            
            print(f"\n⚡ Performance Summary:")
            print(f"  Average time: {avg_time:.2f}s")
            print(f"  Fastest: {min_time:.2f}s")
            print(f"  Slowest: {max_time:.2f}s")
        
        # Save detailed JSON report
        report_file = self.audio_dir / f"combination_test_report_{self.get_timestamp()}.json"
        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📋 Detailed report saved: {report_file}")
        
        # Show audio directory
        audio_files = list(self.audio_dir.glob("*.wav"))
        print(f"\n🔊 Generated {len(audio_files)} audio files in: {self.audio_dir}")
        
        if audio_files:
            print("Recent audio files:")
            for audio_file in sorted(audio_files)[-10:]:  # Show last 10 files
                print(f"  📄 {audio_file.name}")
        
        print(f"\n🎉 Testing complete! Check audio files in: {self.audio_dir}")

async def main():
    """Main entry point"""
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Independent Services TTS/LLM combinations")
    parser.add_argument("--combination", choices=["kokoro_mistral", "kokoro_gpt", "hira_dia_mistral", "hira_dia_gpt"],
                        help="Test specific combination only")
    parser.add_argument("--prompt", type=int, 
                        help="Test specific prompt index only (0-4)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with one prompt per combination")
    
    args = parser.parse_args()
    
    tester = IndependentServicesCombinationTester()
    
    # Filter combinations based on arguments
    if args.combination:
        combo_map = {
            "kokoro_mistral": {"tts": "kokoro_tts", "llm": "mistral_llm", "name": "Kokoro + Mistral"},
            "kokoro_gpt": {"tts": "kokoro_tts", "llm": "gpt_llm", "name": "Kokoro + GPT"},
            "hira_dia_mistral": {"tts": "hira_dia_tts", "llm": "mistral_llm", "name": "Hira Dia + Mistral"},
            "hira_dia_gpt": {"tts": "hira_dia_tts", "llm": "gpt_llm", "name": "Hira Dia + GPT"}
        }
        tester.combinations = [combo_map[args.combination]]
    
    if args.prompt is not None:
        tester.test_prompts = [tester.test_prompts[args.prompt]]
    elif args.quick:
        tester.test_prompts = [tester.test_prompts[0]]  # Use first prompt only
    
    try:
        await tester.run_all_combinations()
    except KeyboardInterrupt:
        print("\n👋 Testing interrupted by user")
    except Exception as e:
        print(f"❌ Testing failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
