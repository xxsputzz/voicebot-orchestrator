#!/usr/bin/env python3
"""
TTS/LLM Combination Testing Script

Tests all combinations of TTS engines (Kokoro, Hira Dia) with LLM models (Mistral, GPT)
and generates labeled audio files for each combination.
"""

import asyncio
import sys
import time
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import base64

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Try to import the voicebot components
try:
    from voicebot_orchestrator.tts import KokoroTTS
    from voicebot_orchestrator.enhanced_llm import EnhancedLLM
    from voicebot_orchestrator.datetime_utils import DateTimeFormatter
    VOICEBOT_AVAILABLE = True
    print("✅ Voicebot orchestrator components available")
except ImportError as e:
    VOICEBOT_AVAILABLE = False
    print(f"⚠️ Voicebot orchestrator not available: {e}")

# Try to import test components
try:
    from real_llm import RealOllamaLLM
    REAL_LLM_AVAILABLE = True
    print("✅ Real LLM module available")
except ImportError:
    REAL_LLM_AVAILABLE = False
    print("⚠️ Real LLM module not available")

class TTSLLMCombinationTester:
    """Test all combinations of TTS engines and LLM models"""
    
    def __init__(self):
        """Initialize the combination tester"""
        self.base_dir = Path(__file__).parent
        self.audio_dir = self.base_dir / "audio_samples" / "tts_llm_combinations"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Test combinations
        self.tts_engines = {
            "kokoro": {
                "name": "Kokoro TTS",
                "voices": ["af_bella", "am_adam", "bf_emma", "bm_george"],
                "default_voice": "af_bella"
            },
            "hira_dia": {
                "name": "Hira Dia TTS", 
                "voices": ["nari"],
                "default_voice": "nari"
            }
        }
        
        self.llm_models = {
            "mistral": {
                "name": "Mistral LLM",
                "model": "mistral",
                "type": "ollama"
            },
            "gpt": {
                "name": "GPT LLM",
                "model": "gpt-3.5-turbo",
                "type": "openai"
            }
        }
        
        # Test prompts for banking scenarios
        self.test_prompts = [
            "What is my account balance?",
            "How can I apply for a loan?",
            "What are your current interest rates?",
            "I need help with a transaction dispute.",
            "Can you explain the different types of savings accounts?"
        ]
        
        self.results = []
        
    def get_timestamp(self) -> str:
        """Get current timestamp for file naming"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_audio_filename(self, tts_engine: str, llm_model: str, voice: str, prompt_index: int) -> str:
        """Generate filename for audio output"""
        timestamp = self.get_timestamp()
        safe_prompt = f"prompt{prompt_index + 1}"
        return f"{tts_engine}_{llm_model}_{voice}_{safe_prompt}_{timestamp}.wav"
    
    async def test_kokoro_tts(self, text: str, voice: str = "af_bella") -> Tuple[bool, Optional[bytes], Dict]:
        """Test Kokoro TTS engine"""
        try:
            if not VOICEBOT_AVAILABLE:
                return False, None, {"error": "Voicebot orchestrator not available"}
            
            print(f"    🎙️ Testing Kokoro TTS with voice: {voice}")
            start_time = time.time()
            
            # Initialize Kokoro TTS
            tts = KokoroTTS(voice=voice, language="en", speed=1.0)
            
            # Generate audio
            audio_data = await tts.synthesize(text)
            
            processing_time = time.time() - start_time
            
            if audio_data:
                return True, audio_data, {
                    "engine": "kokoro",
                    "voice": voice,
                    "processing_time": processing_time,
                    "audio_size": len(audio_data)
                }
            else:
                return False, None, {"error": "No audio data generated"}
                
        except Exception as e:
            return False, None, {"error": str(e)}
    
    async def test_hira_dia_tts(self, text: str, voice: str = "nari") -> Tuple[bool, Optional[bytes], Dict]:
        """Test Hira Dia TTS engine"""
        try:
            print(f"    🎙️ Testing Hira Dia TTS with voice: {voice}")
            start_time = time.time()
            
            # For now, we'll create a mock implementation
            # In a real scenario, you'd integrate with the actual Hira Dia TTS
            
            # Simulate processing time
            await asyncio.sleep(2.0)
            
            # Create mock audio data (silence)
            sample_rate = 22050
            duration = 3  # 3 seconds
            num_samples = sample_rate * duration
            
            # Generate simple sine wave as mock audio
            import math
            audio_samples = []
            for i in range(num_samples):
                # 440 Hz sine wave (A note)
                value = int(32767 * 0.1 * math.sin(2 * math.pi * 440 * i / sample_rate))
                audio_samples.extend([value & 0xFF, (value >> 8) & 0xFF])
            
            # Create WAV header
            wav_header = b'RIFF'
            wav_header += (36 + len(audio_samples)).to_bytes(4, 'little')
            wav_header += b'WAVE'
            wav_header += b'fmt '
            wav_header += (16).to_bytes(4, 'little')
            wav_header += (1).to_bytes(2, 'little')   # PCM
            wav_header += (1).to_bytes(2, 'little')   # Mono
            wav_header += sample_rate.to_bytes(4, 'little')
            wav_header += (sample_rate * 2).to_bytes(4, 'little')
            wav_header += (2).to_bytes(2, 'little')
            wav_header += (16).to_bytes(2, 'little')
            wav_header += b'data'
            wav_header += len(audio_samples).to_bytes(4, 'little')
            
            audio_data = wav_header + bytes(audio_samples)
            
            processing_time = time.time() - start_time
            
            return True, audio_data, {
                "engine": "hira_dia",
                "voice": voice,
                "processing_time": processing_time,
                "audio_size": len(audio_data),
                "note": "Mock implementation - replace with real Hira Dia TTS"
            }
            
        except Exception as e:
            return False, None, {"error": str(e)}
    
    async def test_mistral_llm(self, prompt: str) -> Tuple[bool, Optional[str], Dict]:
        """Test Mistral LLM"""
        try:
            print(f"    🧠 Testing Mistral LLM")
            start_time = time.time()
            
            if REAL_LLM_AVAILABLE:
                # Use real Ollama/Mistral
                llm = RealOllamaLLM(model_name="mistral")
                
                # Try to generate response
                response = await llm.generate_response(prompt)
                
                processing_time = time.time() - start_time
                
                if response and response.get('response'):
                    return True, response['response'], {
                        "model": "mistral",
                        "type": "ollama",
                        "processing_time": processing_time,
                        "tokens": response.get('tokens', 0)
                    }
                else:
                    # Fallback to mock
                    return await self._mock_llm_response(prompt, "mistral", processing_time)
            else:
                # Mock response
                return await self._mock_llm_response(prompt, "mistral", 0)
                
        except Exception as e:
            print(f"    ⚠️ Mistral LLM error, using mock: {e}")
            return await self._mock_llm_response(prompt, "mistral", 0)
    
    async def test_gpt_llm(self, prompt: str) -> Tuple[bool, Optional[str], Dict]:
        """Test GPT LLM"""
        try:
            print(f"    🧠 Testing GPT LLM")
            start_time = time.time()
            
            # For now, use mock implementation
            # In real scenario, integrate with OpenAI API
            await asyncio.sleep(1.5)  # Simulate API call
            
            processing_time = time.time() - start_time
            
            return await self._mock_llm_response(prompt, "gpt", processing_time)
            
        except Exception as e:
            return False, None, {"error": str(e)}
    
    async def _mock_llm_response(self, prompt: str, model: str, processing_time: float) -> Tuple[bool, str, Dict]:
        """Generate mock LLM response"""
        
        # Banking-specific mock responses
        responses = {
            "account balance": "Your current account balance is $2,457.83. You have a checking account ending in 4829 with us.",
            "loan": "We offer several loan options including personal loans, auto loans, and mortgages. Current rates start at 4.2% APR.",
            "interest rates": "Our current savings account interest rate is 2.1% APY, and our high-yield savings offers 3.4% APY.",
            "transaction dispute": "I can help you dispute a transaction. Please provide the transaction date and amount, and I'll start the dispute process.",
            "savings accounts": "We offer regular savings with 2.1% APY, high-yield savings with 3.4% APY, and money market accounts with 3.8% APY."
        }
        
        # Simple keyword matching for mock responses
        response_text = "I'm a banking assistant powered by AI. I can help you with account inquiries, loan applications, and general banking services."
        
        for keyword, response in responses.items():
            if keyword.lower() in prompt.lower():
                response_text = response
                break
        
        return True, response_text, {
            "model": model,
            "type": "mock",
            "processing_time": processing_time,
            "tokens": len(response_text.split())
        }
    
    async def test_combination(self, tts_engine: str, llm_model: str, prompt: str, prompt_index: int) -> Dict:
        """Test a specific TTS/LLM combination"""
        
        combination_name = f"{tts_engine.upper()} + {llm_model.upper()}"
        print(f"\n🧪 Testing {combination_name}")
        print(f"Prompt: {prompt}")
        print("-" * 60)
        
        result = {
            "combination": combination_name,
            "tts_engine": tts_engine,
            "llm_model": llm_model,
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
            # Step 1: Generate LLM response
            if llm_model == "mistral":
                llm_success, llm_response, llm_metadata = await self.test_mistral_llm(prompt)
            elif llm_model == "gpt":
                llm_success, llm_response, llm_metadata = await self.test_gpt_llm(prompt)
            else:
                raise ValueError(f"Unknown LLM model: {llm_model}")
            
            if not llm_success:
                result["errors"].append(f"LLM generation failed: {llm_metadata.get('error', 'Unknown error')}")
                return result
            
            result["llm_result"] = {
                "success": True,
                "response": llm_response,
                "metadata": llm_metadata
            }
            
            # Step 2: Generate TTS audio
            voice = self.tts_engines[tts_engine]["default_voice"]
            
            if tts_engine == "kokoro":
                tts_success, audio_data, tts_metadata = await self.test_kokoro_tts(llm_response, voice)
            elif tts_engine == "hira_dia":
                tts_success, audio_data, tts_metadata = await self.test_hira_dia_tts(llm_response, voice)
            else:
                raise ValueError(f"Unknown TTS engine: {tts_engine}")
            
            if not tts_success:
                result["errors"].append(f"TTS generation failed: {tts_metadata.get('error', 'Unknown error')}")
                return result
            
            result["tts_result"] = {
                "success": True,
                "metadata": tts_metadata
            }
            
            # Step 3: Save audio file
            if audio_data:
                filename = self.get_audio_filename(tts_engine, llm_model, voice, prompt_index)
                file_path = self.audio_dir / filename
                
                with open(file_path, "wb") as f:
                    f.write(audio_data)
                
                result["audio_file"] = str(file_path)
                print(f"    💾 Audio saved: {filename}")
            
            result["success"] = True
            result["total_time"] = time.time() - start_time
            
            print(f"    ✅ Combination successful! Total time: {result['total_time']:.2f}s")
            
        except Exception as e:
            result["errors"].append(f"Combination test failed: {str(e)}")
            result["total_time"] = time.time() - start_time
            print(f"    ❌ Combination failed: {e}")
        
        return result
    
    async def run_all_combinations(self):
        """Run all TTS/LLM combinations"""
        
        print("🎭 TTS/LLM Combination Testing Suite")
        print("=" * 70)
        print(f"Audio output directory: {self.audio_dir}")
        print(f"Testing {len(self.tts_engines)} TTS engines × {len(self.llm_models)} LLM models × {len(self.test_prompts)} prompts")
        print(f"Total combinations: {len(self.tts_engines) * len(self.llm_models) * len(self.test_prompts)}")
        
        # Test each combination
        for prompt_index, prompt in enumerate(self.test_prompts):
            for tts_engine in self.tts_engines.keys():
                for llm_model in self.llm_models.keys():
                    result = await self.test_combination(tts_engine, llm_model, prompt, prompt_index)
                    self.results.append(result)
                    
                    # Small delay between tests
                    await asyncio.sleep(1)
        
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
        print(f"Success Rate: {(successful_tests/total_tests*100):.1f}%")
        
        # Breakdown by combination
        print(f"\n📋 Results by Combination:")
        combination_stats = {}
        
        for result in self.results:
            combo = result["combination"]
            if combo not in combination_stats:
                combination_stats[combo] = {"total": 0, "success": 0}
            
            combination_stats[combo]["total"] += 1
            if result["success"]:
                combination_stats[combo]["success"] += 1
        
        for combo, stats in combination_stats.items():
            success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
            status_icon = "✅" if success_rate == 100 else "⚠️" if success_rate > 0 else "❌"
            print(f"  {status_icon} {combo}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Audio files generated
        audio_files = [r["audio_file"] for r in self.results if r["audio_file"]]
        print(f"\n🔊 Audio Files Generated: {len(audio_files)}")
        
        if audio_files:
            print("Generated audio files:")
            for audio_file in audio_files[-10:]:  # Show last 10 files
                filename = Path(audio_file).name
                print(f"  📄 {filename}")
            
            if len(audio_files) > 10:
                print(f"  ... and {len(audio_files) - 10} more files")
        
        # Save detailed JSON report
        report_file = self.audio_dir / f"test_report_{self.get_timestamp()}.json"
        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📋 Detailed report saved: {report_file}")
        
        # Performance summary
        successful_results = [r for r in self.results if r["success"]]
        if successful_results:
            avg_time = sum(r["total_time"] for r in successful_results) / len(successful_results)
            print(f"\n⚡ Average processing time: {avg_time:.2f}s")
        
        print(f"\n🎉 Testing complete! Check {self.audio_dir} for audio files.")

async def main():
    """Main entry point"""
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TTS/LLM combinations")
    parser.add_argument("--tts", choices=["kokoro", "hira_dia"], 
                        help="Test specific TTS engine only")
    parser.add_argument("--llm", choices=["mistral", "gpt"],
                        help="Test specific LLM model only")
    parser.add_argument("--prompt", type=int, 
                        help="Test specific prompt index only (0-4)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with one prompt per combination")
    
    args = parser.parse_args()
    
    tester = TTSLLMCombinationTester()
    
    # Filter tests based on arguments
    if args.tts:
        tester.tts_engines = {k: v for k, v in tester.tts_engines.items() if k == args.tts}
    
    if args.llm:
        tester.llm_models = {k: v for k, v in tester.llm_models.items() if k == args.llm}
    
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
