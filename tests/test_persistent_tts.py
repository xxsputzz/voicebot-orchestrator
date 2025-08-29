"""
Persistent TTS Manager - Keeps models loaded for fast generation
"""
import asyncio
import time
import torch
import gc
import sys
import os
from typing import Optional, Union
from pathlib import Path

class PersistentTTSManager:
    """
    Manages TTS models with persistent loading for minimal latency.
    Pre-loads models during initialization and keeps them in memory.
    """
    
    def __init__(self):
        self.kokoro_tts = None
        self.nari_model = None
        self.edge_available = False
        self.models_loaded = False
        
        print("🚀 Initializing Persistent TTS Manager...")
    
    async def initialize_models(self, load_kokoro=True, load_nari=True, load_edge=True):
        """Pre-load all TTS models for fast generation"""
        print("⏳ Loading TTS models (one-time setup)...")
        start_time = time.time()
        
        # 1. Load Kokoro TTS
        if load_kokoro:
            print("1️⃣ Loading Kokoro TTS...")
            try:
                sys.path.append('..')
                from voicebot_orchestrator.tts import KokoroTTS
                
                kokoro_start = time.time()
                self.kokoro_tts = KokoroTTS(voice="af_bella")
                # Pre-warm with a test phrase
                await self.kokoro_tts.synthesize_speech("Initializing voice system.")
                kokoro_time = time.time() - kokoro_start
                
                print(f"   ✅ Kokoro loaded and warmed up in {kokoro_time:.2f}s")
            except Exception as e:
                print(f"   ❌ Kokoro failed: {e}")
                self.kokoro_tts = None
        
        # 2. Load Nari Dia (if GPU available)
        if load_nari and torch.cuda.is_available():
            print("2️⃣ Loading Nari Dia-1.6B...")
            try:
                sys.path.append(os.path.join('..', 'tests', 'dia'))
                from dia.model import Dia
                
                nari_start = time.time()
                device = torch.device("cuda")
                self.nari_model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
                
                # Pre-warm with a test phrase using proper format
                test_audio = self.nari_model.generate(
                    text="[S1] Initializing dialogue system. [S2] Voice system ready.",
                    max_tokens=256,  # Short for warm-up
                    cfg_scale=2.0,
                    temperature=1.0,
                    top_p=0.9
                )
                nari_time = time.time() - nari_start
                
                print(f"   ✅ Nari Dia loaded and warmed up in {nari_time:.2f}s")
                
                # Check GPU memory usage
                memory_used = torch.cuda.memory_allocated(0) / 1024**3
                print(f"   📊 GPU memory used: {memory_used:.2f}GB")
                
            except Exception as e:
                print(f"   ❌ Nari Dia failed: {e}")
                self.nari_model = None
        elif load_nari:
            print("   ⚠️ Nari Dia requires CUDA - skipping")
        
        # 3. Check Edge-TTS availability
        if load_edge:
            print("3️⃣ Checking Edge-TTS...")
            try:
                import edge_tts
                self.edge_available = True
                print("   ✅ Edge-TTS available")
            except ImportError:
                print("   ⚠️ Edge-TTS not installed")
                self.edge_available = False
        
        total_time = time.time() - start_time
        self.models_loaded = True
        
        print(f"🎉 All models loaded in {total_time:.2f}s")
        print(f"🎯 Available TTS engines: {self._get_available_engines()}")
        print()
    
    def _get_available_engines(self):
        """Get list of available TTS engines"""
        engines = []
        if self.kokoro_tts:
            engines.append("Kokoro")
        if self.nari_model:
            engines.append("Nari-Dia")
        if self.edge_available:
            engines.append("Edge-TTS")
        return ", ".join(engines) if engines else "None"
    
    async def generate_speech_fast(self, text: str, engine: str = "auto", voice_style: str = "natural") -> tuple[bytes, float, str]:
        """
        Generate speech with minimal latency using pre-loaded models
        
        Returns: (audio_bytes, generation_time, engine_used)
        """
        if not self.models_loaded:
            raise RuntimeError("Models not initialized. Call initialize_models() first.")
        
        # Auto-select best engine
        if engine == "auto":
            if self.nari_model and len(text) > 20:  # Use Nari for longer, more natural text
                engine = "nari"
            elif self.kokoro_tts:  # Use Kokoro for fast, reliable generation
                engine = "kokoro"
            elif self.edge_available:  # Use Edge as fallback
                engine = "edge"
            else:
                raise RuntimeError("No TTS engines available")
        
        start_time = time.time()
        
        try:
            if engine == "nari" and self.nari_model:
                # Format text for Nari Dia (dialogue format)
                formatted_text = f"[S1] {text}"
                if len(text) > 50:  # Add response for longer conversations
                    formatted_text = f"[S1] {text} [S2] I understand."
                
                audio = self.nari_model.generate(
                    text=formatted_text,
                    max_tokens=min(1024, len(text) * 10),  # Scale tokens with text length
                    cfg_scale=2.5,
                    temperature=1.1,
                    top_p=0.92
                )
                
                # Convert to bytes (simplified - would need proper audio encoding)
                import soundfile as sf
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    sf.write(tmp.name, audio, 44100)
                    with open(tmp.name, 'rb') as f:
                        audio_bytes = f.read()
                    os.unlink(tmp.name)
                
                generation_time = time.time() - start_time
                return audio_bytes, generation_time, "Nari-Dia"
            
            elif engine == "kokoro" and self.kokoro_tts:
                audio_bytes = await self.kokoro_tts.synthesize_speech(text)
                generation_time = time.time() - start_time
                return audio_bytes, generation_time, "Kokoro"
            
            elif engine == "edge" and self.edge_available:
                import edge_tts
                import tempfile
                
                # Use natural female voice
                voice = "en-US-AriaNeural"  # Professional, clear
                communicate = edge_tts.Communicate(text, voice)
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    await communicate.save(tmp.name)
                    with open(tmp.name, 'rb') as f:
                        audio_bytes = f.read()
                    os.unlink(tmp.name)
                
                generation_time = time.time() - start_time
                return audio_bytes, generation_time, "Edge-TTS"
            
            else:
                raise RuntimeError(f"Engine '{engine}' not available")
                
        except Exception as e:
            # Fallback to available engine
            if engine != "kokoro" and self.kokoro_tts:
                print(f"⚠️ {engine} failed, falling back to Kokoro: {e}")
                return await self.generate_speech_fast(text, "kokoro", voice_style)
            elif engine != "edge" and self.edge_available:
                print(f"⚠️ {engine} failed, falling back to Edge-TTS: {e}")
                return await self.generate_speech_fast(text, "edge", voice_style)
            else:
                raise e
    
    async def benchmark_speeds(self):
        """Benchmark all available engines with persistent loading"""
        print("🏁 Benchmarking TTS speeds with persistent models...")
        
        test_phrases = [
            "Hello, how can I help you?",  # Short
            "I understand your concern. Let me check your account balance for you.",  # Medium
            "Thank you for calling our banking services. I've found your account information.",  # Long
        ]
        
        results = {}
        
        for engine in ["kokoro", "nari", "edge"]:
            if (engine == "kokoro" and self.kokoro_tts) or \
               (engine == "nari" and self.nari_model) or \
               (engine == "edge" and self.edge_available):
                
                print(f"\n🎭 Testing {engine.upper()}...")
                times = []
                
                for i, text in enumerate(test_phrases, 1):
                    try:
                        _, gen_time, used_engine = await self.generate_speech_fast(text, engine)
                        times.append(gen_time)
                        print(f"   {i}. {gen_time:.3f}s - \"{text[:30]}...\"")
                    except Exception as e:
                        print(f"   {i}. FAILED - {e}")
                
                if times:
                    avg_time = sum(times) / len(times)
                    results[engine] = avg_time
                    print(f"   📊 Average: {avg_time:.3f}s")
        
        print(f"\n🏆 SPEED RANKING (with persistent models):")
        for engine, avg_time in sorted(results.items(), key=lambda x: x[1]):
            status = "🚀 REAL-TIME" if avg_time < 1.0 else "⚡ FAST" if avg_time < 2.0 else "⏳ SLOW"
            print(f"   {status} {engine.upper()}: {avg_time:.3f}s average")
        
        return results
    
    def cleanup(self):
        """Clean up loaded models"""
        print("🧹 Cleaning up TTS models...")
        
        if self.nari_model:
            del self.nari_model
            torch.cuda.empty_cache()
            gc.collect()
        
        self.kokoro_tts = None
        self.models_loaded = False
        print("✅ Cleanup complete")

# Test the persistent TTS manager
async def test_persistent_tts():
    """Test the persistent TTS manager"""
    manager = PersistentTTSManager()
    
    try:
        # Initialize all models
        await manager.initialize_models(load_kokoro=True, load_nari=True, load_edge=True)
        
        # Benchmark speeds
        await manager.benchmark_speeds()
        
        # Test real conversation scenario
        print(f"\n💬 REAL CONVERSATION TEST:")
        conversation = [
            "Hello, welcome to our banking services.",
            "How may I assist you today?",
            "I can help you check your balance.",
            "Your current balance is one thousand two hundred dollars.",
            "Is there anything else I can help you with?"
        ]
        
        total_time = 0
        for i, text in enumerate(conversation, 1):
            audio_bytes, gen_time, engine = await manager.generate_speech_fast(text, "auto")
            total_time += gen_time
            print(f"{i}. [{engine}] {gen_time:.3f}s - \"{text}\"")
        
        print(f"\n🎯 Total conversation time: {total_time:.2f}s")
        print(f"⚡ Average per response: {total_time/len(conversation):.3f}s")
        
        if total_time / len(conversation) < 1.0:
            print("✅ REAL-TIME conversation capable!")
        else:
            print("⚠️ Still needs optimization for real-time")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        manager.cleanup()

if __name__ == "__main__":
    asyncio.run(test_persistent_tts())
