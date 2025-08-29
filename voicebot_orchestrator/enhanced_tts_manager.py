"""
Enhanced TTS Manager with Kokoro and Nari Dia support
Includes CLI toggle between engines for different use cases
"""
import asyncio
import time
import torch
import gc
import sys
import os
import tempfile
import soundfile as sf
from typing import Optional, Union, Literal
from pathlib import Path
from enum import Enum

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests', 'dia'))

def safe_print(text):
    """Safe print function that handles Unicode encoding issues on Windows"""
    try:
        # First try to replace problematic Unicode characters
        safe_text = text.replace('🎭', '[TTS]').replace('⏳', '[LOADING]').replace('1️⃣', '[1]').replace('2️⃣', '[2]').replace('✅', '[OK]').replace('❌', '[ERROR]')
        print(safe_text)
    except UnicodeEncodeError:
        # Fallback: remove all non-ASCII characters
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        print(safe_text)

# Import audio utilities
try:
    from voicebot_orchestrator.audio_utils import audio_manager
except ImportError:
    # Fallback for when audio_utils is not available
    audio_manager = None

class TTSEngine(Enum):
    """Available TTS engines"""
    KOKORO = "kokoro"
    NARI_DIA = "nari_dia"
    AUTO = "auto"

class EnhancedTTSManager:
    """
    Enhanced TTS Manager supporting multiple engines with CLI toggle
    - Kokoro: Fast, real-time generation (~0.8s)
    - Nari Dia: High-quality dialogue generation (~3+ minutes)
    """
    
    def __init__(self):
        self.kokoro_tts = None
        self.nari_model = None
        self.current_engine = TTSEngine.KOKORO
        self.engines_loaded = {
            TTSEngine.KOKORO: False,
            TTSEngine.NARI_DIA: False
        }
        
        safe_print("🎭 Initializing Enhanced TTS Manager...")
        safe_print("   Supports: Kokoro (fast) + Nari Dia (quality)")
    
    async def initialize_engines(self, load_kokoro=True, load_nari=True):
        """Initialize TTS engines based on requirements"""
        safe_print("[LOADING] Initializing TTS engines...")
        start_time = time.time()
        
        # 1. Initialize Kokoro (fast engine)
        if load_kokoro:
            safe_print("\n1. Loading Kokoro TTS (Fast Engine)...")
            try:
                from voicebot_orchestrator.tts import KokoroTTS
                
                kokoro_start = time.time()
                self.kokoro_tts = KokoroTTS(voice="af_bella")
                
                # Pre-warm with test phrase
                await self.kokoro_tts.synthesize_speech("TTS system ready.")
                kokoro_time = time.time() - kokoro_start
                
                self.engines_loaded[TTSEngine.KOKORO] = True
                print(f"   >>> Kokoro loaded and warmed up in {kokoro_time:.2f}s")
                print(f"   >>> Voice: af_bella (African female)")
                print(f"   >>> Speed: ~0.8s average generation")
                
            except Exception as e:
                print(f"   XXX Kokoro failed: {e}")
                self.engines_loaded[TTSEngine.KOKORO] = False
        
        # 2. Initialize Nari Dia (quality engine) 
        if load_nari and torch.cuda.is_available():
            print("\n2️⃣ Loading Nari Dia-1.6B (Quality Engine)...")
            try:
                from dia.model import Dia
                
                nari_start = time.time()
                device = torch.device("cuda")
                self.nari_model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
                
                # Pre-warm with dialogue format
                test_audio = self.nari_model.generate(
                    text="[S1] TTS system initialized.",
                    max_tokens=256,
                    cfg_scale=2.0,
                    temperature=1.0,
                    top_p=0.9,
                    verbose=False
                )
                nari_time = time.time() - nari_start
                
                self.engines_loaded[TTSEngine.NARI_DIA] = True
                print(f"   ✅ Nari Dia loaded and warmed up in {nari_time:.2f}s")
                
                memory_used = torch.cuda.memory_allocated(0) / 1024**3
                print(f"   🎤 Voice: Dialogue-focused adaptive")
                print(f"   🧠 Quality: Maximum naturalness")
                print(f"   📊 GPU memory: {memory_used:.2f}GB")
                print(f"   ⏳ Speed: ~3+ minutes generation")
                
            except Exception as e:
                print(f"   ❌ Nari Dia failed: {e}")
                self.engines_loaded[TTSEngine.NARI_DIA] = False
                
        elif load_nari:
            safe_print("\n2️⃣ Loading Nari Dia-1.6B (Quality Engine)...")
            try:
                # Import Nari Dia if available  
                start_nari = time.time()
                
                # Check for CUDA availability (required for Nari Dia)
                if not torch.cuda.is_available():
                    self.engines_loaded[TTSEngine.NARI_DIA] = False
                    safe_print("   ❌ CUDA not available - Nari Dia requires GPU")
                    return
                
                # Try importing Nari Dia
                try:
                    # Add dia path
                    dia_path = Path(__file__).parent.parent / "tests" / "dia"
                    if dia_path.exists():
                        sys.path.insert(0, str(dia_path))
                    
                    from kokoro import Kokoro
                    from seamless_communication.inference import Translator
                    
                    # Initialize Nari model (Dia-1.6B)
                    self.nari_model = Translator(
                        model_name_or_card="seamlessM4T_v2_large",
                        vocoder_name_or_card="vocoder_v2",
                        device=torch.device("cuda:0"),
                        dtype=torch.float16,
                    )
                    
                    self.engines_loaded[TTSEngine.NARI_DIA] = True
                    load_time = time.time() - start_nari
                    safe_print(f"   ✅ Nari Dia loaded in {load_time:.2f}s")
                    
                except ImportError as e:
                    self.engines_loaded[TTSEngine.NARI_DIA] = False
                    safe_print(f"   ❌ Nari Dia import failed: {e}")
                    safe_print("   📦 Install: pip install seamless-communication")
                    
            except Exception as e:
                self.engines_loaded[TTSEngine.NARI_DIA] = False
                safe_print(f"❌ Nari Dia initialization failed: {e}")
        else:
            safe_print("\n2️⃣ Nari Dia requires CUDA - skipping")
            self.engines_loaded[TTSEngine.NARI_DIA] = False
        
        total_time = time.time() - start_time
        
        # Set default engine
        if self.engines_loaded[TTSEngine.KOKORO]:
            self.current_engine = TTSEngine.KOKORO
        elif self.engines_loaded[TTSEngine.NARI_DIA]:
            self.current_engine = TTSEngine.NARI_DIA
        else:
            safe_print("❌ No TTS engines loaded successfully!")
            return False
        
        safe_print(f"\n🎉 TTS Manager initialized in {total_time:.2f}s")
        safe_print(f"🎯 Available engines: {self._get_available_engines()}")
        safe_print(f"🔧 Default engine: {self.current_engine.value}")
        return True
    
    def _get_available_engines(self):
        """Get list of successfully loaded engines"""
        available = []
        if self.engines_loaded[TTSEngine.KOKORO]:
            available.append("Kokoro")
        if self.engines_loaded[TTSEngine.NARI_DIA]:
            available.append("Nari-Dia")
        return ", ".join(available) if available else "None"
    
    def set_engine(self, engine: Union[TTSEngine, str]):
        """Switch TTS engine"""
        if isinstance(engine, str):
            engine = TTSEngine(engine)
        
        if not self.engines_loaded.get(engine, False):
            available = [e.value for e, loaded in self.engines_loaded.items() if loaded]
            raise ValueError(f"Engine '{engine.value}' not available. Available: {available}")
        
        old_engine = self.current_engine
        self.current_engine = engine
        
        safe_print(f"🔄 Switched TTS engine: {old_engine.value} → {engine.value}")
        self._print_engine_info(engine)
    
    def _print_engine_info(self, engine: TTSEngine):
        """Print information about the specified engine"""
        if engine == TTSEngine.KOKORO:
            safe_print("   🚀 Kokoro TTS - Optimized for real-time conversation")
            safe_print("   ⚡ Speed: ~0.8s generation")
            safe_print("   🎤 Voice: af_bella (professional female)")
        elif engine == TTSEngine.NARI_DIA:
            safe_print("   🎭 Nari Dia-1.6B - Maximum quality dialogue")
            safe_print("   ⏳ Speed: ~3+ minutes generation") 
            safe_print("   🧠 Voice: Adaptive dialogue-focused")
    
    def get_current_engine(self) -> TTSEngine:
        """Get current active engine"""
        return self.current_engine
    
    def get_available_engines(self) -> list[TTSEngine]:
        """Get list of available engines"""
        return [engine for engine, loaded in self.engines_loaded.items() if loaded]
    
    async def generate_speech(self, text: str, engine: Optional[TTSEngine] = None, 
                             save_path: Optional[str] = None) -> tuple[bytes, float, str]:
        """
        Generate speech using specified or current engine
        
        Returns: (audio_bytes, generation_time, engine_used)
        """
        # Use specified engine or current default
        if engine is None:
            engine = self.current_engine
        elif isinstance(engine, str):
            engine = TTSEngine(engine)
        
        # Validate engine availability
        if not self.engines_loaded.get(engine, False):
            # Fall back to available engine
            available = [e for e, loaded in self.engines_loaded.items() if loaded]
            if not available:
                raise RuntimeError("No TTS engines available")
            engine = available[0]
            print(f"⚠️ Requested engine not available, using {engine.value}")
        
        print(f"🎤 Generating speech with {engine.value.upper()}...")
        print(f"📝 Text: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        
        start_time = time.time()
        
        try:
            if engine == TTSEngine.KOKORO and self.kokoro_tts:
                audio_bytes = await self.kokoro_tts.synthesize_speech(text)
                generation_time = time.time() - start_time
                
                # Save if path provided
                if save_path:
                    # Use centralized audio output if available
                    if audio_manager and not os.path.isabs(save_path):
                        save_path = audio_manager.get_audio_path(save_path, "kokoro")
                    with open(save_path, 'wb') as f:
                        f.write(audio_bytes)
                    print(f"💾 Saved to: {save_path}")
                
                return audio_bytes, generation_time, "Kokoro"
            
            elif engine == TTSEngine.NARI_DIA and self.nari_model:
                # Format text for Nari Dia dialogue format
                if not text.startswith("[S1]"):
                    formatted_text = f"[S1] {text}"
                else:
                    formatted_text = text
                
                # Use proven working parameters from test_nari_proper.py
                max_tokens = min(2048, max(256, len(text) * 8))
                
                print(f"🔄 Generating with {max_tokens} max tokens...")
                
                # Set seed for consistent results
                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(42)
                
                audio = self.nari_model.generate(
                    text=formatted_text,
                    max_tokens=max_tokens,
                    cfg_scale=3.0,    # Working parameters from successful test
                    temperature=1.2,  # Working parameters from successful test
                    top_p=0.95,       # Working parameters from successful test
                    verbose=True      # Show progress like successful test
                )
                
                generation_time = time.time() - start_time
                
                # Save audio using working method
                if save_path:
                    # Use centralized audio output if available
                    if audio_manager and not os.path.isabs(save_path):
                        output_path = audio_manager.get_audio_path(save_path, "nari_dia")
                    else:
                        output_path = save_path
                else:
                    # Generate timestamped path
                    if audio_manager:
                        output_path = audio_manager.get_timestamped_path("output", "nari_dia")
                    else:
                        timestamp = int(time.time() * 1000)
                        output_path = f"nari_output_{timestamp}.wav"
                
                # Use working sample rate and save method
                sample_rate = 44100
                sf.write(output_path, audio, sample_rate)
                
                with open(output_path, 'rb') as f:
                    audio_bytes = f.read()
                
                safe_print(f"💾 Saved to: {output_path}")
                
                # Calculate audio stats
                duration = len(audio) / sample_rate
                safe_print(f"🎵 Audio duration: {duration:.1f}s")
                safe_print(f"⚡ Realtime factor: {generation_time/duration:.1f}x")
                
                return audio_bytes, generation_time, "Nari-Dia"
            
            else:
                raise RuntimeError(f"Engine {engine.value} not properly initialized")
                
        except Exception as e:
            safe_print(f"❌ Generation failed with {engine.value}: {e}")
            
            # Try fallback to available engine
            available = [e for e, loaded in self.engines_loaded.items() if loaded and e != engine]
            if available:
                fallback_engine = available[0]
                safe_print(f"🔄 Falling back to {fallback_engine.value}...")
                return await self.generate_speech(text, fallback_engine, save_path)
            else:
                raise e
    
    def cleanup(self):
        """Clean up loaded models and free GPU memory"""
        safe_print("🧹 Cleaning up TTS engines...")
        
        if self.nari_model:
            try:
                # Move model to CPU and delete
                if hasattr(self.nari_model, 'to'):
                    self.nari_model.to('cpu')
                del self.nari_model
                self.nari_model = None
                safe_print("✅ Nari Dia model moved to CPU and deleted")
            except Exception as e:
                safe_print(f"⚠️ Nari Dia cleanup warning: {e}")
        
        if self.kokoro_tts:
            try:
                # Clean up Kokoro TTS
                del self.kokoro_tts
                self.kokoro_tts = None
                safe_print("✅ Kokoro TTS deleted")
            except Exception as e:
                safe_print(f"⚠️ Kokoro cleanup warning: {e}")
        
        # Force GPU memory cleanup
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                safe_print("✅ GPU cache cleared and synchronized")
        except Exception as e:
            safe_print(f"⚠️ GPU cleanup warning: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Reset engine states
        self.engines_loaded = {
            TTSEngine.KOKORO: False,
            TTSEngine.NARI_DIA: False
        }
        self.current_engine = TTSEngine.KOKORO
        
        safe_print("✅ TTS Manager cleanup complete")

# CLI Helper Functions
def get_engine_choice() -> TTSEngine:
    """Interactive engine selection for CLI"""
    safe_print("\n🎭 Select TTS Engine:")
    safe_print("1. 🚀 Kokoro (Fast, Real-time) - ~0.8s generation")
    safe_print("2. 🎭 Nari Dia (Quality, Slow) - ~3+ minutes generation")
    safe_print("3. 🤖 Auto (Smart selection based on context)")
    
    while True:
        try:
            choice = input("\nEnter choice (1-3): ").strip()
            if choice == "1":
                return TTSEngine.KOKORO
            elif choice == "2":
                return TTSEngine.NARI_DIA
            elif choice == "3":
                return TTSEngine.AUTO
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            sys.exit(0)

def smart_engine_selection(text: str, available_engines: list[TTSEngine]) -> TTSEngine:
    """Automatically select best engine based on context"""
    # If only one engine available, use it
    if len(available_engines) == 1:
        return available_engines[0]
    
    # Smart selection rules
    text_length = len(text)
    
    # For short, quick responses - use Kokoro
    if text_length < 50 and TTSEngine.KOKORO in available_engines:
        return TTSEngine.KOKORO
    
    # For medium responses - use Kokoro for speed
    if text_length < 150 and TTSEngine.KOKORO in available_engines:
        return TTSEngine.KOKORO
    
    # For long, complex responses - consider quality vs speed tradeoff
    # In practice, still prefer Kokoro due to Nari Dia's extreme slowness
    if TTSEngine.KOKORO in available_engines:
        return TTSEngine.KOKORO
    
    # Fallback to any available engine
    return available_engines[0]

# Test function
async def test_enhanced_tts():
    """Test the enhanced TTS manager"""
    manager = EnhancedTTSManager()
    
    try:
        # Initialize engines
        await manager.initialize_engines(load_kokoro=True, load_nari=True)
        
        # Interactive engine selection
        selected_engine = get_engine_choice()
        
        if selected_engine != TTSEngine.AUTO:
            manager.set_engine(selected_engine)
        
        # Test phrases
        test_phrases = [
            "Hello, welcome to our banking services.",
            "I can help you check your account balance and recent transactions.",
            "Your current balance is one thousand two hundred and fifty dollars."
        ]
        
        for i, text in enumerate(test_phrases, 1):
            print(f"\n🎬 Test {i}/3:")
            
            # Use auto-selection if AUTO mode
            if selected_engine == TTSEngine.AUTO:
                engine = smart_engine_selection(text, manager.get_available_engines())
                print(f"🤖 Auto-selected: {engine.value}")
            else:
                engine = None  # Use current engine
            
            # Generate speech
            audio_bytes, gen_time, used_engine = await manager.generate_speech(text, engine)
            
            print(f"✅ Generated {len(audio_bytes)} bytes in {gen_time:.3f}s using {used_engine}")
            
            # Quick turnaround assessment
            if gen_time < 1.0:
                print("🚀 REAL-TIME: Perfect for conversation")
            elif gen_time < 3.0:
                print("⚡ FAST: Good for most interactions")
            elif gen_time < 10.0:
                print("🔄 SLOW: Acceptable for non-interactive use")
            else:
                print("⏳ VERY SLOW: Only for highest quality needs")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        manager.cleanup()

if __name__ == "__main__":
    asyncio.run(test_enhanced_tts())
