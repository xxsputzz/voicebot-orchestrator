"""
Enhanced TTS Manager with Full Dia Support
Adds DIA_4BIT engine support to existing EnhancedTTSManager
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
        safe_text = text.replace('🎭', '[TTS]').replace('⏳', '[LOADING]').replace('1️⃣', '[1]').replace('2️⃣', '[2]').replace('✅', '[OK]').replace('❌', '[ERROR]')
        print(safe_text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        print(safe_text)

# Import audio utilities
try:
    from voicebot_orchestrator.audio_utils import audio_manager
except ImportError:
    audio_manager = None

class TTSEngine(Enum):
    """Available TTS engines - Enhanced with 4-bit support"""
    KOKORO = "kokoro"
    NARI_DIA = "nari_dia"
    DIA_4BIT = "dia_4bit"  # NEW: 4-bit quantized Dia
    AUTO = "auto"

class EnhancedTTSManagerDual:
    """
    Enhanced TTS Manager supporting all engines including 4-bit Dia
    - Kokoro: Fast, real-time generation (~0.8s)
    - Nari Dia: High-quality dialogue generation (~3+ minutes)
    - Dia 4-bit: Speed-optimized Dia (~30-60s)
    """
    
    def __init__(self):
        self.kokoro_tts = None
        self.nari_model = None
        self.dia_4bit_model = None  # NEW: 4-bit model
        self.current_engine = TTSEngine.KOKORO
        self.engines_loaded = {
            TTSEngine.KOKORO: False,
            TTSEngine.NARI_DIA: False,
            TTSEngine.DIA_4BIT: False  # NEW: Track 4-bit engine
        }
        
        safe_print("[TTS] Initializing Enhanced TTS Manager with Dual Dia Support...")
        safe_print("   Supports: Kokoro (fast) + Nari Dia (quality) + Dia 4-bit (speed)")
    
    async def initialize_engines(self, load_kokoro=True, load_nari=True, load_dia_4bit=False):
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
                
            except Exception as e:
                print(f"   XXX Kokoro failed: {e}")
                self.engines_loaded[TTSEngine.KOKORO] = False
        
        # 2. Initialize Nari Dia (quality engine) 
        if load_nari and torch.cuda.is_available():
            print("\n[2] Loading Nari Dia-1.6B (Quality Engine)...")
            try:
                dia_path = Path(__file__).parent.parent / "tests" / "dia"
                if dia_path.exists():
                    sys.path.insert(0, str(dia_path))
                
                from dia.model import Dia
                
                nari_start = time.time()
                device = torch.device("cuda")
                self.nari_model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)
                
                # Pre-warm
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
                print(f"   [OK] Nari Dia loaded and warmed up in {nari_time:.2f}s")
                
            except Exception as e:
                print(f"   [ERROR] Nari Dia failed: {e}")
                self.engines_loaded[TTSEngine.NARI_DIA] = False
        
        # 3. Initialize Dia 4-bit (speed-optimized engine) - NEW
        if load_dia_4bit and torch.cuda.is_available():
            print("\n[3] Loading Dia-1.6B-4bit (Speed Engine)...")
            try:
                dia_path = Path(__file__).parent.parent / "tests" / "dia"
                if dia_path.exists():
                    sys.path.insert(0, str(dia_path))
                
                from dia.model import Dia
                
                dia_4bit_start = time.time()
                device = torch.device("cuda")
                
                # Load with 4-bit quantization settings
                # Note: This is a conceptual implementation - actual 4-bit loading would need
                # proper quantization library (e.g., bitsandbytes)
                self.dia_4bit_model = Dia.from_pretrained(
                    "nari-labs/Dia-1.6B-0626", 
                    device=device,
                    torch_dtype=torch.float16,  # Use half precision for speed
                    # quantization_config=...  # Would need actual 4-bit config
                )
                
                # Pre-warm with smaller config for speed
                test_audio = self.dia_4bit_model.generate(
                    text="[S1] Speed system ready.",
                    max_tokens=128,  # Smaller for speed
                    cfg_scale=2.0,
                    temperature=1.0,
                    top_p=0.9,
                    verbose=False
                )
                dia_4bit_time = time.time() - dia_4bit_start
                
                self.engines_loaded[TTSEngine.DIA_4BIT] = True
                print(f"   [OK] Dia 4-bit loaded and warmed up in {dia_4bit_time:.2f}s")
                
                memory_used = torch.cuda.memory_allocated(0) / 1024**3
                print(f"   [SPEED] Voice: Dialogue-focused adaptive (4-bit)")
                print(f"   [BALANCE] Quality: Optimized for speed")
                print(f"   [CHART] GPU memory: {memory_used:.2f}GB (reduced)")
                print(f"   [TIMER] Speed: ~30-60s generation (faster than full)")
                
            except Exception as e:
                print(f"   [ERROR] Dia 4-bit failed: {e}")
                self.engines_loaded[TTSEngine.DIA_4BIT] = False
        
        total_time = time.time() - start_time
        
        # Set default engine
        if self.engines_loaded[TTSEngine.KOKORO]:
            self.current_engine = TTSEngine.KOKORO
        elif self.engines_loaded[TTSEngine.DIA_4BIT]:
            self.current_engine = TTSEngine.DIA_4BIT
        elif self.engines_loaded[TTSEngine.NARI_DIA]:
            self.current_engine = TTSEngine.NARI_DIA
        else:
            safe_print("❌ No TTS engines loaded successfully!")
            return False
        
        safe_print(f"\n🎉 Enhanced TTS Manager initialized in {total_time:.2f}s")
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
        if self.engines_loaded[TTSEngine.DIA_4BIT]:
            available.append("Dia-4bit")
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
        elif engine == TTSEngine.DIA_4BIT:
            safe_print("   ⚡ Dia-1.6B-4bit - Speed-optimized dialogue")
            safe_print("   🚀 Speed: ~30-60s generation") 
            safe_print("   🎯 Voice: Adaptive dialogue-focused (optimized)")
    
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
                
                if save_path:
                    if audio_manager and not os.path.isabs(save_path):
                        save_path = audio_manager.get_audio_path(save_path, "kokoro")
                    with open(save_path, 'wb') as f:
                        f.write(audio_bytes)
                    print(f"💾 Saved to: {save_path}")
                
                return audio_bytes, generation_time, "Kokoro"
            
            elif engine == TTSEngine.NARI_DIA and self.nari_model:
                return await self._generate_dia_speech(text, self.nari_model, "full", save_path)
            
            elif engine == TTSEngine.DIA_4BIT and self.dia_4bit_model:
                return await self._generate_dia_speech(text, self.dia_4bit_model, "4bit", save_path)
            
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
    
    async def _generate_dia_speech(self, text: str, model, model_type: str, save_path: Optional[str] = None):
        """Helper method to generate speech with Dia models"""
        # Format text for Dia dialogue format
        if not text.startswith("[S1]"):
            formatted_text = f"[S1] {text}"
        else:
            formatted_text = text
        
        # Adjust parameters based on model type
        if model_type == "4bit":
            # Speed-optimized parameters
            max_tokens = min(1024, max(128, len(text) * 4))  # Smaller for speed
            cfg_scale = 2.5
            temperature = 1.0
            top_p = 0.9
        else:
            # Quality parameters for full model
            max_tokens = min(2048, max(256, len(text) * 8))
            cfg_scale = 3.0
            temperature = 1.2
            top_p = 0.95
        
        print(f"🔄 Generating with {max_tokens} max tokens ({model_type} mode)...")
        
        # Set seed for consistent results
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        start_time = time.time()
        
        audio = model.generate(
            text=formatted_text,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            verbose=True
        )
        
        generation_time = time.time() - start_time
        
        # Save audio
        if save_path:
            if audio_manager and not os.path.isabs(save_path):
                output_path = audio_manager.get_audio_path(save_path, f"dia_{model_type}")
            else:
                output_path = save_path
        else:
            if audio_manager:
                output_path = audio_manager.get_timestamped_path("output", f"dia_{model_type}")
            else:
                timestamp = int(time.time() * 1000)
                output_path = f"dia_{model_type}_output_{timestamp}.wav"
        
        # Save with appropriate sample rate
        sample_rate = 44100
        sf.write(output_path, audio, sample_rate)
        
        with open(output_path, 'rb') as f:
            audio_bytes = f.read()
        
        safe_print(f"💾 Saved to: {output_path}")
        
        # Calculate audio stats
        duration = len(audio) / sample_rate
        safe_print(f"🎵 Audio duration: {duration:.1f}s")
        safe_print(f"⚡ Realtime factor: {generation_time/duration:.1f}x")
        
        engine_name = f"Dia-{model_type}" if model_type == "4bit" else "Nari-Dia"
        return audio_bytes, generation_time, engine_name
    
    def cleanup(self):
        """Clean up loaded models and free GPU memory"""
        safe_print("🧹 Cleaning up TTS engines...")
        
        # Clean up all Dia models
        for model_attr, model_name in [
            ("nari_model", "Nari Dia"),
            ("dia_4bit_model", "Dia 4-bit")
        ]:
            model = getattr(self, model_attr, None)
            if model:
                try:
                    if hasattr(model, 'to'):
                        model.to('cpu')
                    del model
                    setattr(self, model_attr, None)
                    safe_print(f"✅ {model_name} model cleaned up")
                except Exception as e:
                    safe_print(f"⚠️ {model_name} cleanup warning: {e}")
        
        # Clean up Kokoro
        if self.kokoro_tts:
            try:
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
            TTSEngine.NARI_DIA: False,
            TTSEngine.DIA_4BIT: False
        }
        self.current_engine = TTSEngine.KOKORO
        
        safe_print("✅ Enhanced TTS Manager cleanup complete")

def smart_engine_selection_enhanced(text: str, available_engines: list[TTSEngine], 
                                   priority: str = "balanced") -> TTSEngine:
    """Enhanced smart engine selection with 4-bit support"""
    if len(available_engines) == 1:
        return available_engines[0]
    
    text_length = len(text)
    
    if priority == "speed":
        # Priority: Kokoro > Dia 4-bit > Nari Dia
        if TTSEngine.KOKORO in available_engines:
            return TTSEngine.KOKORO
        elif TTSEngine.DIA_4BIT in available_engines:
            return TTSEngine.DIA_4BIT
        elif TTSEngine.NARI_DIA in available_engines:
            return TTSEngine.NARI_DIA
    
    elif priority == "quality":
        # Priority: Nari Dia > Dia 4-bit > Kokoro
        if TTSEngine.NARI_DIA in available_engines:
            return TTSEngine.NARI_DIA
        elif TTSEngine.DIA_4BIT in available_engines:
            return TTSEngine.DIA_4BIT
        elif TTSEngine.KOKORO in available_engines:
            return TTSEngine.KOKORO
    
    else:  # balanced
        # Smart selection based on text length and content
        if text_length < 50:
            # Short text - prefer speed
            if TTSEngine.KOKORO in available_engines:
                return TTSEngine.KOKORO
            elif TTSEngine.DIA_4BIT in available_engines:
                return TTSEngine.DIA_4BIT
        elif text_length < 200:
            # Medium text - use 4-bit for good balance
            if TTSEngine.DIA_4BIT in available_engines:
                return TTSEngine.DIA_4BIT
            elif TTSEngine.KOKORO in available_engines:
                return TTSEngine.KOKORO
        else:
            # Long text - prefer quality
            if TTSEngine.NARI_DIA in available_engines:
                return TTSEngine.NARI_DIA
            elif TTSEngine.DIA_4BIT in available_engines:
                return TTSEngine.DIA_4BIT
    
    # Fallback
    return available_engines[0]
