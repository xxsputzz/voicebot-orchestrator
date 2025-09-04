"""
Real Tortoise TTS Implementation
High-quality neural text-to-speech with voice cloning capabilities and comprehensive GPU management
"""
import torch
import torchaudio
import numpy as np
import io
import base64
from typing import Optional, Dict, List, Tuple, Any
import os
import sys
import time
import atexit
from datetime import datetime

# Add current directory to path for tortoise imports
sys.path.insert(0, os.path.abspath('.'))

# Import GPU management and progress tracking
try:
    from tortoise_gpu_manager import get_gpu_manager
    GPU_MANAGER_AVAILABLE = True
except ImportError:
    print("Warning: GPU manager not available")
    GPU_MANAGER_AVAILABLE = False

try:
    from tortoise_progress_tracker import get_progress_tracker, ProgressDisplay
    PROGRESS_TRACKING_AVAILABLE = True
except ImportError:
    print("Warning: Progress tracking not available")
    PROGRESS_TRACKING_AVAILABLE = False

try:
    from tortoise.api import TextToSpeech
    TORTOISE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import real Tortoise TTS: {e}")
    TORTOISE_AVAILABLE = False

class TortoiseVoiceConfig:
    """Configuration for different voice personalities"""
    
    # Audio output directory
    AUDIO_OUTPUT_DIR = "audio_output"
    
    VOICE_CONFIGS = {
        # Core original voices
        'angie': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'deniro': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'freeman': {'preset': 'fast', 'voice_samples': None, 'conditioning_latents': None},
        'halle': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'jlaw': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'lj': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'mol': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'myself': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'pat': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'pat2': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'rainbow': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'snakes': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'tim_reynolds': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'tom': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'weaver': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'william': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        
        # Additional character voices
        'applejack': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'daniel': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'emma': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        'geralt': {'preset': 'ultra_fast', 'voice_samples': None, 'conditioning_latents': None},
        
        # Training voices (these have actual voice directories)
        'train_atkins': {'preset': 'fast', 'voice_samples': None, 'conditioning_latents': None},
        'train_daws': {'preset': 'fast', 'voice_samples': None, 'conditioning_latents': None},
        'train_dotrice': {'preset': 'fast', 'voice_samples': None, 'conditioning_latents': None},
        'train_dreams': {'preset': 'fast', 'voice_samples': None, 'conditioning_latents': None},
        'train_empire': {'preset': 'fast', 'voice_samples': None, 'conditioning_latents': None},
        'train_grace': {'preset': 'fast', 'voice_samples': None, 'conditioning_latents': None},
        'train_kennard': {'preset': 'fast', 'voice_samples': None, 'conditioning_latents': None},
        'train_lescault': {'preset': 'standard', 'voice_samples': None, 'conditioning_latents': None},
        'train_mouse': {'preset': 'fast', 'voice_samples': None, 'conditioning_latents': None}
    }

class TortoiseTTSEngine:
    """Real Tortoise TTS Engine using the actual neural models with GPU acceleration"""
    
    def __init__(self, device=None):
        self.tts = None
        self.device = self._get_best_device(device)
        self.gpu_manager = get_gpu_manager() if GPU_MANAGER_AVAILABLE else None
        self.progress_tracker = get_progress_tracker() if PROGRESS_TRACKING_AVAILABLE else None
        self._initialize_tts()
        
        # Register cleanup for this engine instance
        if self.gpu_manager:
            atexit.register(self._cleanup_on_exit)
        
    def _cleanup_on_exit(self):
        """Cleanup function called on exit"""
        try:
            if self.gpu_manager:
                print("[TTS_ENGINE] Cleaning up GPU resources...")
                self.gpu_manager.clear_gpu_cache(aggressive=True)
        except Exception as e:
            print(f"[TTS_ENGINE] Error during exit cleanup: {e}")
    
    def _get_best_device(self, device=None):
        """Determine the best device for Tortoise TTS with robust error handling"""
        if device is not None:
            # Validate the requested device
            if device == "cuda":
                if not torch.cuda.is_available():
                    print("⚠️ CUDA requested but not available, falling back to CPU")
                    return "cpu"
                try:
                    # Test CUDA functionality
                    test_tensor = torch.randn(2, 2).to('cuda')
                    device_name = torch.cuda.get_device_name(0)
                    cuda_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                    print(f"🎮 CUDA Available: {device_name} ({cuda_memory:.1f}GB)")
                    return "cuda"
                except Exception as e:
                    print(f"⚠️ CUDA test failed: {e}, falling back to CPU")
                    return "cpu"
            return device
            
        if torch.cuda.is_available():
            try:
                # Test CUDA functionality
                test_tensor = torch.randn(2, 2).to('cuda')
                device_name = torch.cuda.get_device_name(0)
                cuda_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                print(f"🎮 CUDA Available: {device_name} ({cuda_memory:.1f}GB)")
                return "cuda"
            except Exception as e:
                print(f"⚠️ CUDA test failed: {e}, using CPU")
                return "cpu"
        elif torch.backends.mps.is_available():
            print("🍎 MPS (Apple Silicon) Available")
            return "mps"
        else:
            print("💻 Using CPU (GPU not available)")
            return "cpu"
        
    def _initialize_tts(self):
        """Initialize the real Tortoise TTS system with GPU acceleration and error handling"""
        if not TORTOISE_AVAILABLE:
            raise RuntimeError("Tortoise TTS is not available. Please check installation.")
        
        try:
            print(f"🔄 Initializing Real Tortoise TTS on {self.device}...")
            
            # Try to initialize with the requested device
            try:
                self.tts = TextToSpeech(device=self.device)
            except Exception as device_error:
                if self.device == "cuda":
                    print(f"⚠️ CUDA initialization failed: {device_error}")
                    print("🔄 Falling back to CPU...")
                    self.device = "cpu"
                    self.tts = TextToSpeech(device=self.device)
                else:
                    raise device_error
            
            # CRITICAL: Force models to GPU after initialization (if using GPU)
            # Tortoise TTS loads models on CPU by default, so we need to move them manually
            if self.device == "cuda" or self.device == "mps":
                print(f"🔄 Moving models to {self.device}...")
                
                try:
                    # Move models to GPU quietly
                    if hasattr(self.tts, 'autoregressive') and self.tts.autoregressive is not None:
                        self.tts.autoregressive = self.tts.autoregressive.to(self.device)
                        if self.gpu_manager:
                            self.gpu_manager.track_model(self.tts.autoregressive, "autoregressive")
                    
                    # Move diffusion model to GPU  
                    if hasattr(self.tts, 'diffusion') and self.tts.diffusion is not None:
                        self.tts.diffusion = self.tts.diffusion.to(self.device)
                        if self.gpu_manager:
                            self.gpu_manager.track_model(self.tts.diffusion, "diffusion")
                        
                    # Move vocoder to GPU
                    if hasattr(self.tts, 'vocoder') and self.tts.vocoder is not None:
                        self.tts.vocoder = self.tts.vocoder.to(self.device) 
                        if self.gpu_manager:
                            self.gpu_manager.track_model(self.tts.vocoder, "vocoder")
                    
                    # Move CLVP model to GPU if available
                    if hasattr(self.tts, 'clvp') and self.tts.clvp is not None:
                        self.tts.clvp = self.tts.clvp.to(self.device)
                        if self.gpu_manager:
                            self.gpu_manager.track_model(self.tts.clvp, "clvp")
                        
                except Exception as move_error:
                    print(f"⚠️ Failed to move models to {self.device}: {move_error}")
                    print("🔄 Falling back to CPU...")
                    # Cleanup before fallback
                    if self.gpu_manager:
                        self.gpu_manager.clear_gpu_cache(aggressive=True)
                    self.device = "cpu"
                    # Reinitialize on CPU
                    self.tts = TextToSpeech(device=self.device)
            
            print(f"✅ Real Tortoise TTS initialized successfully on {self.device}!")
            
            # Print device info for debugging
            if self.device == "cuda":
                memory_allocated = torch.cuda.memory_allocated(0) / (1024**2)  # MB
                print(f"📊 GPU Memory Allocated: {memory_allocated:.1f}MB")
                
                # Verify models are on GPU
                if hasattr(self.tts, 'autoregressive') and self.tts.autoregressive is not None:
                    ar_device = next(self.tts.autoregressive.parameters()).device
                    print(f"🔍 Autoregressive model device: {ar_device}")
                
                if hasattr(self.tts, 'diffusion') and self.tts.diffusion is not None:
                    diff_device = next(self.tts.diffusion.parameters()).device
                    print(f"🔍 Diffusion model device: {diff_device}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Tortoise TTS: {e}")
            raise
    
    @property
    def device_info(self):
        """Get current device information"""
        return self.device
    
    def generate_speech(self, text: str, voice: str = 'angie', save_audio: bool = True, timeout_seconds: Optional[float] = None, **kwargs) -> Tuple[torch.Tensor, int]:
        """
        Generate speech using real Tortoise TTS with progress tracking and memory management
        
        Args:
            text: Text to synthesize
            voice: Voice name (must be available in VOICE_CONFIGS)
            save_audio: Whether to automatically save audio to file
            timeout_seconds: Maximum time allowed (None for unlimited)
            **kwargs: Additional parameters for TTS generation
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        if self.tts is None:
            raise RuntimeError("TTS engine not initialized")
        
        # Check memory availability before starting
        if self.gpu_manager and self.device == "cuda":
            if not self.gpu_manager.check_memory_available(required_mb=3000):
                self.gpu_manager.force_cleanup_before_synthesis()
                if not self.gpu_manager.check_memory_available(required_mb=2000):
                    raise RuntimeError("Insufficient GPU memory for synthesis")
        
        # Get voice configuration
        voice_config = TortoiseVoiceConfig.VOICE_CONFIGS.get(voice, 
                                                           TortoiseVoiceConfig.VOICE_CONFIGS['angie'])
        
        # Start progress tracking
        operation_name = f"Tortoise synthesis ({len(text)} chars, voice: {voice})"
        
        if self.progress_tracker:
            progress_context = self.progress_tracker.track_operation(operation_name, timeout_seconds)
        else:
            from contextlib import nullcontext
            progress_context = nullcontext()
        
        try:
            with progress_context as tracker:
                print(f"🔄 Generating speech with voice '{voice}' for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
                # Get preset from kwargs first, then fall back to voice config
                preset = kwargs.pop('preset', voice_config.get('preset', 'fast'))
                
                # Import the load_voices function
                from tortoise.utils.audio import load_voices
                
                # Check for official voices directory
                official_voices_dir = os.path.join(os.getcwd(), 'tortoise_voices')
                extra_voice_dirs = []
                if os.path.exists(official_voices_dir):
                    extra_voice_dirs.append(official_voices_dir)
                
                # Try to load voice samples using the proper Tortoise utility
                try:
                    voice_samples, conditioning_latents = load_voices([voice], extra_voice_dirs)
                    print(f"✅ Loaded voice data for '{voice}' - samples: {voice_samples is not None}, latents: {conditioning_latents is not None}")
                    
                    # Generate audio using loaded voice data with interruption handling
                    try:
                        # Check cancellation before synthesis
                        if tracker and tracker.is_cancelled():
                            raise RuntimeError("Synthesis cancelled by timeout")
                        
                        # CRITICAL: Ensure models are on GPU before synthesis
                        if self.device == "cuda" or self.device == "mps":
                            print(f"🔧 Forcing models to {self.device} before synthesis...")
                            if hasattr(self.tts, 'autoregressive') and self.tts.autoregressive is not None:
                                self.tts.autoregressive = self.tts.autoregressive.to(self.device)
                            if hasattr(self.tts, 'diffusion') and self.tts.diffusion is not None:
                                self.tts.diffusion = self.tts.diffusion.to(self.device)
                            if hasattr(self.tts, 'vocoder') and self.tts.vocoder is not None:
                                self.tts.vocoder = self.tts.vocoder.to(self.device)
                            if hasattr(self.tts, 'clvp') and self.tts.clvp is not None:
                                self.tts.clvp = self.tts.clvp.to(self.device)
                        
                        # Progress callback for synthesis
                        def progress_callback(elapsed, timeout):
                            if PROGRESS_TRACKING_AVAILABLE:
                                ProgressDisplay.show_synthesis_progress(elapsed, timeout)
                            # Check cancellation
                            if tracker and tracker.is_cancelled():
                                raise KeyboardInterrupt("Synthesis cancelled by timeout")
                        
                        # Set up progress monitoring
                        if tracker:
                            tracker.progress_callback = progress_callback
                        
                        with torch.no_grad():
                            audio = self.tts.tts_with_preset(
                                text, 
                                voice_samples=voice_samples,
                                conditioning_latents=conditioning_latents,
                                preset=preset,
                                **kwargs
                            )
                        
                        # Clear progress display
                        if PROGRESS_TRACKING_AVAILABLE:
                            ProgressDisplay.clear_progress()
                        
                        # Clear GPU cache after synthesis if using GPU manager
                        if self.gpu_manager and (self.device == "cuda" or self.device == "mps"):
                            self.gpu_manager.clear_gpu_cache()
                            
                    except KeyboardInterrupt:
                        if PROGRESS_TRACKING_AVAILABLE:
                            ProgressDisplay.clear_progress()
                        print("⚠️ Synthesis interrupted by user or timeout")
                        raise RuntimeError("Synthesis interrupted by user or timeout")
                    except Exception as synthesis_error:
                        if PROGRESS_TRACKING_AVAILABLE:
                            ProgressDisplay.clear_progress()
                        if "control-C" in str(synthesis_error) or "cancelled" in str(synthesis_error):
                            print("⚠️ Synthesis interrupted by control-C event or timeout")
                            raise RuntimeError("Synthesis interrupted by control-C event or timeout")
                        else:
                            raise synthesis_error
                            
                except Exception as e:
                    print(f"⚠️ Could not load voice samples for '{voice}': {e}")
                    # Fallback to random generation with error handling
                    try:
                        # Check cancellation before fallback
                        if tracker and tracker.is_cancelled():
                            raise RuntimeError("Synthesis cancelled by timeout")
                        
                        # CRITICAL: Ensure models are on GPU for fallback synthesis too
                        if self.device == "cuda" or self.device == "mps":
                            print(f"🔧 Forcing models to {self.device} for fallback synthesis...")
                            if hasattr(self.tts, 'autoregressive') and self.tts.autoregressive is not None:
                                self.tts.autoregressive = self.tts.autoregressive.to(self.device)
                            if hasattr(self.tts, 'diffusion') and self.tts.diffusion is not None:
                                self.tts.diffusion = self.tts.diffusion.to(self.device)
                            if hasattr(self.tts, 'vocoder') and self.tts.vocoder is not None:
                                self.tts.vocoder = self.tts.vocoder.to(self.device)
                            if hasattr(self.tts, 'clvp') and self.tts.clvp is not None:
                                self.tts.clvp = self.tts.clvp.to(self.device)
                        
                        with torch.no_grad():
                            audio = self.tts.tts_with_preset(
                                text, 
                                voice_samples=None,
                                conditioning_latents=None,
                                preset=preset,
                                **kwargs
                            )
                        
                        # Clear GPU cache after fallback synthesis
                        if self.gpu_manager and (self.device == "cuda" or self.device == "mps"):
                            self.gpu_manager.clear_gpu_cache()
                            
                    except KeyboardInterrupt:
                        print("⚠️ Fallback synthesis interrupted by user or timeout")
                        raise RuntimeError("Synthesis interrupted by user or timeout")
                    except Exception as fallback_error:
                        if "control-C" in str(fallback_error) or "cancelled" in str(fallback_error):
                            print("⚠️ Fallback synthesis interrupted by control-C event or timeout")
                            raise RuntimeError("Synthesis interrupted by control-C event or timeout")
                        else:
                            raise fallback_error
                            
                print(f"✅ Real speech generated successfully! Audio shape: {audio.shape}")
                
                # Tortoise TTS outputs at 24kHz
                sample_rate = 24000
                
                # Prepare audio tensor for return
                audio_output = audio.squeeze(0).cpu()
                
                # Auto-save audio if requested
                if save_audio:
                    self._save_audio_file(audio_output, sample_rate, text, voice)
                
                return audio_output, sample_rate
                
        except Exception as e:
            print(f"❌ Error generating speech: {e}")
            # Emergency cleanup on error
            if self.gpu_manager:
                try:
                    self.gpu_manager.clear_gpu_cache(aggressive=True)
                except:
                    pass
            # Fallback to silence if generation fails
            sample_rate = 24000
            return torch.zeros(sample_rate), sample_rate
    
    def _save_audio_file(self, audio_tensor: torch.Tensor, sample_rate: int, text: str, voice: str) -> str:
        """
        Save audio tensor to a WAV file in the audio output directory
        
        Args:
            audio_tensor: Audio data as tensor
            sample_rate: Audio sample rate
            text: Original text (for filename)
            voice: Voice name used
            
        Returns:
            Path to saved audio file
        """
        try:
            # Ensure output directory exists
            os.makedirs(TortoiseVoiceConfig.AUDIO_OUTPUT_DIR, exist_ok=True)
            
            # Create filename with timestamp and text preview
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            text_preview = "".join(c for c in text[:30] if c.isalnum() or c in " ._-").strip().replace(" ", "_")
            if not text_preview:
                text_preview = "audio"
            
            filename = f"tortoise_{timestamp}_{voice}_{text_preview}.wav"
            filepath = os.path.join(TortoiseVoiceConfig.AUDIO_OUTPUT_DIR, filename)
            
            # Ensure audio is in the right format for saving
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
            
            # Save the audio file
            torchaudio.save(filepath, audio_tensor, sample_rate)
            print(f"💾 Audio saved to: {filepath}")
            
            return filepath
            
        except Exception as e:
            print(f"⚠️ Failed to save audio file: {e}")
            return ""
    
    def save_audio_to_file(self, audio_tensor: torch.Tensor, sample_rate: int, filename: str = None, text: str = "audio", voice: str = "unknown") -> str:
        """
        Manually save audio tensor to a specific file
        
        Args:
            audio_tensor: Audio data as tensor
            sample_rate: Audio sample rate
            filename: Custom filename (optional)
            text: Text description for auto-generated filename
            voice: Voice name for auto-generated filename
            
        Returns:
            Path to saved audio file
        """
        if filename:
            # Use custom filename
            filepath = os.path.join(TortoiseVoiceConfig.AUDIO_OUTPUT_DIR, filename)
            if not filename.endswith('.wav'):
                filepath += '.wav'
        else:
            # Use auto-generated filename
            filepath = self._save_audio_file(audio_tensor, sample_rate, text, voice)
            
        return filepath
    
    def get_available_voices(self) -> List[str]:
        """Get list of actually available voice names (only those that can be loaded)"""
        if not hasattr(self, '_cached_available_voices'):
            # Cache the result since voice checking can be slow
            self._cached_available_voices = self._discover_working_voices()
        return self._cached_available_voices
    
    def _discover_working_voices(self) -> List[str]:
        """Discover which voices actually work by trying to load them"""
        working_voices = []
        
        try:
            from tortoise.utils.audio import load_voices, get_voices
            
            print("[VOICE_DISCOVERY] Discovering available voices...")
            
            # Check for official voices directory first
            official_voices_dir = os.path.join(os.getcwd(), 'tortoise_voices')
            extra_voice_dirs = []
            if os.path.exists(official_voices_dir):
                extra_voice_dirs.append(official_voices_dir)
                print(f"[VOICE_DISCOVERY] Found official voices directory: {official_voices_dir}")
            
            # Get all available voices including from official directory
            all_voices = get_voices(extra_voice_dirs)
            official_voice_names = list(all_voices.keys()) if all_voices else []
            
            # Priority order: official downloaded voices first, then built-ins
            official_downloaded = ['angie', 'daniel', 'deniro', 'emma', 'freeman', 'geralt', 
                                 'halle', 'jlaw', 'lj', 'mol', 'myself', 'pat', 'rainbow', 
                                 'tom', 'train_atkins', 'train_dotrice', 'train_kennard', 
                                 'weaver', 'william']
            
            # Test official downloaded voices first (highest priority)
            print("[VOICE_DISCOVERY] Testing official downloaded voices...")
            for voice in official_downloaded:
                if voice in official_voice_names:
                    try:
                        voice_samples, conditioning_latents = load_voices([voice], extra_voice_dirs)
                        if voice_samples is not None and len(voice_samples) > 0:
                            working_voices.append(voice)
                            print(f"[VOICE_DISCOVERY] ✅ {voice} - official voice available")
                        else:
                            print(f"[VOICE_DISCOVERY] ❌ {voice} - no samples returned")
                    except Exception as e:
                        print(f"[VOICE_DISCOVERY] ❌ {voice} - error: {str(e)[:50]}")
            
            # Test any remaining built-in voices
            builtin_fallbacks = ['snakes', 'tim_reynolds', 'applejack']
            print("[VOICE_DISCOVERY] Testing built-in fallback voices...")
            
            for voice in builtin_fallbacks:
                if len(working_voices) >= 20:  # Reasonable limit
                    break
                if voice not in working_voices:
                    try:
                        voice_samples, conditioning_latents = load_voices([voice])
                        if voice_samples is not None and len(voice_samples) > 0:
                            working_voices.append(voice)
                            print(f"[VOICE_DISCOVERY] ✅ {voice} - built-in available")
                    except Exception:
                        continue
            
            # Add 'denise' which appeared in your interface (might be a built-in alias)
            if 'denise' not in working_voices:
                try:
                    voice_samples, conditioning_latents = load_voices(['denise'])
                    if voice_samples is not None and len(voice_samples) > 0:
                        working_voices.append('denise')
                        print(f"[VOICE_DISCOVERY] ✅ denise - found as built-in")
                except Exception:
                    pass
            
            if working_voices:
                print(f"[VOICE_DISCOVERY] Found {len(working_voices)} working voices: {', '.join(working_voices)}")
            else:
                print("[VOICE_DISCOVERY] ⚠️ No working voices found, using minimal fallback")
                working_voices = ['angie', 'freeman', 'denise']  # Your interface showed these 3
                
        except ImportError:
            print("[VOICE_DISCOVERY] ⚠️ Cannot import load_voices, using fallback voices")
            working_voices = ['angie', 'freeman', 'denise', 'pat', 'william', 'tom']
        except Exception as e:
            print(f"[VOICE_DISCOVERY] ⚠️ Error discovering voices: {e}, using fallback")
            working_voices = ['angie', 'freeman', 'denise']  # Match what your interface shows
        
        return working_voices
    
    def audio_to_base64(self, audio_tensor: torch.Tensor, sample_rate: int) -> str:
        """Convert audio tensor to base64 encoded WAV"""
        try:
            # Ensure audio is in the right format
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
            
            # Create WAV file in memory
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio_tensor, sample_rate, format='wav')
            buffer.seek(0)
            
            # Encode to base64
            audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return audio_base64
            
        except Exception as e:
            print(f"❌ Error converting audio to base64: {e}")
            return ""

# Legacy compatibility classes (for backward compatibility)
class TortoiseTextProcessor:
    """Text processing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text for TTS"""
        import re
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:-]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()

class TortoiseAudioProcessor:
    """Audio processing utilities"""
    
    @staticmethod
    def audio_to_base64(audio: np.ndarray, sample_rate: int = 24000) -> str:
        """Convert audio array to base64 encoded WAV"""
        # Normalize audio
        audio = np.clip(audio, -1.0, 1.0)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, sample_rate, format="wav")
        
        # Get bytes and encode as base64
        buffer.seek(0)
        audio_bytes = buffer.getvalue()
        return base64.b64encode(audio_bytes).decode('utf-8')

# Main TTS class for backward compatibility
class TortoiseTTS:
    """Legacy wrapper for the real Tortoise TTS engine with GPU acceleration"""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.engine = TortoiseTTSEngine(device=self.device)
        self.voice_configs = TortoiseVoiceConfig.VOICE_CONFIGS
        self.sample_rate = 24000
        
    def get_available_voices(self) -> List[str]:
        """Get list of available voice names"""
        return self.engine.get_available_voices()
    
    @property
    def device_info(self) -> str:
        """Get current device information"""
        return self.engine.device_info
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information if using CUDA"""
        if self.device == "cuda" and torch.cuda.is_available():
            return {
                "device": torch.cuda.get_device_name(0),
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "allocated_memory_mb": torch.cuda.memory_allocated(0) / (1024**2),
                "cached_memory_mb": torch.cuda.memory_reserved(0) / (1024**2),
                "free_memory_mb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / (1024**2)
            }
        return {"device": self.device, "gpu_available": False}
    
    def synthesize(
        self, 
        text: str, 
        voice: str = 'angie', 
        preset: str = 'fast',
        save_audio: bool = True,
        timeout_seconds: Optional[float] = None,
        **kwargs
    ) -> Tuple[np.ndarray, dict]:
        """
        Synthesize speech from text using real Tortoise TTS with timeout support
        
        Args:
            text: Text to synthesize
            voice: Voice name to use
            preset: Quality preset (overrides voice config if provided)
            save_audio: Whether to automatically save audio to file
            timeout_seconds: Maximum time allowed for synthesis
            **kwargs: Additional TTS parameters
            
        Returns:
            Tuple of (audio_array, metadata_dict)
        """
        try:
            # Override preset if provided
            if preset != 'fast':
                kwargs['preset'] = preset
                
            # Generate audio using real TTS with auto-save option and timeout
            audio_tensor, sample_rate = self.engine.generate_speech(
                text, voice, save_audio=save_audio, timeout_seconds=timeout_seconds, **kwargs
            )
            
            # Convert to numpy array
            audio_array = audio_tensor.numpy()
            
            # Prepare metadata
            metadata = {
                'voice': voice,
                'sample_rate': sample_rate,
                'duration': len(audio_array) / sample_rate,
                'engine': 'real_tortoise_tts',
                'timeout_used': timeout_seconds
            }
            
            return audio_array, metadata
            
        except Exception as e:
            print(f"❌ Synthesis error: {e}")
            # Return silence as fallback
            silence = np.zeros(self.sample_rate)
            metadata = {'voice': voice, 'sample_rate': self.sample_rate, 'error': str(e)}
            return silence, metadata

    def synthesize_to_base64(
        self, 
        text: str, 
        voice: str = 'angie', 
        preset: str = 'fast',
        save_audio: bool = True,
        timeout_seconds: Optional[float] = None,
        **kwargs
    ) -> Tuple[str, dict]:
        """
        Synthesize speech and return as base64 encoded audio with timeout support
        
        Args:
            text: Text to synthesize
            voice: Voice name to use
            preset: Quality preset
            save_audio: Whether to automatically save audio to file
            timeout_seconds: Maximum time allowed for synthesis
            **kwargs: Additional TTS parameters
            
        Returns:
            Tuple of (base64_audio, metadata_dict)
        """
        try:
            # Generate audio using the regular synthesize method with timeout
            audio_array, metadata = self.synthesize(
                text, voice, preset, save_audio=save_audio, 
                timeout_seconds=timeout_seconds, **kwargs
            )
            
            # Convert to base64
            audio_base64 = TortoiseAudioProcessor.audio_to_base64(
                audio_array, metadata.get('sample_rate', self.sample_rate)
            )
            
            return audio_base64, metadata
            
        except Exception as e:
            print(f"❌ Base64 synthesis error: {e}")
            # Return empty audio and error metadata
            metadata = {'voice': voice, 'sample_rate': self.sample_rate, 'error': str(e)}
            return "", metadata

# Factory function for creating TTS instances
def create_tortoise_tts(device=None, **kwargs) -> TortoiseTTS:
    """Create a new Tortoise TTS instance with optional GPU acceleration"""
    return TortoiseTTS(device=device, **kwargs)

if __name__ == "__main__":
    # Test the real Tortoise TTS implementation
    print("🧪 Testing Real Tortoise TTS Implementation...")
    
    try:
        tts = TortoiseTTS()
        voices = tts.get_available_voices()
        print(f"✅ Available voices: {voices[:5]}... (showing first 5)")
        
        # Test synthesis
        test_text = "Hello, this is a test of the real Tortoise TTS system."
        audio, metadata = tts.synthesize(test_text, voice='angie')
        print(f"✅ Test synthesis completed: {metadata}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
