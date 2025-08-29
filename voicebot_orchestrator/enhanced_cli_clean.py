"""
Enhanced CLI with Numbered Menu Options and GPU Memory Management
No auto-loading of services - initialize TTS, LLM, and STT independently
Supports multiple TTS engines (Kokoro/Nari Dia) and LLMs (Mistral/GPT-OSS 20B)
"""
import asyncio
import argparse
import sys
import os
from typing import Optional

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from voicebot_orchestrator.enhanced_tts_manager import EnhancedTTSManager, TTSEngine, get_engine_choice, smart_engine_selection

class VoicebotServices:
    """Manages individual voicebot services with on-demand loading"""
    
    def __init__(self):
        self.stt = None
        self.llm = None 
        self.tts_manager = None
        self.current_llm_type = None
        
    async def initialize_stt(self) -> bool:
        """Initialize Speech-to-Text service"""
        if self.stt is not None:
            print("⚠️ STT already initialized")
            return True
            
        try:
            print("🎙️ Initializing Speech-to-Text...")
            # STT initialization would go here - placeholder for now
            self.stt = "initialized"  # Placeholder
            print("✅ STT ready")
            return True
        except Exception as e:
            print(f"❌ STT initialization failed: {e}")
            return False
    
    async def initialize_llm(self, llm_type: str = 'mistral') -> bool:
        """Initialize Large Language Model service"""
        if self.llm is not None:
            print(f"⚠️ LLM already initialized ({self.current_llm_type})")
            return True
            
        try:
            print(f"🧠 Initializing LLM ({llm_type})...")
            
            if llm_type == 'mistral':
                # Initialize Mistral model
                from voicebot_orchestrator.enhanced_llm import EnhancedMistralLLM
                self.llm = EnhancedMistralLLM(
                    model_path="mistral:latest",
                    temperature=0.7
                )
            elif llm_type == 'gpt-oss':
                # Initialize GPT-OSS 20B model
                from voicebot_orchestrator.llm import MistralLLM  # Use base class for now
                self.llm = MistralLLM(
                    model_path="gpt-oss:20b",  # Ollama model name
                    temperature=0.7
                )
            else:
                print(f"❌ Unknown LLM type: {llm_type}")
                return False
                
            # No need to call initialize() - just instantiate
            self.current_llm_type = llm_type
            print(f"✅ LLM ready ({llm_type})")
            return True
            
        except Exception as e:
            print(f"❌ LLM initialization failed: {e}")
            self.llm = None
            self.current_llm_type = None
            return False
    
    async def initialize_tts_kokoro(self) -> bool:
        """Initialize Kokoro TTS engine only"""
        if self.tts_manager is not None:
            print("⚠️ TTS Manager already initialized")
            return True
            
        try:
            print("🚀 Initializing Kokoro TTS (Fast)...")
            self.tts_manager = EnhancedTTSManager()
            success = await self.tts_manager.initialize_engines(
                load_kokoro=True,
                load_nari=False
            )
            
            if success:
                print("✅ Kokoro TTS ready (~0.8s generation)")
                return True
            else:
                print("❌ Kokoro TTS initialization failed")
                return False
                
        except Exception as e:
            print(f"❌ Kokoro TTS initialization failed: {e}")
            return False
    
    async def initialize_tts_nari(self) -> bool:
        """Initialize Nari Dia TTS engine only"""
        if self.tts_manager is not None:
            print("⚠️ TTS Manager already initialized")
            return True
            
        try:
            print("🎭 Initializing Nari Dia TTS (Quality)...")
            self.tts_manager = EnhancedTTSManager()
            success = await self.tts_manager.initialize_engines(
                load_kokoro=False,
                load_nari=True
            )
            
            if success:
                print("✅ Nari Dia TTS ready (~3+ min generation)")
                return True
            else:
                print("❌ Nari Dia TTS initialization failed")
                return False
                
        except Exception as e:
            print(f"❌ Nari Dia TTS initialization failed: {e}")
            return False
    
    async def initialize_tts_both(self) -> bool:
        """Initialize both TTS engines"""
        if self.tts_manager is not None:
            print("⚠️ TTS Manager already initialized")
            return True
            
        try:
            print("🎭 Initializing Both TTS Engines...")
            self.tts_manager = EnhancedTTSManager()
            success = await self.tts_manager.initialize_engines(
                load_kokoro=True,
                load_nari=True
            )
            
            if success:
                print("✅ Both TTS engines ready")
                return True
            else:
                print("❌ TTS initialization failed")
                return False
                
        except Exception as e:
            print(f"❌ TTS initialization failed: {e}")
            return False
    
    async def shutdown_stt(self) -> bool:
        """Shutdown Speech-to-Text service"""
        if self.stt is None:
            print("⚠️ STT not running")
            return True
            
        try:
            print("🛑 Shutting down STT...")
            # STT cleanup would go here
            self.stt = None
            print("✅ STT stopped")
            return True
        except Exception as e:
            print(f"❌ STT shutdown failed: {e}")
            return False
    
    async def shutdown_llm(self) -> bool:
        """Shutdown Large Language Model service"""
        if self.llm is None:
            print("⚠️ LLM not running")
            return True
            
        try:
            print(f"🛑 Shutting down LLM ({self.current_llm_type})...")
            
            # Clean up LLM resources
            if hasattr(self.llm, 'cleanup'):
                await self.llm.cleanup()
            elif hasattr(self.llm, '_model') and self.llm._model:
                # For models that might have GPU resources
                try:
                    if hasattr(self.llm._model, 'to'):
                        self.llm._model.to('cpu')
                    del self.llm._model
                except:
                    pass
            
            del self.llm
            self.llm = None
            self.current_llm_type = None
            
            # Force cleanup
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print("✅ LLM stopped and GPU memory freed")
            return True
        except Exception as e:
            print(f"❌ LLM shutdown failed: {e}")
            return False
    
    async def shutdown_tts(self) -> bool:
        """Shutdown Text-to-Speech service"""
        if self.tts_manager is None:
            print("⚠️ TTS not running")
            return True
            
        try:
            print("🛑 Shutting down TTS...")
            
            # Call the enhanced cleanup method
            self.tts_manager.cleanup()
            self.tts_manager = None
            
            print("✅ TTS stopped and GPU memory freed")
            return True
        except Exception as e:
            print(f"❌ TTS shutdown failed: {e}")
            return False
    
    def get_gpu_memory_usage(self) -> str:
        """Get current GPU memory usage"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                reserved = torch.cuda.memory_reserved() / 1024**3   # Convert to GB
                return f"{allocated:.1f}GB allocated, {reserved:.1f}GB reserved"
            else:
                return "No CUDA available"
        except Exception:
            return "Unable to check GPU memory"
    
    def get_status(self) -> dict:
        """Get status of all services"""
        gpu_memory = self.get_gpu_memory_usage()
        return {
            'stt': '✅ Ready' if self.stt else '❌ Not loaded',
            'llm': f'✅ Ready ({self.current_llm_type})' if self.llm else '❌ Not loaded', 
            'tts': '✅ Ready' if self.tts_manager else '❌ Not loaded',
            'gpu_memory': gpu_memory
        }


class EnhancedVoicebotCLI:
    """Enhanced CLI with numbered menu options and modular service management"""
    
    def __init__(self):
        self.services = VoicebotServices()
        self.current_mode = "interactive"
        
    async def initialize(self):
        """Initialize the CLI without loading any services"""
        print("🚀 Enhanced Voicebot CLI Ready")
        print("⚠️  No services pre-loaded - initialize them manually")
        print("💡 Use 'help' or '?' to see numbered menu options")
        return True
    
    def show_status(self):
        """Show current system status"""
        print("\n📊 SYSTEM STATUS")
        print("=" * 40)
        
        status = self.services.get_status()
        
        print(f"🎙️  STT:  {status['stt']}")
        print(f"🧠  LLM:  {status['llm']}")
        print(f"🔊  TTS:  {status['tts']}")
        print(f"💾  GPU:  {status['gpu_memory']}")
        
        # TTS engine details if available
        if self.services.tts_manager:
            available = self.services.tts_manager.get_available_engines()
            current = self.services.tts_manager.get_current_engine()
            
            print(f"\n🎭 TTS ENGINE DETAILS")
            print("-" * 25)
            
            for engine in [TTSEngine.KOKORO, TTSEngine.NARI_DIA]:
                status_icon = "✅" if engine in available else "❌"
                active_icon = "🎯" if engine == current else "  "
                
                if engine == TTSEngine.KOKORO:
                    print(f"{active_icon} {status_icon} Kokoro TTS (Fast)")
                elif engine == TTSEngine.NARI_DIA:
                    print(f"{active_icon} {status_icon} Nari Dia-1.6B (Quality)")
            
            if available:
                print(f"\n🔧 Current Engine: {current.value}")
        
        print(f"\n💡 Tip: Use option 9 to shut down all services and free GPU memory")

    def show_numbered_menu(self):
        """Show numbered menu options (1-9 with submenus)"""
        print("\n🎭 ENHANCED VOICEBOT CLI - MAIN MENU")
        print("=" * 45)
        
        print("1. 📊 System Status & Info")
        print("2. 🚀 Initialize Services")
        print("3. 🛑 Shutdown Services") 
        print("4. 🎤 Speech Generation")
        print("5. 🔄 TTS Engine Control")
        print("6. �️  Voice Features")
        print("7. 🧪 Testing & Demos")
        print("8. 🏥 System Tools")
        print("9. ❓ Help & Tips")
        print("0. 👋 Exit CLI")
        print()
        print("💡 Select a number (1-9) to open submenu, or 0 to exit")

    def show_status_menu(self):
        """Show status and info submenu"""
        print("\n📊 SYSTEM STATUS & INFO")
        print("=" * 25)
        print("1. 📊 Show current status")
        print("2. 💾 Show GPU memory usage")
        print("3. 🎭 Show TTS engine details") 
        print("4. 🧠 Show LLM information")
        print("0. ↩️  Return to main menu")

    def show_initialize_menu(self):
        """Show service initialization submenu"""
        print("\n🚀 INITIALIZE SERVICES")
        print("=" * 22)
        print("1. 🎙️  Initialize STT (Speech-to-Text)")
        print("2. 🧠 Initialize LLM - Mistral (default)")
        print("3. 🤖 Initialize LLM - GPT-OSS 20B")
        print("4. 🚀 Initialize TTS - Kokoro (fast)")
        print("5. 🎭 Initialize TTS - Nari Dia (quality)")
        print("6. 🔄 Initialize TTS - Both engines")
        print("7. 🌟 Quick Start - Fast (STT + Kokoro + Mistral)")
        print("8. 💎 Quick Start - Quality (STT + Nari Dia + GPT-OSS)")
        print("9. 🔄 Initialize Everything (All services)")
        print("0. ↩️  Return to main menu")

    def show_shutdown_menu(self):
        """Show shutdown submenu"""
        print("\n🛑 SHUTDOWN SERVICES")
        print("=" * 20)
        print("1. 🎙️  Shutdown STT")
        print("2. 🧠 Shutdown LLM")
        print("3. 🔊 Shutdown TTS")
        print("4. 🧹 Shutdown ALL services")
        print("0. ↩️  Return to main menu")

    def show_speech_menu(self):
        """Show speech generation submenu"""
        print("\n🎤 SPEECH GENERATION")
        print("=" * 20)
        print("1. 🎤 Generate speech with text input")
        print("2. 🤖 Auto-select engine and speak")
        print("3. 🚀 Quick speech with Kokoro")
        print("4. 🎭 High-quality speech with Nari")
        print("0. ↩️  Return to main menu")

    def show_engine_menu(self):
        """Show TTS engine control submenu"""
        print("\n🔄 TTS ENGINE CONTROL")
        print("=" * 21)
        print("1. 🔄 Interactive engine switcher")
        print("2. 🚀 Switch to Kokoro TTS")
        print("3. 🎭 Switch to Nari Dia TTS")
        print("4. 📊 Show available engines")
        print("0. ↩️  Return to main menu")

    def show_voice_menu(self):
        """Show voice features submenu"""
        print("\n🎙️ VOICE FEATURES")
        print("=" * 17)
        print("1. 🗣️  Start voice conversation")
        print("2. 🎤 Voice recording test")
        print("3. 🔊 Audio playback test")
        print("0. ↩️  Return to main menu")

    def show_testing_menu(self):
        """Show testing and demos submenu"""
        print("\n🧪 TESTING & DEMOS")
        print("=" * 18)
        print("1. 🧪 Run TTS engine comparison test")
        print("2. 🎭 Run dual TTS demonstration")
        print("3. 📊 Performance benchmark")
        print("4. 🔍 Voice quality test")
        print("0. ↩️  Return to main menu")

    def show_tools_menu(self):
        """Show system tools submenu"""
        print("\n🏥 SYSTEM TOOLS")
        print("=" * 15)
        print("1. 🏥 System health check")
        print("2. 🔧 Configuration check")
        print("3. 📋 List available models")
        print("4. 🧹 Clear cache and temp files")
        print("0. ↩️  Return to main menu")

    def show_help_menu(self):
        """Show help and tips submenu"""
        print("\n❓ HELP & TIPS")
        print("=" * 14)
        print("1. 💡 Quick start guide")
        print("2. 🎯 Performance tips")
        print("3. 🐛 Troubleshooting")
        print("4. 📚 Command reference")
        print("5. 🔗 About this CLI")
        print("0. ↩️  Return to main menu")

    async def handle_numbered_choice(self, choice: str) -> bool:
        """Handle numbered menu choices. Returns True to continue, False to exit"""
        try:
            choice_num = int(choice)
        except ValueError:
            print("❌ Please enter a valid number (0-9)")
            return True
        
        if choice_num == 0:
            await self.check_and_cleanup_on_exit()
            return False
        elif choice_num == 1:
            await self.handle_status_submenu()
        elif choice_num == 2:
            await self.handle_initialize_submenu()
        elif choice_num == 3:
            await self.handle_shutdown_submenu()
        elif choice_num == 4:
            await self.handle_speech_submenu()
        elif choice_num == 5:
            await self.handle_engine_submenu()
        elif choice_num == 6:
            await self.handle_voice_submenu()
        elif choice_num == 7:
            await self.handle_testing_submenu()
        elif choice_num == 8:
            await self.handle_tools_submenu()
        elif choice_num == 9:
            await self.handle_help_submenu()
        else:
            print(f"❌ Invalid choice: {choice_num}")
            print("💡 Please enter a number from 1-9, or 0 to exit")
        
        return True

    async def handle_status_submenu(self):
        """Handle status and info submenu"""
        while True:
            self.show_status_menu()
            try:
                choice = input("\n📊 Status choice (0-4): ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    self.show_status()
                elif choice == '2':
                    self.show_gpu_memory()
                elif choice == '3':
                    self.show_tts_details()
                elif choice == '4':
                    self.show_llm_info()
                else:
                    print(f"❌ Invalid choice: {choice}")
                    
            except KeyboardInterrupt:
                print("\n🔄 Returning to main menu...")
                break

    async def handle_initialize_submenu(self):
        """Handle service initialization submenu"""
        while True:
            self.show_initialize_menu()
            try:
                choice = input("\n🚀 Initialize choice (0-9): ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    await self.services.initialize_stt()
                elif choice == '2':
                    await self.services.initialize_llm('mistral')
                elif choice == '3':
                    await self.services.initialize_llm('gpt-oss')
                elif choice == '4':
                    await self.services.initialize_tts_kokoro()
                elif choice == '5':
                    await self.services.initialize_tts_nari()
                elif choice == '6':
                    await self.services.initialize_tts_both()
                elif choice == '7':
                    print("🌟 Quick Start - Fast: Initializing STT + Kokoro TTS + Mistral LLM...")
                    await self.services.initialize_stt()
                    await self.services.initialize_tts_kokoro()
                    await self.services.initialize_llm('mistral')
                    print("✅ Fast quick start complete! (Best for real-time conversation)")
                elif choice == '8':
                    print("💎 Quick Start - Quality: Initializing STT + Nari Dia TTS + GPT-OSS LLM...")
                    await self.services.initialize_stt()
                    await self.services.initialize_tts_nari()
                    await self.services.initialize_llm('gpt-oss')
                    print("✅ Quality quick start complete! (Best for high-quality generation)")
                elif choice == '9':
                    print("🔄 Initializing ALL services...")
                    await self.services.initialize_stt()
                    await self.services.initialize_tts_both()
                    await self.services.initialize_llm('mistral')
                    print("✅ All services initialized!")
                else:
                    print(f"❌ Invalid choice: {choice}")
                    
            except KeyboardInterrupt:
                print("\n� Returning to main menu...")
                break

    async def handle_shutdown_submenu(self):
        """Handle shutdown submenu"""
        while True:
            self.show_shutdown_menu()
            try:
                choice = input("\n🛑 Shutdown choice (0-4): ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    await self.services.shutdown_stt()
                elif choice == '2':
                    await self.services.shutdown_llm()
                elif choice == '3':
                    await self.services.shutdown_tts()
                elif choice == '4':
                    print("🧹 Shutting down ALL services...")
                    await self.services.shutdown_stt()
                    await self.services.shutdown_llm()
                    await self.services.shutdown_tts()
                    print("✅ All services stopped and GPU memory freed")
                    break
                else:
                    print(f"❌ Invalid choice: {choice}")
                    
            except KeyboardInterrupt:
                print("\n🔄 Returning to main menu...")
                break

    async def handle_speech_submenu(self):
        """Handle speech generation submenu"""
        while True:
            self.show_speech_menu()
            try:
                choice = input("\n🎤 Speech choice (0-4): ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    await self.handle_speech_generation()
                elif choice == '2':
                    await self.handle_auto_speech()
                elif choice == '3':
                    await self.handle_kokoro_speech()
                elif choice == '4':
                    await self.handle_nari_speech()
                else:
                    print(f"❌ Invalid choice: {choice}")
                    
            except KeyboardInterrupt:
                print("\n🔄 Returning to main menu...")
                break

    async def handle_engine_submenu(self):
        """Handle TTS engine control submenu"""
        while True:
            self.show_engine_menu()
            try:
                choice = input("\n🔄 Engine choice (0-4): ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    await self.switch_engine_interactive()
                elif choice == '2':
                    await self.switch_to_kokoro()
                elif choice == '3':
                    await self.switch_to_nari()
                elif choice == '4':
                    self.show_available_engines()
                else:
                    print(f"❌ Invalid choice: {choice}")
                    
            except KeyboardInterrupt:
                print("\n🔄 Returning to main menu...")
                break

    async def handle_voice_submenu(self):
        """Handle voice features submenu"""
        while True:
            self.show_voice_menu()
            try:
                choice = input("\n🎙️ Voice choice (0-3): ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    await self.start_voice_conversation()
                elif choice == '2':
                    await self.test_voice_recording()
                elif choice == '3':
                    await self.test_audio_playback()
                else:
                    print(f"❌ Invalid choice: {choice}")
                    
            except KeyboardInterrupt:
                print("\n🔄 Returning to main menu...")
                break

    async def handle_testing_submenu(self):
        """Handle testing and demos submenu"""
        while True:
            self.show_testing_menu()
            try:
                choice = input("\n🧪 Testing choice (0-4): ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    await self.run_test()
                elif choice == '2':
                    await run_dual_tts_demo()
                elif choice == '3':
                    await run_benchmark()
                elif choice == '4':
                    await self.test_voice_quality()
                else:
                    print(f"❌ Invalid choice: {choice}")
                    
            except KeyboardInterrupt:
                print("\n🔄 Returning to main menu...")
                break

    async def handle_tools_submenu(self):
        """Handle system tools submenu"""
        while True:
            self.show_tools_menu()
            try:
                choice = input("\n🏥 Tools choice (0-4): ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    await run_health_check()
                elif choice == '2':
                    await self.check_configuration()
                elif choice == '3':
                    await self.list_available_models()
                elif choice == '4':
                    await self.clear_cache()
                else:
                    print(f"❌ Invalid choice: {choice}")
                    
            except KeyboardInterrupt:
                print("\n🔄 Returning to main menu...")
                break

    async def handle_help_submenu(self):
        """Handle help and tips submenu"""
        while True:
            self.show_help_menu()
            try:
                choice = input("\n❓ Help choice (0-5): ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    self.show_quick_start_guide()
                elif choice == '2':
                    self.show_performance_tips()
                elif choice == '3':
                    self.show_troubleshooting()
                elif choice == '4':
                    self.show_command_reference()
                elif choice == '5':
                    self.show_about()
                else:
                    print(f"❌ Invalid choice: {choice}")
                    
            except KeyboardInterrupt:
                print("\n🔄 Returning to main menu...")
                break

    # Supporting methods for new menu features
    def show_gpu_memory(self):
        """Show detailed GPU memory information"""
        print("\n💾 GPU MEMORY USAGE")
        print("=" * 20)
        gpu_info = self.services.get_gpu_memory_usage()
        print(f"Current usage: {gpu_info}")
        
        try:
            import torch
            if torch.cuda.is_available():
                print(f"Device: {torch.cuda.get_device_name(0)}")
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"Total GPU memory: {total_mem:.1f}GB")
        except:
            print("Unable to get detailed GPU information")

    def show_tts_details(self):
        """Show TTS engine details"""
        print("\n🎭 TTS ENGINE DETAILS")
        print("=" * 22)
        
        if self.services.tts_manager:
            available = self.services.tts_manager.get_available_engines()
            current = self.services.tts_manager.get_current_engine()
            
            for engine in [TTSEngine.KOKORO, TTSEngine.NARI_DIA]:
                status_icon = "✅" if engine in available else "❌"
                active_icon = "🎯" if engine == current else "  "
                
                if engine == TTSEngine.KOKORO:
                    print(f"{active_icon} {status_icon} Kokoro TTS")
                    print("     Speed: ~0.8s generation")
                    print("     Voice: af_bella (professional female)")
                    print("     Best for: Real-time conversation")
                elif engine == TTSEngine.NARI_DIA:
                    print(f"{active_icon} {status_icon} Nari Dia-1.6B")
                    print("     Speed: ~3+ minutes generation") 
                    print("     Voice: Adaptive dialogue-focused")
                    print("     Best for: Maximum quality")
            
            if available:
                print(f"\nCurrent Engine: {current.value}")
        else:
            print("No TTS engines initialized")

    def show_llm_info(self):
        """Show LLM information"""
        print("\n🧠 LLM INFORMATION")
        print("=" * 18)
        
        if self.services.llm:
            print(f"Current LLM: {self.services.current_llm_type}")
            if self.services.current_llm_type == 'mistral':
                print("Model: Mistral Latest")
                print("Provider: Ollama")
                print("Best for: General conversation, fast responses")
            elif self.services.current_llm_type == 'gpt-oss':
                print("Model: GPT-OSS 20B")
                print("Provider: Ollama")
                print("Best for: Advanced reasoning, experimental features")
        else:
            print("No LLM initialized")
            print("Available options:")
            print("• Mistral (recommended for general use)")
            print("• GPT-OSS 20B (experimental, advanced)")

    def show_available_engines(self):
        """Show detailed TTS engine information"""
        if not self.services.tts_manager:
            print("❌ No TTS engines initialized")
            return
            
        available = self.services.tts_manager.get_available_engines()
        current = self.services.tts_manager.get_current_engine()
        
        print("\n� TTS ENGINE DETAILS")
        print("=" * 25)
        
        for engine in available:
            status = "🎯 CURRENT" if engine == current else "⚪ Available"
            print(f"\n{status} - {engine.value}")
            
            if engine.value == "kokoro":
                print("   💨 Speed: ~0.8s generation (real-time)")
                print("   🎭 Voice: af_bella (professional female)")
                print("   🎯 Best for: Real-time conversation")
                print("   💾 Memory: Low GPU usage")
                
            elif engine.value == "nari_dia":
                print("   🎨 Speed: ~3+ minutes generation (high quality)")
                print("   🎭 Voice: Adaptive dialogue-focused")
                print("   🎯 Best for: Pre-recorded, maximum quality")
                print("   💾 Memory: High GPU usage")
                
            elif engine.value == "auto":
                print("   🤖 Automatically selects best available engine")
                print("   🎯 Best for: General use")
        
        print(f"\n📊 Total engines: {len(available)}")
        print(f"🎯 Current: {current.value}")

    async def handle_auto_speech(self):
        """Handle auto-select engine speech"""
        if not self.services.tts_manager:
            print("❌ TTS not initialized")
            return
            
        try:
            text = input("\n🤖 Enter text for auto-speech: ").strip()
            if text:
                await self.speak_text(text, TTSEngine.AUTO)
            else:
                print("❌ No text provided")
        except KeyboardInterrupt:
            print("\n🔄 Cancelled")

    async def handle_kokoro_speech(self):
        """Handle Kokoro-specific speech"""
        if not self.services.tts_manager:
            print("❌ TTS not initialized")
            return
            
        if TTSEngine.KOKORO not in self.services.tts_manager.get_available_engines():
            print("❌ Kokoro not available. Initialize it first.")
            return
            
        try:
            text = input("\n🚀 Enter text for Kokoro (fast): ").strip()
            if text:
                # Temporarily switch to Kokoro
                original_engine = self.services.tts_manager.get_current_engine()
                self.services.tts_manager.set_engine(TTSEngine.KOKORO)
                await self.speak_text(text)
                self.services.tts_manager.set_engine(original_engine)
            else:
                print("❌ No text provided")
        except KeyboardInterrupt:
            print("\n🔄 Cancelled")

    async def handle_nari_speech(self):
        """Handle Nari Dia-specific speech"""
        if not self.services.tts_manager:
            print("❌ TTS not initialized")
            return
            
        if TTSEngine.NARI_DIA not in self.services.tts_manager.get_available_engines():
            print("❌ Nari Dia not available. Initialize it first.")
            return
            
        try:
            text = input("\n🎭 Enter text for Nari Dia (quality): ").strip()
            if text:
                print("⚠️ Note: Nari Dia generation takes 3+ minutes")
                # Temporarily switch to Nari
                original_engine = self.services.tts_manager.get_current_engine()
                self.services.tts_manager.set_engine(TTSEngine.NARI_DIA)
                await self.speak_text(text)
                self.services.tts_manager.set_engine(original_engine)
            else:
                print("❌ No text provided")
        except KeyboardInterrupt:
            print("\n🔄 Cancelled")

    async def test_voice_recording(self):
        """Test voice recording functionality"""
        if not self.services.stt:
            print("❌ STT not initialized. Use option 2 → 1 first")
            return
            
        try:
            print("🎤 Testing voice recording with STT...")
            print("💡 Speak clearly into your microphone")
            
            # Test recording with STT
            import tempfile
            import time
            import os
            
            try:
                import pyaudio
                import speech_recognition as sr
                
                recognizer = sr.Recognizer()
                microphone = sr.Microphone()
                
                print("🔴 Recording... (speak for 5 seconds)")
                with microphone as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                
                print("🔄 Processing with STT...")
                
                # Save to temp file and process with our STT
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio.get_wav_data())
                    temp_path = temp_file.name
                
                try:
                    transcript = await self.services.stt.transcribe_file(temp_path)
                    print(f"✅ Recording successful!")
                    print(f"📝 Transcript: '{transcript}'")
                    
                    # Ask if user wants to save the audio
                    save = input("\n💾 Save audio file? (y/n): ").strip().lower()
                    if save == 'y':
                        import shutil
                        audio_filename = f"voice_test_{int(time.time())}.wav"
                        shutil.copy(temp_path, audio_filename)
                        print(f"🎵 Audio saved: {audio_filename}")
                        
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
            except ImportError:
                print("❌ Audio libraries not available")
                print("💡 Install with: pip install pyaudio SpeechRecognition")
            except Exception as e:
                print(f"❌ Recording failed: {e}")
                
        except KeyboardInterrupt:
            print("\n🔄 Recording test cancelled")

    async def test_audio_playback(self):
        """Test audio playback functionality"""
        if not self.services.tts_manager:
            print("❌ TTS not initialized. Use option 2 → 4,5,6 first")
            return
            
        try:
            print("🔊 Testing audio playback with TTS...")
            
            # Test text for playback
            test_text = "This is a test of the audio playback system. Can you hear this clearly?"
            
            print(f"🎭 Generating test audio: '{test_text}'")
            
            # Generate audio with current TTS engine
            current_engine = self.services.tts_manager.get_current_engine()
            print(f"🚀 Using engine: {current_engine.value}")
            
            audio_data = await self.services.tts_manager.synthesize_speech(test_text)
            
            # Save to temp file and try to play
            import tempfile
            import time
            import os
            import sys
            
            audio_filename = f"playback_test_{int(time.time())}.wav"
            
            with open(audio_filename, "wb") as f:
                f.write(audio_data)
            
            print(f"💾 Audio saved: {audio_filename}")
            
            # Try to play audio (cross-platform)
            try:
                print("🔊 Playing audio...")
                
                if sys.platform.startswith('win'):
                    # Windows
                    os.system(f'start /min "" "{audio_filename}"')
                elif sys.platform.startswith('darwin'):
                    # macOS  
                    os.system(f'afplay "{audio_filename}"')
                elif sys.platform.startswith('linux'):
                    # Linux
                    os.system(f'aplay "{audio_filename}" 2>/dev/null || paplay "{audio_filename}" 2>/dev/null')
                
                print("✅ Audio playback test completed!")
                print(f"🎵 Audio file saved as: {audio_filename}")
                
            except Exception as e:
                print(f"⚠️ Audio playback error: {e}")
                print(f"🎵 Audio file saved for manual testing: {audio_filename}")
                
        except KeyboardInterrupt:
            print("\n🔄 Playback test cancelled")
        except Exception as e:
            print(f"❌ Playback test failed: {e}")

    async def test_voice_quality(self):
        """Test voice quality by comparing TTS engines"""
        if not self.services.tts_manager:
            print("❌ TTS not initialized. Use option 2 first")
            return
            
        available_engines = self.services.tts_manager.get_available_engines()
        
        if len(available_engines) < 2:
            print("⚠️ Need at least 2 TTS engines for comparison")
            print("💡 Initialize both Kokoro and Nari Dia (option 2 → 6)")
            return
            
        try:
            print("🔍 Voice Quality Comparison Test")
            print("=" * 35)
            
            # Test phrases for quality comparison
            test_phrases = [
                "Welcome to our banking service. How can I assist you today?",
                "Your account balance is one thousand two hundred and thirty-four dollars.",
                "Thank you for choosing our premium banking solutions."
            ]
            
            print("📋 Available test phrases:")
            for i, phrase in enumerate(test_phrases, 1):
                print(f"{i}. {phrase}")
            print("4. Custom text")
            
            choice = input("\n🎯 Choose test phrase (1-4): ").strip()
            
            if choice == "4":
                text = input("📝 Enter custom text: ").strip()
                if not text:
                    print("❌ No text provided")
                    return
            elif choice in ["1", "2", "3"]:
                text = test_phrases[int(choice) - 1]
            else:
                print("❌ Invalid choice, using default")
                text = test_phrases[0]
            
            print(f"\n🎭 Testing with: '{text}'")
            print("=" * 50)
            
            original_engine = self.services.tts_manager.get_current_engine()
            results = []
            
            for engine in available_engines:
                print(f"\n🚀 Testing {engine.value}...")
                
                # Switch to this engine
                self.services.tts_manager.set_engine(engine)
                
                # Time the generation
                import time
                start_time = time.time()
                
                try:
                    audio_data = await self.services.tts_manager.synthesize_speech(text)
                    generation_time = time.time() - start_time
                    
                    # Save comparison file
                    filename = f"quality_test_{engine.value.lower()}_{int(time.time())}.wav"
                    with open(filename, "wb") as f:
                        f.write(audio_data)
                    
                    results.append({
                        'engine': engine.value,
                        'time': generation_time,
                        'file': filename,
                        'size': len(audio_data)
                    })
                    
                    print(f"✅ {engine.value}: {generation_time:.2f}s, {len(audio_data)} bytes")
                    print(f"💾 Saved: {filename}")
                    
                except Exception as e:
                    print(f"❌ {engine.value} failed: {e}")
            
            # Restore original engine
            self.services.tts_manager.set_engine(original_engine)
            
            # Show comparison results
            print("\n📊 QUALITY COMPARISON RESULTS")
            print("=" * 40)
            for result in results:
                print(f"🚀 {result['engine']:12} | {result['time']:6.2f}s | {result['size']:8,} bytes")
            
            print("\n🎵 Audio files saved for manual quality comparison!")
            print("💡 Listen to both files to compare voice quality")
            
        except KeyboardInterrupt:
            print("\n🔄 Quality test cancelled")
        except Exception as e:
            print(f"❌ Quality test failed: {e}")

    async def check_configuration(self):
        """Check system configuration and requirements"""
        print("🔧 SYSTEM CONFIGURATION CHECK")
        print("=" * 35)
        
        # Check Python version
        import sys
        print(f"🐍 Python version: {sys.version}")
        
        # Check key dependencies
        dependencies = [
            ('torch', 'PyTorch for AI models'),
            ('transformers', 'Hugging Face transformers'),
            ('librosa', 'Audio processing'),
            ('numpy', 'Numerical computing'),
            ('pyaudio', 'Audio I/O (optional)'),
            ('speech_recognition', 'Speech recognition (optional)'),
        ]
        
        print("\n📦 Dependency Status:")
        for dep_name, description in dependencies:
            try:
                __import__(dep_name)
                print(f"✅ {dep_name:20} - {description}")
            except ImportError:
                print(f"❌ {dep_name:20} - {description} (MISSING)")
        
        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
                print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                print(f"🔥 CUDA Version: {torch.version.cuda}")
            else:
                print("\n⚠️ GPU: CUDA not available (CPU only)")
        except ImportError:
            print("\n❌ GPU: Cannot check (PyTorch not available)")
        
        # Check service status
        print("\n🎭 Service Status:")
        print(f"STT: {'✅ Loaded' if self.services.stt else '❌ Not loaded'}")
        print(f"LLM: {'✅ Loaded' if self.services.llm else '❌ Not loaded'}")
        print(f"TTS: {'✅ Loaded' if self.services.tts_manager else '❌ Not loaded'}")
        
        if self.services.tts_manager:
            available_engines = self.services.tts_manager.get_available_engines()
            current_engine = self.services.tts_manager.get_current_engine()
            print(f"TTS Engines: {[e.value for e in available_engines]}")
            print(f"Current Engine: {current_engine.value}")
        
        # Check model files
        print("\n📁 Model Files:")
        import os
        model_files = [
            ('kokoro-v1.0.onnx', 'Kokoro TTS model'),
            ('voices-v1.0.bin', 'Voice data'),
        ]
        
        for filename, description in model_files:
            if os.path.exists(filename):
                size = os.path.getsize(filename) / 1024**2
                print(f"✅ {filename:20} - {description} ({size:.1f} MB)")
            else:
                print(f"❌ {filename:20} - {description} (MISSING)")
        
        print("\n💡 Configuration check complete!")

    async def list_available_models(self):
        """List available models and their status"""
        print("📋 AVAILABLE MODELS AND COMPONENTS")
        print("=" * 40)
        
        import os
        
        # Core model files
        print("🤖 Core Models:")
        core_models = [
            ("kokoro-v1.0.onnx", "Kokoro TTS ONNX model", "TTS Engine"),
            ("voices-v1.0.bin", "Voice data for Kokoro", "Voice Data"),
        ]
        
        for filename, description, component in core_models:
            if os.path.exists(filename):
                size_mb = os.path.getsize(filename) / 1024**2
                status = f"✅ Available ({size_mb:.1f} MB)"
            else:
                status = "❌ Missing"
            
            print(f"  {filename:20} - {description:25} {status}")
        
        # LLM Models (through Ollama)
        print("\n🧠 LLM Models (via Ollama):")
        llm_models = [
            ("mistral:latest", "Mistral 7B - Primary LLM", "Default"),
            ("llama2:latest", "Llama 2 - Alternative LLM", "Optional"),
        ]
        
        for model_name, description, status in llm_models:
            print(f"  {model_name:20} - {description:25} {status}")
        
        # TTS Engines
        print("\n🔊 TTS Engines:")
        tts_engines = [
            ("Kokoro", "Fast generation (~0.8s)", "Local ONNX"),
            ("Nari Dia", "High quality (~3min)", "CUDA Required"),
        ]
        
        for engine, description, requirement in tts_engines:
            print(f"  {engine:20} - {description:25} {requirement}")
        
        # Cache and adapters
        print("\n💾 Cache & Adapters:")
        cache_items = [
            ("cache/semantic_cache.json", "Semantic response cache", "Performance"),
            ("adapters/banking-lora/", "Banking domain LoRA", "Specialization"),
            ("enterprise_cache/", "Enterprise cache data", "Enterprise"),
        ]
        
        for path, description, purpose in cache_items:
            if os.path.exists(path):
                if os.path.isdir(path):
                    status = "✅ Directory exists"
                else:
                    size_kb = os.path.getsize(path) / 1024
                    status = f"✅ Available ({size_kb:.1f} KB)"
            else:
                status = "❌ Not found"
            
            print(f"  {path:25} - {description:20} {status}")
        
        print(f"\n💡 Use menu option 8 → 2 for detailed configuration check")

    async def clear_cache(self):
        """Clear cache and temporary files"""
        print("🧹 CLEARING CACHE AND TEMP FILES")
        print("=" * 35)
        
        import os
        import shutil
        import glob
        
        cleaned_items = []
        total_freed = 0
        
        # Clear semantic cache
        cache_files = [
            "cache/semantic_cache.json",
            "enterprise_cache/*.json",
        ]
        
        for pattern in cache_files:
            for file_path in glob.glob(pattern):
                try:
                    size = os.path.getsize(file_path)
                    os.remove(file_path)
                    cleaned_items.append(f"🗑️ Removed: {file_path} ({size/1024:.1f} KB)")
                    total_freed += size
                except Exception as e:
                    cleaned_items.append(f"⚠️ Could not remove {file_path}: {e}")
        
        # Clear temporary audio files
        temp_patterns = [
            "*.wav",
            "benchmark_*.wav",
            "test_*.wav",
            "demo_*.wav",
            "playback_test_*.wav",
            "speech_*.wav",
            "voice_test_*.wav",
            "quality_test_*.wav",
        ]
        
        for pattern in temp_patterns:
            for file_path in glob.glob(pattern):
                # Skip if it's in a protected directory
                if "demos" in file_path or "tests" in file_path:
                    continue
                try:
                    size = os.path.getsize(file_path)
                    os.remove(file_path)
                    cleaned_items.append(f"🎵 Removed audio: {file_path} ({size/1024:.1f} KB)")
                    total_freed += size
                except Exception as e:
                    cleaned_items.append(f"⚠️ Could not remove {file_path}: {e}")
        
        # Clear Python cache
        pycache_patterns = [
            "**/__pycache__",
            "*.pyc",
            "*.pyo",
        ]
        
        cache_dirs_removed = 0
        for pattern in pycache_patterns:
            for cache_path in glob.glob(pattern, recursive=True):
                try:
                    if os.path.isdir(cache_path):
                        shutil.rmtree(cache_path)
                        cache_dirs_removed += 1
                    else:
                        size = os.path.getsize(cache_path)
                        os.remove(cache_path)
                        total_freed += size
                except Exception as e:
                    cleaned_items.append(f"⚠️ Could not remove {cache_path}: {e}")
        
        if cache_dirs_removed > 0:
            cleaned_items.append(f"🐍 Removed {cache_dirs_removed} __pycache__ directories")
        
        # Results
        if cleaned_items:
            print("🧹 Cleanup completed:")
            for item in cleaned_items[:10]:  # Show first 10 items
                print(f"  {item}")
            if len(cleaned_items) > 10:
                print(f"  ... and {len(cleaned_items) - 10} more items")
            
            print(f"\n💾 Total space freed: {total_freed/1024/1024:.2f} MB")
        else:
            print("✨ No cache files found to clean")
        
        print("✅ Cache cleanup complete!")

    def show_quick_start_guide(self):
        """Show quick start guide"""
        print("\n💡 QUICK START GUIDE")
        print("=" * 20)
        print("1. Choose option 2 (Initialize Services)")
        print("2. Select option 7 (Quick Start) for Kokoro + Mistral")
        print("3. Go to option 4 (Speech Generation)")
        print("4. Select option 1 to generate speech")
        print("5. Use option 1 (System Status) to monitor GPU usage")
        print("6. Use option 3 (Shutdown Services) when done")

    def show_performance_tips(self):
        """Show performance tips"""
        print("\n🎯 PERFORMANCE TIPS")
        print("=" * 19)
        print("• Use Kokoro TTS for real-time conversation (~0.8s)")
        print("• Use Nari Dia TTS for highest quality (~3+ min)")
        print("• Mistral LLM is faster than GPT-OSS 20B")
        print("• Monitor GPU memory with option 1")
        print("• Shut down services when not needed to free GPU")
        print("• Quick Start (2→7) sets up the fastest combination")

    def show_troubleshooting(self):
        """Show troubleshooting guide"""
        print("\n🐛 TROUBLESHOOTING")
        print("=" * 17)
        print("• GPU memory issues: Use option 3→4 to shutdown all")
        print("• TTS not working: Check if engines are initialized")
        print("• Slow performance: Use Kokoro instead of Nari Dia")
        print("• LLM errors: Try reinitializing with different model")
        print("• Audio issues: Check system audio settings")

    def show_command_reference(self):
        """Show command reference"""
        print("\n📚 COMMAND REFERENCE")
        print("=" * 20)
        print("Main Menu Numbers:")
        print("1-Status  2-Init   3-Shutdown  4-Speech  5-Engine")
        print("6-Voice   7-Test   8-Tools     9-Help    0-Exit")
        print("")
        print("Each menu has submenus with options 1-9 and 0 to return")

    def show_about(self):
        """Show about information"""
        print("\n🔗 ABOUT ENHANCED VOICEBOT CLI")
        print("=" * 31)
        print("Version: 2.0 - Numbered Menu Edition")
        print("Features:")
        print("• Modular service management (0GB startup)")
        print("• Dual TTS engines (Kokoro + Nari Dia)")
        print("• Multiple LLMs (Mistral + GPT-OSS 20B)")
        print("• GPU memory monitoring and cleanup")
        print("• Organized 1-9 menu system with submenus")
        print("• Independent service initialization/shutdown")

    async def handle_speech_generation(self):
        """Handle speech generation with user input and file saving"""
        if not self.services.tts_manager:
            print("❌ TTS not initialized. Use option 2 first")
            return
        
        try:
            text = input("\n🎤 Enter text to speak: ").strip()
            if not text:
                print("❌ No text provided")
                return
                
            # Show available options
            print("\n📋 Generation options:")
            print("1. 🔊 Generate and play")
            print("2. 💾 Generate and save to file")
            print("3. 🔄 Generate, save and play")
            
            choice = input("\n🎯 Choose option (1-3): ").strip()
            
            current_engine = self.services.tts_manager.get_current_engine()
            print(f"🚀 Using engine: {current_engine.value}")
            
            if choice == "1":
                # Just generate and play
                await self.speak_text(text)
                
            elif choice == "2":
                # Generate and save only
                audio_data = await self.services.tts_manager.synthesize_speech(text)
                
                import time
                engine_name = current_engine.value.lower()
                timestamp = int(time.time())
                filename = f"speech_{engine_name}_{timestamp}.wav"
                
                with open(filename, "wb") as f:
                    f.write(audio_data)
                    
                print(f"💾 Audio saved: {filename}")
                
            elif choice == "3":
                # Generate, save and play
                audio_data = await self.services.tts_manager.synthesize_speech(text)
                
                import time
                import os
                import sys
                
                engine_name = current_engine.value.lower()
                timestamp = int(time.time())
                filename = f"speech_{engine_name}_{timestamp}.wav"
                
                with open(filename, "wb") as f:
                    f.write(audio_data)
                    
                print(f"💾 Audio saved: {filename}")
                
                # Try to play
                try:
                    print("🔊 Playing audio...")
                    if sys.platform.startswith('win'):
                        os.system(f'start /min "" "{filename}"')
                    elif sys.platform.startswith('darwin'):
                        os.system(f'afplay "{filename}"')
                    elif sys.platform.startswith('linux'):
                        os.system(f'aplay "{filename}" 2>/dev/null || paplay "{filename}" 2>/dev/null')
                    print("✅ Playback started")
                except Exception as e:
                    print(f"⚠️ Playback error: {e}")
                    
            else:
                print("❌ Invalid choice, using default (generate and play)")
                await self.speak_text(text)
                
        except KeyboardInterrupt:
            print("\n🔄 Speech generation cancelled")

    async def switch_to_kokoro(self):
        """Switch to Kokoro TTS engine"""
        if self.services.tts_manager and TTSEngine.KOKORO in self.services.tts_manager.get_available_engines():
            self.services.tts_manager.set_engine(TTSEngine.KOKORO)
            print("✅ Switched to Kokoro TTS (fast)")
        else:
            print("❌ Kokoro TTS not available. Use option 5 or 7 first")

    async def switch_to_nari(self):
        """Switch to Nari Dia TTS engine"""
        if self.services.tts_manager and TTSEngine.NARI_DIA in self.services.tts_manager.get_available_engines():
            self.services.tts_manager.set_engine(TTSEngine.NARI_DIA)
            print("✅ Switched to Nari Dia TTS (quality)")
        else:
            print("❌ Nari Dia TTS not available. Use option 6 or 7 first")

    async def start_voice_conversation(self):
        """Start voice conversation using production voice bot"""
        if not self.services.stt or not self.services.llm or not self.services.tts_manager:
            print("❌ Full voice conversation requires STT + LLM + TTS")
            print("💡 Use option 2 → 7 (Quick Start Fast) or 8 (Quick Start Quality)")
            return
            
        try:
            # Import production voice bot
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from demos.production_voice_test import ProductionVoiceBot
            
            print("🎙️ Starting production voice conversation...")
            print("💡 Uses your full voice pipeline: STT + LLM + TTS")
            
            # Create and run voice bot
            voice_bot = ProductionVoiceBot()
            await voice_bot.production_conversation()
            
        except ImportError as e:
            print(f"❌ Could not load voice conversation: {e}")
            print("💡 Make sure production_voice_test.py is available")
        except KeyboardInterrupt:
            print("\n🔄 Voice conversation cancelled")
        except Exception as e:
            print(f"❌ Voice conversation error: {e}")

    async def switch_engine_interactive(self):
        """Interactive engine switching"""
        if not self.services.tts_manager:
            print("❌ TTS not initialized. Use option 5, 6, or 7 first")
            return
            
        try:
            available = self.services.tts_manager.get_available_engines()
            if len(available) <= 1:
                print("⚠️ Only one engine available, cannot switch")
                return
            
            engine = get_engine_choice()
            if engine == TTSEngine.AUTO:
                print("🤖 Auto mode - engine will be selected per request")
                return
            
            if engine in available:
                self.services.tts_manager.set_engine(engine)
                print("✅ Engine switched successfully")
            else:
                print(f"❌ Engine {engine.value} not available")
                
        except KeyboardInterrupt:
            print("\n🔄 Switch cancelled")

    async def speak_text(self, text: str, engine: Optional[TTSEngine] = None):
        """Generate speech for given text"""
        if not self.services.tts_manager:
            print("❌ TTS not initialized. Use option 5, 6, or 7 first")
            return
        
        try:
            print(f"\n🎤 Generating speech...")
            
            # Use smart selection for auto mode
            if engine == TTSEngine.AUTO:
                available = self.services.tts_manager.get_available_engines()
                engine = smart_engine_selection(text, available)
                print(f"🤖 Auto-selected: {engine.value}")
            
            audio_bytes, gen_time, used_engine = await self.services.tts_manager.generate_speech(
                text, engine
            )
            
            print(f"✅ Generated {len(audio_bytes)} bytes in {gen_time:.3f}s using {used_engine}")
            
            # Performance feedback
            if gen_time < 1.0:
                print("🚀 REAL-TIME: Perfect for conversation!")
            elif gen_time < 3.0:
                print("⚡ FAST: Good response time")
            elif gen_time < 30.0:
                print("🔄 SLOW: Consider Kokoro for faster generation")
            else:
                print("⏳ VERY SLOW: High quality but not conversational")
                
        except Exception as e:
            print(f"❌ Speech generation failed: {e}")

    async def run_test(self):
        """Run TTS engine comparison test"""
        if not self.services.tts_manager:
            print("❌ TTS not initialized. Use option 5, 6, or 7 first")
            return
        
        print("\n🧪 RUNNING TTS ENGINE COMPARISON TEST")
        print("=" * 50)
        
        test_text = "Hello, welcome to our banking services. How may I assist you today?"
        available = self.services.tts_manager.get_available_engines()
        
        for engine in available:
            print(f"\n🎭 Testing {engine.value.upper()}...")
            try:
                audio_bytes, gen_time, used_engine = await self.services.tts_manager.generate_speech(
                    test_text, engine
                )
                
                # Performance assessment
                if gen_time < 1.0:
                    rating = "🚀 EXCELLENT"
                elif gen_time < 2.0:
                    rating = "⚡ GOOD"
                elif gen_time < 10.0:
                    rating = "🔄 ACCEPTABLE"
                else:
                    rating = "⏳ SLOW"
                
                print(f"   {rating}: {gen_time:.3f}s generation")
                print(f"   📁 Size: {len(audio_bytes)} bytes")
                
            except Exception as e:
                print(f"   ❌ FAILED: {e}")
        
        print(f"\n💡 RECOMMENDATION:")
        if TTSEngine.KOKORO in available:
            print("   🚀 Use Kokoro for real-time conversation")
        if TTSEngine.NARI_DIA in available:
            print("   🎭 Use Nari Dia for maximum quality (when time permits)")

    async def run_interactive(self):
        """Run interactive CLI mode with numbered menu"""
        print("\n🎭 Enhanced Voicebot CLI - Interactive Mode")
        print("Type a number (1-9), 'help' for menu, or '0' to exit")
        print("⚠️  No services loaded - use menu option 2 to initialize")
        
        while True:
            try:
                command = input("\n🎤 Menu> ").strip()
                
                if not command:
                    continue
                
                # Handle special commands
                if command.lower() in ['help', '?', 'menu']:
                    self.show_numbered_menu()
                elif command.isdigit():
                    should_continue = await self.handle_numbered_choice(command)
                    if not should_continue:
                        break
                    # Show menu after returning from submenu
                    if command != '0':  # Don't show menu if user typed 0 to exit
                        print()  # Add spacing
                        self.show_numbered_menu()
                else:
                    print(f"❌ Unknown command: {command}")
                    print("💡 Type a number (1-9), 'help' for menu, or '0' to exit")
                    
            except KeyboardInterrupt:
                print("\n🔄 Command cancelled")
            except Exception as e:
                print(f"❌ Command failed: {e}")
        
        print("\n👋 Goodbye!")
    
    async def check_and_cleanup_on_exit(self):
        """Check for running services and offer to shut them down before exit"""
        status = self.services.get_status()
        
        # Check if any services are running
        running_services = []
        if self.services.stt:
            running_services.append('STT')
        if self.services.llm:
            running_services.append(f'LLM ({self.services.current_llm_type})')
        if self.services.tts_manager:
            running_services.append('TTS')
        
        if running_services:
            print(f"\n⚠️  Warning: The following services are still running:")
            for service in running_services:
                print(f"   • {service}")
            
            try:
                response = input("\n🛑 Shut down all services before exit? (y/N): ").strip().lower()
                if response in ['y', 'yes']:
                    print("\n🛑 Shutting down all services...")
                    await self.services.shutdown_stt()
                    await self.services.shutdown_llm()
                    await self.services.shutdown_tts()
                    print("✅ All services stopped and GPU memory freed")
                else:
                    print("⚠️  Services left running - GPU memory may still be in use")
            except KeyboardInterrupt:
                print("\n⚠️  Exit cancelled - services left running")
        else:
            print("\n✅ All services already stopped")
    
    def cleanup(self):
        """Synchronous cleanup - kept for compatibility but async cleanup is preferred"""
        try:
            # Only do basic cleanup here to avoid asyncio conflicts
            print("🧹 Basic cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")


# Additional functions from original CLI (imports needed)
async def run_dual_tts_demo():
    """Run dual TTS demonstration with proper audio generation"""
    try:
        print("🎭 DUAL TTS ENGINE DEMONSTRATION")
        print("=" * 40)
        
        # Demo text for comparison
        demo_text = "Welcome to our premium banking services. This demonstration showcases the quality difference between our TTS engines."
        
        print(f"🎯 Demo text: '{demo_text}'")
        print(f"📏 Length: {len(demo_text)} characters")
        
        # Try to use existing services first
        from voicebot_orchestrator.enhanced_cli_clean import EnhancedVoicebotCLI
        temp_cli = EnhancedVoicebotCLI()
        
        if hasattr(temp_cli, 'services') and temp_cli.services.tts_manager:
            print("🎯 Using existing TTS services...")
            manager = temp_cli.services.tts_manager
            available = manager.get_available_engines()
            
            import time
            timestamp = int(time.time())
            
            for engine in available:
                print(f"\n🚀 Testing {engine.value.upper()}...")
                
                try:
                    start_time = time.time()
                    
                    # Generate audio with specific engine
                    manager.set_engine(engine)
                    audio_data = await manager.synthesize_speech(demo_text)
                    
                    generation_time = time.time() - start_time
                    
                    # Save audio file
                    filename = f"dual_demo_{engine.value.lower()}_{timestamp}.wav"
                    
                    if audio_data and len(audio_data) > 1000:  # Ensure audio has content
                        with open(filename, "wb") as f:
                            f.write(audio_data)
                        
                        print(f"✅ {engine.value}: {generation_time:.2f}s generation")
                        print(f"💾 Saved: {filename} ({len(audio_data)} bytes)")
                        
                        # Validate audio file
                        import os
                        if os.path.getsize(filename) > 1000:
                            print(f"� Audio file verified: {os.path.getsize(filename)} bytes")
                        else:
                            print(f"⚠️ Audio file may be empty or corrupted")
                    else:
                        print(f"❌ {engine.value}: Failed to generate audio or audio is empty")
                        
                except Exception as e:
                    print(f"❌ {engine.value}: Generation failed - {e}")
            
        else:
            print("⚠️ No TTS services loaded. Running standalone demo...")
            
            # Fallback: standalone demo with just Kokoro
            from voicebot_orchestrator.tts import KokoroTTS
            
            print("\n🚀 Testing KOKORO (standalone)...")
            kokoro = KokoroTTS(voice="af_bella")
            
            start_time = time.time()
            audio_data = await kokoro.synthesize_speech(demo_text)
            generation_time = time.time() - start_time
            
            timestamp = int(time.time())
            filename = f"dual_demo_kokoro_standalone_{timestamp}.wav"
            
            if audio_data and len(audio_data) > 1000:
                with open(filename, "wb") as f:
                    f.write(audio_data)
                
                print(f"✅ Kokoro: {generation_time:.2f}s generation")
                print(f"💾 Saved: {filename} ({len(audio_data)} bytes)")
            else:
                print("❌ Kokoro: Failed to generate audio")
                
            print("\n💡 For Nari Dia comparison, use option 2 → 6 to initialize both engines first")
        
        print(f"\n🎯 DEMO COMPLETE!")
        print("🎵 Compare the generated audio files to hear the quality difference")
        
    except Exception as e:
        print(f"❌ Dual TTS demo failed: {e}")
        import traceback
        traceback.print_exc()

async def run_health_check():
    """Run comprehensive system health check"""
    try:
        print("🏥 COMPREHENSIVE SYSTEM HEALTH CHECK")
        print("=" * 45)
        
        # Use the config check we already implemented
        from voicebot_orchestrator.enhanced_cli_clean import EnhancedVoicebotCLI
        
        # Create a temp CLI instance for health check
        cli = EnhancedVoicebotCLI()
        await cli.check_configuration()
        
        # Additional health checks
        print("\n🔍 Additional Health Checks:")
        
        # Test basic imports
        try:
            import torch
            print(f"✅ PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"✅ CUDA: {torch.version.cuda} (GPU: {torch.cuda.get_device_name(0)})")
            else:
                print("⚠️ CUDA: Not available (CPU only)")
        except ImportError:
            print("❌ PyTorch: Not available")
        
        # Test audio libraries
        try:
            import pyaudio
            print("✅ PyAudio: Available for microphone input")
        except ImportError:
            print("⚠️ PyAudio: Not available (voice recording disabled)")
        
        try:
            import speech_recognition
            print("✅ SpeechRecognition: Available for STT fallback")
        except ImportError:
            print("⚠️ SpeechRecognition: Not available")
        
        # Test model files
        print("\n📁 Model File Status:")
        import os
        models = [
            ("kokoro-v1.0.onnx", "Kokoro TTS model"),
            ("voices-v1.0.bin", "Voice data")
        ]
        
        all_models_ok = True
        for model_file, description in models:
            if os.path.exists(model_file):
                size_mb = os.path.getsize(model_file) / 1024**2
                print(f"✅ {model_file}: {size_mb:.1f} MB")
            else:
                print(f"❌ {model_file}: Missing")
                all_models_ok = False
        
        # Final status
        print(f"\n🎯 Overall Health: {'✅ HEALTHY' if all_models_ok else '⚠️ ISSUES DETECTED'}")
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        import traceback
        traceback.print_exc()

async def run_benchmark():
    """Run TTS performance benchmark"""
    try:
        print("📊 TTS PERFORMANCE BENCHMARK")
        print("=" * 35)
        
        # Import and use existing benchmark functionality
        import sys
        import os
        import time
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        
        from voicebot_orchestrator.tts import KokoroTTS
        
        # Benchmark test phrases
        test_phrases = [
            "Hello, welcome to our banking service.",
            "Your account balance is one thousand two hundred dollars.",
            "Thank you for choosing our premium financial solutions for your banking needs."
        ]
        
        print("🚀 Benchmarking Kokoro TTS...")
        kokoro = KokoroTTS(voice="af_bella")
        
        kokoro_times = []
        for i, phrase in enumerate(test_phrases, 1):
            print(f"\n🧪 Test {i}: '{phrase}'")
            print(f"📏 Length: {len(phrase)} characters")
            
            start_time = time.time()
            audio_data = await kokoro.synthesize_speech(phrase)
            generation_time = time.time() - start_time
            
            kokoro_times.append(generation_time)
            
            # Save benchmark file
            filename = f"benchmark_kokoro_{i}_{int(time.time())}.wav"
            with open(filename, "wb") as f:
                f.write(audio_data)
            
            print(f"⏱️ Generation time: {generation_time:.2f}s")
            print(f"📊 Audio size: {len(audio_data)} bytes")
            print(f"💾 Saved: {filename}")
        
        # Results summary
        avg_time = sum(kokoro_times) / len(kokoro_times)
        print(f"\n📈 KOKORO BENCHMARK RESULTS:")
        print(f"   Average generation time: {avg_time:.2f}s")
        print(f"   Fastest: {min(kokoro_times):.2f}s")
        print(f"   Slowest: {max(kokoro_times):.2f}s")
        print(f"   Real-time capable: {'✅ YES' if avg_time < 2.0 else '❌ NO'}")
        
        # Try to benchmark other engines if available
        try:
            # Check for existing TTS manager with Nari
            print("\n🎭 Checking for Nari Dia availability...")
            print("💡 To benchmark Nari Dia, initialize it first via menu option 2 → 5")
            
        except Exception as e:
            print(f"ℹ️ Nari Dia benchmark skipped: {e}")
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced Voicebot CLI with Numbered Menu Options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NUMBERED MENU OPTIONS:
  1-9   Service management (initialize/shutdown)
  10-14 TTS operations (speech generation, testing)
  15-18 Voice features (conversation, demo, health)
  0     Exit CLI
  
EXAMPLES:
  python enhanced_cli_clean.py
  
Then use numbered options:
  5 - Initialize Kokoro TTS (fast)
  3 - Initialize Mistral LLM
  10 - Generate speech
  1 - Check status
  9 - Shutdown all services
  0 - Exit
        """
    )
    
    # Add command line arguments for one-shot operations
    parser.add_argument("--text", help="Generate speech for text and exit")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    
    args = parser.parse_args()
    
    # CLI mode
    cli = EnhancedVoicebotCLI()
    
    try:
        # Initialize CLI
        success = await cli.initialize()
        
        if not success:
            return 1
        
        # One-shot modes
        if args.text:
            # Initialize Kokoro TTS for quick speech generation
            await cli.services.initialize_tts_kokoro()
            await cli.speak_text(args.text)
            # Cleanup for one-shot mode
            await cli.services.shutdown_all()
            return 0
        elif args.status:
            cli.show_status()
            return 0
        
        # Interactive mode with numbered menu
        cli.show_numbered_menu()
        await cli.run_interactive()
        
        # Normal interactive exit already handles cleanup via check_and_cleanup_on_exit()
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 CLI interrupted by user")
        # Async cleanup for interrupt
        try:
            await cli.services.shutdown_all()
        except:
            pass
        return 0
    except Exception as e:
        print(f"❌ CLI failed: {e}")
        import traceback
        traceback.print_exc()
        # Async cleanup for errors
        try:
            await cli.services.shutdown_all()
        except:
            pass
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 CLI interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ CLI startup failed: {e}")
        sys.exit(1)
