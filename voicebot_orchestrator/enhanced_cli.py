"""
Enhanced CLI with Modular Service Management
No auto-loading of services - initialize TTS, LLM, and STT independently
Supports multiple TTS engines (Kokoro/Nari Dia) and LLMs (Mistral/GPT-OSS 20B)
"""
import asyncio
import argparse
import sys
import os
from         print(f"🎙️  STT:  {status['stt']}")
        print(f"🧠  LLM:  {status['llm']}")
        print(f"🔊  TTS:  {status['tts']}")
        print(f"💾  GPU:  {status['gpu_memory']}")ing import Optional

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
        if self.tts_manager is None:
            print("🚀 Initializing TTS Manager with Kokoro...")
            self.tts_manager = EnhancedTTSManager()
        else:
            print("🚀 Adding Kokoro to existing TTS Manager...")
            
        try:
            success = await self.tts_manager.initialize_engines(
                load_kokoro=True,
                load_nari=False
            )
            
            if success:
                print("✅ Kokoro TTS ready (fast generation)")
                return True
            else:
                print("❌ Kokoro TTS initialization failed")
                return False
                
        except Exception as e:
            print(f"❌ Kokoro TTS initialization failed: {e}")
            return False
    
    async def initialize_tts_nari(self) -> bool:
        """Initialize Nari Dia TTS engine only"""
        if self.tts_manager is None:
            print("🎭 Initializing TTS Manager with Nari Dia...")
            self.tts_manager = EnhancedTTSManager()
        else:
            print("🎭 Adding Nari Dia to existing TTS Manager...")
            
        try:
            success = await self.tts_manager.initialize_engines(
                load_kokoro=False,
                load_nari=True
            )
            
            if success:
                print("✅ Nari Dia TTS ready (high quality)")
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
            print("⚠️ TTS already initialized")
            return True
            
        try:
            print("🔊 Initializing Text-to-Speech (both engines)...")
            self.tts_manager = EnhancedTTSManager()
            success = await self.tts_manager.initialize_engines(
                load_kokoro=True,
                load_nari=True
            )
            
            if success:
                print("✅ TTS ready (Kokoro + Nari Dia)")
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
    """Enhanced CLI with modular service management"""
    
    def __init__(self):
        self.services = VoicebotServices()
        self.current_mode = "interactive"
    async def initialize(self):
        """Initialize the CLI without loading any services"""
        print("🚀 Enhanced Voicebot CLI Ready")
        print("⚠️  No services pre-loaded - initialize them manually")
        print("💡 Use 'help' to see service management commands")
        return True
    def show_status(self):
        """Show current system status"""
        print("\n📊 SYSTEM STATUS")
        print("=" * 40)
        
        status = self.services.get_status()
        
        print(f"🎙️  STT:  {status['stt']}")
        print(f"🧠  LLM:  {status['llm']}")
        print(f"�  TTS:  {status['tts']}")
        
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
        
        print(f"\n� GPU Memory: 0GB startup, services load on-demand")
    def show_help(self):
        """Show available commands"""
        print("\n💡 ENHANCED VOICEBOT CLI COMMANDS")
        print("=" * 50)
        
        print("🔧 SERVICE MANAGEMENT")
        print("-" * 20)
        print("📊 status                 - Show system status")
        print("🎙️  init-stt               - Initialize Speech-to-Text")
        print("🧠 init-llm <type>        - Initialize LLM (mistral|gpt-oss)")
        print("� init-kokoro            - Initialize Kokoro TTS (fast)")
        print("🎭 init-nari              - Initialize Nari Dia TTS (quality)")
        print("🔊 init-tts-both          - Initialize both TTS engines")
        print("🛑 shutdown <service>     - Shutdown service (stt|llm|tts|all)")
        print("🔄 switch-llm <type>      - Switch LLM model")
        print("")
        
        print("🎤 TTS OPERATIONS (requires init-tts)")
        print("-" * 35)
        print("🎤 speak <text>           - Generate speech with current engine")
        print("🔄 switch                 - Switch TTS engine interactively")
        print("🚀 kokoro                 - Switch to Kokoro TTS (fast)")
        print("🎭 nari                   - Switch to Nari Dia TTS (quality)")
        print("🤖 auto <text>            - Auto-select engine and speak")
        print("🧪 test                   - Run engine comparison test")
        print("")
        
        print("🎙️ VOICE FEATURES")
        print("-" * 15)
        print("🗣️  conversation            - Start voice conversation")
        print("🎭 demo                   - Run dual TTS demonstration")  
        print("🧪 nari-test <type>       - Run Nari Dia tests")
        print("🏥 health                 - System health check")
        print("📊 benchmark              - Performance benchmarks")
        print("")
        
        print("❓ help                   - Show this help")
        print("👋 quit/exit              - Exit the CLI")
        print()
        
        print("💡 TIPS:")
        print("   • Start with 'init-kokoro' for fast TTS")
        print("   • Or 'init-nari' for high quality TTS")
        print("   • Use 'init-llm mistral' for language processing")
        print("   • 'gpt-oss' LLM is experimental 20B model")
        print("   • You can run both TTS engines simultaneously")
        print()
        
        print("🎙️ VOICE CONVERSATION CONTROLS:")
        print("   SPACE - Start/stop recording")
        print("   'k'   - Switch to Kokoro")  
        print("   'n'   - Switch to Nari Dia")
        print("   's'   - Show status")
        print("   'q'   - Quit conversation")
    async def switch_engine_interactive(self):
        """Interactive engine switching"""
        if not self.services.tts_manager:
            print("❌ TTS not initialized. Use 'init-tts' first")
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
            print("❌ TTS not initialized. Use 'init-tts' first")
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
            print("❌ TTS not initialized. Use 'init-tts' first")
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
    async def handle_service_command(self, command_parts):
        """Handle service management commands"""
        if len(command_parts) < 2:
            print("❌ Please specify a service command")
            return
            
        service_cmd = command_parts[1]
        
        # Initialize commands
        if service_cmd == "init-stt":
            await self.services.initialize_stt()
        elif service_cmd == "init-llm":
            llm_type = command_parts[2] if len(command_parts) > 2 else 'mistral'
            if llm_type not in ['mistral', 'gpt-oss']:
                print("❌ LLM type must be 'mistral' or 'gpt-oss'")
                return
            await self.services.initialize_llm(llm_type)
        elif service_cmd == "init-kokoro":
            await self.services.initialize_tts_kokoro()
        elif service_cmd == "init-nari":
            await self.services.initialize_tts_nari()
        elif service_cmd == "init-tts-both":
            await self.services.initialize_tts_both()
        elif service_cmd == "shutdown":
            if len(command_parts) < 3:
                print("❌ Please specify service to shutdown (stt|llm|tts|all)")
                return
            service = command_parts[2]
            if service == "stt":
                await self.services.shutdown_stt()
            elif service == "llm":
                await self.services.shutdown_llm()
            elif service == "tts":
                await self.services.shutdown_tts()
            elif service == "all":
                await self.services.shutdown_stt()
                await self.services.shutdown_llm()
                await self.services.shutdown_tts()
            else:
                print("❌ Unknown service. Use: stt|llm|tts|all")
        elif service_cmd == "switch-llm":
            if len(command_parts) < 3:
                print("❌ Please specify LLM type (mistral|gpt-oss)")
                return
            llm_type = command_parts[2]
            if llm_type not in ['mistral', 'gpt-oss']:
                print("❌ LLM type must be 'mistral' or 'gpt-oss'")
                return
            # Shutdown current and initialize new
            await self.services.shutdown_llm()
            await self.services.initialize_llm(llm_type)
        else:
            print(f"❌ Unknown service command: {service_cmd}")

    async def run_interactive(self):
        """Run interactive CLI mode"""
        print("\n🎭 Enhanced Voicebot CLI - Interactive Mode")
        print("Type 'help' for commands or 'quit' to exit")
        print("⚠️  No services loaded - use service commands to initialize")
        
        while True:
            try:
                command = input("\n🎤 > ").strip()
                
                if not command:
                    continue
                    
                command_parts = command.split()
                base_command = command_parts[0].lower()
                
                if base_command in ['quit', 'exit', 'q']:
                    await self.check_and_cleanup_on_exit()
                    break
                elif base_command == 'help' or base_command == '?':
                    self.show_help()
                elif base_command == 'status':
                    self.show_status()
                elif base_command in ['init-stt', 'init-llm', 'init-kokoro', 'init-nari', 'init-tts-both', 'shutdown', 'switch-llm']:
                    await self.handle_service_command(['service', base_command] + command_parts[1:])
                elif base_command == 'switch':
                    await self.switch_engine_interactive()
                elif base_command == 'kokoro':
                    if self.services.tts_manager and TTSEngine.KOKORO in self.services.tts_manager.get_available_engines():
                        self.services.tts_manager.set_engine(TTSEngine.KOKORO)
                        print("✅ Switched to Kokoro TTS (fast)")
                    else:
                        print("❌ Kokoro TTS not available. Use 'init-tts' first")
                elif base_command == 'nari':
                    if self.services.tts_manager and TTSEngine.NARI_DIA in self.services.tts_manager.get_available_engines():
                        self.services.tts_manager.set_engine(TTSEngine.NARI_DIA)
                        print("✅ Switched to Nari Dia TTS (quality)")
                    else:
                        print("❌ Nari Dia TTS not available. Use 'init-tts' first")
                elif base_command == 'test':
                    await self.run_test()
                elif base_command == 'conversation':
                    if self.services.tts_manager:
                        print("🎙️ Starting voice conversation with current engine...")
                        await run_voice_conversation(self.services.tts_manager.get_current_engine().value)
                    else:
                        print("❌ TTS not initialized. Use 'init-tts' first")
                elif base_command == 'demo':
                    await run_dual_tts_demo()
                elif command.startswith('nari-test '):
                    test_type = command[10:].strip()
                    await run_nari_test(test_type)
                elif base_command == 'health':
                    await run_health_check()
                elif base_command == 'benchmark':
                    await run_benchmark()
                elif base_command == 'speak':
                    text = ' '.join(command_parts[1:])
                    if text:
                        await self.speak_text(text)
                    else:
                        print("❌ Please provide text to speak")
                elif base_command == 'auto':
                    text = ' '.join(command_parts[1:])
                    if text:
                        await self.speak_text(text, TTSEngine.AUTO)
                    else:
                        print("❌ Please provide text for auto-speech")
                else:
                    print(f"❌ Unknown command: {base_command}")
                    print("💡 Type 'help' to see available commands")
                    
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
                    print("✅ All services stopped")
                else:
                    print("⚠️  Services left running")
            except KeyboardInterrupt:
                print("\n⚠️  Exit cancelled - services left running")
        else:
            print("\n✅ All services already stopped")
    
    def cleanup(self):
        """Cleanup all services"""
        # For async cleanup, we'd need to shut down services properly
        # but this is called in finally block so keep it simple
        pass

async def run_voice_conversation(engine: str = "kokoro"):
    """Run the enhanced voice conversation test"""
    print(f"🎙️ Starting Enhanced Voice Conversation with {engine.upper()} TTS...")
    try:
        # Import and run the voice conversation test
        import subprocess
        import sys
        
        venv_python = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv", "Scripts", "python.exe")
        test_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "test_enhanced_voice_conversation.py")
        
        if os.path.exists(venv_python) and os.path.exists(test_script):
            cmd = [venv_python, test_script, "--engine", engine]
            print(f"🚀 Running: {' '.join(cmd)}")
            subprocess.run(cmd)
        else:
            print("❌ Voice conversation test not found")
            
    except Exception as e:
        print(f"❌ Failed to start voice conversation: {e}")

async def run_dual_tts_demo():
    """Run the dual TTS demonstration"""
    print("🎭 Starting Dual TTS Demonstration...")
    try:
        # Import and run dual TTS demo
        import subprocess
        
        venv_python = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv", "Scripts", "python.exe")
        demo_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "demo_dual_tts.py")
        
        if os.path.exists(venv_python) and os.path.exists(demo_script):
            cmd = [venv_python, demo_script]
            print(f"🚀 Running: {' '.join(cmd)}")
            subprocess.run(cmd)
        else:
            print("❌ Dual TTS demo not found")
            
    except Exception as e:
        print(f"❌ Failed to start dual TTS demo: {e}")

async def run_nari_test(test_type: str = "proper"):
    """Run Nari Dia specific tests"""
    test_map = {
        "proper": "test_nari_proper.py",
        "quick": "test_nari_quick.py", 
        "cuda": "test_nari_cuda.py",
        "persistent": "test_nari_persistent.py",
        "comparison": "test_tts_comparison.py"
    }
    
    if test_type not in test_map:
        print(f"❌ Unknown Nari test type: {test_type}")
        print(f"💡 Available: {', '.join(test_map.keys())}")
        return
    
    print(f"🧪 Running Nari Dia Test: {test_type.upper()}...")
    try:
        import subprocess
        
        venv_python = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv", "Scripts", "python.exe")
        test_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", test_map[test_type])
        
        if os.path.exists(venv_python) and os.path.exists(test_script):
            cmd = [venv_python, test_script]
            print(f"🚀 Running: {' '.join(cmd)}")
            subprocess.run(cmd)
        else:
            print(f"❌ Test script not found: {test_map[test_type]}")
            
    except Exception as e:
        print(f"❌ Failed to run Nari test: {e}")

async def run_health_check():
    """Run comprehensive system health check"""
    print("🏥 VOICEBOT SYSTEM HEALTH CHECK")
    print("=" * 50)
    
    # Check Python environment
    print("1️⃣ Python Environment:")
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        print(f"   {'✅' if torch.cuda.is_available() else '❌'} CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   🎯 GPU: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   💾 GPU Memory: {memory_gb:.1f}GB")
    except Exception as e:
        print(f"   ❌ PyTorch issue: {e}")
    
    # Check models
    print("\n2️⃣ Model Files:")
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    kokoro_model = os.path.join(project_root, "kokoro-v1.0.onnx")
    kokoro_voices = os.path.join(project_root, "voices-v1.0.bin")
    
    print(f"   {'✅' if os.path.exists(kokoro_model) else '❌'} Kokoro Model: {kokoro_model}")
    print(f"   {'✅' if os.path.exists(kokoro_voices) else '❌'} Kokoro Voices: {kokoro_voices}")
    
    # Check Ollama
    print("\n3️⃣ LLM Service:")
    try:
        import ollama
        client = ollama.Client(host="localhost:11434")
        models = client.list()
        print("   ✅ Ollama service accessible")
        print(f"   📋 Available models: {len(models.get('models', []))}")
    except Exception as e:
        print(f"   ⚠️ Ollama issue: {e}")
    
    # Test TTS Manager
    print("\n4️⃣ TTS Manager:")
    try:
        cli = EnhancedVoicebotCLI()
        await cli.initialize()
        success = await cli.services.initialize_tts_both()
        if success:
            available = cli.services.tts_manager.get_available_engines()
            print(f"   ✅ TTS Manager initialized")
            print(f"   🎭 Available engines: {[e.value for e in available]}")
        else:
            print("   ❌ TTS Manager initialization failed")
        cli.cleanup()
    except Exception as e:
        print(f"   ❌ TTS Manager error: {e}")
    
    print("\n✅ Health check complete!")

async def run_benchmark(engines: str = "all"):
    """Run performance benchmarks"""
    print("📊 VOICEBOT PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    test_phrases = [
        "Hello, how are you?",  # Short
        "Welcome to our banking services. How may I assist you today?",  # Medium
        "Thank you for calling our customer service. I understand you need help with your account. Let me access your information and provide you with the assistance you need."  # Long
    ]
    
    try:
        cli = EnhancedVoicebotCLI()
        success = await cli.initialize(load_kokoro=True, load_nari=True)
        
        if not success:
            print("❌ Failed to initialize TTS engines")
            return
        
        available = cli.tts_manager.get_available_engines()
        
        # Filter engines based on request
        if engines == "kokoro" and TTSEngine.KOKORO in available:
            test_engines = [TTSEngine.KOKORO]
        elif engines == "nari_dia" and TTSEngine.NARI_DIA in available:
            test_engines = [TTSEngine.NARI_DIA]
        else:
            test_engines = available
        
        for engine in test_engines:
            print(f"\n🎭 BENCHMARKING {engine.value.upper()}")
            print("-" * 30)
            
            total_time = 0
            total_chars = 0
            
            for i, phrase in enumerate(test_phrases, 1):
                print(f"Test {i}: {len(phrase)} chars...")
                
                try:
                    audio_bytes, gen_time, used_engine = await cli.tts_manager.generate_speech(
                        phrase, engine, save_path=f"benchmark_{engine.value}_{i}.wav"
                    )
                    
                    total_time += gen_time
                    total_chars += len(phrase)
                    
                    chars_per_sec = len(phrase) / gen_time
                    print(f"   ⏱️  Time: {gen_time:.2f}s")
                    print(f"   📊 Speed: {chars_per_sec:.1f} chars/sec")
                    print(f"   💾 Size: {len(audio_bytes):,} bytes")
                    
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
            
            # Summary
            if total_time > 0:
                avg_speed = total_chars / total_time
                print(f"\n📈 SUMMARY for {engine.value.upper()}:")
                print(f"   ⏱️  Total time: {total_time:.2f}s")
                print(f"   📊 Average speed: {avg_speed:.1f} chars/sec")
                
                if avg_speed > 50:
                    print("   🚀 EXCELLENT for real-time use")
                elif avg_speed > 10:
                    print("   ⚡ GOOD for interactive use")
                elif avg_speed > 2:
                    print("   🔄 ACCEPTABLE for non-real-time")
                else:
                    print("   ⏳ SLOW - quality over speed")
        
        cli.cleanup()
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced Voicebot CLI with TTS Engine Toggle and Testing Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
VOICEBOT COMMANDS:
  speak <text>              Generate speech with current engine
  conversation              Start voice conversation (interactive)
  demo                      Run dual TTS demonstration
  test <type>               Run specific tests
  health-check              Check system health
  benchmark                 Run performance benchmarks
  
EXAMPLES:
  # Basic speech generation
  python enhanced_cli.py --text "Hello world"
  python enhanced_cli.py --auto "Auto-select engine for this text"
  
  # Voice conversation
  python enhanced_cli.py conversation --engine kokoro
  python enhanced_cli.py conversation --engine nari_dia
  
  # Testing and demos
  python enhanced_cli.py demo
  python enhanced_cli.py test nari-proper
  python enhanced_cli.py health-check
  python enhanced_cli.py benchmark --engines kokoro
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Conversation command
    conv_parser = subparsers.add_parser('conversation', help='Start voice conversation')
    conv_parser.add_argument('--engine', choices=['kokoro', 'nari_dia'], default='kokoro',
                           help='TTS engine to use')
    
    # Demo command  
    demo_parser = subparsers.add_parser('demo', help='Run dual TTS demonstration')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run specific tests')
    test_parser.add_argument('type', choices=['nari-proper', 'nari-quick', 'nari-cuda', 
                           'nari-persistent', 'tts-comparison'], 
                           help='Type of test to run')
    
    # Health check command
    health_parser = subparsers.add_parser('health-check', help='Check system health')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    bench_parser.add_argument('--engines', choices=['kokoro', 'nari_dia', 'all'], 
                            default='all', help='Engines to benchmark')
    
    # Original arguments for backward compatibility
    parser.add_argument("--no-kokoro", action="store_true", help="Skip loading Kokoro TTS")
    parser.add_argument("--no-nari", action="store_true", help="Skip loading Nari Dia TTS")
    parser.add_argument("--engine", choices=["kokoro", "nari_dia"], help="Set default engine")
    parser.add_argument("--text", help="Generate speech for text and exit")
    parser.add_argument("--auto", help="Auto-select engine and generate speech")
    
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'conversation':
        await run_voice_conversation(args.engine)
        return 0
    elif args.command == 'demo':
        await run_dual_tts_demo()
        return 0
    elif args.command == 'test':
        # Map the test command to proper format
        test_type_map = {
            'nari-proper': 'proper',
            'nari-quick': 'quick', 
            'nari-cuda': 'cuda',
            'nari-persistent': 'persistent',
            'tts-comparison': 'comparison'
        }
        test_type = test_type_map.get(args.type, args.type)
        await run_nari_test(test_type)
        return 0
    elif args.command == 'health-check':
        await run_health_check()
        return 0
    elif args.command == 'benchmark':
        await run_benchmark(args.engines)
        return 0
    
    # Original functionality - CLI mode
    cli = EnhancedVoicebotCLI()
    
    try:
        # Initialize CLI without loading services
        success = await cli.initialize()
        
        if not success:
            return 1
        
        # One-shot modes with service initialization if needed
        if args.text:
            # Initialize TTS for text generation
            if not args.no_kokoro and not args.no_nari:
                await cli.services.initialize_tts_both()
            elif not args.no_kokoro:
                await cli.services.initialize_tts_kokoro()
            elif not args.no_nari:
                await cli.services.initialize_tts_nari()
            else:
                print("❌ At least one TTS engine must be enabled")
                return 1
                
            # Set engine if specified
            if args.engine:
                try:
                    cli.services.tts_manager.set_engine(TTSEngine(args.engine))
                except ValueError as e:
                    print(f"❌ {e}")
                    return 1
            await cli.speak_text(args.text)
            return 0
        elif args.auto:
            # Initialize TTS for auto mode
            if not args.no_kokoro and not args.no_nari:
                await cli.services.initialize_tts_both()
            elif not args.no_kokoro:
                await cli.services.initialize_tts_kokoro()
            elif not args.no_nari:
                await cli.services.initialize_tts_nari()
            else:
                print("❌ At least one TTS engine must be enabled")
                return 1
            await cli.speak_text(args.auto, TTSEngine.AUTO)
            return 0
        
        # Interactive mode - show status and start
        cli.show_status()
        await cli.run_interactive()
        
    except Exception as e:
        print(f"❌ CLI failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cli.cleanup()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
