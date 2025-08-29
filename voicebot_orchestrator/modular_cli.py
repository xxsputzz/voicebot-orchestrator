"""
Enhanced Voicebot CLI - Modular Service Architecture
Clean landing page with on-demand service initialization
"""
import asyncio
import argparse
import sys
import os
import subprocess
from typing import Optional
from datetime import datetime

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class VoicebotServices:
    """Manages voicebot microservices"""
    
    def __init__(self):
        self.stt_initialized = False
        self.llm_initialized = False
        self.llm_type = None  # Track which LLM is loaded
        self.tts_engines = {
            'kokoro': False,
            'nari_dia': False
        }
        self.current_tts = None
        
        # Service instances
        self.stt_service = None
        self.llm_service = None
        self.tts_manager = None
    
    def get_status(self):
        """Get current service status"""
        status = {
            'stt': '✅ Ready' if self.stt_initialized else '❌ Not loaded',
            'llm': f'✅ Ready ({self.llm_type})' if self.llm_initialized else '❌ Not loaded',
            'tts_kokoro': '✅ Ready' if self.tts_engines['kokoro'] else '❌ Not loaded',
            'tts_nari': '✅ Ready' if self.tts_engines['nari_dia'] else '❌ Not loaded',
            'current_tts': self.current_tts or 'None'
        }
        return status
    
    async def initialize_stt(self):
        """Initialize Speech-to-Text service"""
        if self.stt_initialized:
            print("🎤 STT already initialized")
            return True
            
        print("🎤 Initializing Speech-to-Text...")
        try:
            # Add tests directory to path for imports
            import sys
            import os
            tests_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests')
            if tests_path not in sys.path:
                sys.path.append(tests_path)
            
            from faster_whisper_stt import FasterWhisperSTT
            self.stt_service = FasterWhisperSTT()
            self.stt_initialized = True
            print("✅ STT initialized successfully")
            return True
        except Exception as e:
            print(f"❌ STT initialization failed: {e}")
            return False
    
    async def initialize_llm(self, llm_type='mistral'):
        """Initialize Large Language Model service"""
        if self.llm_initialized and self.llm_type == llm_type:
            print(f"🧠 LLM ({llm_type}) already initialized")
            return True
        elif self.llm_initialized:
            print(f"🧠 Switching LLM from {self.llm_type} to {llm_type}...")
            await self.shutdown_llm()
            
        print(f"🧠 Initializing Large Language Model ({llm_type.upper()})...")
        try:
            if llm_type == 'mistral':
                from voicebot_orchestrator.enhanced_llm import EnhancedMistralLLM
                self.llm_service = EnhancedMistralLLM(model_path="mistral:latest")
            elif llm_type == 'gpt-oss':
                # Initialize GPT-OSS 20B model
                from voicebot_orchestrator.llm import MistralLLM  # Use base class for now
                self.llm_service = MistralLLM(
                    model_path="gpt-oss:20b",  # Ollama model name
                    max_tokens=2048,
                    temperature=0.7
                )
            else:
                raise ValueError(f"Unknown LLM type: {llm_type}")
                
            self.llm_initialized = True
            self.llm_type = llm_type
            print(f"✅ LLM ({llm_type}) initialized successfully")
            return True
        except Exception as e:
            print(f"❌ LLM ({llm_type}) initialization failed: {e}")
            return False
    
    async def initialize_tts(self, engine='kokoro'):
        """Initialize Text-to-Speech engine"""
        if self.tts_engines.get(engine, False):
            print(f"🔊 TTS {engine} already initialized")
            return True
            
        print(f"🔊 Initializing TTS Engine: {engine.upper()}...")
        
        try:
            if not self.tts_manager:
                from voicebot_orchestrator.enhanced_tts_manager import EnhancedTTSManager
                self.tts_manager = EnhancedTTSManager()
            
            # Initialize only the requested engine
            load_kokoro = (engine == 'kokoro')
            load_nari = (engine == 'nari_dia')
            
            success = await self.tts_manager.initialize_engines(
                load_kokoro=load_kokoro,
                load_nari=load_nari
            )
            
            if success:
                self.tts_engines[engine] = True
                self.current_tts = engine
                print(f"✅ TTS {engine} initialized successfully")
                return True
            else:
                print(f"❌ TTS {engine} initialization failed")
                return False
                
        except Exception as e:
            print(f"❌ TTS {engine} initialization failed: {e}")
            return False
    
    # SHUTDOWN METHODS
    
    async def shutdown_stt(self):
        """Shutdown Speech-to-Text service"""
        if not self.stt_initialized:
            print("🎤 STT already stopped")
            return True
            
        print("🎤 Shutting down Speech-to-Text...")
        try:
            if self.stt_service:
                # STT cleanup if needed
                self.stt_service = None
            self.stt_initialized = False
            print("✅ STT shutdown successfully")
            return True
        except Exception as e:
            print(f"❌ STT shutdown failed: {e}")
            return False
    
    async def shutdown_llm(self):
        """Shutdown Large Language Model service"""
        if not self.llm_initialized:
            print("🧠 LLM already stopped")
            return True
            
        print(f"🧠 Shutting down LLM ({self.llm_type})...")
        try:
            if self.llm_service:
                # LLM cleanup if needed
                self.llm_service = None
            self.llm_initialized = False
            self.llm_type = None
            print("✅ LLM shutdown successfully")
            return True
        except Exception as e:
            print(f"❌ LLM shutdown failed: {e}")
            return False
    
    async def shutdown_tts(self, engine=None):
        """Shutdown Text-to-Speech engine(s)"""
        if engine is None:
            # Shutdown all TTS engines
            print("🔊 Shutting down all TTS engines...")
            try:
                if self.tts_manager:
                    self.tts_manager.cleanup()
                    self.tts_manager = None
                self.tts_engines = {'kokoro': False, 'nari_dia': False}
                self.current_tts = None
                print("✅ All TTS engines shutdown successfully")
                return True
            except Exception as e:
                print(f"❌ TTS shutdown failed: {e}")
                return False
        else:
            # Shutdown specific engine
            if not self.tts_engines.get(engine, False):
                print(f"🔊 TTS {engine} already stopped")
                return True
                
            print(f"🔊 Shutting down TTS {engine}...")
            try:
                # For now, we'll shutdown all TTS since they share the manager
                if self.tts_manager:
                    self.tts_manager.cleanup()
                    self.tts_manager = None
                self.tts_engines = {'kokoro': False, 'nari_dia': False}
                self.current_tts = None
                print(f"✅ TTS {engine} shutdown successfully")
                return True
            except Exception as e:
                print(f"❌ TTS {engine} shutdown failed: {e}")
                return False
    
    def cleanup(self):
        """Clean up all services (legacy method)"""
        asyncio.create_task(self.cleanup_all())
    
    async def cleanup_all(self):
        """Clean up all services"""
        print("🧹 Cleaning up all services...")
        
        # Shutdown in reverse order of dependencies
        await self.shutdown_tts()
        await self.shutdown_llm() 
        await self.shutdown_stt()
        
        print("✅ All services cleaned up")

class EnhancedVoicebotCLI:
    """Enhanced CLI with modular service architecture"""
    
    def __init__(self):
        self.services = VoicebotServices()
        
    def show_landing_page(self):
        """Show clean landing page with main options"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "="*60)
        print("🎭 ORKESTRA VOICEBOT ORCHESTRATOR")
        print("="*60)
        print(f"📅 {timestamp}")
        
        # Show service status
        status = self.services.get_status()
        print(f"\n📊 Service Status:")
        print(f"   🎤 STT: {status['stt']}")
        print(f"   🧠 LLM: {status['llm']}")
        print(f"   🔊 TTS Kokoro: {status['tts_kokoro']}")
        print(f"   🔊 TTS Nari Dia: {status['tts_nari']}")
        print(f"   🎯 Active TTS: {status['current_tts']}")
        
        print(f"\n🚀 MAIN OPTIONS")
        print("-"*30)
        print("1. 🎙️  Voice Pipeline       - Full conversation (STT→LLM→TTS)")
        print("2. 🔧 Service Management   - Initialize/manage microservices")
        print("3. 🧪 Testing & Demos      - Run tests and demonstrations")
        print("4. 🎵 Audio Generation     - Direct TTS text-to-speech")
        print("5. 🏥 Health & Diagnostics - System health and benchmarks")
        print("6. 📚 Documentation        - Help and guides")
        print("7. ⚙️  Settings & Config   - Configuration options")
        print("8. 👋 Exit                - Quit the CLI")
        
        print(f"\n💡 Quick Commands: 'help', 'status', 'quick-test'")
        
    async def handle_voice_pipeline(self):
        """Handle voice pipeline submenu"""
        while True:
            print("\n🎙️ VOICE PIPELINE OPTIONS")
            print("-"*40)
            print("1. 🗣️  Start Voice Conversation")
            print("2. 🎤 Test STT Only")
            print("3. 🧠 Test LLM Only") 
            print("4. 🔊 Test TTS Only")
            print("5. 🔄 Test Full Pipeline")
            print("6. ⚙️  Configure Pipeline")
            print("0. 🔙 Back to Main Menu")
            
            choice = input("\n🎯 Choose option (0-6): ").strip()
            
            try:
                if choice == "1":
                    await self.start_voice_conversation()
                elif choice == "2":
                    await self.test_stt()
                elif choice == "3":
                    await self.test_llm()
                elif choice == "4":
                    await self.test_tts()
                elif choice == "5":
                    await self.test_full_pipeline()
                elif choice == "6":
                    await self.configure_pipeline()
                elif choice == "0":
                    break
                else:
                    print("❌ Invalid choice")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def handle_service_management(self):
        """Handle service management submenu"""
        while True:
            print("\n🔧 SERVICE MANAGEMENT")
            print("-"*30)
            print("1. 🎤 Initialize STT")
            print("2. 🧠 Initialize LLM (Mistral)")
            print("3. 🤖 Initialize LLM (GPT-OSS 20B)")
            print("4. 🔊 Initialize TTS Kokoro (Fast)")
            print("5. 🎭 Initialize TTS Nari Dia (Quality)")
            print("6. 🚀 Initialize All Services")
            print("7. 📊 Service Status")
            print("8. 🔴 SHUTDOWN SERVICES")
            print("9. 🧹 Cleanup All Services")
            print("0. 🔙 Back to Main Menu")
            
            choice = input("\n🎯 Choose option (0-9): ").strip()
            
            try:
                if choice == "1":
                    await self.services.initialize_stt()
                elif choice == "2":
                    await self.services.initialize_llm('mistral')
                elif choice == "3":
                    await self.services.initialize_llm('gpt-oss')
                elif choice == "4":
                    await self.services.initialize_tts('kokoro')
                elif choice == "5":
                    await self.services.initialize_tts('nari_dia')
                elif choice == "6":
                    await self.initialize_all_services()
                elif choice == "7":
                    self.show_service_status()
                elif choice == "8":
                    await self.handle_service_shutdown()
                elif choice == "9":
                    await self.services.cleanup_all()
                    print("🧹 All services cleaned up")
                elif choice == "0":
                    break
                else:
                    print("❌ Invalid choice")
            except Exception as e:
                print(f"❌ Error: {e}")
            
    async def handle_service_shutdown(self):
        """Handle service shutdown submenu"""
        while True:
            print("\n🔴 SERVICE SHUTDOWN")
            print("-"*25)
            print("1. 🎤 Shutdown STT")
            print("2. 🧠 Shutdown LLM")
            print("3. 🔊 Shutdown TTS Kokoro")
            print("4. 🎭 Shutdown TTS Nari Dia")
            print("5. 🔊 Shutdown All TTS")
            print("6. 🧹 Shutdown All Services")
            print("0. 🔙 Back to Service Management")
            
            choice = input("\n🎯 Choose option (0-6): ").strip()
            
            try:
                if choice == "1":
                    await self.services.shutdown_stt()
                elif choice == "2":
                    await self.services.shutdown_llm()
                elif choice == "3":
                    await self.services.shutdown_tts('kokoro')
                elif choice == "4":
                    await self.services.shutdown_tts('nari_dia')
                elif choice == "5":
                    await self.services.shutdown_tts()
                elif choice == "6":
                    await self.services.cleanup_all()
                    print("🧹 All services shutdown")
                elif choice == "0":
                    break
                else:
                    print("❌ Invalid choice")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def handle_testing_demos(self):
        """Handle testing and demos submenu"""
        print("\n🧪 TESTING & DEMOS")
        print("-"*25)
        print("1. 🎭 Dual TTS Demo")
        print("2. 🔊 TTS Comparison Test")
        print("3. 🎤 STT Accuracy Test")
        print("4. 🧠 LLM Response Test")
        print("5. ⚡ Performance Benchmarks")
        print("6. 🎵 Audio Quality Test")
        print("7. 🔧 Integration Tests")
        print("8. 🔙 Back to Main Menu")
        
        choice = input("\n🎯 Choose option (1-8): ").strip()
        
        if choice == "1":
            await self.run_dual_tts_demo()
        elif choice == "2":
            await self.run_tts_comparison()
        elif choice == "3":
            await self.run_stt_test()
        elif choice == "4":
            await self.run_llm_test()
        elif choice == "5":
            await self.run_benchmarks()
        elif choice == "6":
            await self.run_audio_test()
        elif choice == "7":
            await self.run_integration_tests()
        elif choice == "8":
            return
        else:
            print("❌ Invalid choice")
    
    async def handle_audio_generation(self):
        """Handle audio generation submenu"""
        print("\n🎵 AUDIO GENERATION")
        print("-"*25)
        print("1. 🚀 Generate with Kokoro (Fast)")
        print("2. 🎭 Generate with Nari Dia (Quality)")
        print("3. 🤖 Auto-select Engine")
        print("4. 📝 Batch Text Processing")
        print("5. 🎤 Voice Cloning (if available)")
        print("6. 🔙 Back to Main Menu")
        
        choice = input("\n🎯 Choose option (1-6): ").strip()
        
        if choice == "1":
            await self.generate_kokoro()
        elif choice == "2":
            await self.generate_nari()
        elif choice == "3":
            await self.generate_auto()
        elif choice == "4":
            await self.batch_generate()
        elif choice == "5":
            print("🚧 Voice cloning not yet implemented")
        elif choice == "6":
            return
        else:
            print("❌ Invalid choice")
    
    async def handle_health_diagnostics(self):
        """Handle health and diagnostics submenu"""
        print("\n🏥 HEALTH & DIAGNOSTICS")
        print("-"*30)
        print("1. 🏥 System Health Check")
        print("2. 📊 Performance Benchmarks")
        print("3. 🔍 CUDA/GPU Status")
        print("4. 💾 Memory Usage")
        print("5. 🔧 Configuration Check")
        print("6. 📈 Analytics Report")
        print("7. 🔙 Back to Main Menu")
        
        choice = input("\n🎯 Choose option (1-7): ").strip()
        
        if choice == "1":
            await self.run_health_check()
        elif choice == "2":
            await self.run_benchmarks()
        elif choice == "3":
            self.check_cuda_status()
        elif choice == "4":
            self.check_memory_usage()
        elif choice == "5":
            self.check_configuration()
        elif choice == "6":
            self.show_analytics()
        elif choice == "7":
            return
        else:
            print("❌ Invalid choice")
    
    def show_service_status(self):
        """Show detailed service status"""
        status = self.services.get_status()
        print("\n📊 DETAILED SERVICE STATUS")
        print("="*40)
        print(f"🎤 Speech-to-Text:    {status['stt']}")
        print(f"🧠 Language Model:    {status['llm']}")
        print(f"🚀 TTS Kokoro:        {status['tts_kokoro']}")
        print(f"🎭 TTS Nari Dia:      {status['tts_nari']}")
        print(f"🎯 Active TTS:        {status['current_tts']}")
        
        if any([status['stt'] == '✅ Ready', status['llm'] == '✅ Ready', 
                status['tts_kokoro'] == '✅ Ready', status['tts_nari'] == '✅ Ready']):
            print("\n💡 Some services are running and using GPU memory")
        else:
            print("\n💡 No services loaded - GPU memory available")
    
    async def initialize_all_services(self):
        """Initialize all services with user confirmation"""
        print("\n⚠️  WARNING: Initializing all services will use significant GPU memory")
        print("   Recommended: Initialize services as needed")
        
        confirm = input("Continue with full initialization? (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ Canceled")
            return
        
        print("\n🚀 Initializing all services...")
        await self.services.initialize_stt()
        await self.services.initialize_llm()
        await self.services.initialize_tts('kokoro')
        
        # Ask about Nari Dia separately due to memory usage
        nari_confirm = input("\n🎭 Also initialize Nari Dia TTS? (Warning: Uses ~8GB GPU memory) (y/N): ").strip().lower()
        if nari_confirm == 'y':
            await self.services.initialize_tts('nari_dia')
        
        print("\n✅ Service initialization complete!")
    
    async def start_voice_conversation(self):
        """Start voice conversation with pre-checks"""
        print("\n🎙️ STARTING VOICE CONVERSATION")
        print("-"*35)
        
        # Check required services
        if not self.services.stt_initialized:
            print("🎤 STT not initialized. Initializing now...")
            if not await self.services.initialize_stt():
                print("❌ Cannot start conversation without STT")
                return
        
        if not self.services.llm_initialized:
            print("\n🧠 LLM not initialized. Please select LLM:")
            print("1. 🤖 Mistral (Default)")
            print("2. 🧠 GPT-OSS 20B (Experimental)")
            
            llm_choice = input("Choose LLM (1-2): ").strip()
            if llm_choice == "1":
                if not await self.services.initialize_llm('mistral'):
                    print("❌ Cannot start conversation without LLM")
                    return
            elif llm_choice == "2":
                if not await self.services.initialize_llm('gpt-oss'):
                    print("❌ Cannot start conversation without LLM")
                    return
            else:
                print("❌ Invalid choice, using Mistral")
                if not await self.services.initialize_llm('mistral'):
                    print("❌ Cannot start conversation without LLM")
                    return
        
        # TTS selection
        if not any(self.services.tts_engines.values()):
            print("\n🔊 No TTS engine loaded. Please select:")
            print("1. 🚀 Kokoro (Fast, real-time)")
            print("2. 🎭 Nari Dia (Quality, slow)")
            
            tts_choice = input("Choose TTS engine (1-2): ").strip()
            if tts_choice == "1":
                await self.services.initialize_tts('kokoro')
            elif tts_choice == "2":
                await self.services.initialize_tts('nari_dia')
            else:
                print("❌ Invalid choice, using Kokoro")
                await self.services.initialize_tts('kokoro')
        
        print("\n✅ All services ready! Starting conversation...")
        print(f"   🎤 STT: Ready")
        print(f"   🧠 LLM: {self.services.llm_type}")
        print(f"   🔊 TTS: {self.services.current_tts}")
        
        # Run the actual conversation
        try:
            # Import the conversation test
            import sys
            import os
            tests_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests')
            if tests_path not in sys.path:
                sys.path.append(tests_path)
            
            from test_enhanced_voice_conversation import run_voice_conversation_test
            await run_voice_conversation_test()
        except Exception as e:
            print(f"❌ Conversation failed: {e}")
            print("💡 You can manually test the services through other menu options")
    
    async def run_interactive(self):
        """Run interactive CLI"""
        while True:
            try:
                self.show_landing_page()
                choice = input("\n🎯 Choose option (1-8): ").strip()
                
                if choice == "1":
                    await self.handle_voice_pipeline()
                elif choice == "2":
                    await self.handle_service_management()
                elif choice == "3":
                    await self.handle_testing_demos()
                elif choice == "4":
                    await self.handle_audio_generation()
                elif choice == "5":
                    await self.handle_health_diagnostics()
                elif choice == "6":
                    self.show_documentation()
                elif choice == "7":
                    self.show_settings()
                elif choice == "8":
                    print("\n👋 Goodbye!")
                    break
                elif choice.lower() in ['help', 'h']:
                    self.show_help()
                elif choice.lower() in ['status', 's']:
                    self.show_service_status()
                elif choice.lower() == 'quick-test':
                    await self.run_quick_test()
                else:
                    print("❌ Invalid choice. Try 'help' for available commands.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("⏸️  Press Enter to continue...")
    
    def show_help(self):
        """Show help information"""
        print("\n💡 ENHANCED VOICEBOT CLI HELP")
        print("="*40)
        print("🎭 Main Features:")
        print("   • Modular service initialization")
        print("   • On-demand GPU memory usage")
        print("   • Full voice pipeline (STT→LLM→TTS)")
        print("   • Dual TTS engines (fast vs quality)")
        print("   • Comprehensive testing suite")
        print()
        print("⚡ Quick Commands:")
        print("   help     - Show this help")
        print("   status   - Show service status")
        print("   quick-test - Run quick system test")
        print()
        print("🎯 Navigation:")
        print("   Use number keys to navigate menus")
        print("   Most menus have 'Back to Main Menu' option")
        print("   Ctrl+C to exit anytime")
    
    def show_documentation(self):
        """Show documentation menu"""
        print("\n📚 DOCUMENTATION")
        print("-"*20)
        print("1. 📖 User Guide")
        print("2. 🔧 Technical Documentation") 
        print("3. 🎤 Voice Commands Reference")
        print("4. 🧪 Testing Guide")
        print("5. ⚠️  Troubleshooting")
        print("6. 🔙 Back to Main Menu")
        
        choice = input("\n🎯 Choose option (1-6): ").strip()
        
        if choice == "1":
            self.show_user_guide()
        elif choice == "2":
            self.show_technical_docs()
        elif choice == "3":
            self.show_voice_commands()
        elif choice == "4":
            self.show_testing_guide()
        elif choice == "5":
            self.show_troubleshooting()
        elif choice == "6":
            return
        else:
            print("❌ Invalid choice")
    
    def show_settings(self):
        """Show settings menu"""
        print("\n⚙️  SETTINGS & CONFIGURATION")
        print("-"*30)
        print("1. 🎤 STT Configuration")
        print("2. 🧠 LLM Configuration")
        print("3. 🔊 TTS Configuration")
        print("4. 📁 Audio Output Settings")
        print("5. 🔧 System Preferences")
        print("6. 🔙 Back to Main Menu")
        
        choice = input("\n🎯 Choose option (1-6): ").strip()
        
        if choice == "1":
            self.configure_stt()
        elif choice == "2":
            self.configure_llm()
        elif choice == "3":
            self.configure_tts()
        elif choice == "4":
            self.configure_audio_output()
        elif choice == "5":
            self.configure_system()
        elif choice == "6":
            return
        else:
            print("❌ Invalid choice")
    
    # Placeholder methods for functionality to be implemented
    async def test_stt(self):
        print("🧪 Running STT test...")
        # Implementation to be added
    
    async def test_llm(self):
        print("🧪 Running LLM test...")
        # Implementation to be added
    
    async def test_tts(self):
        print("🧪 Running TTS test...")
        # Implementation to be added
    
    async def test_full_pipeline(self):
        print("🧪 Running full pipeline test...")
        # Implementation to be added
    
    async def configure_pipeline(self):
        print("⚙️ Configuring pipeline...")
        # Implementation to be added
    
    async def run_dual_tts_demo(self):
        print("🎭 Running dual TTS demo...")
        try:
            subprocess.run([sys.executable, "tests/demo_dual_tts.py"], check=True)
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    
    async def run_quick_test(self):
        """Run quick system test"""
        print("\n🚀 QUICK SYSTEM TEST")
        print("-"*25)
        
        # Check CUDA
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA available")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("❌ CUDA not available")
        
        # Check virtual environment
        venv_path = os.environ.get('VIRTUAL_ENV')
        if venv_path:
            print("✅ Virtual environment active")
        else:
            print("⚠️  Virtual environment not detected")
        
        print("✅ Quick test complete")
    
    def check_cuda_status(self):
        """Check CUDA/GPU status"""
        import torch
        print("\n🔍 CUDA/GPU STATUS")
        print("-"*25)
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.current_device()}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f}GB")
    
    def check_memory_usage(self):
        """Check memory usage"""
        import psutil
        print("\n💾 MEMORY USAGE")
        print("-"*20)
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / 1024**3:.1f}GB")
        print(f"Available: {memory.available / 1024**3:.1f}GB")
        print(f"Used: {memory.used / 1024**3:.1f}GB ({memory.percent}%)")
    
    def cleanup(self):
        """Clean up resources"""
        self.services.cleanup()

# Command line argument parsing
async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Voicebot CLI with Modular Services")
    parser.add_argument('--quick-start', action='store_true', help='Quick start with minimal services')
    parser.add_argument('--health-check', action='store_true', help='Run health check and exit')
    parser.add_argument('--status', action='store_true', help='Show status and exit')
    
    args = parser.parse_args()
    
    cli = EnhancedVoicebotCLI()
    
    try:
        if args.health_check:
            await cli.run_quick_test()
            return 0
        elif args.status:
            cli.show_service_status()
            return 0
        else:
            # Start interactive mode
            await cli.run_interactive()
            return 0
            
    except Exception as e:
        print(f"❌ CLI failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cli.cleanup()

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
