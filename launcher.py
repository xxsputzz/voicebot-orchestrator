1
0
#!/usr/bin/env python3
"""
Voicebot Orchestrator Launcher

Main entry point for launching different components of the voicebot orchestration platform.
Provides a unified interface for testing, services, and deployment.
"""

import asyncio
import sys
import os
import subprocess
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_banner():
    """Print application banner."""
    print("=" * 60)
    print("🤖 VOICEBOT ORCHESTRATION PLATFORM LAUNCHER")
    print("   Enterprise-Grade Banking Voice AI")
    print("   Sprint 6 Complete | Independent Services Ready")
    print("=" * 60)
    print()

def show_main_menu():
    """Show the main menu."""
    print("=" * 60)
    print("📋 MAIN MENU")
    print("=" * 60)
    print()
    print("🔧 SERVICE MANAGEMENT:")
    print("  1. Launch Enhanced Service Manager (Independent Services)")
    print("  2. Check Service Status")
    print()
    print("🧪 TESTING SUITES:")
    print("  3. Run All Tests")
    print("  4. Run TTS/LLM Combination Tests")
    print("  5. Run Independent Services Tests")
    print("  6. Run Pipeline Tests (Currently Running Services)")
    print("  7. Test Interactive Pipeline (Select Specific Services)")
    print("  8. Run Specific Test Suite")
    print("  9. Test Menu (Batch Scripts)")
    print()
    print("🚀 DEMOS & EXAMPLES:")
    print("  10. Enterprise CLI Demo (Production Validation)")
    print("  11. Run Sprint 6 Demo")
    print("  12. Voice Conversation Demo")
    print()
    print("📊 ANALYTICS & MONITORING:")
    print("  13. Analytics Dashboard")
    print("  14. Performance Report")
    print("  15. Cache Statistics")
    print()
    print("🐳 DEPLOYMENT:")
    print("  16. Docker Compose Setup")
    print("  17. Kubernetes Instructions")
    print("  18. Installation Guide")
    print()
    print("  0. Exit")
    print()

def launch_enhanced_service_manager():
    """Launch the enhanced service manager."""
    script_path = project_root / "aws_microservices" / "enhanced_service_manager.py"
    if script_path.exists():
        subprocess.run([sys.executable, str(script_path)], cwd=project_root)
    else:
        print(f"❌ Enhanced service manager not found at: {script_path}")
        print("=" * 60)
        print("🎭 Enhanced Independent Microservices Manager")
        print("=" * 60)
        print()
        print("This provides numbered menu for independent services:")
        print("- Fast: Kokoro TTS + Mistral LLM")
        print("- Balanced: Kokoro TTS + GPT LLM") 
        print("- Efficient: Hira Dia TTS + Mistral LLM")
        print("- Quality: Hira Dia TTS + GPT LLM")
        print("- Premium: Tortoise TTS + GPT LLM")
        print("- Individual service management")
        print("- Comprehensive testing")
        print()

def launch_original_orchestrator():
    """Launch the original FastAPI orchestrator."""
    print("🚀 Launching Original Orchestrator (FastAPI)...")
    print("-" * 40)
    print("Starting FastAPI server on http://localhost:8000")
    print("WebSocket endpoint: ws://localhost:8000/ws")
    print()
    
    try:
        script_path = project_root / "start_server.py"
        if script_path.exists():
            subprocess.run([sys.executable, str(script_path)], cwd=project_root)
        else:
            # Try alternative methods
            subprocess.run([sys.executable, "-m", "voicebot_orchestrator.main"], cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 Orchestrator closed by user")
    except Exception as e:
        print(f"❌ Error launching orchestrator: {e}")
        print("Alternative: Try running 'python start_server.py' or 'uvicorn voicebot_orchestrator.main:app'")

def check_service_status():
    """Check status of all services."""
    print("-" * 40)
    print("🔍 Checking Service Status...")
    print("-" * 40)
    
    services = {
        "Main Orchestrator": "http://localhost:8000",
        "Whisper STT": "http://localhost:8003", 
        "Kokoro TTS": "http://localhost:8011",
        "Hira Dia TTS": "http://localhost:8012",
        "Dia 4-bit TTS": "http://localhost:8013",
        "Zonos TTS": "http://localhost:8014", 
        "Tortoise TTS": "http://localhost:8015",
        "Mistral LLM": "http://localhost:8021",
        "GPT LLM": "http://localhost:8022"
    }
    
    try:
        import requests
        
        for service_name, url in services.items():
            try:
                response = requests.get(f"{url}/health", timeout=3)
                if response.status_code == 200:
                    print(f"✅ {service_name}: Running ({url})")
                else:
                    print(f"⚠️ {service_name}: Unhealthy (Status: {response.status_code})")
            except requests.exceptions.RequestException:
                print(f"❌ {service_name}: Not running ({url})")
    
    except ImportError:
        print("❌ requests library not available. Cannot check services.")
        print("Install with: pip install requests")

async def run_all_tests():
    """Run all test suites."""
    print("🧪 Running All Test Suites...")
    print("-" * 40)
    
    try:
        # Change to tests directory
        tests_dir = project_root / "tests"
        os.chdir(tests_dir)
        
        # Run the comprehensive test suite
        subprocess.run([sys.executable, "run_tests.py", "all"], cwd=tests_dir)
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")

async def run_combination_tests():
    """Run TTS/LLM combination tests."""
    print("🎭 Running TTS/LLM Combination Tests...")
    print("=" * 40)
    
    tests_dir = project_root / "tests"
    print("1. General TTS/LLM combinations")
    print("2. Independent services combinations")
    print("3. Both")
    choice = input("\nSelect test type (1-3): ").strip()
    if choice == "1":
        subprocess.run([sys.executable, "test_tts_llm_combinations.py"], cwd=tests_dir)
    elif choice == "2":
        subprocess.run([sys.executable, "test_independent_combinations.py"], cwd=tests_dir)
    elif choice == "3":
        subprocess.run([sys.executable, "test_tts_llm_combinations.py"], cwd=tests_dir)
        subprocess.run([sys.executable, "test_independent_combinations.py"], cwd=tests_dir)
    else:
        print("❌ Invalid choice")

async def run_pipeline_tests():
    """Run pipeline tests for currently running services."""
    print("🔧 Running Pipeline Tests for Currently Running Services...")
    print("=" * 50)
    print("This will test:")
    print("  🎙️ STT → LLM (Speech to Language)")
    print("  🧠 LLM → TTS (Language to Speech)")
    print("  🎯 Full STT → LLM → TTS Pipeline")
    print("Only tests services that are currently running!")
    print()
    
    confirm = input("Continue with pipeline tests? (y/n): ").strip().lower()
    if confirm == 'y':
        tests_dir = project_root / "tests"
        subprocess.run([sys.executable, "test_running_services_pipeline.py"], cwd=tests_dir)
    else:
        print("❌ Pipeline tests cancelled")

async def run_interactive_pipeline_tests():
    """Run interactive pipeline tests with service selection."""
    print("🎯 Running Interactive Pipeline Tests...")
    print("=" * 50)
    print("This will let you:")
    print("  🎙️ Select specific STT service")
    print("  🧠 Select specific LLM service") 
    print("  🔊 Select specific TTS service")
    print("  🎯 Test individual components or full pipeline")
    print("Only shows services that are currently running!")
    print()
    
    tests_dir = project_root / "tests"
    subprocess.run([sys.executable, "test_interactive_pipeline.py"], cwd=tests_dir)

def launch_test_menu():
    """Launch the batch test menu."""
    print("🎮 Launching Batch Test Menu...")
    print("-" * 40)
    
    try:
        tests_dir = project_root / "tests"
        batch_script = tests_dir / "run_tts_llm_tests.bat"
        
        if batch_script.exists():
            subprocess.run([str(batch_script)], cwd=tests_dir, shell=True)
        else:
            print(f"❌ Batch script not found: {batch_script}")
    except Exception as e:
        print(f"❌ Error launching test menu: {e}")

def run_cli_demo():
    """Run Enterprise CLI demonstration with comprehensive validation."""
    print("🚀 Running Enterprise CLI Demo...")
    print("-" * 40)
    print("This runs comprehensive production validation of all enterprise features:")
    print("✅ Session monitoring & analytics")
    print("✅ System health & diagnostics")  
    print("✅ Security & compliance auditing")
    print("✅ Enterprise management features")
    print("✅ Performance benchmarking")
    print("✅ AWS deployment readiness")
    print()
    
    try:
        # Run the comprehensive enterprise CLI demo
        demo_script = project_root / "demos" / "cli_enterprise_demo.py"
        if demo_script.exists():
            subprocess.run([sys.executable, str(demo_script)], cwd=project_root)
        else:
            print("❌ Enterprise CLI demo script not found")
            print("Falling back to comparison demo...")
            comparison_script = project_root / "demos" / "cli_demo_comparison.py"
            if comparison_script.exists():
                subprocess.run([sys.executable, str(comparison_script)], cwd=project_root)
            else:
                print("❌ No CLI demo scripts found")
    except Exception as e:
        print(f"❌ Error running Enterprise CLI demo: {e}")

def run_sprint6_demo():
    """Run Sprint 6 demonstration."""
    print("🎯 Running Sprint 6 Demo...")
    print("-" * 40)
    
    try:
        demo_script = project_root / "demos" / "sprint6_demo.py"
        if demo_script.exists():
            subprocess.run([sys.executable, str(demo_script)], cwd=project_root)
        else:
            subprocess.run([sys.executable, "run_app.py", "demo"], cwd=project_root)
    except Exception as e:
        print(f"❌ Error running Sprint 6 demo: {e}")

def run_voice_demo():
    """Run voice conversation demo."""
    print("🎙️ Running Voice Conversation Demo...")
    print("-" * 40)
    
    demos_dir = project_root / "demos"
    demo_files = [
        "sprint5_complete_demo.py",
        "production_conversation_demo.py", 
        "voice_test.py"
    ]
    
    for demo_file in demo_files:
        demo_path = demos_dir / demo_file
        if demo_path.exists():
            try:
                subprocess.run([sys.executable, str(demo_path)], cwd=project_root)
                break
            except Exception as e:
                print(f"❌ Error running {demo_file}: {e}")
                continue
    else:
        print("❌ No voice demo scripts found")

def show_analytics():
    """Show analytics dashboard."""
    print("📊 Analytics Dashboard...")
    print("-" * 40)
    
    try:
        # Try Sprint 6 CLI
        subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint6_cli", 
            "analytics-report", "--type", "summary"
        ], cwd=project_root)
    except Exception as e:
        print(f"❌ Error showing analytics: {e}")

def show_performance_report():
    """Show performance report."""
    print("⚡ Performance Report...")
    print("-" * 40)
    
    try:
        subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint6_cli",
            "orchestrator-health"
        ], cwd=project_root)
    except Exception as e:
        print(f"❌ Error showing performance report: {e}")

def show_cache_stats():
    """Show cache statistics."""
    print("🗃️ Cache Statistics...")
    print("-" * 40)
    
    try:
        subprocess.run([
            sys.executable, "-m", "voicebot_orchestrator.sprint6_cli",
            "cache-manager", "stats"
        ], cwd=project_root)
    except Exception as e:
        print(f"❌ Error showing cache stats: {e}")

def show_docker_setup():
    """Show Docker Compose setup."""
    print("🐳 Docker Compose Setup...")
    print("-" * 40)
    print()
    print("Available Docker profiles:")
    print("  docker-compose up                        # Basic services")
    print("  docker-compose --profile monitoring up   # With Prometheus/Grafana")
    print("  docker-compose --profile loadbalancer up # With NGINX load balancer")
    print()
    print("Service URLs (when running):")
    print("  http://localhost:8000  # Main orchestrator")
    print("  http://localhost:3000  # Grafana dashboard") 
    print("  http://localhost:9090  # Prometheus metrics")
    print()

def show_kubernetes_instructions():
    """Show Kubernetes deployment instructions."""
    print("☸️ Kubernetes Deployment...")
    print("-" * 40)
    print()
    print("Deploy to Kubernetes:")
    print("  kubectl apply -f k8s/orchestrator-core.yaml")
    print("  kubectl get pods -n voicebot-orchestrator")
    print("  kubectl logs -f deployment/orchestrator-core")
    print()
    print("Scale deployment:")
    print("  kubectl scale deployment orchestrator-core --replicas=3")
    print()
    print("Access services:")
    print("  kubectl port-forward service/orchestrator-core 8000:8000")
    print()

def show_installation_guide():
    """Show installation guide."""
    print("📦 Installation Guide...")
    print("-" * 40)
    print()
    print("Option 1: Package Installation")
    print("  pip install voicebot-orchestrator")
    print("  poetry install voicebot-orchestrator")
    print()
    print("Option 2: Development Setup")
    print("  git clone <repository>")
    print("  cd voicebot-orchestrator")
    print("  pip install -r requirements.txt")
    print("  python launcher.py")
    print()
    print("Option 3: Docker")
    print("  docker-compose up")
    print()

async def main():
    """Main launcher function."""
    print_banner()
    
    # Clear any buffered input that might be contaminating stdin
    import sys
    if hasattr(sys.stdin, 'flush'):
        sys.stdin.flush()
    
    while True:
        show_main_menu()
        
        try:
            # Clear any remaining input buffer before prompting
            if sys.stdin.isatty():  # Only flush if running in a terminal
                try:
                    import msvcrt
                    while msvcrt.kbhit():
                        msvcrt.getch()
                except ImportError:
                    pass  # Not Windows or msvcrt not available
            
            choice = input("Enter your choice (0-18): ").strip()
            
            if choice == "0":
                print("\n👋 Goodbye!\n")
                break
            elif choice == "1":
                launch_enhanced_service_manager()
            elif choice == "2":
                check_service_status()
            elif choice == "3":
                await run_all_tests()
            elif choice == "4":
                await run_combination_tests()
            elif choice == "5":
                # Run independent services tests specifically
                tests_dir = project_root / "tests"
                subprocess.run([sys.executable, "test_independent_combinations.py"], cwd=tests_dir)
            elif choice == "6":
                await run_pipeline_tests()
            elif choice == "7":
                await run_interactive_pipeline_tests()
            elif choice == "8":
                # Run specific test suite placeholder
                print("🔧 Run Specific Test Suite - Feature coming soon!")
            elif choice == "9":
                launch_test_menu()
            elif choice == "10":
                run_cli_demo()
            elif choice == "11":
                run_sprint6_demo()
            elif choice == "12":
                run_voice_demo()
            elif choice == "13":
                show_analytics()
            elif choice == "14":
                show_performance_report()
            elif choice == "15":
                show_cache_stats()
            elif choice == "16":
                show_docker_setup()
            elif choice == "17":
                show_kubernetes_instructions()
            elif choice == "18":
                show_installation_guide()
            else:
                print("❌ Invalid choice. Please enter 0-18.")
            
            # Only pause for informational displays, not for actions that launch other programs
            if choice in ["2", "16", "17", "18"]:  # Status checks and info displays
                input("\nPress Enter to continue...")
                print("\n" + "="*50 + "\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!\n")
    except asyncio.CancelledError:
        print("\n👋 Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("👋 Goodbye!\n")
