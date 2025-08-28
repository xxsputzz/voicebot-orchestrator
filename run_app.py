#!/usr/bin/env python3
"""
Voicebot Orchestrator Application Runner

This script provides multiple ways to run the voicebot orchestration platform:
1. CLI commands for individual operations
2. Development mode with mock services
3. Production deployment instructions
"""

import sys
import os
import subprocess
from pathlib import Path

def print_banner():
    """Print application banner."""
    print("=" * 80)
    print("🤖 VOICEBOT ORCHESTRATION PLATFORM")
    print("   Enterprise-Grade Banking Voice AI")
    print("   Version: 1.0.0 | Sprint 6 Complete")
    print("=" * 80)
    print()

def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import requests
    except ImportError:
        missing.append("requests")
    
    return missing

def install_dependencies():
    """Install missing dependencies."""
    print("📦 Installing required dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_cli_demo():
    """Run CLI demonstration."""
    print("🚀 RUNNING CLI DEMONSTRATION")
    print("-" * 40)
    
    commands = [
        ("System Health Check", "python -m voicebot_orchestrator.sprint6_cli orchestrator-health"),
        ("Cache Statistics", "python -m voicebot_orchestrator.sprint6_cli cache-manager stats"),
        ("Start Demo Session", "python -m voicebot_orchestrator.sprint6_cli start-call demo-session --phone +1234567890 --domain banking"),
        ("Monitor Session", "python -m voicebot_orchestrator.sprint6_cli monitor-session --session-id demo-session"),
        ("Analytics Report", "python -m voicebot_orchestrator.sprint6_cli analytics-report --type summary --time-range 24h"),
        ("Adapter Control", "python -m voicebot_orchestrator.sprint6_cli adapter-control list"),
    ]
    
    for name, cmd in commands:
        print(f"\n🔧 {name}")
        print(f"Command: {cmd}")
        print("-" * 40)
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("✅ Success!")
                if result.stdout:
                    # Show first few lines of output
                    lines = result.stdout.strip().split('\n')
                    for line in lines[:10]:  # Show first 10 lines
                        if not line.startswith("Warning:"):
                            print(f"   {line}")
                    if len(lines) > 10:
                        print(f"   ... ({len(lines)-10} more lines)")
            else:
                print(f"❌ Failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("⏰ Command timed out")
        except Exception as e:
            print(f"❌ Error: {e}")

def run_comprehensive_demo():
    """Run the comprehensive Sprint 6 demo."""
    print("🎯 RUNNING COMPREHENSIVE SPRINT 6 DEMO")
    print("-" * 40)
    
    try:
        result = subprocess.run([sys.executable, "sprint6_demo.py"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Demo completed successfully!")
            print("\nDemo Output:")
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if not line.startswith("Warning:"):
                    print(f"   {line}")
        else:
            print(f"❌ Demo failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⏰ Demo timed out")
    except Exception as e:
        print(f"❌ Error running demo: {e}")

def show_deployment_options():
    """Show deployment options."""
    print("🚢 DEPLOYMENT OPTIONS")
    print("-" * 40)
    print()
    
    print("1. 📦 Package Installation:")
    print("   pip install voicebot-orchestrator")
    print("   poetry install voicebot-orchestrator")
    print()
    
    print("2. 🐳 Docker Compose (Recommended):")
    print("   docker-compose up                     # Basic services")
    print("   docker-compose --profile monitoring up  # With monitoring")
    print("   docker-compose --profile loadbalancer up  # With load balancer")
    print()
    
    print("3. ☸️  Kubernetes:")
    print("   kubectl apply -f k8s/orchestrator-core.yaml")
    print("   kubectl get pods -n voicebot-orchestrator")
    print("   kubectl logs -f deployment/orchestrator-core")
    print()
    
    print("4. 🔧 Individual Microservices:")
    print("   python -m voicebot_orchestrator.microservices.orchestrator_core")
    print("   python -m voicebot_orchestrator.microservices.stt_service")
    print("   python -m voicebot_orchestrator.microservices.llm_service")
    print("   python -m voicebot_orchestrator.microservices.tts_service")
    print("   python -m voicebot_orchestrator.microservices.cache_service")
    print("   python -m voicebot_orchestrator.microservices.analytics_service")
    print()
    
    print("5. 🎮 CLI Commands:")
    print("   orchestrator orchestrator-health        # System status")
    print("   orchestrator start-call <session> --phone <phone> --domain banking")
    print("   orchestrator monitor-session --session-id <session>")
    print("   orchestrator analytics-report --type summary")
    print("   orchestrator cache-manager stats")
    print("   orchestrator adapter-control list")

def main():
    """Main application runner."""
    print_banner()
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("Available modes:")
        print("  python run_app.py cli          # Run CLI demonstration")
        print("  python run_app.py demo         # Run comprehensive demo")
        print("  python run_app.py deploy       # Show deployment options")
        print("  python run_app.py install      # Install dependencies")
        print()
        mode = input("Select mode (cli/demo/deploy/install): ").lower().strip()
    
    if mode == "install":
        missing = check_dependencies()
        if missing:
            print(f"Missing dependencies: {', '.join(missing)}")
            if install_dependencies():
                print("✅ Ready to run!")
            else:
                print("❌ Installation failed. Please install manually:")
                print("   pip install pandas numpy requests fastapi uvicorn websockets")
        else:
            print("✅ All dependencies are already installed!")
    
    elif mode == "cli":
        missing = check_dependencies()
        if missing:
            print(f"⚠️  Warning: Missing dependencies: {', '.join(missing)}")
            print("   Running with mock implementations...")
        run_cli_demo()
    
    elif mode == "demo":
        missing = check_dependencies()
        if missing:
            print(f"⚠️  Warning: Missing dependencies: {', '.join(missing)}")
            print("   Running with mock implementations...")
        run_comprehensive_demo()
    
    elif mode == "deploy":
        show_deployment_options()
    
    else:
        print("❌ Invalid mode. Use: cli, demo, deploy, or install")
    
    print()
    print("🎉 Voicebot Orchestrator - Sprint 6 Complete!")
    print("   For more information, see SPRINT6_SUMMARY.md")

if __name__ == "__main__":
    main()
