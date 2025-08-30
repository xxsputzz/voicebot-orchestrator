#!/usr/bin/env python3
"""
CLI Demo Comparison Script
Shows the difference between legacy and enterprise CLI systems
"""

import subprocess
import sys
import time
from datetime import datetime

def print_header(title, color="blue"):
    colors = {
        "blue": "\033[94m",
        "green": "\033[92m", 
        "yellow": "\033[93m",
        "red": "\033[91m",
        "reset": "\033[0m"
    }
    
    # Remove colors for title headers and equals symbols - use plain text
    if title in ["CLI SYSTEMS DEMONSTRATION", "ENTERPRISE CLI DEMO (RECOMMENDED)", "CLI COMPARISON SUMMARY", "DOCUMENTATION LINKS"]:
        print(f"\n{'='*60}")
        print(f"🎯 {title}")
        print(f"{'='*60}")
    else:
        # Keep colors for other headers
        print(f"\n{colors.get(color, colors['blue'])}{'='*60}")
        print(f"🎯 {title}")
        print(f"{'='*60}{colors['reset']}")

def run_demo():
    print_header("CLI SYSTEMS DEMONSTRATION")
    print(f"🕐 Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📋 AVAILABLE CLI SYSTEMS:")
    print("1. 🚀 Enterprise CLI (sprint6_cli.py) - PRODUCTION READY")
    print("2. 🎭 Modular Voice CLI (modular_cli.py) - DEVELOPMENT")  
    print("3. 🎵 Enhanced TTS CLI (enhanced_cli.py) - SPECIALIZED")
    
    print_header("ENTERPRISE CLI DEMO (RECOMMENDED)")
    print("This is our production-ready CLI with enterprise features...")
    print("\n🚀 Running: python demos/cli_enterprise_demo.py")
    print("⏳ This will show comprehensive validation with checkmarks...")
    
    choice = input("\n❓ Run Enterprise CLI Demo? (y/n): ").lower().strip()
    
    if choice == 'y':
        try:
            print("\n🚀 Starting Enterprise CLI Demo...")
            subprocess.run([
                sys.executable, 
                "demos/cli_enterprise_demo.py"
            ], cwd="C:\\Users\\miken\\Desktop\\Orkestra", check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Demo failed with error: {e}")
        except FileNotFoundError:
            print("❌ Demo script not found. Please ensure you're in the correct directory.")
    
    print_header("ALTERNATIVE DEMOS", "yellow")
    print("These are specialized CLIs for specific use cases:")
    
    print("\n🎭 Modular Voice CLI:")
    print("   Command: .venv\\Scripts\\python.exe voicebot_orchestrator\\modular_cli.py")
    print("   Purpose: Interactive voice conversation testing")
    
    print("\n🎵 Enhanced TTS CLI:")
    print("   Command: .venv\\Scripts\\python.exe voicebot_orchestrator\\enhanced_cli.py demo")  
    print("   Purpose: TTS engine comparison and quality testing")
    
    print_header("CLI COMPARISON SUMMARY")
    print("📊 FEATURE COMPARISON:")
    print("┌─────────────────────┬─────────────┬─────────────────┐")
    print("│ Feature             │ Legacy CLIs │ Enterprise CLI  │")
    print("├─────────────────────┼─────────────┼─────────────────┤")
    print("│ Commands            │ 6 basic     │ 15+ enterprise  │")
    print("│ Validation          │ None        │ ✅ Checkmarks   │")
    print("│ Production Ready    │ No          │ ✅ Yes          │")
    print("│ AWS Deployment      │ No          │ ✅ Yes          │")
    print("│ Security Audit      │ None        │ ✅ Built-in     │")
    print("│ Performance Testing │ None        │ ✅ Load Testing │")
    print("│ Enterprise Features │ None        │ ✅ Complete     │")
    print("└─────────────────────┴─────────────┴─────────────────┘")
    
    print("\n✨ RECOMMENDATION:")
    print("🚀 Use Enterprise CLI Demo (Option 1) for production validation")
    print("🎭 Use Modular CLI for voice conversation development")
    print("🎵 Use Enhanced TTS CLI for speech synthesis research")
    
    print_header("DOCUMENTATION LINKS")
    print("📚 Complete guides available:")
    print("   • docs/CLI_DEMO_GUIDE.md - Enterprise CLI command reference")
    print("   • docs/ENTERPRISE_CLI_FEATURES.md - Production capabilities")
    print("   • docs/CLI_SYSTEMS_OVERVIEW.md - Complete architecture overview")
    
    print(f"\n🕐 Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎉 Thank you for exploring our CLI systems!")

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        print("\n👋 Goodbye!")
