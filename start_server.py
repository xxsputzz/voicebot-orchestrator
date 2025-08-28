#!/usr/bin/env python3
"""
Startup script for Voicebot Orchestrator.
"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point for the application."""
    try:
        import uvicorn
        from voicebot_orchestrator.config import settings
        
        print("🚀 Starting Voicebot Orchestrator")
        print(f"   Server: {settings.host}:{settings.port}")
        print(f"   Log Level: {settings.log_level}")
        print("   Press Ctrl+C to stop")
        print("-" * 40)
        
        uvicorn.run(
            "voicebot_orchestrator.main:app",
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
            reload=False,
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down Voicebot Orchestrator")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
