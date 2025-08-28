"""
Test runner for all voicebot orchestrator tests.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test modules
from test_session_manager import run_tests as run_session_tests
from test_stt import run_tests as run_stt_tests
from test_llm import run_tests as run_llm_tests
from test_tts import run_tests as run_tts_tests
from test_integration import run_tests as run_integration_tests

# Import Sprint 5 validation
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from sprint5_validation import main as run_sprint5_validation
    SPRINT5_AVAILABLE = True
except ImportError:
    SPRINT5_AVAILABLE = False


async def run_all_tests():
    """Run all test suites."""
    print("🧪 Running Voicebot Orchestrator Test Suite")
    print("=" * 50)
    
    test_suites = [
        ("Session Manager", run_session_tests),
        ("Speech-to-Text (STT)", run_stt_tests),
        ("Large Language Model (LLM)", run_llm_tests),
        ("Text-to-Speech (TTS)", run_tts_tests),
        ("Integration", run_integration_tests),
    ]
    
    # Add Sprint 5 tests if available
    if SPRINT5_AVAILABLE:
        # Wrap synchronous function to work with async test runner
        async def run_sprint5_async():
            return run_sprint5_validation()
        test_suites.append(("Sprint 5 (Semantic Cache & LoRA)", run_sprint5_async))
    
    total_passed = 0
    total_failed = 0
    suite_results = []
    
    for suite_name, test_runner in test_suites:
        print(f"\n🔄 Running {suite_name} Tests...")
        print("-" * 30)
        
        try:
            success = await test_runner()
            suite_results.append((suite_name, success))
            
            if success:
                print(f"✅ {suite_name} tests passed")
            else:
                print(f"❌ {suite_name} tests failed")
                
        except Exception as e:
            print(f"💥 {suite_name} test suite crashed: {e}")
            suite_results.append((suite_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Suite Summary")
    print("=" * 50)
    
    passed_suites = 0
    failed_suites = 0
    
    for suite_name, success in suite_results:
        if success:
            print(f"✅ {suite_name}")
            passed_suites += 1
        else:
            print(f"❌ {suite_name}")
            failed_suites += 1
    
    print(f"\nTotal: {passed_suites} passed, {failed_suites} failed")
    
    if failed_suites == 0:
        print("🎉 All test suites passed!")
        return True
    else:
        print("⚠️  Some test suites failed")
        return False


def run_specific_tests():
    """Run specific test suites based on command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run voicebot orchestrator tests")
    parser.add_argument(
        "suites",
        nargs="*",
        choices=["session", "stt", "llm", "tts", "integration", "sprint5", "all"],
        default=["all"],
        help="Test suites to run (default: all)"
    )
    
    args = parser.parse_args()
    
    if "all" in args.suites:
        return asyncio.run(run_all_tests())
    
    # Run specific suites
    suite_map = {
        "session": ("Session Manager", run_session_tests),
        "stt": ("Speech-to-Text (STT)", run_stt_tests),
        "llm": ("Large Language Model (LLM)", run_llm_tests),
        "tts": ("Text-to-Speech (TTS)", run_tts_tests),
        "integration": ("Integration", run_integration_tests),
    }
    
    # Add Sprint 5 if available
    if SPRINT5_AVAILABLE:
        async def run_sprint5_async():
            return run_sprint5_validation()
        suite_map["sprint5"] = ("Sprint 5 (Semantic Cache & LoRA)", run_sprint5_async)
    
    async def run_selected():
        all_passed = True
        
        for suite_key in args.suites:
            if suite_key in suite_map:
                suite_name, test_runner = suite_map[suite_key]
                print(f"\n🔄 Running {suite_name} Tests...")
                print("-" * 30)
                
                try:
                    success = await test_runner()
                    if not success:
                        all_passed = False
                except Exception as e:
                    print(f"💥 {suite_name} test suite crashed: {e}")
                    all_passed = False
        
        return all_passed
    
    return asyncio.run(run_selected())


if __name__ == "__main__":
    success = run_specific_tests()
    sys.exit(0 if success else 1)
