"""
Sprint 5: Simple Validation Script

Basic validation of Sprint 5 functionality without external dependencies.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_semantic_cache():
    """Test semantic cache basic functionality."""
    try:
        from voicebot_orchestrator.semantic_cache import SemanticCache, get_semantic_cache_analytics
        
        print("✓ Semantic cache imports successful")
        
        # Test initialization
        cache = SemanticCache(
            model_name="test-model",
            similarity_threshold=0.3,
            max_cache_size=100
        )
        
        print("✓ Semantic cache initialization successful")
        
        # Test adding to cache
        cache.add_to_cache("test query", "test response")
        assert len(cache.cache_queries) == 1
        print("✓ Add to cache works")
        
        # Test cache stats
        stats = cache.get_cache_stats()
        assert "total_entries" in stats
        assert stats["total_entries"] == 1
        print("✓ Cache stats work")
        
        # Test analytics
        analytics = get_semantic_cache_analytics()
        assert "service_name" in analytics
        print("✓ Cache analytics work")
        
        return True
        
    except Exception as e:
        print(f"❌ Semantic cache test failed: {e}")
        traceback.print_exc()
        return False


def test_lora_adapter():
    """Test LoRA adapter basic functionality."""
    try:
        from voicebot_orchestrator.lora_adapter import LoraAdapter, LoraAdapterManager, get_lora_analytics
        
        print("✓ LoRA adapter imports successful")
        
        # Test adapter initialization
        adapter = LoraAdapter("test-adapter", "test-model", r=8)
        assert adapter.adapter_name == "test-adapter"
        print("✓ LoRA adapter initialization successful")
        
        # Test adapter creation
        success = adapter.create_adapter()
        assert success is True
        print("✓ LoRA adapter creation works")
        
        # Test manager
        manager = LoraAdapterManager()
        success = manager.create_adapter("test-mgr", "test-model")
        assert success is True
        print("✓ LoRA adapter manager works")
        
        # Test analytics
        analytics = get_lora_analytics()
        assert "service_name" in analytics
        print("✓ LoRA analytics work")
        
        return True
        
    except Exception as e:
        print(f"❌ LoRA adapter test failed: {e}")
        traceback.print_exc()
        return False


def test_sprint5_cli():
    """Test Sprint 5 CLI functionality."""
    try:
        from voicebot_orchestrator.sprint5_cli import create_sprint5_parser
        
        print("✓ Sprint 5 CLI imports successful")
        
        # Test parser creation
        parser = create_sprint5_parser()
        assert parser is not None
        print("✓ CLI parser creation works")
        
        # Test parsing cache command
        args = parser.parse_args(["cache-manager", "--stats"])
        assert args.command == "cache-manager"
        assert args.stats is True
        print("✓ CLI cache command parsing works")
        
        # Test parsing adapter command
        args = parser.parse_args(["adapter-control", "--list"])
        assert args.command == "adapter-control"
        assert args.list is True
        print("✓ CLI adapter command parsing works")
        
        # Test parsing orchestrator command
        args = parser.parse_args(["orchestrator-log", "--all"])
        assert args.command == "orchestrator-log"
        assert args.all is True
        print("✓ CLI orchestrator command parsing works")
        
        return True
        
    except Exception as e:
        print(f"❌ Sprint 5 CLI test failed: {e}")
        traceback.print_exc()
        return False


def test_integration():
    """Test Sprint 5 integration."""
    try:
        from voicebot_orchestrator.semantic_cache import SemanticCache
        from voicebot_orchestrator.lora_adapter import LoraAdapterManager
        
        print("✓ Integration imports successful")
        
        # Test cache and adapter working together
        cache = SemanticCache()
        manager = LoraAdapterManager()
        
        # Add some data
        cache.add_to_cache("integration test", "integration response")
        manager.create_adapter("integration-adapter", "test-model")
        
        # Check results
        cache_stats = cache.get_cache_stats()
        adapter_status = manager.get_adapter_status()
        
        assert cache_stats["total_entries"] == 1
        assert "integration-adapter" in adapter_status["loaded_adapters"]
        
        print("✓ Integration works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("🚀 Sprint 5 Validation Tests")
    print("=" * 40)
    
    tests = [
        ("Semantic Cache", test_semantic_cache),
        ("LoRA Adapter", test_lora_adapter),
        ("CLI Commands", test_sprint5_cli),
        ("Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}:")
        if test_func():
            passed += 1
            print(f"✅ {test_name} tests passed")
        else:
            print(f"❌ {test_name} tests failed")
    
    print(f"\n📊 Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All Sprint 5 components are working correctly!")
        return True
    else:
        print("⚠️  Some components need attention")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
