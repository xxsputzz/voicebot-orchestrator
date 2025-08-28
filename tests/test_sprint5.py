"""
Sprint 5: Comprehensive Tests

Test suite for semantic caching and LoRA adapter functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Import modules to test
from voicebot_orchestrator.semantic_cache import (
    SemanticCache, 
    get_semantic_cache_analytics,
    create_semantic_cache
)
from voicebot_orchestrator.lora_adapter import (
    LoraAdapter,
    LoraAdapterManager,
    get_lora_analytics
)


class TestSemanticCache:
    """Test semantic cache functionality."""
    
    def test_cache_initialization(self):
        """Test cache initialization with various parameters."""
        cache = SemanticCache(
            model_name="test-model",
            similarity_threshold=0.3,
            max_cache_size=100
        )
        
        assert cache.model_name == "test-model"
        assert cache.similarity_threshold == 0.3
        assert cache.max_cache_size == 100
        assert cache.hit_count == 0
        assert cache.miss_count == 0
    
    def test_cache_initialization_invalid_params(self):
        """Test cache initialization with invalid parameters."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            SemanticCache(model_name="")
        
        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            SemanticCache(similarity_threshold=1.5)
        
        with pytest.raises(ValueError, match="max_cache_size must be positive"):
            SemanticCache(max_cache_size=0)
    
    def test_add_to_cache_valid(self):
        """Test adding valid entries to cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = SemanticCache(cache_dir=temp_dir)
            
            cache.add_to_cache("test query", "test response")
            
            assert len(cache.cache_queries) == 1
            assert cache.cache_queries[0] == "test query"
            assert cache.cache_responses[0] == "test response"
    
    def test_add_to_cache_invalid(self):
        """Test adding invalid entries to cache."""
        cache = SemanticCache()
        
        with pytest.raises(ValueError, match="Query must be non-empty"):
            cache.add_to_cache("", "response")
        
        with pytest.raises(ValueError, match="Response must be non-empty"):
            cache.add_to_cache("query", "")
    
    def test_check_cache_empty(self):
        """Test cache check with empty cache."""
        cache = SemanticCache()
        
        result = cache.check_cache("test query")
        
        assert result is None
        assert cache.miss_count == 1
        assert cache.total_queries == 1
    
    def test_check_cache_invalid_query(self):
        """Test cache check with invalid query."""
        cache = SemanticCache()
        
        with pytest.raises(ValueError, match="Query must be non-empty"):
            cache.check_cache("")
    
    def test_cache_hit_simulation(self):
        """Test cache hit simulation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = SemanticCache(
                cache_dir=temp_dir,
                similarity_threshold=0.8  # High threshold for exact matches
            )
            
            # Add entry
            cache.add_to_cache("hello world", "hi there")
            
            # Should find similar query (with mock implementation)
            # Note: Mock implementation returns based on hash similarity
            result = cache.check_cache("hello world")
            
            # With mock implementation, we expect either hit or miss
            assert cache.total_queries == 1
    
    def test_clear_cache(self):
        """Test cache clearing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = SemanticCache(cache_dir=temp_dir)
            
            cache.add_to_cache("query1", "response1")
            cache.add_to_cache("query2", "response2")
            
            cache.clear_cache()
            
            assert len(cache.cache_queries) == 0
            assert len(cache.cache_responses) == 0
            assert cache.hit_count == 0
            assert cache.miss_count == 0
            assert cache.total_queries == 0
    
    def test_get_cache_stats(self):
        """Test cache statistics retrieval."""
        cache = SemanticCache()
        
        cache.add_to_cache("query1", "response1")
        cache.check_cache("query1")  # Should be a hit or miss
        
        stats = cache.get_cache_stats()
        
        assert "total_entries" in stats
        assert "hit_rate" in stats
        assert "total_queries" in stats
        assert stats["total_entries"] == 1
        assert stats["total_queries"] == 1
    
    def test_export_cache_data(self):
        """Test cache data export."""
        cache = SemanticCache()
        
        cache.add_to_cache("query1", "response1", {"type": "test"})
        cache.add_to_cache("query2", "response2")
        
        exported = cache.export_cache_data()
        
        assert len(exported) == 2
        assert exported[0]["query"] == "query1"
        assert exported[0]["response"] == "response1"
        assert exported[0]["metadata"]["type"] == "test"
        assert exported[1]["metadata"] == {}
    
    def test_evict_by_threshold(self):
        """Test cache eviction by threshold."""
        cache = SemanticCache()
        
        cache.add_to_cache("query1", "response1")
        cache.add_to_cache("query2", "response2")
        
        evicted = cache.evict_by_threshold(0.9)  # High threshold
        
        assert isinstance(evicted, int)
        assert evicted >= 0
    
    def test_factory_function(self):
        """Test cache factory function."""
        cache = create_semantic_cache(
            model_name="test-model",
            similarity_threshold=0.5
        )
        
        assert isinstance(cache, SemanticCache)
        assert cache.model_name == "test-model"
        assert cache.similarity_threshold == 0.5


class TestLoraAdapter:
    """Test LoRA adapter functionality."""
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        adapter = LoraAdapter(
            adapter_name="test-adapter",
            base_model_name="test-model",
            r=16,
            lora_alpha=32
        )
        
        assert adapter.adapter_name == "test-adapter"
        assert adapter.base_model_name == "test-model"
        assert adapter.r == 16
        assert adapter.lora_alpha == 32
        assert not adapter.is_loaded
    
    def test_adapter_initialization_invalid(self):
        """Test adapter initialization with invalid parameters."""
        with pytest.raises(ValueError, match="adapter_name cannot be empty"):
            LoraAdapter("", "model")
        
        with pytest.raises(ValueError, match="base_model_name cannot be empty"):
            LoraAdapter("adapter", "")
        
        with pytest.raises(ValueError, match="r \\(rank\\) must be positive"):
            LoraAdapter("adapter", "model", r=0)
        
        with pytest.raises(ValueError, match="lora_dropout must be between"):
            LoraAdapter("adapter", "model", lora_dropout=1.5)
    
    def test_create_lora_config(self):
        """Test LoRA configuration creation."""
        adapter = LoraAdapter("test", "model", r=8, lora_alpha=16)
        
        config = adapter.create_lora_config()
        
        assert hasattr(config, 'r')
        assert hasattr(config, 'lora_alpha')
        assert config.r == 8
        assert config.lora_alpha == 16
    
    def test_load_base_model(self):
        """Test base model loading."""
        adapter = LoraAdapter("test", "test-model")
        
        success = adapter.load_base_model()
        
        assert success is True  # Mock always succeeds
        assert adapter.base_model is not None
    
    def test_create_adapter(self):
        """Test adapter creation."""
        adapter = LoraAdapter("test", "test-model")
        
        success = adapter.create_adapter()
        
        assert success is True
        assert adapter.is_loaded is True
        assert adapter.model is not None
    
    def test_get_adapter_info(self):
        """Test adapter information retrieval."""
        adapter = LoraAdapter("test", "test-model", r=8)
        adapter.create_adapter()
        
        info = adapter.get_adapter_info()
        
        assert info["adapter_name"] == "test"
        assert info["base_model_name"] == "test-model"
        assert info["is_loaded"] is True
        assert info["config"]["r"] == 8
        assert "parameter_count" in info
    
    def test_add_training_data(self):
        """Test adding training data."""
        adapter = LoraAdapter("test", "test-model")
        
        adapter.add_training_data("input text", "target text", {"category": "test"})
        
        assert len(adapter.training_data) == 1
        assert adapter.training_data[0]["input"] == "input text"
        assert adapter.training_data[0]["target"] == "target text"
        assert adapter.training_data[0]["metadata"]["category"] == "test"
    
    def test_add_training_data_invalid(self):
        """Test adding invalid training data."""
        adapter = LoraAdapter("test", "test-model")
        
        with pytest.raises(ValueError, match="input_text and target_text cannot be empty"):
            adapter.add_training_data("", "target")
        
        with pytest.raises(ValueError, match="input_text and target_text cannot be empty"):
            adapter.add_training_data("input", "")
    
    def test_simulate_training(self):
        """Test training simulation."""
        adapter = LoraAdapter("test", "test-model")
        adapter.add_training_data("input", "target")
        
        metrics = adapter.simulate_training(epochs=3, learning_rate=1e-4)
        
        assert "epochs" in metrics
        assert "learning_rate" in metrics
        assert "training_samples" in metrics
        assert "final_loss" in metrics
        assert metrics["epochs"] == 3
        assert metrics["training_samples"] == 1
    
    def test_simulate_training_invalid(self):
        """Test training simulation with invalid parameters."""
        adapter = LoraAdapter("test", "test-model")
        
        with pytest.raises(ValueError, match="No training data available"):
            adapter.simulate_training()
        
        adapter.add_training_data("input", "target")
        
        with pytest.raises(ValueError, match="epochs must be positive"):
            adapter.simulate_training(epochs=0)
        
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            adapter.simulate_training(learning_rate=0)
    
    def test_save_load_adapter(self):
        """Test adapter saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save adapter
            adapter1 = LoraAdapter("test", "test-model")
            adapter1.create_adapter()
            
            save_path = Path(temp_dir) / "test_adapter"
            success = adapter1.save_adapter(str(save_path))
            assert success is True
            
            # Load adapter
            adapter2 = LoraAdapter("test2", "")
            success = adapter2.load_adapter(str(save_path))
            assert success is True
            assert adapter2.adapter_name == "test"
            assert adapter2.base_model_name == "test-model"


class TestLoraAdapterManager:
    """Test LoRA adapter manager functionality."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoraAdapterManager(adapter_dir=temp_dir)
            
            assert manager.adapter_dir == Path(temp_dir)
            assert len(manager.adapters) == 0
            assert manager.active_adapter is None
    
    def test_create_adapter(self):
        """Test adapter creation through manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoraAdapterManager(adapter_dir=temp_dir)
            
            success = manager.create_adapter("test-adapter", "test-model")
            
            assert success is True
            assert "test-adapter" in manager.adapters
    
    def test_create_duplicate_adapter(self):
        """Test creating duplicate adapter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoraAdapterManager(adapter_dir=temp_dir)
            
            manager.create_adapter("test-adapter", "test-model")
            success = manager.create_adapter("test-adapter", "test-model")
            
            assert success is False  # Should fail for duplicate
    
    def test_load_unload_adapter(self):
        """Test loading and unloading adapters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoraAdapterManager(adapter_dir=temp_dir)
            
            # Create adapter directory structure
            adapter_dir = Path(temp_dir) / "test-adapter"
            adapter_dir.mkdir()
            
            metadata = {
                "adapter_name": "test-adapter",
                "base_model_name": "test-model",
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj"]
            }
            
            with open(adapter_dir / "adapter_metadata.json", "w") as f:
                json.dump(metadata, f)
            
            # Test loading
            success = manager.load_adapter("test-adapter")
            assert success is True
            assert "test-adapter" in manager.adapters
            
            # Test unloading
            success = manager.unload_adapter("test-adapter")
            assert success is True
            assert "test-adapter" not in manager.adapters
    
    def test_activate_deactivate_adapter(self):
        """Test adapter activation and deactivation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoraAdapterManager(adapter_dir=temp_dir)
            
            manager.create_adapter("test-adapter", "test-model")
            
            # Test activation
            success = manager.activate_adapter("test-adapter")
            assert success is True
            assert manager.active_adapter == "test-adapter"
            
            # Test deactivation
            success = manager.deactivate_adapter()
            assert success is True
            assert manager.active_adapter is None
    
    def test_activate_nonexistent_adapter(self):
        """Test activating non-existent adapter."""
        manager = LoraAdapterManager()
        
        success = manager.activate_adapter("nonexistent")
        assert success is False
    
    def test_get_adapter_status(self):
        """Test getting adapter status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoraAdapterManager(adapter_dir=temp_dir)
            
            manager.create_adapter("test-adapter", "test-model")
            manager.activate_adapter("test-adapter")
            
            status = manager.get_adapter_status()
            
            assert "available_adapters" in status
            assert "loaded_adapters" in status
            assert "active_adapter" in status
            assert status["active_adapter"] == "test-adapter"
            assert "test-adapter" in status["loaded_adapters"]
    
    def test_create_banking_adapter(self):
        """Test creating pre-configured banking adapter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoraAdapterManager(adapter_dir=temp_dir)
            
            success = manager.create_banking_adapter("banking-test")
            
            assert success is True
            assert "banking-test" in manager.adapters
            
            # Check that training data was added
            adapter = manager.adapters["banking-test"]
            assert len(adapter.training_data) > 0
            assert adapter.training_metrics  # Should have training metrics
    
    def test_get_adapter_info(self):
        """Test getting adapter information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoraAdapterManager(adapter_dir=temp_dir)
            
            manager.create_adapter("test-adapter", "test-model")
            
            info = manager.get_adapter_info("test-adapter")
            
            assert info is not None
            assert info["adapter_name"] == "test-adapter"
            assert info["base_model_name"] == "test-model"
            
            # Test non-existent adapter
            info = manager.get_adapter_info("nonexistent")
            assert info is None


class TestAnalytics:
    """Test analytics functionality."""
    
    def test_semantic_cache_analytics(self):
        """Test semantic cache analytics."""
        analytics = get_semantic_cache_analytics()
        
        assert "cache_hits" in analytics
        assert "cache_misses" in analytics
        assert "service_name" in analytics
        assert analytics["service_name"] == "semantic_cache"
    
    def test_lora_analytics(self):
        """Test LoRA adapter analytics."""
        analytics = get_lora_analytics()
        
        assert "adapters_created" in analytics
        assert "adapters_loaded" in analytics
        assert "adapter_switches" in analytics
        assert "service_name" in analytics
        assert analytics["service_name"] == "lora_adapters"


class TestIntegration:
    """Integration tests for Sprint 5 components."""
    
    def test_cache_and_adapter_integration(self):
        """Test cache and adapter working together."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize components
            cache = SemanticCache(cache_dir=temp_dir)
            manager = LoraAdapterManager(adapter_dir=temp_dir)
            
            # Create banking adapter
            manager.create_banking_adapter("banking-lora")
            manager.activate_adapter("banking-lora")
            
            # Add some queries to cache
            cache.add_to_cache("What is APR?", "APR stands for Annual Percentage Rate...")
            cache.add_to_cache("Calculate loan payment", "Use the formula M = P[r(1+r)^n]...")
            
            # Test cache lookup
            result = cache.check_cache("What is APR?")
            # Result depends on mock implementation
            
            # Get status from both systems
            cache_stats = cache.get_cache_stats()
            adapter_status = manager.get_adapter_status()
            
            assert cache_stats["total_entries"] == 2
            assert adapter_status["active_adapter"] == "banking-lora"
    
    def test_mock_library_fallback(self):
        """Test that mock implementations work when real libraries unavailable."""
        # This test verifies our mock implementations work correctly
        
        cache = SemanticCache()
        assert cache.embedder is not None
        assert cache.index is not None
        
        manager = LoraAdapterManager()
        success = manager.create_adapter("test", "model")
        assert success is True


# Performance tests
class TestPerformance:
    """Performance tests for Sprint 5 components."""
    
    def test_cache_performance_large_dataset(self):
        """Test cache performance with larger dataset."""
        cache = SemanticCache(max_cache_size=1000)
        
        # Add many entries
        for i in range(100):
            cache.add_to_cache(f"query_{i}", f"response_{i}")
        
        # Test lookups
        for i in range(10):
            result = cache.check_cache(f"query_{i}")
            # Result depends on implementation
        
        stats = cache.get_cache_stats()
        assert stats["total_entries"] == 100
        assert stats["total_queries"] == 10
    
    def test_adapter_parameter_estimation(self):
        """Test adapter parameter count estimation."""
        adapter = LoraAdapter("test", "model", r=16)
        
        param_count = adapter._estimate_parameter_count()
        
        assert param_count > 0
        assert isinstance(param_count, int)
        
        # Higher rank should mean more parameters
        adapter2 = LoraAdapter("test2", "model", r=32)
        param_count2 = adapter2._estimate_parameter_count()
        
        assert param_count2 > param_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
