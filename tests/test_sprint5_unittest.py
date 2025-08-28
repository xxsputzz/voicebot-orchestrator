"""
Sprint 5 Tests - unittest compatible version
"""

import unittest
import asyncio
import tempfile
import os
import shutil
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from voicebot_orchestrator.semantic_cache import SemanticCache
from voicebot_orchestrator.lora_adapter import LoraAdapter, LoraAdapterManager
from voicebot_orchestrator.enhanced_llm import EnhancedMistralLLM, create_enhanced_llm


class TestSemanticCache(unittest.TestCase):
    """Test semantic cache functionality."""
    
    def setUp(self):
        """Set up test cache."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = SemanticCache(
            model_name="all-MiniLM-L6-v2",
            cache_dir=self.temp_dir,
            similarity_threshold=0.8,
            max_cache_size=100
        )
    
    def tearDown(self):
        """Clean up test cache."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        self.assertEqual(self.cache.model_name, "all-MiniLM-L6-v2")
        self.assertEqual(self.cache.similarity_threshold, 0.8)
        self.assertEqual(self.cache.max_cache_size, 100)
        self.assertIsNotNone(self.cache.model)
        self.assertIsNotNone(self.cache.index)
    
    def test_add_to_cache(self):
        """Test adding entries to cache."""
        query = "What is my balance?"
        response = "Your balance is $1000"
        
        self.cache.add_to_cache(query, response)
        
        # Verify cache has entry
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats["total_entries"], 1)
    
    def test_check_cache_miss(self):
        """Test cache miss."""
        result = self.cache.check_cache("Non-existent query")
        self.assertIsNone(result)
        
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats["miss_count"], 1)
    
    def test_check_cache_hit(self):
        """Test cache hit."""
        query = "What is my balance?"
        response = "Your balance is $1000"
        
        # Add to cache
        self.cache.add_to_cache(query, response)
        
        # Check cache - should be a hit for identical query
        result = self.cache.check_cache(query)
        self.assertEqual(result, response)
        
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats["hit_count"], 1)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        self.cache.add_to_cache("test", "response")
        
        # Verify cache has entries
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats["total_entries"], 1)
        
        # Clear cache
        self.cache.clear_cache()
        
        # Verify cache is empty
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats["total_entries"], 0)


class TestLoraAdapter(unittest.TestCase):
    """Test LoRA adapter functionality."""
    
    def setUp(self):
        """Set up test adapter."""
        self.temp_dir = tempfile.mkdtemp()
        self.adapter = LoraAdapter(
            name="test-adapter",
            target_modules=["query", "value"],
            r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        self.assertEqual(self.adapter.name, "test-adapter")
        self.assertEqual(self.adapter.target_modules, ["query", "value"])
        self.assertEqual(self.adapter.r, 16)
        self.assertEqual(self.adapter.lora_alpha, 32)
        self.assertEqual(self.adapter.lora_dropout, 0.1)
        self.assertFalse(self.adapter.is_trained)
    
    def test_simulate_training(self):
        """Test adapter training simulation."""
        initial_state = self.adapter.is_trained
        self.assertFalse(initial_state)
        
        # Simulate training
        self.adapter.simulate_training(epochs=1, learning_rate=1e-4)
        
        self.assertTrue(self.adapter.is_trained)
        self.assertIsNotNone(self.adapter.training_metrics)
    
    def test_save_load_adapter(self):
        """Test adapter save/load."""
        adapter_path = os.path.join(self.temp_dir, "test_adapter")
        
        # Train and save
        self.adapter.simulate_training(epochs=1)
        success = self.adapter.save(adapter_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(adapter_path))
        
        # Load adapter
        loaded_success = self.adapter.load(adapter_path)
        self.assertTrue(loaded_success)


class TestLoraAdapterManager(unittest.TestCase):
    """Test LoRA adapter manager."""
    
    def setUp(self):
        """Set up test manager."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = LoraAdapterManager(adapter_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        self.assertEqual(self.manager.adapter_dir, self.temp_dir)
        self.assertEqual(len(self.manager.adapters), 0)
        self.assertIsNone(self.manager.active_adapter)
    
    def test_create_adapter(self):
        """Test adapter creation."""
        success = self.manager.create_adapter(
            name="test-adapter",
            target_modules=["query", "value"]
        )
        self.assertTrue(success)
        self.assertIn("test-adapter", self.manager.adapters)
    
    def test_create_banking_adapter(self):
        """Test banking adapter creation."""
        success = self.manager.create_banking_adapter("banking-test")
        self.assertTrue(success)
        self.assertIn("banking-test", self.manager.adapters)
        
        # Verify banking-specific configuration
        adapter = self.manager.adapters["banking-test"]
        self.assertEqual(adapter.domain, "banking")
    
    def test_load_activate_adapter(self):
        """Test adapter loading and activation."""
        # Create adapter first
        self.manager.create_adapter("test-adapter")
        
        # Load adapter
        success = self.manager.load_adapter("test-adapter")
        self.assertTrue(success)
        
        # Activate adapter
        success = self.manager.activate_adapter("test-adapter")
        self.assertTrue(success)
        self.assertEqual(self.manager.active_adapter, "test-adapter")
    
    def test_get_adapter_status(self):
        """Test adapter status retrieval."""
        # Create and load adapter
        self.manager.create_adapter("test-adapter")
        self.manager.load_adapter("test-adapter")
        self.manager.activate_adapter("test-adapter")
        
        status = self.manager.get_adapter_status()
        
        self.assertIn("available_adapters", status)
        self.assertIn("loaded_adapters", status)
        self.assertIn("active_adapter", status)
        self.assertEqual(status["active_adapter"], "test-adapter")


class TestEnhancedLLM(unittest.TestCase):
    """Test enhanced LLM functionality."""
    
    def setUp(self):
        """Set up test LLM."""
        self.temp_dir = tempfile.mkdtemp()
        self.llm = EnhancedMistralLLM(
            model_path="test-model",
            enable_cache=True,
            enable_adapters=True,
            cache_dir=self.temp_dir,
            adapter_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_llm_initialization(self):
        """Test LLM initialization."""
        self.assertEqual(self.llm.model_path, "test-model")
        self.assertTrue(self.llm.enable_cache)
        self.assertTrue(self.llm.enable_adapters)
        self.assertIsNotNone(self.llm.cache)
        self.assertIsNotNone(self.llm.adapter_manager)
    
    def test_generate_response_sync(self):
        """Test response generation (sync version for unittest)."""
        async def async_test():
            response = await self.llm.generate_response("Hello")
            self.assertIsNotNone(response)
            self.assertGreater(len(response), 0)
            self.assertEqual(self.llm.llm_calls, 1)
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_empty_input_validation(self):
        """Test empty input validation."""
        async def async_test():
            with self.assertRaises(ValueError):
                await self.llm.generate_response("")
            
            with self.assertRaises(ValueError):
                await self.llm.generate_response("   ")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()
    
    def test_banking_domain_response(self):
        """Test banking domain response."""
        response = self.llm._generate_banking_response("What is my balance?")
        self.assertIn("Enhanced by banking domain adapter", response)
        self.assertIn("balance", response.lower())
    
    def test_compliance_domain_response(self):
        """Test compliance domain response."""
        response = self.llm._generate_compliance_response("Are calls recorded?")
        self.assertIn("Enhanced by compliance adapter", response)
        self.assertIn("recorded", response.lower())
    
    def test_adapter_selection(self):
        """Test adapter selection logic."""
        # Banking domain
        adapter = self.llm._select_adapter("banking")
        self.assertEqual(adapter, "banking-lora")
        
        # Compliance domain
        adapter = self.llm._select_adapter("compliance")
        self.assertEqual(adapter, "compliance-lora")
        
        # Unknown domain
        adapter = self.llm._select_adapter("unknown")
        self.assertIsNone(adapter)
    
    def test_performance_metrics(self):
        """Test performance metrics."""
        metrics = self.llm.get_performance_metrics()
        
        required_keys = [
            "total_queries", "cache_hits", "cache_misses", 
            "cache_hit_rate", "llm_calls", "adapter_enhanced_calls"
        ]
        
        for key in required_keys:
            self.assertIn(key, metrics)
        
        self.assertIn("performance_improvement", metrics)
        perf = metrics["performance_improvement"]
        self.assertIn("latency_reduction_pct", perf)
        self.assertIn("cost_reduction_pct", perf)
        self.assertIn("domain_accuracy_improvement", perf)
    
    def test_cache_operations(self):
        """Test cache operations."""
        # Get cache stats
        stats = self.llm.get_cache_stats()
        self.assertIsNotNone(stats)
        self.assertIsInstance(stats, dict)
        
        # Clear cache
        result = self.llm.clear_cache()
        self.assertTrue(result)
    
    def test_adapter_operations(self):
        """Test adapter operations."""
        # Get adapter status
        status = self.llm.get_adapter_status()
        self.assertIsNotNone(status)
        self.assertIsInstance(status, dict)
        
        # Setup banking domain
        result = self.llm.setup_banking_domain()
        self.assertTrue(result)


class TestFactoryFunction(unittest.TestCase):
    """Test factory function."""
    
    def test_create_enhanced_llm_defaults(self):
        """Test creating LLM with defaults."""
        llm = create_enhanced_llm()
        self.assertEqual(llm.model_path, "mistralai/Mistral-7B-v0.1")
        self.assertTrue(llm.enable_cache)
        self.assertTrue(llm.enable_adapters)
    
    def test_create_enhanced_llm_custom(self):
        """Test creating LLM with custom settings."""
        llm = create_enhanced_llm(
            model_path="custom-model",
            enable_cache=False,
            enable_adapters=False
        )
        self.assertEqual(llm.model_path, "custom-model")
        self.assertFalse(llm.enable_cache)
        self.assertFalse(llm.enable_adapters)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios."""
    
    def setUp(self):
        """Set up integration test."""
        self.temp_dir = tempfile.mkdtemp()
        self.llm = create_enhanced_llm()
    
    def tearDown(self):
        """Clean up integration test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_banking_flow_integration(self):
        """Test banking conversation flow."""
        async def async_test():
            # Setup banking domain
            setup_result = self.llm.setup_banking_domain()
            self.assertTrue(setup_result)
            
            # Banking queries
            queries = [
                "What is my account balance?",
                "How do I transfer money?",
                "What is my account balance?",  # Should hit cache
            ]
            
            responses = []
            for query in queries:
                response = await self.llm.generate_response(
                    query,
                    domain_context="banking"
                )
                responses.append(response)
            
            # Verify responses
            self.assertEqual(len(responses), 3)
            
            # Verify adapter enhancement
            for response in responses:
                self.assertIn("Enhanced by banking domain adapter", response)
            
            # Verify cache hit
            self.assertGreaterEqual(self.llm.cache_hits, 1)
            self.assertEqual(responses[0], responses[2])  # Same query, same response
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_test())
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
