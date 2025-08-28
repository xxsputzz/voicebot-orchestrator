"""
Tests for Enhanced LLM with Semantic Cache & LoRA Integration
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from voicebot_orchestrator.enhanced_llm import EnhancedMistralLLM, create_enhanced_llm


class TestEnhancedMistralLLM:
    """Test cases for Enhanced Mistral LLM."""
    
    @pytest.fixture
    def llm(self):
        """Create test LLM instance."""
        return EnhancedMistralLLM(
            model_path="test-model",
            enable_cache=True,
            enable_adapters=True
        )
    
    @pytest.fixture
    def llm_no_features(self):
        """Create LLM with features disabled."""
        return EnhancedMistralLLM(
            model_path="test-model",
            enable_cache=False,
            enable_adapters=False
        )
    
    def test_initialization(self, llm):
        """Test LLM initialization."""
        assert llm.model_path == "test-model"
        assert llm.enable_cache is True
        assert llm.enable_adapters is True
        assert llm.cache is not None
        assert llm.adapter_manager is not None
        assert llm.cache_hits == 0
        assert llm.cache_misses == 0
    
    def test_initialization_disabled_features(self, llm_no_features):
        """Test LLM initialization with features disabled."""
        assert llm_no_features.cache is None
        assert llm_no_features.adapter_manager is None
    
    @pytest.mark.asyncio
    async def test_generate_response_basic(self, llm):
        """Test basic response generation."""
        response = await llm.generate_response("Hello")
        assert response is not None
        assert len(response) > 0
        assert llm.llm_calls == 1
    
    @pytest.mark.asyncio
    async def test_generate_response_empty_input(self, llm):
        """Test response generation with empty input."""
        with pytest.raises(ValueError, match="User input cannot be empty"):
            await llm.generate_response("")
        
        with pytest.raises(ValueError, match="User input cannot be empty"):
            await llm.generate_response("   ")
    
    @pytest.mark.asyncio
    async def test_generate_response_with_cache(self, llm):
        """Test response generation with caching."""
        # First call should be cache miss
        response1 = await llm.generate_response("What is my balance?")
        assert llm.cache_misses == 1
        assert llm.cache_hits == 0
        
        # Second call should be cache hit
        response2 = await llm.generate_response("What is my balance?")
        assert llm.cache_misses == 1
        assert llm.cache_hits == 1
        assert response1 == response2
    
    @pytest.mark.asyncio
    async def test_generate_response_banking_domain(self, llm):
        """Test banking domain adapter selection."""
        response = await llm.generate_response(
            "What is my account balance?",
            domain_context="banking"
        )
        assert "Enhanced by banking domain adapter" in response
        assert llm.adapter_enhanced_calls == 1
    
    @pytest.mark.asyncio
    async def test_generate_response_compliance_domain(self, llm):
        """Test compliance domain adapter selection."""
        response = await llm.generate_response(
            "Are calls recorded?",
            domain_context="compliance"
        )
        assert "Enhanced by compliance adapter" in response
        assert llm.adapter_enhanced_calls == 1
    
    @pytest.mark.asyncio
    async def test_generate_response_no_domain(self, llm):
        """Test response without domain context."""
        response = await llm.generate_response("Hello there")
        assert "Enhanced by" not in response
        assert llm.adapter_enhanced_calls == 0
    
    def test_select_adapter_banking(self, llm):
        """Test banking adapter selection."""
        adapter = llm._select_adapter("banking")
        assert adapter == "banking-lora"
        
        adapter = llm._select_adapter("loans")
        assert adapter == "banking-lora"
        
        adapter = llm._select_adapter("mortgage")
        assert adapter == "banking-lora"
    
    def test_select_adapter_compliance(self, llm):
        """Test compliance adapter selection."""
        adapter = llm._select_adapter("compliance")
        assert adapter == "compliance-lora"
        
        adapter = llm._select_adapter("legal")
        assert adapter == "compliance-lora"
        
        adapter = llm._select_adapter("kyc")
        assert adapter == "compliance-lora"
    
    def test_select_adapter_unknown(self, llm):
        """Test unknown domain adapter selection."""
        adapter = llm._select_adapter("unknown")
        assert adapter is None
    
    def test_select_adapter_disabled(self, llm_no_features):
        """Test adapter selection when disabled."""
        adapter = llm_no_features._select_adapter("banking")
        assert adapter is None
    
    def test_generate_banking_response(self, llm):
        """Test banking-specific response generation."""
        response = llm._generate_banking_response("What is my balance?")
        assert "balance" in response.lower()
        assert "Enhanced by banking domain adapter" in response
        
        response = llm._generate_banking_response("I need a loan")
        assert "loan" in response.lower()
        assert "Enhanced by banking domain adapter" in response
    
    def test_generate_compliance_response(self, llm):
        """Test compliance-specific response generation."""
        response = llm._generate_compliance_response("Are calls recorded?")
        assert "recorded" in response.lower()
        assert "Enhanced by compliance adapter" in response
        
        response = llm._generate_compliance_response("How do you handle my data?")
        assert "data" in response.lower()
        assert "Enhanced by compliance adapter" in response
    
    def test_generate_general_response(self, llm):
        """Test general response generation."""
        response = llm._generate_general_response("Hello")
        assert "Hello" in response
        assert "Enhanced by" not in response
    
    def test_get_performance_metrics(self, llm):
        """Test performance metrics retrieval."""
        metrics = llm.get_performance_metrics()
        
        assert "total_queries" in metrics
        assert "cache_hits" in metrics
        assert "cache_misses" in metrics
        assert "cache_hit_rate" in metrics
        assert "llm_calls" in metrics
        assert "adapter_enhanced_calls" in metrics
        assert "performance_improvement" in metrics
        
        perf = metrics["performance_improvement"]
        assert "latency_reduction_pct" in perf
        assert "cost_reduction_pct" in perf
        assert "domain_accuracy_improvement" in perf
    
    def test_get_cache_stats(self, llm):
        """Test cache statistics retrieval."""
        stats = llm.get_cache_stats()
        assert stats is not None
        assert isinstance(stats, dict)
    
    def test_get_cache_stats_disabled(self, llm_no_features):
        """Test cache statistics when cache disabled."""
        stats = llm_no_features.get_cache_stats()
        assert stats is None
    
    def test_get_adapter_status(self, llm):
        """Test adapter status retrieval."""
        status = llm.get_adapter_status()
        assert status is not None
        assert isinstance(status, dict)
    
    def test_get_adapter_status_disabled(self, llm_no_features):
        """Test adapter status when adapters disabled."""
        status = llm_no_features.get_adapter_status()
        assert status is None
    
    def test_clear_cache(self, llm):
        """Test cache clearing."""
        result = llm.clear_cache()
        assert result is True
    
    def test_clear_cache_disabled(self, llm_no_features):
        """Test cache clearing when cache disabled."""
        result = llm_no_features.clear_cache()
        assert result is False
    
    def test_setup_banking_domain(self, llm):
        """Test banking domain setup."""
        result = llm.setup_banking_domain()
        assert result is True
    
    def test_setup_banking_domain_disabled(self, llm_no_features):
        """Test banking domain setup when adapters disabled."""
        result = llm_no_features.setup_banking_domain()
        assert result is False


class TestCreateEnhancedLLM:
    """Test factory function."""
    
    def test_create_enhanced_llm_defaults(self):
        """Test creating LLM with defaults."""
        llm = create_enhanced_llm()
        assert llm.model_path == "mistralai/Mistral-7B-v0.1"
        assert llm.enable_cache is True
        assert llm.enable_adapters is True
        assert llm.cache is not None
        assert llm.adapter_manager is not None
    
    def test_create_enhanced_llm_custom(self):
        """Test creating LLM with custom settings."""
        llm = create_enhanced_llm(
            model_path="custom-model",
            enable_cache=False,
            enable_adapters=False
        )
        assert llm.model_path == "custom-model"
        assert llm.enable_cache is False
        assert llm.enable_adapters is False
        assert llm.cache is None
        assert llm.adapter_manager is None


class TestIntegrationScenarios:
    """Integration test scenarios."""
    
    @pytest.mark.asyncio
    async def test_banking_conversation_flow(self):
        """Test full banking conversation with caching and adapters."""
        llm = create_enhanced_llm()
        
        # Setup banking domain
        llm.setup_banking_domain()
        
        # Conversation flow
        queries = [
            "What is my account balance?",
            "How do I transfer money?",
            "What are your loan rates?",
            "What is my account balance?",  # Should hit cache
        ]
        
        responses = []
        for query in queries:
            response = await llm.generate_response(
                query,
                domain_context="banking"
            )
            responses.append(response)
        
        # Verify responses
        assert len(responses) == 4
        assert all("Enhanced by banking domain adapter" in r for r in responses)
        
        # Verify cache hit
        assert llm.cache_hits >= 1
        assert responses[0] == responses[3]  # Same query should get same response
    
    @pytest.mark.asyncio
    async def test_mixed_domain_conversation(self):
        """Test conversation with mixed domains."""
        llm = create_enhanced_llm()
        
        queries_domains = [
            ("What is my balance?", "banking"),
            ("Are calls recorded?", "compliance"),
            ("I need a loan", "banking"),
            ("What are KYC requirements?", "compliance"),
        ]
        
        for query, domain in queries_domains:
            response = await llm.generate_response(
                query,
                domain_context=domain
            )
            
            if domain == "banking":
                assert "Enhanced by banking domain adapter" in response
            elif domain == "compliance":
                assert "Enhanced by compliance adapter" in response
        
        # Should have used adapters for all calls
        assert llm.adapter_enhanced_calls == 4
    
    @pytest.mark.asyncio
    async def test_performance_optimization(self):
        """Test performance optimization features."""
        llm = create_enhanced_llm()
        
        # Generate some responses to populate metrics
        await llm.generate_response("Test query 1", domain_context="banking")
        await llm.generate_response("Test query 2", domain_context="compliance")
        await llm.generate_response("Test query 1", domain_context="banking")  # Cache hit
        
        metrics = llm.get_performance_metrics()
        
        # Verify metrics
        assert metrics["total_queries"] == 3
        assert metrics["cache_hits"] == 1
        assert metrics["cache_misses"] == 2
        assert metrics["cache_hit_rate"] > 0
        assert metrics["adapter_enhanced_calls"] == 3
        
        # Verify performance improvements
        perf = metrics["performance_improvement"]
        assert perf["latency_reduction_pct"] > 0
        assert perf["cost_reduction_pct"] > 0
        assert perf["domain_accuracy_improvement"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
