"""
Sprint 5: Complete Integration Demo

Demonstrates the full Sprint 5 implementation with semantic caching,
LoRA adapters, enhanced LLM, and enterprise orchestration.
"""

import asyncio
import json
import time
from voicebot_orchestrator.enhanced_llm import create_enhanced_llm
from voicebot_orchestrator.semantic_cache import SemanticCache
from voicebot_orchestrator.lora_adapter import LoraAdapterManager


async def main():
    """Run comprehensive Sprint 5 integration demo."""
    print("=" * 80)
    print("SPRINT 5: COMPLETE INTEGRATION DEMO")
    print("Semantic Cache Tuning & LoRA Adapter Training")
    print("=" * 80)
    
    # Initialize enhanced LLM with all features
    print("\n1. Initializing Enhanced Mistral LLM...")
    llm = create_enhanced_llm(
        model_path="mistralai/Mistral-7B-v0.1",
        enable_cache=True,
        enable_adapters=True
    )
    
    # Set up banking domain optimization
    print("2. Setting up Banking Domain Optimization...")
    banking_setup = llm.setup_banking_domain()
    print(f"   Banking domain setup: {'✓ Success' if banking_setup else '✗ Failed'}")
    
    # Demo banking conversation with caching and adapters
    print("\n3. Banking Conversation Demo (with caching & adapters):")
    banking_queries = [
        "What is my account balance?",
        "How do I transfer money between accounts?",
        "What are your current mortgage rates?",
        "What is my account balance?",  # Repeat for cache demo
        "I need help with a loan application",
        "What is APR and how is it calculated?",
        "What are your current mortgage rates?",  # Repeat for cache demo
    ]
    
    conversation_history = []
    for i, query in enumerate(banking_queries, 1):
        print(f"\n   Query {i}: {query}")
        
        start_time = time.time()
        response = await llm.generate_response(
            user_input=query,
            conversation_history=conversation_history,
            domain_context="banking"
        )
        end_time = time.time()
        
        print(f"   Response: {response}")
        print(f"   Time: {end_time - start_time:.3f}s")
        
        # Add to conversation history
        conversation_history.append({
            "human": query,
            "assistant": response
        })
    
    # Show performance metrics
    print("\n4. Performance Metrics:")
    metrics = llm.get_performance_metrics()
    print(json.dumps(metrics, indent=2))
    
    # Show cache statistics
    print("\n5. Semantic Cache Statistics:")
    cache_stats = llm.get_cache_stats()
    if cache_stats:
        print(json.dumps(cache_stats, indent=2))
    
    # Show adapter status
    print("\n6. LoRA Adapter Status:")
    adapter_status = llm.get_adapter_status()
    if adapter_status:
        print(json.dumps(adapter_status, indent=2))
    
    # Demo compliance conversation
    print("\n7. Compliance Conversation Demo:")
    compliance_queries = [
        "Are our calls being recorded?",
        "How do you handle my personal data?",
        "What are the KYC requirements?",
        "Is this transaction subject to audit?"
    ]
    
    for i, query in enumerate(compliance_queries, 1):
        print(f"\n   Query {i}: {query}")
        
        response = await llm.generate_response(
            user_input=query,
            domain_context="compliance"
        )
        
        print(f"   Response: {response}")
    
    # Cache efficiency test
    print("\n8. Cache Efficiency Test (repeated queries):")
    test_query = "What is my checking account balance?"
    
    # First call (cache miss)
    start_time = time.time()
    response1 = await llm.generate_response(test_query, domain_context="banking")
    time1 = time.time() - start_time
    
    # Second call (cache hit)
    start_time = time.time()
    response2 = await llm.generate_response(test_query, domain_context="banking")
    time2 = time.time() - start_time
    
    print(f"   First call (miss):  {time1:.3f}s - {response1}")
    print(f"   Second call (hit):  {time2:.3f}s - {response2}")
    print(f"   Speed improvement:  {((time1 - time2) / time1 * 100):.1f}%")
    
    # Final metrics summary
    print("\n9. Final Performance Summary:")
    final_metrics = llm.get_performance_metrics()
    
    print(f"   Total Queries:           {final_metrics['total_queries']}")
    print(f"   Cache Hit Rate:          {final_metrics['cache_hit_rate']:.1%}")
    print(f"   Adapter Enhanced Calls:  {final_metrics['adapter_enhanced_calls']}")
    print(f"   Latency Reduction:       {final_metrics['performance_improvement']['latency_reduction_pct']:.1f}%")
    print(f"   Cost Reduction:          {final_metrics['performance_improvement']['cost_reduction_pct']:.1f}%")
    print(f"   Domain Accuracy Boost:   {final_metrics['performance_improvement']['domain_accuracy_improvement']:.1f}%")
    
    print("\n" + "=" * 80)
    print("SPRINT 5 INTEGRATION DEMO COMPLETE")
    print("✓ Semantic caching with Faiss vector similarity")
    print("✓ LoRA adapters for banking and compliance domains")
    print("✓ Enhanced LLM with automatic cache/adapter integration")
    print("✓ Performance optimization and cost reduction")
    print("✓ Enterprise-ready analytics and monitoring")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
