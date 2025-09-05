"""
Test the new Ollama-enhanced LLM integration.
"""
import asyncio
import logging
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from voicebot_orchestrator.ollama_enhanced_llm import create_ollama_enhanced_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

async def test_ollama_enhanced_llm():
    """Test the Ollama-enhanced LLM with banking scenarios."""
    print("🚀 Testing Ollama Enhanced LLM Integration")
    print("=" * 60)
    
    # Create enhanced LLM with Ollama
    llm = create_ollama_enhanced_llm(
        model_name="mistral:latest",
        enable_cache=True,
        enable_adapters=True,
        fallback_enabled=True
    )
    
    # Test connection first
    print("\n🔗 Testing Ollama connection...")
    connected = await llm.test_ollama_connection()
    if not connected:
        print("❌ Ollama connection failed - testing fallback mode")
    else:
        print("✅ Ollama connection successful")
    
    # Test scenarios that should work with real AI
    test_scenarios = [
        "My name is Mike and I have $15,000 in credit card debt",
        "What's the difference between APR and interest rate?",
        "I'm self-employed, can I still get a personal loan?",
        "How much could I save with debt consolidation?",
        "Are there any fees I should know about?"
    ]
    
    print(f"\n🧠 Testing {len(test_scenarios)} scenarios")
    print("-" * 40)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📝 Test {i}: '{scenario}'")
        
        try:
            # Test with banking domain context
            response = await llm.generate_response(
                user_input=scenario,
                domain_context="banking",
                call_type="inbound"
            )
            
            print(f"🤖 Response ({len(response)} chars): {response}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Show performance metrics
    print(f"\n📊 Performance Metrics")
    print("-" * 30)
    metrics = llm.get_performance_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    print(f"\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_ollama_enhanced_llm())
