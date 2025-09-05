"""
Test Simple GPT LLM - Clean, Direct Approach
"""
import asyncio
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simple_gpt_llm import create_simple_gpt_llm

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

async def test_simple_gpt():
    """Test simple, clean GPT responses."""
    print("🤖 Simple GPT LLM Test")
    print("Clean, Direct Responses")
    print("=" * 40)
    
    # Create simple GPT LLM
    llm = create_simple_gpt_llm(model_name="gpt-oss:20b")
    
    # Simple, realistic test scenarios
    test_scenarios = [
        {
            "input": "My name is Mike and I have $15,000 in credit card debt",
            "user_name": "Mike",
            "description": "Basic debt inquiry"
        },
        {
            "input": "What's the difference between APR and interest rate?",
            "description": "Financial education question"
        },
        {
            "input": "I make $4,500 per month, can I qualify for a loan?",
            "description": "Income qualification"
        },
        {
            "input": "Are there any fees I should know about?",
            "description": "Fee inquiry"
        },
        {
            "input": "How much could I save by consolidating my debt?",
            "description": "Savings question"
        }
    ]
    
    print(f"\n💬 Testing {len(test_scenarios)} Simple Scenarios")
    print("-" * 45)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📝 Test {i}: {scenario['description']}")
        print(f"Input: '{scenario['input']}'")
        
        try:
            response = await llm.generate_response(
                user_input=scenario['input'],
                user_name=scenario.get('user_name'),
                call_type="inbound"
            )
            
            print(f"🤖 GPT Response:")
            print(f"   {response}")
            print(f"   ({len(response)} characters)")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Show simple stats
    stats = llm.get_stats()
    print(f"\n📊 Simple Performance Stats")
    print("-" * 30)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\n✅ Simple GPT Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_simple_gpt())
