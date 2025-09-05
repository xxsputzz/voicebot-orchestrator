"""
FINAL TEST: Breakthrough 95%+ Accuracy Achievement
"""
import asyncio
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from breakthrough_accuracy_llm import create_breakthrough_accuracy_llm

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

async def test_breakthrough_accuracy():
    """Final breakthrough accuracy test."""
    print("🚀 BREAKTHROUGH ACCURACY TEST")
    print("Target: 95%+ Accuracy Achievement")
    print("=" * 60)
    
    llm = create_breakthrough_accuracy_llm()
    
    # Precision test scenarios
    test_scenarios = [
        {
            "input": "My name is Sarah and I have $25,000 in credit card debt",
            "user_profile": {"name": "Sarah", "debt_amount": 25000},
            "description": "Name + Amount Recognition"
        },
        {
            "input": "I make $6,000 monthly but worried about qualifying",
            "user_profile": {"income": 6000},
            "description": "Income + Emotion Recognition"
        },
        {
            "input": "Can you calculate savings from 24% to 12% on $15K?",
            "description": "Complex Financial Calculation"
        },
        {
            "input": "I'm a nurse with $35,000 debt across 4 cards, need help urgently",
            "user_profile": {"profession": "nurse", "debt_amount": 35000},
            "description": "Multi-Detail High-Stakes"
        },
        {
            "input": "Hi Alex, what's your best rate for debt consolidation?",
            "description": "Direct Product Inquiry"
        }
    ]
    
    print(f"\n🎯 Testing {len(test_scenarios)} Breakthrough Scenarios")
    print("-" * 50)
    
    results = []
    total_accuracy = 0
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📝 Test {i}: {scenario['description']}")
        print(f"Input: '{scenario['input']}'")
        
        try:
            response, accuracy = await llm.generate_response(
                user_input=scenario['input'],
                user_profile=scenario.get('user_profile')
            )
            
            results.append(accuracy)
            total_accuracy += accuracy
            
            print(f"🤖 Response: {response[:200]}...")
            print(f"🎯 Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            if accuracy >= 0.98:
                grade = "🏆 BREAKTHROUGH"
            elif accuracy >= 0.95:
                grade = "🥇 EXCEPTIONAL"
            elif accuracy >= 0.90:
                grade = "🥈 EXCELLENT"
            elif accuracy >= 0.85:
                grade = "🥉 VERY GOOD"
            else:
                grade = "📈 IMPROVING"
            
            print(f"   {grade}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append(0.0)
    
    # Final analysis
    print(f"\n🏆 BREAKTHROUGH ACCURACY RESULTS")
    print("=" * 45)
    
    if results:
        avg_accuracy = total_accuracy / len(results)
        
        breakthrough_count = sum(1 for r in results if r >= 0.98)
        exceptional_count = sum(1 for r in results if r >= 0.95)
        excellent_count = sum(1 for r in results if r >= 0.90)
        
        print(f"Average Accuracy: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)")
        print(f"🏆 Breakthrough (≥98%): {breakthrough_count}/{len(results)} ({(breakthrough_count/len(results))*100:.1f}%)")
        print(f"🥇 Exceptional (≥95%): {exceptional_count}/{len(results)} ({(exceptional_count/len(results))*100:.1f}%)")
        print(f"🥈 Excellent (≥90%): {excellent_count}/{len(results)} ({(excellent_count/len(results))*100:.1f}%)")
        
        # Achievement status
        if avg_accuracy >= 0.95:
            print(f"\n🎉 SUCCESS: ACHIEVED 95%+ ACCURACY!")
            print(f"🚀 Final Score: {avg_accuracy*100:.1f}%")
            if breakthrough_count >= len(results) * 0.6:
                print("💫 BONUS: 60%+ responses achieved breakthrough accuracy!")
        else:
            print(f"\n📈 Progress: {avg_accuracy*100:.1f}% accuracy achieved")
            print(f"🎯 Target: 95%+ accuracy")
        
        # System metrics
        metrics = llm.get_performance_metrics()
        print(f"\n⚙️ SYSTEM PERFORMANCE")
        print("-" * 25)
        for key, value in metrics.items():
            if key != 'model_performance':
                print(f"{key}: {value}")
    
    print(f"\n✅ Breakthrough Accuracy Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_breakthrough_accuracy())
