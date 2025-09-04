#!/usr/bin/env python3
"""
Test emoji purging function
"""

import re

def _purge_emojis_from_llm_response(text: str) -> str:
    """
    Remove all emojis from LLM responses to prevent TTS encoding issues.
    Nuclear option - removes all Unicode emoji ranges
    """
    if not text:
        return text
    
    print(f"Original text: {repr(text)}")
    
    # Common emoji replacements
    emoji_replacements = {
        '🎤': 'microphone',
        '🔊': 'speaker',
        '👋': 'hello',
        '😊': 'happy',
        '😄': 'smile',
        '👍': 'thumbs up',
        '❤️': 'heart',
        '💯': 'perfect',
        '🔥': 'fire',
        '✨': 'sparkle',
        '🎉': 'celebration'
    }
    
    # Apply replacements first
    for emoji, replacement in emoji_replacements.items():
        if emoji in text:
            print(f"Replacing {emoji} with {replacement}")
            text = text.replace(emoji, replacement)
    
    print(f"After emoji replacement: {repr(text)}")
    
    # Nuclear emoji removal - comprehensive Unicode ranges
    emoji_patterns = [
        r'[\U0001F600-\U0001F64F]',  # Emoticons
        r'[\U0001F300-\U0001F5FF]',  # Misc Symbols
        r'[\U0001F680-\U0001F6FF]',  # Transport
        r'[\U0001F1E0-\U0001F1FF]',  # Country flags
        r'[\U00002600-\U000027BF]',  # Misc symbols
        r'[\U0001F900-\U0001F9FF]',  # Supplemental Symbols
        r'[\U00002702-\U000027B0]',  # Dingbats
        r'[\U000024C2-\U0001F251]',  # Various symbols
        r'[\U0001F170-\U0001F171]',  # Enclosed alphanumerics
        r'[\U0001F17E-\U0001F17F]',  # More enclosed
        r'[\U0001F18E]',             # Negative squared
        r'[\U0001F191-\U0001F19A]',  # Squared symbols
        r'[\U0001F201-\U0001F202]',  # Squared katakana
        r'[\U0001F21A]',             # Squared CJK
        r'[\U0001F22F]',             # Squared finger
        r'[\U0001F232-\U0001F23A]',  # Squared CJK symbols
        r'[\U0001F250-\U0001F251]',  # Circled ideographs
        r'[\U0000FE0F]',             # Variation selector (fixed escape)
        r'[\U0000200D]',             # Zero width joiner
    ]
    
    # Apply all patterns
    for pattern in emoji_patterns:
        text = re.sub(pattern, '', text)
    
    print(f"After pattern removal: {repr(text)}")
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    print(f"Final cleaned text: {repr(text)}")
    
    return text

if __name__ == "__main__":
    # Test with the specific text that's causing issues
    test_texts = [
        "I understand you're asking about: steal ball bearings. How can I help you with this?🎤",
        "🎤 LLM Input: 'steal ball bearings'",
        "📝 LLM Input: 'steal ball bearings'",
        "✅ LLM Output: 'I understand you're asking about this...'",
        "Hello 🎤 World 🔊 Test 👋"
    ]
    
    print("Testing emoji purging function:")
    print("=" * 50)
    
    for i, test_text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print("-" * 20)
        cleaned = _purge_emojis_from_llm_response(test_text)
        print(f"SUCCESS: '{cleaned}'")
        print()
