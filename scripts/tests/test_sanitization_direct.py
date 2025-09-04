#!/usr/bin/env python3
"""
Direct test of Unicode sanitization function
"""
import re

def sanitize_text_for_synthesis(text: str) -> str:
    """
    Sanitize text for TTS synthesis by removing/replacing problematic Unicode characters
    """
    
    # Replace common emojis with text equivalents
    emoji_replacements = {
        '🎤': 'microphone',
        '🎵': 'music',
        '🎶': 'notes',
        '🔊': 'speaker',
        '🎧': 'headphones',
        '📢': 'announcement',
        '📣': 'megaphone',
        '✅': 'checkmark',
        '❌': 'X',
        '⚠️': 'warning',
        '🚀': 'rocket',
        '🔥': 'fire',
        '💡': 'lightbulb',
        '🎭': 'theater masks',
        '⏳': 'hourglass',
        '🔍': 'magnifying glass',
        '📊': 'chart',
        '📈': 'trending up',
        '📉': 'trending down',
        '💰': 'money',
        '🏦': 'bank',
        '💳': 'credit card',
        '💵': 'dollar bill',
        '📱': 'phone',
        '💻': 'laptop',
        '🌐': 'globe',
        '🔒': 'lock',
        '🔓': 'unlock'
    }
    
    # Replace emojis with text
    sanitized_text = text
    for emoji, replacement in emoji_replacements.items():
        sanitized_text = sanitized_text.replace(emoji, replacement)
    
    # Remove any remaining Unicode characters that could cause encoding issues
    # Keep only ASCII printable characters, spaces, and basic punctuation
    sanitized_text = re.sub(r'[^\x20-\x7E]', '', sanitized_text)
    
    # Clean up extra spaces
    sanitized_text = ' '.join(sanitized_text.split())
    
    # If text is now empty or very short, provide a fallback
    if not sanitized_text or len(sanitized_text.strip()) < 2:
        sanitized_text = "Text contains unsupported characters."
    
    return sanitized_text

def test_unicode_sanitization():
    """Test the Unicode sanitization function"""
    
    test_cases = [
        "I understand you're asking about: townhouse. Let me help you with that! 🎤",
        "Here's your response with music 🎵 and sound 🔊",
        "Banking information 🏦 with charts 📊 and trends 📈",
        "Simple text without emojis",
        "Mixed content: phone 📱 laptop 💻 and network 🌐",
        "Warning ⚠️ and checkmark ✅ and X ❌",
        "Complex unicode: café naïve résumé",
        "Only emojis: 🎤🎵🔊",
        "",
        "   "
    ]
    
    print("🧪 Testing Unicode Sanitization Function")
    print("=" * 60)
    
    for i, original in enumerate(test_cases, 1):
        sanitized = sanitize_text_for_synthesis(original)
        
        print(f"\nTest {i}:")
        print(f"  Original : '{original}'")
        print(f"  Sanitized: '{sanitized}'")
        print(f"  Length   : {len(original)} → {len(sanitized)}")
        
        # Check if sanitization worked
        try:
            # Try to encode as cp1252 (Windows default) to simulate the error
            sanitized.encode('cp1252')
            print("  Status   : ✅ Safe for cp1252 encoding")
        except UnicodeEncodeError as e:
            print(f"  Status   : ❌ Still has encoding issues: {e}")
        
        # Check ASCII safety
        if all(ord(c) < 128 for c in sanitized):
            print("  ASCII    : ✅ Pure ASCII")
        else:
            print("  ASCII    : ⚠️ Contains non-ASCII characters")
    
    print("\n" + "=" * 60)
    print("🎯 Sanitization Test Complete")

if __name__ == "__main__":
    test_unicode_sanitization()
