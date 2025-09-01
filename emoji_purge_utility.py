#!/usr/bin/env python3
"""
Comprehensive Emoji Purge Utility
Removes ALL emoji characters from text at multiple pipeline stages
"""

import re
import unicodedata

def purge_all_emojis(text: str) -> str:
    """
    Aggressively remove ALL emoji and unicode characters that could cause encoding issues
    """
    if not text:
        return ""
    
    # Step 1: Replace common emojis with text equivalents first (preserve meaning)
    emoji_replacements = {
        '🎤': 'microphone',
        '🎵': 'music',
        '🎶': 'notes', 
        '🔊': 'speaker',
        '🎧': 'headphones',
        '📢': 'announcement',
        '📣': 'megaphone',
        '✅': 'yes',
        '❌': 'no',
        '⚠️': 'warning',
        '🚀': 'rocket',
        '🔥': 'fire',
        '💡': 'idea',
        '🎭': 'theater',
        '⏳': 'waiting',
        '🔍': 'search',
        '📊': 'chart',
        '📈': 'trending up',
        '📉': 'trending down',
        '💰': 'money',
        '🏦': 'bank',
        '💳': 'credit card',
        '💵': 'dollar',
        '📱': 'phone',
        '💻': 'computer',
        '🌐': 'internet',
        '🔒': 'secure',
        '🔓': 'unlock',
        '🎯': 'target',
        '🔄': 'refresh',
        '🔧': 'settings',
        '🎉': 'celebration',
        '🧹': 'cleanup',
        '🏥': 'hospital',
        '🧠': 'brain',
        '⚡': 'lightning',
        '🛑': 'stop',
        '👋': 'hello',
        '🤖': 'robot',
        '💬': 'chat',
        '📝': 'note',
        '📤': 'send',
        '📥': 'receive',
        '🔔': 'notification'
    }
    
    # Replace known emojis first
    cleaned_text = text
    for emoji, replacement in emoji_replacements.items():
        cleaned_text = cleaned_text.replace(emoji, replacement)
    
    # Step 2: Remove numbered emojis (1️⃣, 2️⃣, etc.)
    cleaned_text = re.sub(r'[0-9]️⃣', '', cleaned_text)
    
    # Step 3: Remove ALL emoji characters using Unicode categories
    # This is the nuclear option - removes ALL emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+"
    )
    cleaned_text = emoji_pattern.sub('', cleaned_text)
    
    # Step 4: Remove any remaining high Unicode characters that could cause issues
    # Keep only ASCII printable characters and basic punctuation
    cleaned_text = re.sub(r'[^\x20-\x7E\r\n\t]', '', cleaned_text)
    
    # Step 5: Clean up extra whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    
    # Step 6: Ensure we still have meaningful text
    if not cleaned_text.strip() or len(cleaned_text.strip()) < 2:
        cleaned_text = "Text contains unsupported characters and has been sanitized."
    
    return cleaned_text

def test_emoji_purge():
    """Test the emoji purging function"""
    test_cases = [
        "Hello 🎤 this is a test",
        "I understand you're asking about: The hobbit books 🎤 How can I help?",
        "✅ Success! 🎉 Your transaction has been processed.",
        "🏦 Banking services 💰 are available 24/7 📱",
        "🤖 Hello! I'm here to help with 🎯 your needs.",
        "Text with various emojis: 🚀🔥💡⚡🧠🎭⏳🔍",
        "Normal text without any emojis should remain unchanged."
    ]
    
    print("🧹 Testing Emoji Purge Function")
    print("=" * 60)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Original: {text}")
        purged = purge_all_emojis(text)
        print(f"Purged:   {purged}")
        
        # Check if any emojis remain
        has_emoji = any(ord(char) > 127 for char in purged)
        print(f"Status:   {'❌ Still has Unicode' if has_emoji else '✅ Clean ASCII'}")

if __name__ == "__main__":
    test_emoji_purge()
