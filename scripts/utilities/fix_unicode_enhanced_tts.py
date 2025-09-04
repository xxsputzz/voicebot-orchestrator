#!/usr/bin/env python3
"""
Quick Unicode/Emoji removal script for enhanced_tts_manager.py
"""

import re

def remove_emojis_from_file(file_path):
    """Remove all emojis and special Unicode characters from print statements"""
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define Unicode emoji patterns
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese characters
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+", flags=re.UNICODE)
    
    # Also replace specific problem characters
    replacements = {
        '→': '->',
        '←': '<-',
        '↑': '^',
        '↓': 'v',
        '⇒': '=>',
        '⇐': '<=',
        '🎭': '',
        '⏳': '',
        '1️⃣': '',
        '2️⃣': '',
        '✅': '',
        '🎤': '',
        '⚡': '',
        '❌': '',
        '🧠': '',
        '📊': '',
        '🎉': '',
        '🎯': '',
        '🔧': '',
        '🔄': '',
        '📝': '',
        '⚠️': '',
        '💾': '',
        '🎵': '',
        '🧹': '',
        '🚀': '',
        '🤖': '',
        '🎬': '',
        '👋': '',
    }
    
    # Apply replacements
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    # Remove any remaining emojis
    content = emoji_pattern.sub('', content)
    
    # Clean up multiple spaces that might result from emoji removal
    content = re.sub(r'  +', ' ', content)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Removed emojis from {file_path}")

if __name__ == "__main__":
    file_path = r"c:\Users\miken\Desktop\Orkestra\voicebot_orchestrator\enhanced_tts_manager.py"
    remove_emojis_from_file(file_path)
    print("Unicode/emoji removal complete!")
