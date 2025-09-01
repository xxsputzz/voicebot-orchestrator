#!/usr/bin/env python3
"""
Fix Unicode emoji characters in logging statements that cause encoding issues on Windows
"""

import re

def fix_unicode_in_file(file_path):
    """Fix Unicode emojis in the specified file"""
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define emoji replacements for logging statements
    emoji_replacements = {
        '💡': '[INFO]',
        '🔧': '[FIX]',
        '✅': '[OK]',
        '🛑': '[STOP]',
        '⚠️': '[WARNING]',
        '🎤': '[MIC]',
        '🚀': '[FAST]',
        '⚡': '[SPEED]',
        '🧠': '[BRAIN]',
        '🎯': '[TARGET]',
        '🔄': '[SWITCH]',
        '🔧': '[CONFIG]',
        '🎉': '[SUCCESS]'
    }
    
    # Replace emojis with text equivalents
    for emoji, replacement in emoji_replacements.items():
        content = content.replace(emoji, replacement)
    
    # Write the file back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed Unicode characters in {file_path}")

if __name__ == "__main__":
    # Fix the TTS service file
    fix_unicode_in_file("aws_microservices/tts_hira_dia_service.py")
    print("Unicode logging fixes applied successfully!")
