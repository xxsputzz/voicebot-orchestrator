#!/usr/bin/env python3
"""
Fix Unicode emoji characters in zonos_tts.py for Windows terminal compatibility
"""

import re

def fix_unicode_chars():
    file_path = 'voicebot_orchestrator/zonos_tts.py'
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace Unicode emoji characters with plain text equivalents
    replacements = {
        '🎙️': '',
        '✅': '',
        '❌': '',
        '⚠️': 'WARNING:',
        '📥': '',
        '🎭': '',
        '🎯': '',
        '🚀': '',
        '📊': '',
        '⏳': '',
        '💫': '',
        '🔄': '',
    }
    
    for emoji, replacement in replacements.items():
        content = content.replace(emoji, replacement)
    
    # Clean up any double spaces created by emoji removal
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'^\s+', '', content, flags=re.MULTILINE)
    
    # Write the fixed content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed Unicode characters in zonos_tts.py")

if __name__ == "__main__":
    fix_unicode_chars()
