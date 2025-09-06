#!/usr/bin/env python3
import re

def _purge_emojis_from_llm_response(text: str) -> str:
    """Test the emoji purging function"""
    if not text:
        return text
    
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
    ]
    for pattern in emoji_patterns:
        text = re.sub(pattern, '', text)
    
    # Clean up extra spaces but preserve trailing spaces for token streaming
    text = re.sub(r'\s+', ' ', text)
    # Don't strip() here as it removes trailing spaces needed for token joining
    
    return text

# Test tokens
test_tokens = ["Hey", "there!", "I'm", "Alex", "from", "Finally", "Pay", "off", "Debt."]

print("=== Testing Token Processing ===")
for i, word in enumerate(test_tokens):
    # Simulate how tokens are created in _stream_llm_tokens
    token = word + (" " if i < len(test_tokens) - 1 else "")
    clean_token = _purge_emojis_from_llm_response(token)
    
    print(f"Original token: '{token}' (len={len(token)})")
    print(f"Clean token:    '{clean_token}' (len={len(clean_token)})")
    print(f"Space removed:  {len(token) != len(clean_token)}")
    print()

print("=== Full Response Test ===")
response = "Hey there! I'm Alex from Finally Pay off Debt."
words = response.split()
tokens = []
for i, word in enumerate(words):
    token = word + (" " if i < len(words) - 1 else "")
    clean_token = _purge_emojis_from_llm_response(token)
    tokens.append(clean_token)

joined_response = ''.join(tokens)
print(f"Original response: '{response}'")
print(f"Joined tokens:     '{joined_response}'")
print(f"Spaces missing:    {joined_response == response}")
