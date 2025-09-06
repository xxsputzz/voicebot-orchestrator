#!/usr/bin/env python3
"""
Fix deprecated datetime.utcnow() warnings in orchestrator
"""

import re

# Read the file
with open('ws_orchestrator_service.py', 'r') as f:
    content = f.read()

# Replace all instances of datetime.utcnow() with datetime.now(datetime.timezone.utc)
content = re.sub(
    r'datetime\.utcnow\(\)\.isoformat\(\)',
    'datetime.now(datetime.timezone.utc).isoformat()',
    content
)

# Write back
with open('ws_orchestrator_service.py', 'w') as f:
    f.write(content)

print("✅ Fixed all datetime.utcnow() deprecated warnings")
