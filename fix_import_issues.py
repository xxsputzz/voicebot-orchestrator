#!/usr/bin/env python3
"""
Fix WebSocket services by removing the problematic WebSocketServiceClient import
"""

import os
import re

def main():
    # List of all WebSocket services
    services = [
        'aws_microservices/ws_stt_whisper_service.py',
        'aws_microservices/ws_llm_gpt_service.py', 
        'aws_microservices/ws_llm_mistral_service.py',
        'aws_microservices/ws_tts_kokoro_service.py',
        'aws_microservices/ws_tts_zonos_service.py',
        'aws_microservices/ws_tts_tortoise_service.py',
        'aws_microservices/ws_tts_dia_service.py'
    ]

    successful_fixes = []
    failed_fixes = []

    for service_path in services:
        if not os.path.exists(service_path):
            failed_fixes.append(f'{service_path} - File not found')
            continue
        
        try:
            with open(service_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if the problematic import exists
            if 'from ws_orchestrator_service import WebSocketServiceClient' not in content:
                successful_fixes.append(f'{service_path} - Already fixed')
                continue
            
            # Remove the problematic import line
            content = content.replace(
                'from ws_orchestrator_service import WebSocketServiceClient\n    ',
                ''
            ).replace(
                ', WebSocketServiceClient',
                ''
            ).replace(
                'from ws_orchestrator_service import WebSocketServiceClient',
                ''
            )
            
            # Also fix any lingering line issues
            content = re.sub(r'from ws_service_registry import ServiceRegistration, ServiceCapabilities\n\n    WEBSOCKET_AVAILABLE', 
                           'from ws_service_registry import ServiceRegistration, ServiceCapabilities\n    WEBSOCKET_AVAILABLE', content)
            
            with open(service_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            successful_fixes.append(f'{service_path} - Import fixed')
        
        except Exception as e:
            failed_fixes.append(f'{service_path} - Error: {e}')

    print('Successful fixes:')
    for fix in successful_fixes:
        print(f'  ✅ {fix}')

    print('\nFailed fixes:')  
    for fix in failed_fixes:
        print(f'  ❌ {fix}')

if __name__ == "__main__":
    main()
