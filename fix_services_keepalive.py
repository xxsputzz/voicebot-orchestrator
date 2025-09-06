#!/usr/bin/env python3
"""
Fix WebSocket services to use keep-alive message loops instead of exiting immediately
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

    # New message loop pattern (with keep-alive)
    new_message_loop = '''    async def message_loop(self):
        """Main message handling loop with keep-alive"""
        try:
            while self.running:
                try:
                    # Wait for messages with timeout to allow for keep-alive checks
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
                    try:
                        data = json.loads(message)
                        await self.handle_message(data)
                    except json.JSONDecodeError:
                        logging.error(f"[WS] Invalid JSON received: {message}")
                    except Exception as e:
                        logging.error(f"[WS] Error processing message: {e}")
                        
                except asyncio.TimeoutError:
                    # No message received, continue loop (keep-alive)
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logging.info("[WS] WebSocket connection closed")
                    break
                    
        except Exception as e:
            logging.error(f"[WS] Message loop error: {e}")
        finally:
            self.running = False'''

    successful_fixes = []
    failed_fixes = []

    for service_path in services:
        if not os.path.exists(service_path):
            failed_fixes.append(f'{service_path} - File not found')
            continue
        
        try:
            with open(service_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Only fix services that haven't been fixed yet (avoid corrupted ones)
            if 'await asyncio.wait_for(self.websocket.recv(), timeout=10.0)' in content:
                successful_fixes.append(f'{service_path} - Already fixed')
                continue
            
            # Check if content is too corrupted to fix
            if 'sys.path.insert(0, parent_di    async def message_loop' in content:
                failed_fixes.append(f'{service_path} - File corrupted, needs manual fix')
                continue
                
            # Pattern to match the old message_loop implementation 
            old_pattern = r'    async def message_loop\(self\):[^:]*:.*?async for message in self\.websocket:.*?finally:\s*self\.running = False'
            
            if re.search(old_pattern, content, re.DOTALL):
                # Replace the old pattern with new one
                content = re.sub(old_pattern, new_message_loop, content, flags=re.DOTALL)
                
                with open(service_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                successful_fixes.append(f'{service_path} - Fixed successfully')
            else:
                failed_fixes.append(f'{service_path} - Pattern not found')
        
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
