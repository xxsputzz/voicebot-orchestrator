#!/usr/bin/env python3
"""
Debug the exact script matching logic
"""
import psutil

services = [
    'ws_stt_whisper_service.py', 
    'ws_llm_gpt_service.py', 
    'ws_llm_mistral_service.py', 
    'ws_tts_kokoro_service.py', 
    'ws_tts_zonos_service.py', 
    'ws_tts_tortoise_service.py', 
    'ws_tts_dia_service.py'
]

print('🔍 Actual running processes:')
running_processes = []
for process in psutil.process_iter(['pid', 'name', 'cmdline']):
    if process.info['name'] == 'python.exe':
        try:
            cmdline = ' '.join(process.info['cmdline'])
            print(f'  PID {process.info["pid"]}: {cmdline}')
            running_processes.append((process.info['pid'], cmdline))
        except:
            pass

print(f'\n🔍 Checking each service script:')
for service_script in services:
    found = False
    for pid, cmdline in running_processes:
        if service_script in cmdline:
            print(f'  ✅ {service_script}: FOUND in PID {pid}')
            found = True
            break
    if not found:
        print(f'  ❌ {service_script}: NOT FOUND')
