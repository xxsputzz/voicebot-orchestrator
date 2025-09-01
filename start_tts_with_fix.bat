@echo off
cd /d "c:\Users\miken\Desktop\Orkestra\aws_microservices"
echo Starting TTS service with Unicode fix...
python tts_hira_dia_service.py --engine full
pause
