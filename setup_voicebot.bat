@echo off
echo 🎭 Enhanced Voicebot Setup
echo ========================

echo.
echo Installing required packages...
pip install requests pyaudio wave

echo.
echo Checking microphone access...
python -c "try: import pyaudio; print('✅ PyAudio available'); except: print('❌ PyAudio not available')"

echo.
echo Starting Enhanced Voicebot Interface...
python enhanced_voicebot_interface.py

echo.
echo Setup completed!
pause
