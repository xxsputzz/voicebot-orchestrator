@echo off
echo ========================================
echo 🎙️ ENHANCED REAL TTS INSTALLATION
echo ========================================
echo.
echo Installing required packages for real TTS...
echo.

echo 📦 Installing edge-tts (Microsoft Neural Voices)...
pip install edge-tts

echo.
echo 📦 Installing gTTS (Google Text-to-Speech)...
pip install gtts

echo.
echo 📦 Installing pyttsx3 (Offline TTS)...
pip install pyttsx3

echo.
echo 📦 Installing pydub (Audio processing)...
pip install pydub

echo.
echo 📦 Installing additional audio dependencies...
pip install scipy soundfile

echo.
echo ✅ Installation complete!
echo.
echo 🎯 Testing installation...
python enhanced_real_tts.py

echo.
echo ========================================
echo 🚀 READY TO USE ENHANCED REAL TTS!
echo ========================================
pause
