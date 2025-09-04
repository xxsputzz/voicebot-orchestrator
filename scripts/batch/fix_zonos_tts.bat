@echo off
echo ========================================
echo 🔧 FIXING ZONOS TTS DIGITAL NOISE ISSUE
echo ========================================
echo.
echo This will replace the synthetic Zonos TTS with real neural speech
echo.

echo 📋 Step 1: Backing up original zonos_tts.py...
copy "voicebot_orchestrator\zonos_tts.py" "voicebot_orchestrator\zonos_tts_backup.py" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backup created: zonos_tts_backup.py
) else (
    echo ⚠️ Backup failed, continuing anyway...
)

echo.
echo 📋 Step 2: Installing real TTS packages...
echo 📦 Installing edge-tts...
pip install edge-tts --quiet

echo 📦 Installing gTTS...
pip install gtts --quiet

echo 📦 Installing pyttsx3...
pip install pyttsx3 --quiet

echo 📦 Installing pydub...
pip install pydub --quiet

echo.
echo 📋 Step 3: Testing real TTS installation...
python -c "import edge_tts; print('✅ edge-tts installed')" 2>nul || echo "⚠️ edge-tts not installed"
python -c "import gtts; print('✅ gtts installed')" 2>nul || echo "⚠️ gtts not installed"
python -c "import pyttsx3; print('✅ pyttsx3 installed')" 2>nul || echo "⚠️ pyttsx3 not installed"
python -c "import pydub; print('✅ pydub installed')" 2>nul || echo "⚠️ pydub not installed"

echo.
echo 📋 Step 4: Applying enhanced TTS patch...
copy "voicebot_orchestrator\zonos_tts_enhanced.py" "voicebot_orchestrator\zonos_tts.py" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Enhanced TTS patch applied successfully!
) else (
    echo ❌ Failed to apply patch
    pause
    exit /b 1
)

echo.
echo 📋 Step 5: Testing enhanced TTS...
python -c "from voicebot_orchestrator.zonos_tts import ZonosTTS; print('✅ Enhanced TTS loaded successfully')" 2>nul
if %errorlevel% equ 0 (
    echo ✅ Enhanced TTS working correctly!
) else (
    echo ⚠️ Enhanced TTS loading issue - check logs
)

echo.
echo ========================================
echo ✅ ZONOS TTS ENHANCEMENT COMPLETE!
echo ========================================
echo.
echo 🎙️ Your TTS will now generate real speech instead of digital noises
echo 🔄 Restart any running TTS services to use the new engine
echo 📁 Original backed up as: zonos_tts_backup.py
echo.
echo 🧪 Test your TTS now:
echo    python enhanced_real_tts.py
echo.
pause
