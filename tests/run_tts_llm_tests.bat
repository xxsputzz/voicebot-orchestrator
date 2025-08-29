@echo off
REM TTS/LLM Combination Testing Script
REM Tests different combinations of TTS engines and LLM models

echo.
echo 🎭 TTS/LLM Combination Testing Suite
echo =====================================
echo.

cd /d "%~dp0"

echo Current directory: %CD%
echo.

:MENU
echo Available Test Options:
echo.
echo 1. Test All Combinations (General)
echo 2. Test All Combinations (Independent Services)
echo 3. Test Kokoro TTS only
echo 4. Test Hira Dia TTS only
echo 5. Test Mistral LLM only
echo 6. Test GPT LLM only
echo 7. Quick Test (One prompt per combination)
echo 8. Test Specific Independent Combination
echo 9. Check Service Status
echo 0. Exit
echo.

set /p choice="Enter your choice (0-9): "

if "%choice%"=="0" goto EXIT
if "%choice%"=="1" goto TEST_ALL_GENERAL
if "%choice%"=="2" goto TEST_ALL_INDEPENDENT
if "%choice%"=="3" goto TEST_KOKORO
if "%choice%"=="4" goto TEST_HIRA_DIA
if "%choice%"=="5" goto TEST_MISTRAL
if "%choice%"=="6" goto TEST_GPT
if "%choice%"=="7" goto TEST_QUICK
if "%choice%"=="8" goto TEST_SPECIFIC
if "%choice%"=="9" goto CHECK_SERVICES

echo Invalid choice. Please enter 0-9.
pause
goto MENU

:TEST_ALL_GENERAL
echo.
echo 🧪 Running All Combinations Test (General)...
python test_tts_llm_combinations.py
pause
goto MENU

:TEST_ALL_INDEPENDENT
echo.
echo 🧪 Running All Combinations Test (Independent Services)...
echo Note: This requires independent microservices to be running
echo Start them with: cd aws_microservices && python enhanced_service_manager.py
echo.
python test_independent_combinations.py
pause
goto MENU

:TEST_KOKORO
echo.
echo 🎙️ Testing Kokoro TTS only...
python test_tts_llm_combinations.py --tts kokoro
pause
goto MENU

:TEST_HIRA_DIA
echo.
echo 🎙️ Testing Hira Dia TTS only...
python test_tts_llm_combinations.py --tts hira_dia
pause
goto MENU

:TEST_MISTRAL
echo.
echo 🧠 Testing Mistral LLM only...
python test_tts_llm_combinations.py --llm mistral
pause
goto MENU

:TEST_GPT
echo.
echo 🧠 Testing GPT LLM only...
python test_tts_llm_combinations.py --llm gpt
pause
goto MENU

:TEST_QUICK
echo.
echo ⚡ Running Quick Test...
python test_tts_llm_combinations.py --quick
pause
goto MENU

:TEST_SPECIFIC
echo.
echo Available Independent Service Combinations:
echo 1. Kokoro + Mistral
echo 2. Kokoro + GPT
echo 3. Hira Dia + Mistral
echo 4. Hira Dia + GPT
echo.
set /p combo_choice="Enter combination choice (1-4): "

if "%combo_choice%"=="1" set combo_name=kokoro_mistral
if "%combo_choice%"=="2" set combo_name=kokoro_gpt
if "%combo_choice%"=="3" set combo_name=hira_dia_mistral
if "%combo_choice%"=="4" set combo_name=hira_dia_gpt

if defined combo_name (
    echo.
    echo 🧪 Testing %combo_name% combination...
    python test_independent_combinations.py --combination %combo_name%
) else (
    echo Invalid choice.
)
pause
goto MENU

:CHECK_SERVICES
echo.
echo 🔍 Checking Independent Service Status...
echo.

REM Check each service
echo Checking STT Service (Port 8001)...
curl -s -o nul -w "STT Service: %%{http_code}\n" http://localhost:8001/health

echo Checking Kokoro TTS (Port 8011)...
curl -s -o nul -w "Kokoro TTS: %%{http_code}\n" http://localhost:8011/health

echo Checking Hira Dia TTS (Port 8012)...
curl -s -o nul -w "Hira Dia TTS: %%{http_code}\n" http://localhost:8012/health

echo Checking Mistral LLM (Port 8021)...
curl -s -o nul -w "Mistral LLM: %%{http_code}\n" http://localhost:8021/health

echo Checking GPT LLM (Port 8022)...
curl -s -o nul -w "GPT LLM: %%{http_code}\n" http://localhost:8022/health

echo.
echo Note: 200 = Service Available, 000 = Service Not Running
echo.
echo To start services:
echo   cd aws_microservices
echo   python enhanced_service_manager.py
echo.
pause
goto MENU

:EXIT
echo.
echo 👋 Goodbye!
echo.
echo Generated audio files can be found in:
echo   tests\audio_samples\tts_llm_combinations\
echo   tests\audio_samples\independent_combinations\
echo.
pause
