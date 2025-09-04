@echo off
echo Emergency GPU Cleanup for Tortoise TTS
echo =====================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run emergency cleanup
python emergency_gpu_cleanup.py

echo.
echo Checking GPU status...
nvidia-smi

echo.
pause
