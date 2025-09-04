@echo off
echo 🧪 Voicebot Local Microservices Test
echo ====================================

echo.
echo Installing required packages...
pip install requests fastapi uvicorn aiohttp aiofiles

echo.
echo Running test suite...
python test_local_microservices.py

echo.
echo Test completed!
pause
