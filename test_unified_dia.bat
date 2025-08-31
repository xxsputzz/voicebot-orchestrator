@echo off
echo ========================================
echo Testing Unified Hira Dia TTS Service
echo ========================================
echo.
echo This test verifies the dual-engine Hira Dia service
echo supporting both Full Dia (quality) and 4-bit Dia (speed) engines.
echo.

cd /d "%~dp0"

echo Starting unified Hira Dia service test...
python tests\test_unified_dia_service.py

echo.
echo Test complete! Check the output above for results.
pause
