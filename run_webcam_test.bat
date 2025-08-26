@echo off
echo 🎓 CheatGPT3 Webcam Test Launcher
echo ================================
echo.
echo Choose test type:
echo 1. Simple webcam test (verify camera works)
echo 2. Full CheatGPT3 real-time test
echo.

set /p choice=Enter choice (1 or 2): 

if "%choice%"=="1" (
    echo.
    echo 📹 Starting simple webcam test...
    echo Press 'q' or ESC in the camera window to quit
    echo.
    C:/Users/admin/miniconda3/Scripts/conda.exe run -p "d:\CHEATGPT CAPSTONE\CheatGPT3\.conda" --no-capture-output python test_webcam_simple.py
) else if "%choice%"=="2" (
    echo.
    echo 🎯 Starting CheatGPT3 real-time test...
    echo This will load the complete detection pipeline...
    echo.
    echo Controls:
    echo - ESC or 'q': Quit
    echo - SPACE: Pause/Resume
    echo - 's': Save frame
    echo - 'r': Reset stats
    echo - 'h': Toggle help
    echo.
    pause
    C:/Users/admin/miniconda3/Scripts/conda.exe run -p "d:\CHEATGPT CAPSTONE\CheatGPT3\.conda" --no-capture-output python test_webcam_realtime.py
) else (
    echo Invalid choice. Please run again and choose 1 or 2.
)

echo.
echo Test complete! Check the output above for any issues.
pause
