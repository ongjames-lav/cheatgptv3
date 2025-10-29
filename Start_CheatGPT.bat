@echo off
REM ========================================
REM CheatGPT System Launcher
REM Automatically starts the complete system
REM ========================================

title CheatGPT System - Starting...

echo.
echo ========================================
echo    CheatGPT System Launcher v3.0
echo ========================================
echo.

REM Change to the project directory
cd /d "D:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3"

echo [1/3] Initializing conda environment...
echo.

REM Initialize conda from known location
IF EXIST "D:\Miniconda3\Scripts\activate.bat" (
    CALL "D:\Miniconda3\Scripts\activate.bat" "D:\Miniconda3"
) ELSE IF EXIST "D:\Miniconda3\condabin\conda.bat" (
    CALL "D:\Miniconda3\condabin\conda.bat" activate cheatgpt
    goto :conda_activated
) ELSE (
    echo [ERROR] Conda not found at D:\Miniconda3
    echo Please update the path in this script.
    pause
    exit /b 1
)

REM Activate the environment
CALL conda activate cheatgpt

:conda_activated
IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] Failed to activate 'cheatgpt' environment
    echo Please ensure the environment exists.
    echo.
    pause
    exit /b 1
)

echo [SUCCESS] Environment activated!
echo.
echo [2/3] Checking Python...
python --version
echo.

echo [3/3] Starting CheatGPT web server...
cd web_app
echo.
echo ========================================
echo   System is starting...
echo   Please wait for the server to load...
echo ========================================
echo.
echo Once running, the system will be available at:
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Open browser after 25 seconds (giving time for full GPU initialization)
start "" cmd /c "timeout /t 25 /nobreak >nul && start http://localhost:5000"

REM Start the Flask application (this keeps the window open)
python app.py

REM If the app stops or crashes
echo.
echo ========================================
echo   CheatGPT System Stopped
echo ========================================
echo.
pause
