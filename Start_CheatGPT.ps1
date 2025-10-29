# CheatGPT System Launcher (PowerShell Version)
# Run with: powershell -ExecutionPolicy Bypass -File Start_CheatGPT.ps1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   CheatGPT System Launcher v3.0" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to project directory
Set-Location "D:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3"

Write-Host "[1/3] Activating conda environment..." -ForegroundColor Yellow
Write-Host ""

# Initialize conda for PowerShell
$condaPath = "D:\Miniconda3"

if (Test-Path "$condaPath\Scripts\conda.exe") {
    Write-Host "Found conda at: $condaPath" -ForegroundColor Green
    
    # Initialize conda for PowerShell
    (& "$condaPath\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression
    
    # Activate environment
    conda activate cheatgpt
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] Environment activated!" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Failed to activate 'cheatgpt' environment" -ForegroundColor Red
        Write-Host "Please ensure the environment exists." -ForegroundColor Red
        pause
        exit 1
    }
} else {
    Write-Host "[ERROR] Conda not found at $condaPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run manually:" -ForegroundColor Yellow
    Write-Host "  1. Open Anaconda Prompt or PowerShell" -ForegroundColor White
    Write-Host "  2. Run: conda activate cheatgpt" -ForegroundColor White
    Write-Host "  3. Run: cd 'D:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3\web_app'" -ForegroundColor White
    Write-Host "  4. Run: python app.py" -ForegroundColor White
    Write-Host ""
    pause
    exit 1
}

Write-Host ""
Write-Host "[2/3] Checking Python..." -ForegroundColor Yellow
python --version
Write-Host ""

Write-Host "[3/3] Starting CheatGPT web server..." -ForegroundColor Yellow
Set-Location "web_app"
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  System is starting..." -ForegroundColor Green
Write-Host "  Please wait for the server to load..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Once running, the system will be available at:" -ForegroundColor White
Write-Host "  http://localhost:5000" -ForegroundColor Green
Write-Host ""
Write-Host "Browser will open automatically in 25 seconds..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Open browser after 25 seconds in background (enough time for full GPU initialization)
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 25
    Start-Process "http://localhost:5000"
} | Out-Null

# Start Flask application (this keeps the window open and shows logs)
python app.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Red
Write-Host "   CheatGPT System Stopped" -ForegroundColor Red
Write-Host "========================================" -ForegroundColor Red
Write-Host ""
pause
