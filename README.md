<img width="1890" height="867" alt="image" src="https://github.com/user-attachments/assets/f360d7f8-9685-41fa-a811-5763e94e7eba" />
<img width="1899" height="869" alt="image" src="https://github.com/user-attachments/assets/b78daec1-f76e-4e76-8bff-321efc919e6b" />
<img width="1897" height="862" alt="image" src="https://github.com/user-attachments/assets/2b73c66a-3411-4262-843d-b992eee1ca69" />

<img width="1889" height="865" alt="image" src="https://github.com/user-attachments/assets/f491aa29-98ae-46f3-aa16-aa4abbfe1e66" />
<img width="1893" height="869" alt="image" src="https://github.com/user-attachments/assets/4ea93a4d-dbaa-4d77-b88e-6d5305393fe9" />
<img width="1902" height="871" alt="image" src="https://github.com/user-attachments/assets/65ea66b7-875c-4aab-b502-663118b1ec35" />






# CheatGPT3 - AI-Powered Exam Monitoring System

CheatGPT3 is an advanced AI-powered exam monitoring system that uses computer vision and machine learning to detect suspicious behaviors during online examinations. The system provides real-time monitoring, session recording, and comprehensive analytics with a YouTube-style interface.

## 🎯 Features

- **Real-time Behavior Detection**: Detects suspicious behaviors like looking around, leaning, hand gestures, and unauthorized device usage
- **Live Video Monitoring**: Real-time webcam feed with overlay detection markers
- **Session Recording**: Automatic video recording with synchronized event timelines
- **Analytics Dashboard**: YouTube-style analytics interface with video playback and event analysis
- **Event Deduplication**: Intelligent grouping of sustained behaviors to reduce redundancy
- **Multi-format Reports**: Generate PDF and JSON reports of detected events
- **Web Interface**: Modern Flask-based web application with real-time updates

## 📋 System Requirements

### Hardware Requirements
#### Minimum Requirements
- **CPU**: Intel i5-6400 / AMD Ryzen 5 1600 or equivalent (4 cores)
- **RAM**: 8GB DDR4 (16GB recommended for optimal performance)
- **Storage**: 10GB free space (SSD recommended for better I/O performance)
- **Camera**: USB 2.0 webcam (720p minimum, 1080p recommended)

#### Recommended Requirements
- **CPU**: Intel i7-8700K / AMD Ryzen 7 3700X or better (8+ cores)
- **RAM**: 16GB DDR4 or higher
- **GPU**: NVIDIA GTX 1060 6GB / AMD RX 580 8GB or better
- **Storage**: 20GB free space on SSD
- **Camera**: USB 3.0 webcam with 1080p @ 30fps

#### GPU Requirements for Acceleration
**NVIDIA GPUs (CUDA Support)**:
- GTX 1050 Ti or better (4GB+ VRAM recommended)
- RTX 20/30/40 series (optimal performance)
- Compute Capability 6.1+ required

**AMD GPUs (ROCm Support)**:
- RX 5700 XT or better
- RX 6000/7000 series (optimal performance)
- 8GB+ VRAM recommended for large models

### Software Requirements
- **Operating System**: Windows 10/11 (64-bit) - **Primary Support**
- **Python**: Version 3.8 to 3.11 (3.10 recommended)
- **Git**: Latest version for repository cloning
- **Visual Studio Build Tools**: Required for compiling native dependencies
- **CUDA Toolkit** (NVIDIA GPUs): Version 11.8 or 12.x
- **Windows C++ Redistributables**: Latest version

## 🚀 Complete Installation Guide

This guide will walk you through installing CheatGPT3 from scratch on a fresh system, including all external dependencies and requirements.

## 📦 Windows Prerequisites Installation

### Step 1: Install System Prerequisites

#### **1.1 Install Visual Studio Build Tools (Required)**

**Option A: Visual Studio Build Tools 2022 (Recommended)**
```powershell
# Download Visual Studio Build Tools 2022
# URL: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Run the installer and select:
# - C++ build tools
# - Windows 11 SDK (latest version)  
# - MSVC v143 - VS 2022 C++ x64/x86 build tools
# - CMake tools for C++
```

**Option B: Visual Studio Community (Full IDE)**
```powershell
# Download from: https://visualstudio.microsoft.com/vs/community/
# During installation, select:
# - Desktop development with C++
# - Python development (optional but helpful)
```

**Option C: Package Manager Installation**
```powershell
# Using Chocolatey (if installed)
choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools"

# Using winget
winget install Microsoft.VisualStudio.2022.BuildTools

# Verify installation
where cl
# Should show: C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\...\bin\Hostx64\x64\cl.exe
```

#### **1.2 Install Git for Windows**
```powershell
# Option A: Direct download
# URL: https://git-scm.com/download/win
# Use default settings during installation

# Option B: Package managers
# Chocolatey:
choco install git

# winget:
winget install Git.Git

# Verify installation
git --version
```

#### **1.3 Install Python 3.10**
```powershell
# Option A: Official Python installer (Recommended)
# Download from: https://www.python.org/downloads/windows/
# IMPORTANT: Check "Add Python to PATH" during installation

# Option B: Microsoft Store
# Search "Python 3.10" in Microsoft Store and install

# Option C: Package managers
# Chocolatey:
choco install python310

# winget:
winget install Python.Python.3.10

# Verify installation
python --version
pip --version

# If 'python' command not found, try:
py --version
py -m pip --version
```

#### **1.4 Install Windows C++ Redistributables**
```powershell
# Download and install Microsoft Visual C++ Redistributable
# URL: https://aka.ms/vs/17/release/vc_redist.x64.exe

# Or via winget:
winget install Microsoft.VCRedist.2015+.x64

# This is required for some Python packages with native dependencies
```

### Step 2: Windows GPU Acceleration Setup

#### **NVIDIA CUDA Setup (Recommended for NVIDIA GPUs)**

**2.1 Install NVIDIA Graphics Drivers**
```powershell
# Option A: NVIDIA GeForce Experience (Recommended)
# Download from: https://www.nvidia.com/en-us/geforce/geforce-experience/
# Automatically detects and installs latest drivers

# Option B: Manual driver download
# Go to: https://www.nvidia.com/drivers/
# Select your GPU model and download latest Game Ready Driver

# Option C: Windows Update
# Settings > Windows Update > Check for updates
# Often includes NVIDIA drivers

# Verify installation
nvidia-smi
```

**2.2 Install CUDA Toolkit 11.8**
```powershell
# Download CUDA 11.8 for Windows
# URL: https://developer.nvidia.com/cuda-11-8-0-download-archive
# Select: Windows > x86_64 > 11 > exe (local)

# Run the installer with these options:
# - Custom installation (Advanced)
# - Select: CUDA Toolkit, Documentation, Samples
# - Uncheck: Visual Studio Integration (if you don't need it)
# - Install to default location: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

# Verify installation
nvcc --version
# Should show: Cuda compilation tools, release 11.8

# Check environment variables (should be set automatically):
echo $env:CUDA_PATH
echo $env:PATH | Select-String CUDA
```

**2.3 Install cuDNN (Deep Learning Library)**
```powershell
# 1. Register and download cuDNN from:
# https://developer.nvidia.com/cudnn (requires free NVIDIA account)
# Download: cuDNN Library for Windows (x86) for CUDA 11.x

# 2. Extract cuDNN files
# Extract the downloaded zip file
# Copy files to CUDA installation directory:

# From extracted cuDNN folder:
# Copy bin\cudnn*.dll → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\
# Copy include\cudnn*.h → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include\
# Copy lib\x64\cudnn*.lib → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64\

# 3. Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**2.4 Test CUDA Installation**
```powershell
# Test with deviceQuery sample (if CUDA samples installed)
cd "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.8\1_Utilities\deviceQuery"
# Open deviceQuery.sln in Visual Studio and build, or use pre-built executable

# Quick Python test
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU device:', torch.cuda.get_device_name(0))
    print('GPU memory:', f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

#### **AMD Radeon GPU Setup (Windows)**

**Note:** AMD ROCm is primarily supported on Linux. For Windows AMD GPUs, the system will use CPU processing or DirectML (if available).

```powershell
# Install latest AMD Radeon drivers
# Download from: https://www.amd.com/en/support
# Or use AMD Radeon Software

# For DirectML support (experimental):
pip install torch-directml
# This provides some GPU acceleration for AMD GPUs on Windows
```

### Step 3: Clone the Repository

```powershell
# Open PowerShell or Command Prompt
# Navigate to your desired directory (e.g., D:\Projects)
cd D:\Projects

# Clone the repository
git clone https://github.com/ongjames-lav/cheatgptv3.git
cd cheatgptv3

# Verify repository structure
dir
# You should see folders like: cheatgpt, web_app, weights, etc.
```

### Step 4: Create Python Virtual Environment

#### **Option A: Using Conda (Recommended for GPU setups)**

**4.1 Install Miniconda for Windows**
```powershell
# Option A: Direct download
# URL: https://docs.conda.io/en/latest/miniconda.html
# Download: Miniconda3 Windows 64-bit installer
# Run the installer with default settings

# Option B: Using package managers
# Chocolatey:
choco install miniconda3

# winget:
winget install Anaconda.Miniconda3

# After installation, restart PowerShell or Command Prompt
```

**4.2 Create and Activate Conda Environment**
```powershell
# Create environment with Python 3.10
conda create -n cheatgpt python=3.10 -y

# Activate environment
conda activate cheatgpt

# Verify Python version
python --version
# Should show: Python 3.10.x

# Update conda and pip
conda update conda -y
python -m pip install --upgrade pip
```

#### **Option B: Using Python venv (Alternative)**

**4.1 Create Virtual Environment**
```powershell
# Navigate to project directory
cd cheatgptv3

# Create virtual environment
python -m venv cheatgpt_env

# Activate environment (PowerShell)
.\cheatgpt_env\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Alternative: Command Prompt activation
# cheatgpt_env\Scripts\activate.bat

# Verify activation (should show (cheatgpt_env) in prompt)
where python
# Should point to: .\cheatgpt_env\Scripts\python.exe
```

**4.2 Upgrade pip and setuptools**
```powershell
# Ensure latest pip and setuptools
python -m pip install --upgrade pip setuptools wheel
```

### Step 5: Install Python Dependencies

#### **5.1 Install PyTorch with GPU Support**

**For NVIDIA GPUs with CUDA 11.8 (Recommended)**
```powershell
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

**For NVIDIA GPUs with CUDA 12.1**
```powershell
# If you installed CUDA 12.1 instead
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

**For CPU-only Installation (Fallback)**
```powershell
# Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

**For AMD GPUs (Experimental)**
```powershell
# Install DirectML support for AMD GPUs on Windows
pip install torch-directml
pip install torchvision torchaudio

# Note: Limited support compared to CUDA
```

#### **5.2 Install Computer Vision Dependencies**

```powershell
# Install OpenCV with full support
pip install opencv-python opencv-contrib-python

# Install image processing libraries
pip install Pillow numpy scipy

# Install YOLO and object detection
pip install ultralytics

# Verify YOLO installation
python -c "from ultralytics import YOLO; print('YOLO imported successfully')"
```

#### **5.3 Install Web Application Dependencies**

```powershell
# Flask web framework and extensions
pip install flask flask-socketio

# Async support for SocketIO
pip install eventlet

# PDF report generation
pip install reportlab

# Environment variable management
pip install python-dotenv

# Note: sqlite3 is included with Python by default
```

#### **5.4 Install Machine Learning Dependencies**

```powershell
# Scientific computing
pip install numpy pandas scikit-learn

# Deep learning utilities (optional)
pip install tensorboard  # For training monitoring

# Additional ML libraries
pip install matplotlib seaborn  # For plotting and visualization
```

#### **5.5 Install Additional System Dependencies**

```powershell
# Video processing (if needed)
pip install imageio imageio-ffmpeg

# Performance monitoring
pip install psutil

# Development tools (optional)
pip install jupyter notebook ipython  # For development and testing
```

#### **5.6 One-Command Installation (All Dependencies)**

```powershell
# Install all dependencies at once
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && `
pip install opencv-python opencv-contrib-python ultralytics Pillow numpy scipy && `
pip install flask flask-socketio eventlet reportlab python-dotenv && `
pip install pandas scikit-learn matplotlib seaborn psutil imageio imageio-ffmpeg

# Note: Use ` (backtick) for line continuation in PowerShell
```

#### **5.7 Verify Complete Installation**

```powershell
# Create a comprehensive test script
python -c "
import sys
import platform
print('🔍 CheatGPT3 Installation Verification')
print('=' * 40)
print(f'Platform: {platform.platform()}')
print(f'Python: {sys.version}')
print()

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'   CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
        print(f'   CUDA version: {torch.version.cuda}')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')

try:
    import cv2
    print(f'✅ OpenCV: {cv2.__version__}')
except ImportError as e:
    print(f'❌ OpenCV import failed: {e}')

try:
    from ultralytics import YOLO
    print('✅ YOLO/Ultralytics imported successfully')
except ImportError as e:
    print(f'❌ YOLO import failed: {e}')

try:
    import flask
    from flask_socketio import SocketIO
    print('✅ Flask and SocketIO imported successfully')
except ImportError as e:
    print(f'❌ Flask/SocketIO import failed: {e}')

try:
    import numpy as np
    import pandas as pd
    import PIL
    print('✅ Scientific libraries (numpy, pandas, PIL) imported successfully')
except ImportError as e:
    print(f'❌ Scientific libraries import failed: {e}')

print()
print('✅ Installation verification complete!')
"
```

### Step 6: Download Model Weights and Setup

#### **6.1 Download YOLO Models**

```powershell
# Navigate to project directory
cd cheatgptv3

# Create weights directory if it doesn't exist
if (!(Test-Path "weights")) { New-Item -ItemType Directory -Path "weights" }

# Models will be downloaded automatically on first run, but you can pre-download:
python -c "
from ultralytics import YOLO
print('Downloading YOLO11 models...')
print('This may take a few minutes depending on your internet connection...')
model1 = YOLO('yolo11m.pt')  # Object detection model (~50MB)
model2 = YOLO('yolo11m-pose.pt')  # Pose detection model (~50MB)
print('✅ Models downloaded successfully!')
print('Models saved to:', model1.ckpt_path if hasattr(model1, 'ckpt_path') else 'default location')
"
```

#### **6.2 Environment Configuration**

```powershell
# Create .env file for configuration
# Check if .env.example exists, otherwise create new .env
if (Test-Path ".env.example") {
    Copy-Item ".env.example" ".env"
} else {
    New-Item -ItemType File -Path ".env"
}

# Edit .env file with notepad
notepad .env

# Or use PowerShell to create basic configuration
@"
# CheatGPT3 Configuration for Windows

# Performance settings
FORCE_CPU=false
DEBUG_ENGINE=true
USE_GPU=true

# Detection sensitivity (lower values = more sensitive)
LEAN_ANGLE_THRESH=12.0
HEAD_TURN_THRESH=15.0
POSE_CONFIDENCE_THRESH=0.25

# Behavior analysis
BEHAVIOR_REPEAT_WINDOW=5.0
ALERT_PERSIST_FRAMES=2

# Video recording settings
RECORD_VIDEO=true
VIDEO_FPS=30
VIDEO_RESOLUTION=1280x720

# Database settings (Windows paths)
DATABASE_PATH=web_app/cheatgpt.db
SESSION_DATABASE_PATH=web_app/cheatgpt_sessions.db

# Windows-specific settings
OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0
"@ | Out-File -FilePath ".env" -Encoding UTF8
```

**Manual .env Configuration (if needed):**
```ini
# Performance settings
FORCE_CPU=false  # Set to true if you don't have CUDA GPU
DEBUG_ENGINE=true  # Enable debug logging
USE_GPU=true  # Enable GPU acceleration

# Detection sensitivity (lower values = more sensitive)
LEAN_ANGLE_THRESH=12.0
HEAD_TURN_THRESH=15.0
POSE_CONFIDENCE_THRESH=0.25

# Behavior analysis
BEHAVIOR_REPEAT_WINDOW=5.0
ALERT_PERSIST_FRAMES=2

# Video recording settings
RECORD_VIDEO=true
VIDEO_FPS=30
VIDEO_RESOLUTION=1920x1080

# Database settings
DATABASE_PATH=web_app/cheatgpt.db
SESSION_DATABASE_PATH=web_app/cheatgpt_sessions.db

# Windows camera optimization
OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0
```

### Step 7: Initialize Database and Test Setup

#### **7.1 Initialize Database**
```powershell
# Navigate to web application directory
cd web_app

# Test database initialization
python -c "
import sys
import os
sys.path.append('..')
try:
    from web_app.db_manager import db
    print('✅ Database manager imported successfully!')
    print('Database will be created automatically on first app run.')
except Exception as e:
    print(f'⚠️  Database manager import issue: {e}')
    print('This is normal - database will be created when app starts.')
"
```

#### **7.2 Test Core Components**
```powershell
# Test the core detection engine
cd ..  # Back to root directory
python -c "
import sys
import os
sys.path.append('.')
print('Testing core engine import...')
try:
    from cheatgpt.engine import Engine
    print('✅ Core engine imported successfully!')
    
    # Test engine initialization
    print('Testing engine initialization...')
    engine = Engine()
    print('✅ Engine initialized successfully!')
    
except Exception as e:
    print(f'❌ Engine test failed: {e}')
    print('This might be due to missing dependencies or GPU issues.')
"

# Test webcam access
python -c "
import cv2
print('Testing webcam access...')
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print('✅ Webcam access successful!')
        print(f'Frame resolution: {frame.shape[1]}x{frame.shape[0]}')
        print(f'Frame channels: {frame.shape[2]}')
    else:
        print('❌ Could not read from webcam')
    cap.release()
else:
    print('❌ Could not open webcam')
    print('Check if camera is being used by another application')
    print('Or try different camera index: cv2.VideoCapture(1), etc.')
"
```

#### **7.3 Windows Firewall and Antivirus Configuration**
```powershell
# Allow Python through Windows Firewall
Write-Host "Configuring Windows Firewall..."

# Add firewall rules for Python and the web application
New-NetFirewallRule -DisplayName "Python CheatGPT3" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
New-NetFirewallRule -DisplayName "Python CheatGPT3 Outbound" -Direction Outbound -Protocol TCP -LocalPort 5000 -Action Allow

Write-Host "✅ Firewall rules added for port 5000"
Write-Host "Note: You may need to run PowerShell as Administrator for firewall commands"
```

**Windows Defender/Antivirus Notes:**
- Some antivirus software may flag Python AI applications
- Add the project folder to your antivirus exclusions if needed
- Add Python.exe to exclusions if necessary

#### Key Environment Variables
```bash
# Performance settings
FORCE_CPU=false  # Set to true if you don't have CUDA GPU
DEBUG_ENGINE=true  # Enable debug logging

# Detection sensitivity (lower values = more sensitive)
LEAN_ANGLE_THRESH=12.0
HEAD_TURN_THRESH=15.0
POSE_CONFIDENCE_THRESH=0.25

# Behavior analysis
BEHAVIOR_REPEAT_WINDOW=5.0
ALERT_PERSIST_FRAMES=2
```

### Step 8: Final Testing and Launch

#### **8.1 Test Installation Components**

**Test Available Scripts**
```powershell
# Check if test scripts exist and run them
if (Test-Path "test_engine_hybrid.py") {
    Write-Host "Running hybrid engine test..."
    python test_engine_hybrid.py
}

if (Test-Path "demo_hybrid_engine.py") {
    Write-Host "Running demo..."
    python demo_hybrid_engine.py
}

# Basic engine test
python -c "
from cheatgpt.engine import Engine
import cv2
print('🔧 Testing engine initialization...')
try:
    engine = Engine()
    print('✅ Engine initialized successfully!')
    print('Engine components loaded:')
    if hasattr(engine, 'yolo_detector'):
        print('  - YOLO detector: ✅')
    if hasattr(engine, 'pose_detector'):
        print('  - Pose detector: ✅')
    if hasattr(engine, 'device'):
        print(f'  - Using device: {engine.device}')
except Exception as e:
    print(f'❌ Engine initialization failed: {e}')
"
```

**Test Webcam Integration**
```powershell
# Comprehensive webcam and detection test
python -c "
import cv2
import torch
print('🎥 Testing webcam and detection pipeline...')

# Test webcam access
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('❌ Primary webcam not accessible, trying alternative...')
    cap = cv2.VideoCapture(1)

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f'✅ Webcam working - Resolution: {frame.shape[1]}x{frame.shape[0]}')
        
        # Test basic detection
        try:
            from cheatgpt.engine import Engine
            engine = Engine()
            
            print('🔍 Testing detection pipeline...')
            # This may take a moment on first run
            results = engine.process_frame(frame)
            print('✅ Detection pipeline working successfully!')
            
        except Exception as e:
            print(f'⚠️  Detection test failed: {e}')
    else:
        print('❌ Could not read from webcam')
    cap.release()
else:
    print('❌ No webcam accessible')
    print('Please check:')
    print('  - Camera privacy settings in Windows')
    print('  - Camera not in use by another application')
    print('  - Camera drivers are installed')
"
```

#### **8.2 Launch Web Application**

```powershell
# Navigate to web app directory
cd web_app

# Start the web application
Write-Host "🚀 Starting CheatGPT3 Web Application..."
Write-Host "This may take a moment to initialize all components..."

python app.py

# You should see output like:
# * Running on http://127.0.0.1:5000
# * Restarting with stat
# * Debugger is active!
```

**Alternative: Run with specific configuration**
```powershell
# Set environment variables and run
$env:FLASK_ENV="development"
$env:FLASK_DEBUG="1"
python app.py
```

**Access the Web Interface:**
1. **Open your web browser** (Chrome, Edge, or Firefox recommended)
2. **Navigate to**: `http://localhost:5000` or `http://127.0.0.1:5000`
3. **You should see** the CheatGPT3 dashboard with screenshots and interface
4. **Click "Start Camera"** to begin live monitoring
5. **Grant camera permissions** when Windows prompts you
6. **Test the system** by moving around in front of the camera

#### **8.3 Windows System Verification**

**Complete System Check**
```powershell
# Run comprehensive system verification
python -c "
import sys
import platform
import torch
import cv2
import numpy as np
from pathlib import Path
import psutil
import os

print('🔍 CheatGPT3 Windows System Check')
print('=' * 50)

# System Information
print(f'OS: {platform.platform()}')
print(f'Architecture: {platform.architecture()[0]}')
print(f'Processor: {platform.processor()}')
print(f'Python: {sys.version.split()[0]}')
print(f'Python Path: {sys.executable}')
print()

# Memory and CPU
memory = psutil.virtual_memory()
print(f'RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available')
print(f'CPU Cores: {psutil.cpu_count()} logical, {psutil.cpu_count(logical=False)} physical')
print()

# Core Libraries
try:
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'   CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
        print(f'   CUDA version: {torch.version.cuda}')
        print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
except Exception as e:
    print(f'❌ PyTorch issue: {e}')

try:
    print(f'✅ OpenCV: {cv2.__version__}')
    # Test camera backends
    backends = []
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]:
        if cv2.videoio_registry.hasBackend(backend):
            backends.append(cv2.videoio_registry.getBackendName(backend))
    print(f'   Available backends: {', '.join(backends)}')
except Exception as e:
    print(f'❌ OpenCV issue: {e}')

try:
    from ultralytics import YOLO
    print('✅ YOLO/Ultralytics available')
except Exception as e:
    print(f'❌ YOLO issue: {e}')

try:
    import flask
    from flask_socketio import SocketIO
    print('✅ Flask and SocketIO available')
except Exception as e:
    print(f'❌ Flask/SocketIO issue: {e}')

# Project Structure
print()
print('📁 Project Structure:')
required_paths = [
    'cheatgpt/',
    'cheatgpt/engine.py',
    'web_app/',
    'web_app/app.py',
    'weights/',
    '.env'
]

for path in required_paths:
    path_obj = Path(path)
    if path_obj.exists():
        if path_obj.is_file():
            size = path_obj.stat().st_size
            print(f'✅ {path} ({size} bytes)')
        else:
            print(f'✅ {path}/')
    else:
        print(f'❌ {path} missing')

# Camera Test
print()
print('📷 Camera Test:')
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f'✅ Primary camera working - {frame.shape[1]}x{frame.shape[0]}')
    else:
        print('⚠️  Camera detected but cannot capture frames')
    cap.release()
else:
    print('❌ Primary camera not accessible')

print()
print('🎯 CheatGPT3 Windows System Check Complete!')
print('If most checks show ✅, your installation should work properly.')
"
```

#### **8.4 Quick Launch Script**

**Create a batch file for easy launching:**
```powershell
# Create launch script
@"
@echo off
echo Starting CheatGPT3...
cd /d "%~dp0"

REM Activate conda environment if using conda
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" cheatgpt
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" cheatgpt
)

REM Or activate venv if using venv
if exist "cheatgpt_env\Scripts\activate.bat" (
    call cheatgpt_env\Scripts\activate.bat
)

cd web_app
python app.py
pause
"@ | Out-File -FilePath "start_cheatgpt.bat" -Encoding ASCII

Write-Host "✅ Created start_cheatgpt.bat"
Write-Host "Double-click this file to start the application easily!"
```

## 🏃‍♂️ Quick Start

### Running the Web Application

1. **Activate Environment**:
   ```bash
   conda activate cheatgpt  # or source cheatgpt_env/bin/activate
   ```

2. **Navigate to Web App**:
   ```bash
   cd web_app
   ```

3. **Start the Server**:
   ```bash
   python app.py
   ```

4. **Access Web Interface**:
   Open `http://localhost:5000` in your browser

### Running Standalone Detection

1. **Activate Environment**:
   ```bash
   conda activate cheatgpt
   ```

2. **Run Detection Script**:
   ```bash
   python run_enhanced_detection.py
   ```

3. **Controls**:
   - `ESC` or `q`: Quit
   - `s`: Start/stop recording
   - `SPACE`: Pause/resume

## 📱 Using the Web Interface

### Live Monitoring
1. Navigate to the main dashboard (`http://localhost:5000`)
2. Click **"Start Camera"** to begin live monitoring
3. Grant camera permissions when prompted
4. Monitor real-time detection results
5. Click **"Stop Camera"** to end session

### Analytics Dashboard
1. Navigate to **"Analytics"** from the main menu
2. Browse recorded sessions in YouTube-style grid
3. Click on any session to view detailed analytics
4. Use the video player to review events with timestamps
5. Generate reports using the **"Generate Report"** button

### Session Management
1. All sessions are automatically saved with timestamps
2. Videos are stored in `web_app/videos/` directory
3. Event data is stored in SQLite database
4. Reports can be exported as PDF or JSON

## 🛠️ Configuration

### Performance Optimization

#### GPU Configuration
```bash
# Check if CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# For NVIDIA GPUs, ensure CUDA drivers are installed
# Download from: https://developer.nvidia.com/cuda-downloads
```

#### CPU-Only Mode
```bash
# In .env file:
FORCE_CPU=true

# Or set environment variable:
export FORCE_CPU=true  # Linux/macOS
set FORCE_CPU=true     # Windows
```

### Detection Sensitivity

Adjust detection thresholds in `.env`:

```bash
# More sensitive detection (lower values)
LEAN_ANGLE_THRESH=8.0      # Default: 12.0
HEAD_TURN_THRESH=10.0      # Default: 15.0
POSE_CONFIDENCE_THRESH=0.2 # Default: 0.25

# Less sensitive detection (higher values)
LEAN_ANGLE_THRESH=20.0
HEAD_TURN_THRESH=25.0
POSE_CONFIDENCE_THRESH=0.4
```

## 📁 Project Structure

```
cheatgptv3/
├── cheatgpt/                 # Core detection engine
│   ├── detectors/           # YOLO and pose detectors
│   ├── engine.py           # Main detection engine
│   └── requirements.txt    # Core dependencies
├── web_app/                 # Flask web application
│   ├── app.py              # Main web server
│   ├── db_manager.py       # Database management
│   ├── templates/          # HTML templates
│   ├── static/             # CSS, JS, images
│   └── videos/             # Recorded session videos
├── weights/                 # Model weight files
├── config/                  # Configuration files
├── data/                    # Training data and datasets
├── logs/                    # Application logs
└── .env                     # Environment configuration
```

## 🔧 Comprehensive Troubleshooting Guide

### Installation Issues

#### 1. Visual Studio Build Tools / C++ Compiler Issues

**Missing Visual Studio Build Tools:**
```powershell
# Error messages you might see:
# "Microsoft Visual C++ 14.0 is required"
# "error: Microsoft Visual C++ 14.0 or greater is required"
# "building wheel for [package] failed"

# Solution 1: Install Visual Studio Build Tools 2022
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
# Run installer and select "C++ build tools"

# Solution 2: Install Visual Studio Community (full IDE)
# Download from: https://visualstudio.microsoft.com/vs/community/
# Select "Desktop development with C++" during installation

# Solution 3: Package manager installation
choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools"
# OR
winget install Microsoft.VisualStudio.2022.BuildTools

# Verify installation:
where cl
# Should output: C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\...\cl.exe

# If still not working, restart PowerShell/Command Prompt
```

**Build Tools Detection Issues:**
```powershell
# Sometimes Python can't find the build tools
# Set environment variables manually:

# Find your VS installation path
$vsPath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools"
$vsPath2 = "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community"

# Add to PATH temporarily
$env:PATH += ";$vsPath\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64"
$env:PATH += ";$vsPath\Common7\IDE"

# Or set VS environment variables
$env:VS160COMNTOOLS = "$vsPath\Common7\Tools\"
```

#### 2. CUDA Installation Issues

**CUDA Not Detected:**
```bash
# Check NVIDIA driver installation
nvidia-smi

# If command not found, install/reinstall NVIDIA drivers:
# Windows: Download from https://www.nvidia.com/drivers/
# Linux Ubuntu:
sudo apt purge nvidia* libnvidia*
sudo apt autoremove
sudo apt install nvidia-driver-535  # or latest stable version
sudo reboot

# Check CUDA installation
nvcc --version

# If CUDA not found, add to PATH:
# Windows: Add to System Environment Variables:
#   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
#   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp

# Linux: Add to ~/.bashrc
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source ~/.bashrc
```

**PyTorch CUDA Version Mismatch:**
```bash
# Check current PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Check system CUDA version
nvcc --version

# Reinstall matching PyTorch version:
pip uninstall torch torchvision torchaudio
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 3. AMD ROCm Issues (Linux)

**ROCm Not Working:**
```bash
# Check AMD GPU detection
lspci | grep -i amd

# Check ROCm installation
rocm-smi
rocminfo

# If not working, reinstall ROCm:
sudo apt remove rocm-dev rocm-libs
sudo apt autoremove
sudo apt install rocm-dev rocm-libs rocm-utils

# Add user to groups
sudo usermod -a -G render,video $USER
# Logout and login again

# Test ROCm with PyTorch
python -c "
import torch
print('ROCm available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device name:', torch.cuda.get_device_name(0))
"
```

#### 4. Python Environment Issues

**Wrong Python Version:**
```bash
# Check current Python version
python --version

# If not 3.10, install Python 3.10:
# Windows: Download from python.org or use winget
winget install Python.Python.3.10

# Linux Ubuntu:
sudo apt install python3.10 python3.10-venv python3.10-dev

# macOS:
brew install python@3.10

# Create virtual environment with specific Python version
python3.10 -m venv cheatgpt_env
```

**Virtual Environment Issues:**
```bash
# Cannot activate virtual environment
# Windows PowerShell execution policy issue:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Corrupted virtual environment - recreate:
rm -rf cheatgpt_env  # or delete folder on Windows
python -m venv cheatgpt_env
# Reactivate and reinstall dependencies
```

#### 5. Package Installation Issues

**Pip Install Failures:**
```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Clear pip cache
pip cache purge

# Install with verbose output to see errors
pip install torch --verbose

# Use conda instead of pip for problematic packages
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**OpenCV Installation Problems:**
```bash
# Uninstall and reinstall OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python

# Linux: Install system dependencies
sudo apt install -y python3-opencv
sudo apt install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libglib2.0-0

# macOS: Install via homebrew
brew install opencv
```

### Runtime Issues

#### 6. Windows Camera Access Issues

**Camera Not Detected:**
```powershell
# Test camera access with different methods
python -c "
import cv2
print('Testing camera access...')

# Try different camera indices
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f'✅ Camera {i} working - Resolution: {frame.shape[1]}x{frame.shape[0]}')
        else:
            print(f'⚠️  Camera {i} detected but cannot read frames')
        cap.release()
    else:
        print(f'❌ Camera {i} not accessible')
"

# Try different backends
python -c "
import cv2
backends = [
    (cv2.CAP_DSHOW, 'DirectShow'),
    (cv2.CAP_MSMF, 'Media Foundation'),
    (cv2.CAP_V4L2, 'Video4Linux')
]

for backend_id, backend_name in backends:
    try:
        cap = cv2.VideoCapture(0, backend_id)
        if cap.isOpened():
            ret, frame = cap.read()
            print(f'✅ {backend_name} backend working' if ret else f'⚠️  {backend_name} backend detected but no frames')
            cap.release()
        else:
            print(f'❌ {backend_name} backend not working')
    except:
        print(f'❌ {backend_name} backend not available')
"
```

**Windows Camera Permission Issues:**
```powershell
# Check Windows camera privacy settings
Write-Host "Checking Windows camera permissions..."

# Open Windows camera settings
start ms-settings:privacy-webcam

# Manual steps:
# 1. Go to Settings > Privacy & Security > Camera
# 2. Make sure "Camera access" is ON
# 3. Make sure "Let apps access your camera" is ON
# 4. Scroll down and make sure "Let desktop apps access your camera" is ON

# Check if camera is being used by another app
Get-Process | Where-Object {$_.ProcessName -like "*camera*" -or $_.ProcessName -like "*skype*" -or $_.ProcessName -like "*teams*" -or $_.ProcessName -like "*zoom*"}

# Kill processes that might be using camera (be careful!)
# Stop-Process -Name "Skype" -Force
# Stop-Process -Name "Teams" -Force
```

**Camera Driver Issues:**
```powershell
# Check camera in Device Manager
devmgmt.msc

# Update camera drivers
# 1. Open Device Manager
# 2. Expand "Cameras" or "Imaging devices"
# 3. Right-click your camera
# 4. Select "Update driver"
# 5. Choose "Search automatically for drivers"

# Alternative: Use Windows Update
# Settings > Windows Update > Check for updates

# For integrated cameras, also check:
# Settings > Windows Update > View optional updates > Driver updates
```

**Camera Hardware Issues:**
```powershell
# Test camera with Windows Camera app
start microsoft.windows.camera:

# If Windows Camera app doesn't work, the issue is hardware/driver related
# If it works but CheatGPT3 doesn't, it's a software configuration issue

# Check USB connections for external cameras
# Try different USB ports
# Try USB 2.0 ports instead of USB 3.0 (sometimes more compatible)
```

#### 7. Windows Web Application Issues

**Port Already in Use:**
```powershell
# Find what's using port 5000
netstat -ano | findstr :5000

# Kill the process using the port
# Replace <PID> with the actual Process ID from above command
taskkill /PID <PID> /F

# Alternative: Use PowerShell to find and kill
$processId = (Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue).OwningProcess
if ($processId) {
    Stop-Process -Id $processId -Force
    Write-Host "Killed process using port 5000"
} else {
    Write-Host "No process found using port 5000"
}

# Or change port in web_app/app.py:
# Find the line: app.run(host='0.0.0.0', port=5000, debug=True)
# Change to: app.run(host='0.0.0.0', port=5001, debug=True)
```

**Windows Firewall Blocking:**
```powershell
# Add firewall exception for Python
New-NetFirewallRule -DisplayName "CheatGPT3 Python" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
New-NetFirewallRule -DisplayName "CheatGPT3 Python Outbound" -Direction Outbound -Protocol TCP -LocalPort 5000 -Action Allow

# Alternative: Disable Windows Firewall temporarily (NOT RECOMMENDED for production)
# Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False

# Re-enable firewall later:
# Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True
```

**SocketIO Connection Issues:**
```powershell
# Install specific compatible versions
pip uninstall flask-socketio python-socketio python-engineio -y
pip install flask-socketio==5.3.6 python-socketio==5.8.0 python-engineio==4.7.1

# If still having issues, try alternative approach:
pip install flask-socketio==4.3.4 python-socketio==4.6.1 python-engineio==3.14.2

# Test SocketIO connectivity
python -c "
from flask import Flask
from flask_socketio import SocketIO
import eventlet

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

@app.route('/')
def index():
    return 'SocketIO Test Server'

print('Starting test SocketIO server on http://localhost:5001')
print('Press Ctrl+C to stop')
socketio.run(app, host='0.0.0.0', port=5001, debug=True)
"

# Open browser to http://localhost:5001 to test
```

**Browser Compatibility Issues:**
```powershell
# Test in different browsers
# Recommended order: Chrome > Edge > Firefox

# Clear browser cache and cookies
# Chrome: Ctrl+Shift+Delete
# Edge: Ctrl+Shift+Delete  
# Firefox: Ctrl+Shift+Delete

# Check browser console for errors:
# Press F12 > Console tab > look for red error messages

# Common browser issues:
# 1. WebSocket connections blocked by corporate firewall
# 2. Browser security settings too strict
# 3. Ad blockers interfering with SocketIO
# 4. HTTPS/HTTP mixed content issues
```

#### 8. GPU Memory Issues

**CUDA Out of Memory:**
```bash
# Check GPU memory usage
nvidia-smi

# In .env file, force CPU mode temporarily:
FORCE_CPU=true

# Or reduce batch size in detection code
# Edit cheatgpt/engine.py and reduce model inference batch size

# Clear GPU cache in Python:
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU cache cleared')
"
```

#### 9. Model Loading Issues

**YOLO Model Download Failures:**
```bash
# Manually download models
mkdir -p weights
cd weights

# Download manually:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11m.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11m-pose.pt

# Or use Python:
python -c "
from ultralytics import YOLO
model = YOLO('yolo11m.pt')  # Will download if not exists
pose_model = YOLO('yolo11m-pose.pt')
print('Models downloaded successfully')
"
```

#### 10. Database Issues

**Database Corruption:**
```bash
# Backup and recreate database
cd web_app
cp cheatgpt.db cheatgpt.db.backup  # Backup if needed
rm cheatgpt.db cheatgpt_sessions.db

# Clear WAL files
rm -f *.db-wal *.db-shm

# Restart application - database will be recreated
python app.py
```

### Performance Optimization

#### 11. Low Performance / FPS Issues

**GPU Acceleration Not Working:**
```bash
# Verify GPU is being used
python -c "
import torch
from ultralytics import YOLO
print('CUDA available:', torch.cuda.is_available())
model = YOLO('yolo11m.pt')
print('Model device:', next(model.model.parameters()).device)
"

# Force GPU usage in engine
# Edit cheatgpt/engine.py and ensure:
# self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**System Resource Issues:**
```bash
# Monitor system resources
# Windows: Task Manager > Performance
# Linux: htop or top
# macOS: Activity Monitor

# Close unnecessary applications
# Reduce video resolution in .env:
VIDEO_RESOLUTION=1280x720  # Instead of 1920x1080

# Reduce detection frequency
DETECTION_INTERVAL=2  # Process every 2nd frame
```

### Debug and Logging

#### 12. Enable Debug Mode

```bash
# Create detailed .env configuration
cat > .env << EOF
# Debug settings
DEBUG_ENGINE=true
DEBUG_POSE=true
DEBUG_POLICY=true
DEBUG_VIDEO=true
VERBOSE_LOGGING=true

# Performance settings
FORCE_CPU=false
USE_GPU=true
DETECTION_INTERVAL=1

# Detection thresholds
LEAN_ANGLE_THRESH=12.0
HEAD_TURN_THRESH=15.0
POSE_CONFIDENCE_THRESH=0.25
EOF

# Check logs directory
mkdir -p logs
ls -la logs/
```

#### 13. System Information for Support

```bash
# Generate system report for troubleshooting
python -c "
import sys
import platform
import torch
import cv2
import numpy as np
from pathlib import Path

print('=== CheatGPT3 System Report ===')
print(f'Date: {__import__('datetime').datetime.now()}')
print(f'Platform: {platform.platform()}')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'NumPy: {np.__version__}')

if torch.cuda.is_available():
    print(f'CUDA: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('CUDA: Not available')

# Check critical files
critical_files = [
    'cheatgpt/engine.py',
    'web_app/app.py',
    'weights/',
    '.env'
]

for file_path in critical_files:
    path = Path(file_path)
    status = '✅' if path.exists() else '❌'
    print(f'{status} {file_path}')

print('=== End Report ===')
"
```

**Getting Help:**
1. Run the system report above
2. Check the `logs/` directory for error messages
3. Include both when reporting issues
4. Specify your OS, GPU model, and Python version

### Quick Fix Commands

```bash
# Complete environment reset (nuclear option)
conda deactivate
conda remove -n cheatgpt --all -y
conda create -n cheatgpt python=3.10 -y
conda activate cheatgpt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python flask flask-socketio eventlet reportlab python-dotenv

# GPU memory cleanup
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Database reset
cd web_app && rm -f *.db *.db-* && cd ..

# Clear all caches
pip cache purge
conda clean --all -y
```

## 📖 Additional Documentation

- [Engine Implementation](ENGINE_COMPLETE.md) - Core detection engine details
- [Hotspot Overlay Guide](HOTSPOT_OVERLAY_GUIDE.md) - Event visualization system
- [Pose Detector](POSE_DETECTOR.md) - Pose detection implementation
- [Video Recording Guide](VIDEO_RECORDING_GUIDE.md) - Recording system details
- [Policy Rules](POLICY_RULES.md) - Behavior detection rules

## 🤝 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review log files in `logs/` directory
3. Ensure all dependencies are correctly installed
4. Verify camera permissions and hardware compatibility

## 📄 License

This project is for educational and research purposes. Please ensure compliance with privacy laws and institutional policies when using this system for exam monitoring.

## 🔄 Updates

To update the system:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r cheatgpt/requirements.txt

# Restart the application
cd web_app
python app.py
```

---

**CheatGPT3** - Advanced AI-Powered Exam Monitoring System
