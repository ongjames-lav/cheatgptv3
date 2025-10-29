<img width="1890" height="867" alt="image" src="https://github.com/user-attachments/assets/f360d7f8-9685-41fa-a811-5763e94e7eba" />
<img width="1899" height="869" alt="image" src="https://github.com/user-attachments/assets/b78daec1-f76e-4e76-8bff-321efc919e6b" />
<img width="1897" height="862" alt="image" src="https://github.com/user-attachments/assets/2b73c66a-3411-4262-843d-b992eee1ca69" />

<img width="1889" height="865" alt="image" src="https://github.com/user-attachments/assets/f491aa29-98ae-46f3-aa16-aa4abbfe1e66" />
<img width="1893" height="869" alt="image" src="https://github.com/user-attachments/assets/4ea93a4d-dbaa-4d77-b88e-6d5305393fe9" />
<img width="1902" height="871" alt="image" src="https://github.com/user-attachments/assets/65ea66b7-875c-4aab-b502-663118b1ec35" />






# CheatGPT3 - AI-Powered Exam Monitoring System

CheatGPT3 is an advanced AI-powered exam monitoring system that uses computer vision, deep learning (YOLO11), and behavioral analysis to detect cheating gestures during examinations. The system provides real-time monitoring, session recording, and comprehensive analytics with a modern web interface.

## 🎯 Features

- **Real-time Cheating Detection**: Detects phone usage, sustained head turning, and suspicious hand activity
- **Multi-Modal Detection**: Combines YOLO11 object detection, MediaPipe pose estimation, and motion analysis
- **LSTM Temporal Analysis**: Context-aware classification to reduce false positives (83.87% accuracy)
- **Live Video Monitoring**: Real-time webcam feed with detection overlays and bounding boxes
- **Session Recording**: Automatic video recording with synchronized event timelines
- **Analytics Dashboard**: YouTube-style interface with video playback, heatmaps, and event analysis
- **Event Deduplication**: Intelligent 3-second window to prevent event spam
- **Comprehensive Reports**: Generate PDF reports with behavior scores and recommendations
- **GPU Acceleration**: CUDA support for real-time processing (60+ FPS on GPU)

## 📋 System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: Minimum 8GB (16GB recommended for smooth operation)
- **GPU**: NVIDIA GPU with 4GB+ VRAM (GTX 1650 Ti or better for real-time processing)
  - Optional: System works on CPU but at reduced speed (15-20 FPS)
- **Storage**: At least 10GB free space (for videos and models)
- **Camera**: Webcam or external USB camera (720p minimum, 1080p recommended)

### Software Requirements
- **Operating System**: Windows 10/11 (64-bit)
- **Python**: Version 3.11.x (3.11.13 tested and recommended)
- **Conda/Miniconda**: For environment management
- **Git**: For version control and updates
- **CUDA Toolkit**: 11.8 or 12.x (if using GPU acceleration)

## 🚀 Quick Start (Desktop Launcher)

### ⚡ Easiest Method - One-Click Desktop Launcher

1. **Convert favicon to icon** (one-time setup):
   ```powershell
   conda activate cheatgpt
   cd "D:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3"
   python convert_favicon.py
   ```

2. **Create desktop shortcut**:
   - Double-click `Create_Desktop_Shortcut.vbs`
   - Choose **YES** for PowerShell launcher (recommended)
   - A "CheatGPT System" shortcut will appear on your desktop

3. **Launch the system**:
   - Double-click the **"CheatGPT System"** icon on your desktop
   - Wait 25 seconds for GPU models to load
   - Browser opens automatically to `http://localhost:5000`

**That's it!** The system handles everything automatically:
- ✅ Activates conda environment
- ✅ Loads YOLO11 models on GPU
- ✅ Starts Flask web server
- ✅ Opens browser when ready

## 📦 Full Installation Guide

### Step 1: Install Prerequisites

#### 1.1 Install Miniconda
```powershell
# Download Miniconda from: https://docs.conda.io/en/latest/miniconda.html
# Install to default location (D:\Miniconda3 or C:\Users\<Username>\Miniconda3)
# Check "Add to PATH" during installation
```

#### 1.2 Install Git (if not installed)
```powershell
# Download from: https://git-scm.com/download/win
# Use default settings during installation
```

#### 1.3 Install CUDA Toolkit (for GPU acceleration)
```powershell
# Download CUDA 11.8 from: https://developer.nvidia.com/cuda-11-8-0-download-archive
# Or CUDA 12.x from: https://developer.nvidia.com/cuda-downloads
# Verify installation:
nvidia-smi
```

### Step 2: Clone Repository

```powershell
# Open PowerShell or Command Prompt
cd "D:\CHEATGPT CAPSTONE\Cheatgpt4"
git clone <repository-url> cheatgptv3
cd cheatgptv3
```

### Step 3: Create Conda Environment

```powershell
# Create environment with Python 3.11
conda create -n cheatgpt python=3.11 -y

# Activate environment
conda activate cheatgpt
```

### Step 4: Install Dependencies

#### 4.1 Install PyTorch with CUDA
```powershell
# For CUDA 11.8 (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only (not recommended)
pip install torch torchvision
```

Verify PyTorch installation:
```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
```

#### 4.2 Install Core Dependencies
```powershell
# Install from requirements file
pip install -r cheatgpt/requirements.txt

# Key packages installed:
# - ultralytics (YOLO11)
# - opencv-python (video processing)
# - mediapipe (pose detection)
# - flask, flask-socketio (web server)
# - numpy, pillow (image processing)
# - supervision (tracking)
# - reportlab (PDF generation)
```

#### 4.3 Install Additional Dependencies
```powershell
pip install eventlet python-dotenv pygame
```

### Step 5: Download Model Weights

Model weights are downloaded automatically on first run, but you can verify:

```powershell
# Check weights directory
dir cheatgpt\cheatgpt\weights

# Should contain:
# - yolo11m.pt (~40MB) - Object detection
# - yolo11m-pose.pt (~50MB) - Pose estimation
```

If missing, they'll download automatically when you start the system.

### Step 6: Verify Installation

```powershell
# Test Python environment
python --version
# Should show: Python 3.11.x

# Test GPU detection
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# Test imports
python -c "import ultralytics; import cv2; import mediapipe; print('All imports successful!')"
```

### Step 7: Initialize Database (Automatic)

The database is created automatically on first run. Verify after first launch:

```powershell
# Check database files
dir web_app\*.db
# Should show: cheatgpt.db, cheatgpt_sessions.db
```

## 🏃‍♂️ Running the System

### Method 1: Desktop Launcher (Recommended)

Double-click the **"CheatGPT System"** desktop shortcut.

### Method 2: PowerShell Script

```powershell
cd "D:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3"
powershell -ExecutionPolicy Bypass -File Start_CheatGPT.ps1
```

### Method 3: Batch File

```cmd
cd "D:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3"
Start_CheatGPT.bat
```

### Method 4: Manual (Terminal)

```powershell
# Open PowerShell
conda activate cheatgpt
cd "D:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3\web_app"
python app.py
```

Then open browser to `http://localhost:5000`

## 🖥️ Using the Web Interface

### Dashboard (Home)
1. Navigate to `http://localhost:5000/analytics/home`
2. View all recorded sessions in grid layout
3. Search and filter sessions
4. Select multiple sessions for batch deletion
5. Click any session to view details

### Live Monitoring
1. Click "Live" from navigation menu
2. Click **"Start Camera"** to begin monitoring
3. Grant camera permissions when prompted
4. Real-time detection overlay shows:
   - Bounding boxes around phones
   - Head turning angles
   - Motion detection zones
5. Click **"Stop Camera"** to end and save session

### Video Player
1. Navigate to "Player" from menu
2. Select a recorded session
3. Features:
   - Video playback with controls
   - Event timeline with markers
   - Heatmap overlay showing activity
   - Jump to specific events
   - Frame-by-frame navigation

### Analytics & Reports
1. Navigate to "Reports" from menu
2. Select session for analysis
3. View:
   - Behavior score (0-100)
   - Event distribution charts
   - Time-based heatmaps
   - Detailed event log
4. Click **"Export PDF"** to generate report

### Upload Videos
1. Navigate to "Upload" from menu
2. Drag and drop video files or click to browse
3. Supported formats: MP4, AVI, MOV
4. System processes video offline
5. View results in Analytics after processing

## ⚙️ Configuration

### Detection Settings

Edit `.env` file (create from template if missing):

```bash
# Performance
FORCE_CPU=false              # Set true to force CPU (slower)
DEBUG_ENGINE=true            # Enable detailed logging

# Detection Thresholds
PHONE_CONFIDENCE=0.50        # Phone detection minimum confidence (50%)
HEAD_TURN_ANGLE=40.0         # Head turning angle threshold (degrees)
HEAD_TURN_DURATION=3.0       # Sustained head turn duration (seconds)
MOTION_THRESHOLD=0.5         # Hand motion detection sensitivity
MOTION_DURATION=5.0          # Hand activity duration (seconds)

# LSTM Classification
LSTM_CONFIDENCE=0.65         # Temporal analysis threshold (65%)

# Event Management
DEDUPLICATION_WINDOW=3.0     # Prevent duplicate events (3 seconds)
```

### System Settings

```bash
# Frame Processing
LIVE_FPS=30.0               # Live stream frame rate
DETECTION_FPS=10.0          # Detection processing rate

# Video Recording
VIDEO_CODEC=mp4v            # Codec for recordings
VIDEO_QUALITY=90            # Quality (0-100)

# Database
DB_PATH=web_app/cheatgpt_sessions.db
ENABLE_AUTO_CLEANUP=false   # Auto-delete old sessions

# Web Server
HOST=0.0.0.0               # Accept connections from any IP
PORT=5000                  # Default port
DEBUG=false                # Flask debug mode (disable in production)
```

### Security Settings

```bash
# Deletion Password (web_app/config.py)
DELETION_PASSWORD=cheatgpt2024   # Required to delete sessions

# Change password:
# Edit web_app/config.py
# Update DELETION_PASSWORD value
```

## 📊 Performance Optimization

### GPU Acceleration (Recommended)

```powershell
# Verify CUDA setup
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Expected performance:
# - GPU (GTX 1650 Ti): 60+ FPS detection
# - CPU (i5/Ryzen 5): 15-20 FPS detection
```

### CPU-Only Mode

```bash
# In .env file:
FORCE_CPU=true

# Adjust settings for better CPU performance:
DETECTION_FPS=5.0          # Reduce to 5 FPS
LIVE_FPS=15.0              # Reduce live stream
```

### Reduce Memory Usage

```bash
# Use smaller YOLO models (edit detection code)
# yolo11m.pt (default, ~40MB) → yolo11s.pt (~20MB)
# yolo11m-pose.pt → yolo11s-pose.pt

# Clear old sessions regularly
# Disable auto-recording if not needed
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

## 🔧 Troubleshooting

### Common Issues

#### 1. Camera Access Issues
```bash
# Check camera permissions
# Windows: Settings > Privacy > Camera
# macOS: System Preferences > Security & Privacy > Camera
# Linux: Ensure user is in video group
sudo usermod -a -G video $USER
```

#### 2. CUDA/GPU Issues
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Port Already in Use
```bash
# Change port in app.py or kill existing process
# Linux/macOS:
lsof -ti:5000 | xargs kill -9

# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

#### 4. Missing Dependencies
```bash
# Reinstall all dependencies
pip install --force-reinstall -r cheatgpt/requirements.txt
pip install flask-socketio eventlet reportlab
```

#### 5. Database Issues
```bash
# Delete and recreate database
cd web_app
rm cheatgpt.db cheatgpt_sessions.db
python app.py  # Will recreate databases
```

### Performance Issues

#### Low FPS
- Enable GPU acceleration (`FORCE_CPU=false`)
- Reduce camera resolution
- Close other applications
- Use faster storage (SSD)

#### High Memory Usage
- Restart application periodically
- Clear old session data
- Reduce video retention period

### Debug Mode

Enable detailed logging:

```bash
# In .env file:
DEBUG_ENGINE=true
DEBUG_POSE=true
DEBUG_POLICY=true
```

Check logs in `logs/` directory for detailed error information.

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
