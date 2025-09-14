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
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better recommended)
- **RAM**: Minimum 8GB (16GB recommended for optimal performance)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for better performance)
- **Storage**: At least 5GB free space
- **Camera**: Webcam or external camera for monitoring

### Software Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 to 3.11 (3.10 recommended)
- **Git**: For cloning the repository

## 🚀 Installation Guide

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd cheatgptv3
```

### Step 2: Create Python Virtual Environment

#### Using Conda (Recommended)
```bash
# Install Miniconda if you haven't already
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Create environment
conda create -n cheatgpt python=3.10
conda activate cheatgpt
```

#### Using venv (Alternative)
```bash
# Create virtual environment
python -m venv cheatgpt_env

# Activate environment
# On Windows:
cheatgpt_env\Scripts\activate
# On macOS/Linux:
source cheatgpt_env/bin/activate
```

### Step 3: Install Dependencies

#### Core Dependencies
```bash
# Install core Python packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA support
# OR for CPU only:
# pip install torch torchvision

# Install other dependencies
pip install ultralytics
pip install opencv-python
pip install numpy
pip install Pillow
pip install flask
pip install flask-socketio
pip install eventlet
pip install python-dotenv
pip install reportlab
pip install sqlite3  # Usually included with Python
```

#### Alternative: Install from requirements file
```bash
# Install from the provided requirements
pip install -r cheatgpt/requirements.txt

# Additional web app dependencies
pip install flask-socketio eventlet reportlab
```

### Step 4: Download Model Weights

The system requires YOLO models for object and pose detection:

```bash
# Navigate to weights directory
cd weights

# Download YOLO models (these will be downloaded automatically on first run)
# You can also manually download:
# - yolo11m.pt (object detection)
# - yolo11m-pose.pt (pose detection)
```

### Step 5: Environment Configuration

Copy the example environment file and configure it:

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your preferred settings
# Use any text editor like notepad, vim, or nano
```

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

### Step 6: Initialize Database

The system uses SQLite for data storage. The database will be created automatically on first run, but you can initialize it manually:

```bash
# Navigate to web_app directory
cd web_app

# Run the application once to create database
python app.py
# Stop with Ctrl+C after seeing "Running on http://127.0.0.1:5000"
```

### Step 7: Test Installation

#### Test Core Engine
```bash
# Test the detection engine
python test_engine_complete.py

# Test webcam integration
python test_enhanced_webcam.py

# Test pose detection
python test_enhanced_pose.py
```

#### Test Web Application
```bash
# Navigate to web app
cd web_app

# Start the web application
python app.py
```

Open your browser and go to `http://localhost:5000` to access the web interface.

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
