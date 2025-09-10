# CheatGPT Video Recording with Overlays

This document describes the video recording functionality added to the CheatGPT system.

## Overview

The enhanced CheatGPT system now supports:
- **Session-based video recording** with overlays
- **Real-time frame processing** with bounding boxes and labels
- **Database storage** of session metadata and hotspots
- **Automatic video management** with start/stop functionality

## Architecture

### Components

1. **VideoRecorder** (`cheatgpt/video_recorder.py`)
   - Handles MP4 video creation with OpenCV
   - Records frames with overlays already applied
   - Manages video file lifecycle

2. **Enhanced Engine** (`cheatgpt/engine.py`)
   - Integrated video recording into frame processing
   - Session management with unique IDs
   - Hotspot tracking for spatial analysis

3. **Enhanced Database** (`cheatgpt/db/db_manager.py`)
   - New `sessions` table for session metadata
   - New `hotspots` table for spatial event tracking
   - Session lifecycle management

4. **Enhanced App** (`cheatgpt/app.py`)
   - Complete application with video recording
   - Real-time webcam monitoring
   - Batch video processing capabilities

## Usage

### Starting a Recording Session

```python
from cheatgpt.engine import Engine

# Initialize engine
engine = Engine()

# Start session with video recording
session_id = engine.start_session(
    cam_id="main_camera",
    frame_size=(1280, 720)  # Required for video recording
)

# Process frames (video recording happens automatically)
overlay_frame, events = engine.process_frame(frame, cam_id="main_camera")

# Stop session
session_info = engine.stop_session()
```

### Real-time Webcam Application

```python
from cheatgpt.app import CheatGPTApp

app = CheatGPTApp()
app.start_webcam_session(
    duration_minutes=5,    # 0 = infinite
    record_video=True      # Enable video recording
)
```

### Batch Video Processing

```python
app = CheatGPTApp()
app.demo_batch_processing("input_video.mp4")
```

## Video Output

### Format
- **Container**: MP4
- **Codec**: mp4v (compatible with most players)
- **FPS**: 30 (configurable)
- **Quality**: Original camera resolution

### Content
Each recorded frame includes:
- **Person bounding boxes** (green=normal, yellow=suspicious, orange=concerning, red=critical)
- **Event labels** with confidence scores
- **Pose indicators** (lean angles, head turns)
- **Phone detection** markers
- **System status** overlays

### File Naming
```
videos/session_{session_id}_{cam_id}_{timestamp}.mp4
```

Example: `videos/session_20240903_142350_abcd1234_main_camera_20240903_142350.mp4`

## Database Schema

### Sessions Table
```sql
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    video_path TEXT,
    start_timestamp REAL NOT NULL,
    end_timestamp REAL,
    cam_id TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    frame_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Hotspots Table
```sql
CREATE TABLE hotspots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    width REAL NOT NULL,
    height REAL NOT NULL,
    event_count INTEGER DEFAULT 1,
    severity_level TEXT,
    first_detection_time REAL,
    last_detection_time REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
);
```

## Features

### Session Management
- **Unique session IDs** with timestamp and UUID
- **Automatic video path tracking** in database
- **Session lifecycle** (active → completed)
- **Frame counting** and duration tracking

### Hotspot Tracking
- **Spatial event clustering** (50-pixel grid)
- **Event count accumulation** per location
- **Severity level tracking** (escalation)
- **Temporal analysis** (first/last detection times)

### Video Recording
- **Real-time overlay recording** - frames with bounding boxes are saved
- **Automatic file management** - videos directory creation
- **Error handling** - graceful fallbacks if recording fails
- **Memory efficient** - streaming write to disk

### Performance Optimizations
- **GPU acceleration** for frame processing
- **Efficient video encoding** with OpenCV
- **Minimal overhead** - recording doesn't slow detection
- **Background processing** - non-blocking video writes

## Testing

### Quick Test
```bash
python test_video_recording.py
```

### Manual Testing
1. **Start the application**:
   ```bash
   python -m cheatgpt.app
   ```

2. **Perform test behaviors**:
   - Sit normally (green boxes)
   - Wave hands near face (yellow/orange)
   - Look around frequently (yellow/orange)
   - Lean to side repeatedly (yellow/orange)
   - Combine behaviors (red alerts)

3. **Check results**:
   - Video file in `videos/` directory
   - Database entries in `cheatgpt.db`
   - Console logs show session statistics

### Verification
- **Video playback**: Open recorded MP4 files
- **Database inspection**: Check sessions and hotspots tables
- **Overlay quality**: Verify bounding boxes and labels in video
- **Performance**: Monitor FPS and processing times

## Configuration

### Environment Variables
```bash
# Video recording settings
VIDEO_FPS=30
VIDEO_CODEC=mp4v
VIDEOS_DIR=videos

# Database settings
DATABASE_URL=cheatgpt.db

# Detection thresholds
PERSON_CONF_THRESH=0.4
PHONE_CONF_THRESH=0.4
```

### VideoRecorder Settings
```python
recorder = VideoRecorder(videos_dir="custom_videos")
recorder.fps = 25  # Custom FPS
recorder.fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Custom codec
```

## Troubleshooting

### Common Issues

1. **Video file not created**
   - Check permissions for videos directory
   - Verify frame_size parameter is provided
   - Check OpenCV VideoWriter codec support

2. **Recording fails silently**
   - Check console logs for VideoRecorder errors
   - Verify disk space availability
   - Try different video codec (XVID, H264)

3. **Performance issues**
   - Monitor GPU memory usage
   - Reduce video resolution if needed
   - Check CPU usage during recording

4. **Database errors**
   - Verify SQLite file permissions
   - Check for table creation errors in logs
   - Ensure unique session IDs

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging for all components
```

## Future Enhancements

- **H.264 encoding** for better compression
- **Multiple camera support** with synchronized recording
- **Audio recording** integration
- **Streaming protocols** (RTMP, WebRTC)
- **Cloud storage** integration (AWS S3, Google Cloud)
- **Real-time analytics** dashboard
- **Video annotation** tools
- **Export formats** (AVI, MOV, WebM)

## API Reference

### Engine.start_session()
```python
def start_session(self, cam_id: str = "webcam", frame_size: Optional[Tuple[int, int]] = None) -> str
```

### Engine.stop_session()
```python
def stop_session(self) -> dict
```

### Engine.get_session_status()
```python
def get_session_status(self) -> dict
```

### VideoRecorder.start_recording()
```python
def start_recording(self, session_id: str, frame_size: Tuple[int, int], cam_id: str = "webcam") -> Tuple[bool, str]
```

### VideoRecorder.write_frame()
```python
def write_frame(self, overlay_frame) -> bool
```

### VideoRecorder.stop_recording()
```python
def stop_recording(self) -> Tuple[bool, dict]
```
