# CheatGPT3 Engine Implementation

## Overview
The CheatGPT3 Engine is the core orchestration component that coordinates the complete detection pipeline. It integrates YOLO object detection, pose analysis, tracking, and policy evaluation into a unified system with automatic evidence saving and real-time visualization.

## Architecture

### Full Pipeline Flow
```
Frame Input
    ↓
[1] YOLOv11 Detection (persons, phones)
    ↓
[2] Tracker Updates IDs  
    ↓
[3] Pose Detector Extracts Behavior Features
    ↓
[4] Policy Module Applies Rules
    ↓
[5] Draw Bounding Boxes with Labels & Severity Colors
    ↓
[6] Save Evidence Frames (when cheating detected)
    ↓
Output: overlay_frame, events
```

## Implementation Details

### Files
- `cheatgpt/engine.py` - Main engine implementation
- `cheatgpt/detectors/tracker.py` - Object tracking system (implemented)

### Core Components

#### 1. Engine Class
The main orchestration class that coordinates all pipeline components:
- **YOLO11Detector**: Object detection for persons and phones
- **PoseDetector**: Pose estimation and behavior analysis
- **Tracker**: Multi-object tracking for ID consistency
- **PolicyEngine**: Rule-based violation detection
- **DBManager**: Database integration for event storage

#### 2. Tracker Implementation
Custom multi-object tracker optimized for exam scenarios:
- **Distance-based Association**: IoU + centroid distance matching
- **Track Lifecycle Management**: Creation, updating, and cleanup
- **Linear Prediction**: Simple motion prediction for better association
- **Configurable Parameters**: Adjustable disappearance and distance thresholds

### API Specification

#### Primary API
```python
overlay_frame, events = engine.process_frame(frame, cam_id, ts)
```

**Parameters:**
- `frame` (np.ndarray): Input video frame in BGR format
- `cam_id` (str): Camera identifier (e.g., "classroom_01")
- `ts` (float, optional): Timestamp (uses current time if None)

**Returns:**
- `overlay_frame` (np.ndarray): Annotated frame with bounding boxes and labels
- `events` (List[Dict]): List of violation events detected

#### Event Structure
```python
{
    'timestamp': float,        # Event timestamp
    'cam_id': str,            # Camera identifier  
    'track_id': str,          # Person track ID
    'event_type': str,        # Violation type
    'severity': str,          # Severity level
    'confidence': float,      # Detection confidence (0-1)
    'bbox': List[float],      # Bounding box [x1,y1,x2,y2]
    'details': str            # Additional information
}
```

### Visualization System

#### Bounding Box Colors
Color-coded severity system for immediate visual assessment:
- 🟢 **Green**: Normal behavior
- 🟡 **Yellow**: Minor violations (Leaning, Looking Around)
- 🟠 **Orange**: Phone Use detected
- 🔴 **Red**: Cheating detected

#### Visual Elements
- **Bounding Boxes**: Person detection with severity colors
- **Track IDs**: Persistent identification labels
- **Behavior Indicators**: L=Lean, H=Head turn, P=Phone
- **Head Direction**: Arrow indicators for significant head turns
- **Phone Detection**: Magenta boxes for detected phones
- **System Status**: Real-time statistics overlay

### Evidence Saving System

#### Automatic Evidence Capture
When cheating is detected:
1. **Frame Saving**: Original frame saved to `uploads/evidence/`
2. **Filename Format**: `cheating_{cam_id}_{timestamp}_{track_ids}.jpg`
3. **Database Entry**: Event stored with evidence path reference
4. **Logging**: Warning logs for administrator alerts

#### Evidence Directory Structure
```
uploads/evidence/
├── cheating_cam01_20250826_143022_123_person_001.jpg
├── cheating_classroom_20250826_143055_456_person_002_person_003.jpg
└── ...
```

### Performance Characteristics

#### Processing Pipeline
- **Real-time Capable**: Optimized for live video processing
- **Average FPS**: 3-6 FPS on CPU (depends on image size and complexity)
- **Memory Efficient**: Automatic cleanup of old tracking data
- **Scalable**: Supports multiple concurrent camera streams

#### Tracking Performance
- **Association Accuracy**: High precision with IoU + distance matching
- **ID Consistency**: Persistent tracking across temporary occlusions
- **Recovery**: Automatic track recovery after brief disappearances
- **Cleanup**: Configurable track timeout (30 frames default)

### Configuration

#### Tracker Parameters
- `max_disappeared`: Maximum frames before track deletion (default: 30)
- `max_distance`: Maximum distance for track association (default: 100.0)

#### Performance Tuning
- Process every Nth frame for higher FPS if needed
- Adjust image resolution for speed vs. accuracy trade-off
- Configure tracker parameters based on camera setup

### Usage Examples

#### Basic Usage
```python
from cheatgpt.engine import Engine
import cv2

# Initialize engine
engine = Engine()

# Process video frame
frame = cv2.imread("exam_room.jpg")
overlay_frame, events = engine.process_frame(frame, "cam_01")

# Check for violations
for event in events:
    if event['severity'] == 'Cheating':
        print(f"ALERT: {event['track_id']} cheating detected!")
```

#### Video Stream Processing
```python
cap = cv2.VideoCapture(0)  # Camera input

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    overlay_frame, events = engine.process_frame(frame, "live_cam")
    
    # Display result
    cv2.imshow("CheatGPT3 Monitor", overlay_frame)
    
    # Handle events
    for event in events:
        print(f"{event['track_id']}: {event['event_type']}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

#### Statistics Monitoring
```python
# Get comprehensive statistics
stats = engine.get_statistics()

print(f"Processing {stats['performance']['fps']:.1f} FPS")
print(f"Tracking {stats['active_tracks']} students")
print(f"Active violations: {stats['active_violations']}")
print(f"Evidence saved to: {stats['evidence_directory']}")
```

### Integration Points

#### Input Sources
- **Live Cameras**: Real-time video streams
- **Video Files**: Recorded exam sessions
- **Image Sequences**: Batch processing capability

#### Output Consumers
- **Web Interface**: Live monitoring dashboard
- **Alert System**: Real-time notifications
- **Database**: Event storage and history
- **Recording System**: Evidence archival

### Testing and Validation

#### Comprehensive Test Suite
- ✅ **Component Integration**: All pipeline components working together
- ✅ **API Compliance**: Exact specification adherence
- ✅ **Performance Testing**: FPS and memory usage validation
- ✅ **Error Handling**: Robust failure recovery
- ✅ **Evidence System**: Automatic saving verification
- ✅ **Visualization Quality**: Proper annotation rendering

#### Test Results
```
🎯 Engine Test Summary:
✅ Engine initialization: PASSED
✅ Single frame processing: PASSED  
✅ Multi-frame processing: PASSED
✅ Statistics retrieval: PASSED
✅ Evidence saving: PASSED
✅ API format: PASSED
✅ Visualization: PASSED
✅ Engine reset: PASSED

📈 Performance Metrics:
Average FPS: 3.8
Processing reliability: 100%
Evidence capture: Automatic
```

### Production Deployment

#### System Requirements
- **Python 3.8+** with required packages
- **CPU**: Multi-core recommended for real-time processing
- **GPU**: CUDA support for enhanced performance (optional)
- **Storage**: Evidence directory with adequate space
- **Network**: If processing remote camera streams

#### Monitoring
- **Performance Metrics**: Built-in FPS and processing time tracking
- **Health Checks**: Component status validation
- **Log Monitoring**: Comprehensive logging for debugging
- **Alert Integration**: Events can trigger external notification systems

### Key Achievements

✅ **Complete Pipeline Integration**
- All components working seamlessly together
- Efficient data flow from detection to policy evaluation

✅ **Robust Tracking System**
- Persistent ID assignment across frames
- Handles occlusions and temporary disappearances

✅ **Automatic Evidence System**
- Saves frames when cheating detected
- Database integration for audit trails

✅ **Real-time Visualization**
- Color-coded severity indicators
- Comprehensive status information
- Professional monitoring interface

✅ **Production-Ready Performance**
- Optimized for live video processing
- Comprehensive error handling
- Configurable parameters

The CheatGPT3 Engine successfully implements the complete detection pipeline with all specified features and is ready for production deployment in exam monitoring scenarios.
