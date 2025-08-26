# Pose Detector Implementation

## Overview
The `PoseDetector` class implements YOLOv11-Pose based human pose estimation and behavior analysis for the CheatGPT3 system. It detects suspicious behaviors during exams including leaning, looking around, and phone usage.

## Implementation Details

### Files
- `cheatgpt/detectors/pose_detector.py` - Main implementation
- `cheatgpt/.env` - Configuration (updated with pose settings)

### Key Features

#### 1. Model Loading
- Loads YOLOv11-Pose from `POSE_MODEL_PATH` environment variable
- Automatic GPU/CPU device selection
- Configurable via environment variables

#### 2. Keypoint Extraction
Extracts and processes COCO-format keypoints:
- **Head**: nose, left_eye, right_eye
- **Shoulders**: left_shoulder, right_shoulder  
- **Hips**: left_hip, right_hip

#### 3. Derived Features

##### Leaning Detection
- Computes torso angle from shoulder-hip alignment
- Flags as leaning if angle > `LEAN_ANGLE_THRESH` (default: 20°)

##### Looking Around Detection
- Estimates head yaw angle from eye positions
- Flags as looking around if |yaw| > `HEAD_TURN_THRESH` (default: 30°)

##### Phone Near Detection
- Calculates IoU between person torso bbox and phone detections
- Flags as phone near if IoU > `PHONE_IOU_THRESH` (default: 0.3)

#### 4. Output Format
Each detected person returns:
```python
{
    'track_id': str,        # Person tracking ID
    'bbox': [x1,y1,x2,y2],  # Bounding box coordinates
    'yaw': float,           # Head yaw angle in degrees
    'pitch': float,         # Head pitch angle in degrees  
    'lean_flag': bool,      # True if leaning detected
    'look_flag': bool,      # True if looking around
    'phone_flag': bool,     # True if phone near torso
    'confidence': float     # Detection confidence
}
```

### Configuration

Environment variables in `.env`:
```bash
# Pose Model
POSE_MODEL_PATH=weights/yolo11n-pose.pt

# Detection Thresholds
LEAN_ANGLE_THRESH=20.0      # Degrees for leaning detection
HEAD_TURN_THRESH=30.0       # Degrees for head turn detection  
PHONE_IOU_THRESH=0.3        # IoU threshold for phone proximity

# Device Settings
FORCE_CPU=false             # Force CPU usage
```

### Usage Example

```python
from cheatgpt.detectors.pose_detector import PoseDetector
from cheatgpt.detectors.yolo11_detector import YOLO11Detector

# Initialize detectors
pose_detector = PoseDetector()
yolo_detector = YOLO11Detector()

# Process frame
frame = cv2.imread("image.jpg")
phone_detections = yolo_detector.detect(frame)
phones = [d for d in phone_detections if d['cls_name'] == 'cell phone']

# Get pose estimates
poses = pose_detector.estimate(frame, phones)

# Analyze results
for pose in poses:
    if pose['lean_flag'] or pose['look_flag'] or pose['phone_flag']:
        print(f"Suspicious behavior detected for {pose['track_id']}")
```

### Testing

Three comprehensive test files validate the implementation:

1. **`test_pose_detector.py`** - Basic functionality test
2. **`test_pose_comprehensive.py`** - Detailed feature testing with visualization
3. **`validate_pose_detector.py`** - Full requirements validation

### Performance

- **GPU Support**: Automatic CUDA detection with CPU fallback
- **Memory Management**: Tensors kept on device until final conversion
- **Error Handling**: Robust error handling for edge cases
- **Efficiency**: Batch processing of multiple persons per frame

### Integration Points

The pose detector integrates with:
- **YOLO11Detector**: Uses phone detections for proximity analysis
- **Tracker**: Will receive track IDs from tracking system
- **Analytics**: Provides behavior data for reporting
- **Database**: Stores detection results and events

### Validation Results

✅ All requirements successfully implemented:
- YOLOv11-Pose model loading from environment
- Keypoint extraction for head, shoulders, hips
- Behavior analysis (leaning, looking, phone proximity)
- Required output format with all fields
- GPU tensor management
- Configurable thresholds
- Edge case handling

The pose detector is ready for integration into the main CheatGPT3 system.
