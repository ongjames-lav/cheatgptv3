# CheatGPT System Workflow: Complete Detection Pipeline

## 🎯 System Overview

The CheatGPT system uses a **multi-layered detection pipeline** that combines computer vision, deep learning, and behavioral analysis to identify cheating gestures in classroom environments.

---

## 📊 Complete System Flow

```
Webcam/Upload → Frame Extraction → Preprocessing → Detection Engine → LSTM Analysis → Event Filtering → Database → Analytics → Reports
```

---

## 🔄 Detailed Component Flow

### **Main Processing Pipeline**

```
Video Input
    ↓
Frame Capture (30 FPS)
    ↓
Resolution Normalization (640x480)
    ↓
Color Conversion (BGR → RGB)
    ↓
┌─────────────────────────────────────┐
│     Parallel Detection (3 paths)    │
├─────────────┬───────────┬───────────┤
│   YOLO11    │ MediaPipe │  Motion   │
│   Phone     │   Head    │   Hand    │
│ Detection   │  Turning  │ Activity  │
│   (~95%)    │  (~85%)   │  (~75%)   │
└─────────────┴───────────┴───────────┘
    ↓           ↓           ↓
Feature Vector [phone_conf, head_angle, motion_mag]
    ↓
LSTM Temporal Analysis (10-frame windows)
    ↓
Classification Confidence (0.0 - 1.0)
    ↓
Confidence Threshold Check (≥0.65)
    ↓
Deduplication (3-second window)
    ↓
Event Creation
    ↓
Database Storage (SQLite)
    ↓
Real-Time Dashboard Update
    ↓
Heatmap Visualization
    ↓
PDF Report Generation
```

---

## 🎯 Detection Path Breakdown

### **Path 1: Phone Detection**

```
Frame → YOLO11 Model → Bounding Box Detection → Class Filter (cell phone) → Confidence Check (≥50%) → Duration Check (≥1s) → Phone Event
```

### **Path 2: Head Turning Detection**

```
Frame → MediaPipe Pose → 33 Landmarks → Facial Points (nose, ears) → Angle Calculation → Threshold Check (≥40°) → Duration Check (≥3s) → Head Turning Event
```

### **Path 3: Hand Activity Detection**

```
Frame Sequence → Optical Flow → Motion Vectors → Magnitude Calculation → ROI Check (desk area) → Duration Check (≥5s) → Hand Activity Event
```

---

## 🧠 LSTM Integration Flow

```
Detection Results (10 frames)
    ↓
Feature Extraction
    ↓
Sequence Window [t-9, t-8, ..., t-1, t]
    ↓
LSTM Layer 1 (128 units)
    ↓
Dropout (0.3)
    ↓
LSTM Layer 2 (128 units)
    ↓
Dense Layer (64 units)
    ↓
Output Layer (Binary Classification)
    ↓
Sigmoid Activation (0.0 - 1.0)
    ↓
Cheating Probability
```

---

## 💾 Database Flow

```
Event Detection
    ↓
Event Object Creation
    ↓
INSERT INTO hotspots (session_id, event_type, confidence, timestamp_offset, frame_no, bbox_data)
    ↓
UPDATE sessions SET hotspot_count = hotspot_count + 1
    ↓
Commit Transaction
    ↓
Return Success
```

---

## 📊 Analytics Flow

```
Database Query (SELECT * FROM hotspots WHERE session_id = ?)
    ↓
Event Aggregation
    ↓
Statistical Calculation (counts, averages, distributions)
    ↓
Behavior Score Calculation (100 - events_per_minute * 10)
    ↓
Heatmap Generation (time buckets + color mapping)
    ↓
Chart Creation (event timeline, distribution)
    ↓
PDF Report Compilation
    ↓
Export Download
```

---

## ⚡ Real-Time Processing Timeline

```
t=0ms:    Frame Capture
t=10ms:   YOLO11 Detection → Phone Found (0.87 confidence)
t=10ms:   MediaPipe Detection → Head Angle = 15°
t=10ms:   Motion Detection → Magnitude = 0.45
t=20ms:   Feature Vector Created [0.87, 15.0, 0.45]
t=30ms:   LSTM Classification → 0.78 probability
t=35ms:   Confidence Check → PASS
t=35ms:   Deduplication Check → PASS
t=40ms:   Database INSERT
t=50ms:   WebSocket Broadcast
t=50ms:   Dashboard Update
Total:    50ms per detection cycle
```

---

## 🔀 Parallel Processing Flow

```
                    Frame Input
                        |
        ┌───────────────┼───────────────┐
        |               |               |
    YOLO11          MediaPipe        Motion
   (Thread 1)       (Thread 2)     (Thread 3)
        |               |               |
    Phone Bbox      Landmarks       Optical Flow
        |               |               |
    Confidence      Head Angle      Magnitude
        └───────────────┼───────────────┘
                        |
                Feature Vector
                        |
                   LSTM Model
                        |
                Classification
```

---

## 📈 Data Flow Summary

### **Input → Processing → Output**

```
Live Stream:
Webcam → Frames → Detection → Events → Database → Dashboard

Uploaded Video:
File Upload → Validation → Processing Queue → Detection → Events → Database → Reports

Analytics:
Database → Queries → Aggregation → Calculations → Visualizations → PDF Export
```

---

## 🔧 Component-Level Workflow

### **1. Video Input Layer**

#### **1.1 Live Stream Processing**
```python
# Module: web_app/app.py → VideoStream class
VideoCapture → Frame Buffer → Processing Queue
```

**Process:**
1. **Webcam Initialization**: `cv2.VideoCapture(0)` captures live video
2. **Frame Rate Control**: Processes 30 FPS (can be configured)
3. **Buffer Management**: Maintains 10-frame rolling buffer
4. **Thread Safety**: Uses threading locks for concurrent access

**Key Files:**
- `web_app/app.py` - Main video streaming logic
- `detection/video_processor.py` - Frame preprocessing

---

#### **1.2 Uploaded Video Processing**
```python
# Module: web_app/upload_routes.py
File Upload → Validation → Storage → Processing Queue
```

**Process:**
1. **File Validation**: Checks format (MP4, AVI, MOV), size (<500MB)
2. **Video Encoding**: Converts to H.264 codec if needed
3. **Storage**: Saves to `uploads/` directory
4. **Metadata Extraction**: Duration, resolution, frame count
5. **Queue for Processing**: Adds to background processing queue

**Key Files:**
- `web_app/upload_routes.py` - Upload handling
- `reencode_video_h264.py` - Video encoding utility
- `auto_migrate_uploads.py` - Upload migration system

---

### **2. Detection Engine Components**

#### **2.1 YOLO11 Phone Detection**

**Purpose:** Detect mobile phones held by students

```python
# Module: detection/phone_detector.py
Frame → YOLO11 Model → Bounding Boxes → Confidence Scores
```

**Workflow:**
1. **Model Loading**: `YOLO('yolo11x.pt')` - Pretrained on COCO dataset
2. **Inference**: Processes each frame at 640x640 resolution
3. **Class Filtering**: Extracts only "cell phone" class (class_id=67)
4. **Confidence Threshold**: Minimum 50% confidence
5. **Bounding Box Output**: `[x1, y1, x2, y2, confidence, class_id]`

**Detection Criteria:**
- Phone must be visible for >1 second (30 consecutive frames)
- Confidence score ≥ 50%
- Not in designated "phone zone" (if configured)

**Key Files:**
- `detection/phone_detector.py` - Main phone detection logic
- `download_yolo11x_models.py` - Model download utility
- `demo_classroom_phone_detection.py` - Standalone demo

**Performance:**
- **Estimated Precision:** ~95%
- **False Positive Rate:** ~5% (often triggered by tablets, calculators)
- **Processing Speed:** 15-20 FPS on CPU, 60+ FPS on GPU

---

#### **2.2 MediaPipe Pose Estimation (Head Turning)**

**Purpose:** Detect sustained head turning (looking at neighbor's paper)

```python
# Module: detection/head_turning_detector.py
Frame → MediaPipe Pose → Facial Landmarks → Angle Calculation → Event Detection
```

**Workflow:**
1. **Pose Detection**: MediaPipe extracts 33 body landmarks
2. **Facial Landmarks**: Focuses on nose (0), left ear (7), right ear (8)
3. **Angle Calculation**:
   ```python
   # Calculate head rotation angle
   ear_vector = right_ear - left_ear
   nose_offset = nose - ear_midpoint
   angle = atan2(nose_offset.y, nose_offset.x)
   ```
4. **Threshold Check**: Angle > 40° (sustained for >3 seconds)
5. **Direction Classification**: Left turn vs Right turn

**Detection Criteria:**
- Head turned >40° from forward-facing position
- Sustained for ≥3 seconds (90 consecutive frames)
- Not during normal "looking up" or "stretching"

**Key Files:**
- `detection/head_turning_detector.py` - Main head turning logic
- `demo_head_turning_keypoints.py` - Visualization demo
- `POSE_DETECTOR.md` - Technical documentation

**Performance:**
- **Estimated Precision:** ~85%
- **False Positive Rate:** ~15% (triggered by stretching, looking at clock)
- **Processing Speed:** 25-30 FPS

---

#### **2.3 Motion Analysis (Hand Activity)**

**Purpose:** Detect suspicious hand movements (writing on hidden notes, gestures)

```python
# Module: detection/motion_detector.py
Frame Sequence → Optical Flow → Motion Magnitude → Pattern Analysis
```

**Workflow:**
1. **Frame Differencing**: Compare consecutive frames
2. **Optical Flow**: `cv2.calcOpticalFlowFarneback()` for motion vectors
3. **Region of Interest (ROI)**: Focus on desk area (lower 50% of frame)
4. **Motion Magnitude**: Calculate pixel displacement
5. **Pattern Recognition**:
   - **High-frequency motion**: Rapid hand movements
   - **Localized motion**: Activity in specific area (under desk)
   - **Sustained activity**: Movement for >5 seconds

**Detection Criteria:**
- Motion magnitude > threshold (calibrated per environment)
- Activity localized to "suspicious zones" (under desk, lap)
- Duration ≥ 5 seconds
- Not normal "writing on exam paper" motion

**Key Files:**
- `detection/motion_detector.py` - Motion analysis
- `analyze_tracking.py` - Motion pattern analysis utility

**Performance:**
- **Estimated Precision:** ~75%
- **False Positive Rate:** ~25% (triggered by fidgeting, adjusting position)
- **Processing Speed:** 20-25 FPS

---

### **3. LSTM Temporal Analysis**

**Purpose:** Understand context and reduce false positives through temporal reasoning

```python
# Module: lstm_model/lstm_classifier.py
Detection Sequence → Feature Extraction → LSTM Network → Classification
```

**Workflow:**
1. **Sequence Creation**: Group detections into 10-frame windows
2. **Feature Vector**: Combine YOLO + Pose + Motion features
   ```python
   features = [
       phone_confidence,      # YOLO output
       head_angle,            # Pose estimation
       motion_magnitude,      # Motion analysis
       temporal_context       # Previous frames
   ]
   ```
3. **LSTM Processing**: 
   - **Input Layer**: 10 timesteps × 4 features
   - **LSTM Layers**: 2 layers, 128 units each
   - **Output Layer**: Binary classification (cheating / not cheating)
4. **Confidence Score**: Sigmoid activation (0-1 probability)

**Model Architecture:**
```
Input (10, 4) → LSTM(128) → Dropout(0.3) → LSTM(128) → Dense(64) → Output(1)
```

**Training Details:**
- **Dataset**: 1,500+ labeled sequences
- **Training Accuracy**: 83.87%
- **Validation F1-Score**: 0.7651
- **Loss Function**: Binary Cross-Entropy

**Key Files:**
- `lstm_model/lstm_classifier.py` - Model architecture
- `lstm_model/train_lstm.py` - Training script
- `lstm_model/training_log.txt` - Training metrics
- `LSTM_Integration_Explained.md` - Integration guide

**Performance:**
- **Reduces False Positives**: ~30% reduction vs. single-frame detection
- **Context Understanding**: Distinguishes intentional vs. accidental movements
- **Processing Speed**: 10-15 FPS (batch processing)

---

### **4. Event Detection & Filtering**

**Purpose:** Clean up detections and create actionable events

```python
# Module: detection/event_manager.py
Raw Detections → Filtering → Deduplication → Event Creation
```

**Workflow:**

#### **4.1 Confidence Filtering**
```python
# Minimum confidence thresholds
thresholds = {
    'phone_detection': 0.50,      # 50% confidence
    'head_turning': 0.60,          # 60% confidence
    'hand_activity': 0.55,         # 55% confidence
    'lstm_classification': 0.65    # 65% confidence
}
```

#### **4.2 Deduplication**
- **Time Window**: 3-second window (90 frames)
- **Logic**: If same event type detected within 3s, keep only highest confidence
- **Purpose**: Prevent event spam from continuous detection

#### **4.3 Event Creation**
```python
event = {
    'session_id': session_id,
    'event_type': 'phone_detection',  # or 'head_turning', 'hand_activity'
    'confidence': 0.87,
    'timestamp_offset': 45.2,         # Seconds from session start
    'frame_no': 1356,
    'bbox_data': '[100, 200, 300, 400]',  # Bounding box if applicable
    'created_at': datetime.now()
}
```

#### **4.4 Database Storage**
```sql
-- Table: hotspots
INSERT INTO hotspots (
    session_id, event_type, confidence, 
    timestamp_offset, frame_no, bbox_data
) VALUES (?, ?, ?, ?, ?, ?);
```

**Key Files:**
- `detection/event_manager.py` - Event handling
- `database/schema.sql` - Database schema
- `add_events_retroactively.py` - Bulk event insertion

---

### **5. Database Layer**

**Purpose:** Persistent storage for sessions, events, and metadata

```python
# Module: database/db_manager.py
SQLite Database → Sessions Table + Hotspots Table
```

#### **5.1 Schema Design**

**Sessions Table:**
```sql
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    session_id TEXT UNIQUE,
    session_type TEXT,           -- 'recorded' or 'uploaded'
    video_path TEXT,
    thumbnail_path TEXT,
    start_time REAL,
    duration REAL,
    status TEXT,                 -- 'active', 'completed', 'processing'
    hotspot_count INTEGER,
    created_at TIMESTAMP
);
```

**Hotspots Table:**
```sql
CREATE TABLE hotspots (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    event_type TEXT,             -- 'phone_detection', 'head_turning', 'hand_activity'
    confidence REAL,
    timestamp_offset REAL,       -- Seconds from session start
    frame_no INTEGER,
    bbox_data TEXT,              -- JSON: [x1, y1, x2, y2]
    created_at TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
```

#### **5.2 Database Operations**

**Session Creation:**
```python
# On session start (live or upload)
create_session(session_id, video_path, start_time)
```

**Event Insertion:**
```python
# As events are detected
insert_hotspot(session_id, event_type, confidence, timestamp, frame_no, bbox)
```

**Session Completion:**
```python
# On session end
update_session_status(session_id, 'completed', duration, hotspot_count)
```

**Key Files:**
- `database/db_manager.py` - Database operations
- `check_databases.py` - Database integrity checker
- `consolidate_databases.py` - Database migration utility

---

### **6. Analytics & Reporting Layer**

**Purpose:** Transform raw detection data into actionable insights

```python
# Module: web_app/analytics_routes.py
Database Queries → Statistical Analysis → Visualization → Report Generation
```

#### **6.1 Real-Time Dashboard**

**Features:**
- **Live Session Monitoring**: Active webcam sessions
- **Event Timeline**: Chronological list of detections
- **Session Statistics**: Total events, duration, behavior score

**Calculation Example:**
```python
# Behavior Score (0-100)
total_events = session.hotspot_count
duration_minutes = session.duration / 60

events_per_minute = total_events / duration_minutes
behavior_score = max(0, 100 - (events_per_minute * 10))

# Score ranges:
# 90-100: Excellent (0-1 events/min)
# 70-89:  Good (1-3 events/min)
# 50-69:  Fair (3-5 events/min)
# 0-49:   Poor (>5 events/min)
```

**Key Files:**
- `web_app/templates/analytics_home.html` - Dashboard UI
- `web_app/analytics_routes.py` - API endpoints

---

#### **6.2 Heatmap Generation**

**Purpose:** Visualize temporal distribution of cheating events

```python
# Module: analytics/heatmap_generator.py
Hotspot Events → Time Buckets → Color Mapping → Overlay on Video
```

**Workflow:**
1. **Time Bucketing**: Divide session into N-second buckets (default 10s)
2. **Event Aggregation**: Count events per bucket
3. **Color Intensity**: Map event count to color gradient
   ```python
   colors = {
       0: (0, 255, 0),      # Green - No events
       1-2: (255, 255, 0),  # Yellow - Low activity
       3-5: (255, 165, 0),  # Orange - Moderate activity
       >5: (255, 0, 0)      # Red - High activity
   }
   ```
4. **Overlay**: Draw heatmap bars on video frames

**Key Files:**
- `analytics/heatmap_generator.py` - Heatmap logic
- `HOTSPOT_OVERLAY_GUIDE.md` - Implementation guide
- `debug_hotspots.py` - Heatmap debugging utility

---

#### **6.3 PDF Report Export**

**Purpose:** Generate professional reports for instructors

```python
# Module: web_app/report_generator.py
Session Data → Report Template → PDF Generation
```

**Report Contents:**
1. **Session Summary**: Date, duration, total events
2. **Event Timeline**: Detailed list with timestamps
3. **Behavior Analysis**: Score, trends, patterns
4. **Visualizations**: Event distribution chart, heatmap
5. **Recommendations**: Based on detected patterns

**Key Files:**
- `web_app/report_generator.py` - PDF generation
- `web_app/templates/report_template.html` - Report layout

---

## 🔄 Complete System Flow (Real-Time Example)

### **Scenario: Student Uses Phone During Exam**

```
Step 1: Frame Capture (t=0ms)
├─ Webcam captures frame at 30 FPS
└─ Frame stored in buffer: (640x480 RGB)

Step 2: Parallel Detection (t=10ms)
├─ YOLO11: Detects phone → confidence=0.87 → bbox=[150, 200, 250, 350]
├─ MediaPipe: Head angle=15° → No turning detected
└─ Motion: Moderate hand motion → magnitude=0.45

Step 3: Feature Extraction (t=20ms)
└─ Feature vector: [0.87, 15.0, 0.45, 0]

Step 4: LSTM Classification (t=30ms - every 10 frames)
├─ Input: Last 10 feature vectors
├─ LSTM Output: 0.78 (78% cheating probability)
└─ Classification: CHEATING DETECTED

Step 5: Event Filtering (t=35ms)
├─ Confidence check: 0.78 > 0.65 ✓ PASS
├─ Deduplication: No recent phone events ✓ PASS
└─ CREATE EVENT

Step 6: Database Storage (t=40ms)
└─ INSERT INTO hotspots (
    session_id='session_20251026_143022',
    event_type='phone_detection',
    confidence=0.78,
    timestamp_offset=45.2,
    frame_no=1356,
    bbox_data='[150, 200, 250, 350]'
  )

Step 7: UI Update (t=50ms)
├─ WebSocket broadcast to frontend
├─ Dashboard updates event counter
├─ Video overlay shows bounding box
└─ Alert notification (if enabled)

Total Processing Time: ~50ms per frame (20 FPS)
```

---

## 🎓 Component Interaction Matrix

| Component | Inputs | Outputs | Dependencies |
|-----------|--------|---------|--------------|
| **YOLO11** | Raw frame (640x480) | Bounding boxes, confidence | CUDA (optional) |
| **MediaPipe** | Raw frame (any size) | 33 body landmarks | TensorFlow Lite |
| **Motion** | Frame sequence (2-10 frames) | Motion magnitude | OpenCV |
| **LSTM** | Feature vector sequence (10 frames) | Cheating probability | Keras/TensorFlow |
| **Event Manager** | Detection results | Filtered events | SQLite |
| **Database** | Events, session metadata | Stored records | File system |
| **Analytics** | Database queries | Statistics, visualizations | Matplotlib |

---

## 📈 Performance Metrics by Component

| Component | Accuracy/Precision | False Positive Rate | Processing Speed |
|-----------|-------------------|---------------------|------------------|
| **YOLO11 Phone** | ~95% | ~5% | 15-20 FPS (CPU), 60+ FPS (GPU) |
| **MediaPipe Head** | ~85% | ~15% | 25-30 FPS |
| **Motion Analysis** | ~75% | ~25% | 20-25 FPS |
| **LSTM Temporal** | 83.87% | Reduces FP by 30% | 10-15 FPS |
| **Combined System** | ~87% (est.) | ~13% (est.) | 10-15 FPS |

---

## 🔍 Critical Detection Thresholds

### **Phone Detection**
- **Confidence:** ≥50%
- **Duration:** ≥1 second (30 frames)
- **Size:** Min 5% of frame area

### **Head Turning**
- **Angle:** ≥40° from center
- **Duration:** ≥3 seconds (90 frames)
- **Consistency:** 80% of frames in window

### **Hand Activity**
- **Motion Magnitude:** ≥0.5 (normalized)
- **Duration:** ≥5 seconds (150 frames)
- **Localization:** ROI coverage >20%

### **LSTM Classification**
- **Confidence:** ≥65%
- **Sequence Length:** 10 frames (0.33 seconds)
- **Feature Threshold:** All features >0.1

---

## 🛠️ Key Configuration Files

| File | Purpose | Key Parameters |
|------|---------|----------------|
| `config/detection_config.json` | Detection thresholds | confidence_min, angle_threshold, motion_threshold |
| `config/lstm_config.json` | LSTM model settings | sequence_length, batch_size, learning_rate |
| `config/system_config.json` | System-wide settings | frame_rate, resolution, buffer_size |
| `database/schema.sql` | Database schema | Table structures, indexes |

---

## 📚 Module Dependency Tree

```
web_app/app.py (Main Application)
├── detection/
│   ├── phone_detector.py (YOLO11)
│   ├── head_turning_detector.py (MediaPipe)
│   ├── motion_detector.py (Optical Flow)
│   └── event_manager.py (Event Handling)
├── lstm_model/
│   ├── lstm_classifier.py (Temporal Analysis)
│   └── feature_extractor.py (Feature Engineering)
├── database/
│   ├── db_manager.py (SQLite Operations)
│   └── schema.sql (Database Schema)
├── analytics/
│   ├── heatmap_generator.py (Visualization)
│   └── statistics.py (Metrics Calculation)
└── web_app/
    ├── analytics_routes.py (API Endpoints)
    ├── upload_routes.py (File Upload)
    └── templates/ (Frontend UI)
```

---

## 🚀 Processing Pipeline Summary

### **Real-Time (Live Stream)**
```
Webcam → 30 FPS → Detection (10-15 FPS) → LSTM → Event → Database → Dashboard (Real-time)
```

### **Batch (Uploaded Video)**
```
Upload → Validation → Queue → Detection (Offline) → LSTM → Events → Database → Reports
```

### **Analytics (Post-Processing)**
```
Database → Queries → Statistics → Heatmaps → PDF Reports → Export
```

---

## 🎯 System Strengths

✅ **Multi-Modal Detection**: Combines 3 detection methods for robustness  
✅ **Temporal Context**: LSTM reduces false positives by understanding sequences  
✅ **High Precision**: ~87% estimated system precision (pending ground truth)  
✅ **Real-Time Capable**: 10-15 FPS end-to-end processing  
✅ **Scalable**: Can handle multiple concurrent sessions  
✅ **Comprehensive Analytics**: Rich reporting and visualization  

---

## ⚠️ System Limitations

❌ **Environmental Sensitivity**: Camera angle, lighting, distance affect accuracy  
❌ **False Positives**: ~13% FP rate (adjusting position, fidgeting)  
❌ **Occlusions**: Cannot detect hidden phones or behaviors outside frame  
❌ **Calibration Required**: Thresholds need tuning per classroom setup  
❌ **Ground Truth Needed**: Estimated metrics require manual validation  

---

## 📖 Related Documentation

- **MODULES_OVERVIEW.md** - Detailed module descriptions
- **ENGINE_COMPLETE.md** - Detection engine architecture
- **LSTM_Integration_Explained.md** - LSTM model integration
- **BENCHMARK_RESULTS_EXPLAINED.md** - Performance metrics
- **PERFORMANCE_SUMMARY.md** - Quick reference for thesis defense
- **BENCHMARKING_GUIDE.md** - How to validate system accuracy

---

## 🔗 API Endpoints Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/sessions/list` | GET | List all sessions (recorded + uploaded) |
| `/api/sessions/start` | POST | Start live recording session |
| `/api/sessions/stop` | POST | Stop active session |
| `/api/hotspots/{session_id}` | GET | Get all events for session |
| `/api/reports/export/session_pdf` | GET | Generate PDF report |
| `/upload` | POST | Upload video for processing |

---

## 💡 Future Improvements

1. **Multi-Person Tracking**: Track individual students in crowded classrooms
2. **Voice Analysis**: Detect whispering or verbal communication
3. **Gaze Tracking**: Eye movement analysis for looking at others' papers
4. **Federated Learning**: Privacy-preserving model training across institutions
5. **Real-Time Alerts**: SMS/Email notifications for instructors
6. **Mobile App**: Remote monitoring from instructor's phone

---

**Last Updated:** October 26, 2025  
**System Version:** CheatGPT v3.0  
**Documentation Status:** Complete & Defense-Ready
