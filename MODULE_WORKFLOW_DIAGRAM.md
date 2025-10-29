# CheatGPT Module-by-Module Workflow Diagram

## 🎯 How Core Components Function Collectively to Detect Classroom Cheating Gestures

---

## 📐 Complete Module Interaction Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MODULE 1: VIDEO INPUT                             │
│                                                                              │
│  web_app/app.py (VideoStream class)                                         │
│  • cv2.VideoCapture(0) → Captures webcam frames                             │
│  • Frame buffer (10 frames) → Temporary storage                             │
│  • Threading locks → Ensures thread safety                                  │
│                                                                              │
│  OR                                                                          │
│                                                                              │
│  web_app/upload_routes.py (File Upload)                                     │
│  • File validation → Checks MP4/AVI/MOV format                              │
│  • Video encoding → Converts to H.264 codec                                 │
│  • Storage → Saves to uploads/ directory                                    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MODULE 2: FRAME PREPROCESSING                         │
│                                                                              │
│  detection/video_processor.py                                               │
│  • Frame extraction → 30 FPS sampling rate                                  │
│  • Resolution normalization → Resize to 640x480                             │
│  • Color space conversion → BGR to RGB                                      │
│  • Frame buffering → Store last 10 frames                                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MODULE 3: PARALLEL DETECTION ENGINE                      │
│                          (3 Independent Threads)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ↓                  ↓                  ↓
    ┌───────────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
    │  MODULE 3A: YOLO11    │  │ MODULE 3B:       │  │ MODULE 3C:          │
    │  PHONE DETECTION      │  │ MEDIAPIPE POSE   │  │ MOTION ANALYSIS     │
    │                       │  │ HEAD TURNING     │  │ HAND ACTIVITY       │
    └───────────────────────┘  └──────────────────┘  └─────────────────────┘
                │                      │                      │
                ↓                      ↓                      ↓
    ┌───────────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
    │ detection/            │  │ detection/       │  │ detection/          │
    │ phone_detector.py     │  │ head_turning_    │  │ motion_detector.py  │
    │                       │  │ detector.py      │  │                     │
    │ PROCESS:              │  │                  │  │ PROCESS:            │
    │ 1. Load YOLO11x model │  │ PROCESS:         │  │ 1. Optical flow     │
    │    ↓                  │  │ 1. MediaPipe     │  │    calculation      │
    │ 2. Frame inference    │  │    pose init     │  │    ↓                │
    │    ↓                  │  │    ↓             │  │ 2. Frame difference │
    │ 3. Object detection   │  │ 2. Extract 33    │  │    ↓                │
    │    ↓                  │  │    landmarks     │  │ 3. Motion vectors   │
    │ 4. Filter class=67    │  │    ↓             │  │    ↓                │
    │    (cell phone)       │  │ 3. Get facial    │  │ 4. Magnitude calc   │
    │    ↓                  │  │    points        │  │    ↓                │
    │ 5. Confidence ≥50%    │  │    (nose, ears)  │  │ 5. ROI filtering    │
    │    ↓                  │  │    ↓             │  │    (desk area)      │
    │ 6. Bounding box       │  │ 4. Angle calc    │  │    ↓                │
    │    [x1,y1,x2,y2]      │  │    atan2()       │  │ 6. Threshold check  │
    │    ↓                  │  │    ↓             │  │    ↓                │
    │ OUTPUT:               │  │ 5. Check ≥40°    │  │ OUTPUT:             │
    │ • confidence: 0.87    │  │    ↓             │  │ • magnitude: 0.45   │
    │ • bbox: coordinates   │  │ 6. Duration 3s   │  │ • location: ROI     │
    │ • class: phone        │  │    ↓             │  │ • duration: time    │
    │                       │  │ OUTPUT:          │  │                     │
    │ ~95% precision        │  │ • angle: 15°     │  │ ~75% precision      │
    │ 15-20 FPS (CPU)       │  │ • direction: L/R │  │ 20-25 FPS           │
    │ 60+ FPS (GPU)         │  │                  │  │                     │
    │                       │  │ ~85% precision   │  │                     │
    │                       │  │ 25-30 FPS        │  │                     │
    └───────────────────────┘  └──────────────────┘  └─────────────────────┘
                │                      │                      │
                └──────────────────────┼──────────────────────┘
                                       │
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODULE 4: FEATURE EXTRACTION                              │
│                                                                              │
│  lstm_model/feature_extractor.py                                            │
│  • Combine detection results → Create unified feature vector                │
│  • Feature vector = [phone_conf, head_angle, motion_mag, context]          │
│  • Normalization → Scale values to [0, 1] range                             │
│  • Sequence creation → Group into 10-frame windows                          │
│                                                                              │
│  EXAMPLE OUTPUT:                                                             │
│  features = [0.87, 15.0, 0.45, 0.0]                                         │
│  sequence = [[f1], [f2], ..., [f10]]  (10 timesteps)                       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                  MODULE 5: LSTM TEMPORAL ANALYSIS                            │
│                                                                              │
│  lstm_model/lstm_classifier.py                                              │
│                                                                              │
│  ARCHITECTURE:                                                               │
│  Input Layer (10 timesteps × 4 features)                                    │
│       ↓                                                                      │
│  LSTM Layer 1 (128 units, return_sequences=True)                            │
│       ↓                                                                      │
│  Dropout Layer (0.3)                                                         │
│       ↓                                                                      │
│  LSTM Layer 2 (128 units)                                                   │
│       ↓                                                                      │
│  Dense Layer (64 units, ReLU activation)                                    │
│       ↓                                                                      │
│  Output Layer (1 unit, Sigmoid activation)                                  │
│       ↓                                                                      │
│  Cheating Probability [0.0 - 1.0]                                           │
│                                                                              │
│  TRAINING:                                                                   │
│  • Dataset: 1,500+ labeled sequences                                        │
│  • Accuracy: 83.87%                                                          │
│  • F1-Score: 0.7651                                                          │
│  • Loss: Binary Cross-Entropy                                               │
│                                                                              │
│  OUTPUT:                                                                     │
│  • probability = 0.78 (78% cheating likelihood)                             │
│  • Reduces false positives by ~30%                                          │
│                                                                              │
│  Processing Speed: 10-15 FPS                                                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODULE 6: EVENT FILTERING                                 │
│                                                                              │
│  detection/event_manager.py                                                 │
│                                                                              │
│  STEP 1: Confidence Thresholding                                            │
│  • Phone detection: ≥0.50                                                   │
│  • Head turning: ≥0.60                                                      │
│  • Hand activity: ≥0.55                                                     │
│  • LSTM classification: ≥0.65                                               │
│       ↓                                                                      │
│  STEP 2: Deduplication                                                      │
│  • Check last 3 seconds (90 frames)                                         │
│  • If duplicate event type → Keep highest confidence                        │
│  • Purpose: Prevent event spam                                              │
│       ↓                                                                      │
│  STEP 3: Event Creation                                                     │
│  • session_id: Unique identifier                                            │
│  • event_type: phone_detection / head_turning / hand_activity              │
│  • confidence: Final score                                                  │
│  • timestamp_offset: Seconds from session start                             │
│  • frame_no: Frame number                                                   │
│  • bbox_data: Bounding box (if applicable)                                  │
│       ↓                                                                      │
│  OUTPUT: Filtered, deduplicated event object                                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MODULE 7: DATABASE STORAGE                              │
│                                                                              │
│  database/db_manager.py                                                     │
│                                                                              │
│  DATABASE: cheatgpt_sessions.db (SQLite)                                    │
│                                                                              │
│  TABLE 1: sessions                                                           │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │ session_id | session_type | video_path | start_time | ...    │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                              │
│  TABLE 2: hotspots                                                           │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │ id | session_id | event_type | confidence | timestamp_offset │          │
│  │ frame_no | bbox_data | created_at                            │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                              │
│  OPERATIONS:                                                                 │
│  1. INSERT event into hotspots table                                        │
│       ↓                                                                      │
│  2. UPDATE sessions.hotspot_count += 1                                      │
│       ↓                                                                      │
│  3. COMMIT transaction                                                       │
│       ↓                                                                      │
│  4. Return success/failure status                                            │
│                                                                              │
│  Storage Location: database/cheatgpt_sessions.db                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                  MODULE 8: REAL-TIME NOTIFICATION                            │
│                                                                              │
│  web_app/websocket_handler.py                                               │
│                                                                              │
│  WebSocket Communication:                                                    │
│  Server → Client broadcast                                                   │
│       ↓                                                                      │
│  EVENT MESSAGE:                                                              │
│  {                                                                           │
│    "type": "event_detected",                                                │
│    "session_id": "session_20251026_143022",                                 │
│    "event_type": "phone_detection",                                         │
│    "confidence": 0.78,                                                       │
│    "timestamp": 45.2,                                                        │
│    "bbox": [150, 200, 250, 350]                                             │
│  }                                                                           │
│       ↓                                                                      │
│  Frontend receives → Updates dashboard in real-time                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MODULE 9: ANALYTICS ENGINE                               │
│                                                                              │
│  web_app/analytics_routes.py                                                │
│                                                                              │
│  SUBMODULE 9A: Dashboard Updates                                            │
│  • Real-time event counter                                                  │
│  • Session statistics (duration, events)                                    │
│  • Behavior score calculation                                               │
│       ↓                                                                      │
│  SUBMODULE 9B: Statistical Analysis                                         │
│  analytics/statistics.py                                                    │
│  • Event aggregation by type                                                │
│  • Time-series analysis                                                     │
│  • Behavioral patterns identification                                       │
│       ↓                                                                      │
│  SUBMODULE 9C: Behavior Score                                               │
│  Formula: score = max(0, 100 - events_per_minute * 10)                     │
│  • 90-100: Excellent (0-1 events/min)                                       │
│  • 70-89:  Good (1-3 events/min)                                            │
│  • 50-69:  Fair (3-5 events/min)                                            │
│  • 0-49:   Poor (>5 events/min)                                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODULE 10: VISUALIZATION                                  │
│                                                                              │
│  SUBMODULE 10A: Heatmap Generation                                          │
│  analytics/heatmap_generator.py                                             │
│       ↓                                                                      │
│  PROCESS:                                                                    │
│  1. Divide session into time buckets (10s intervals)                        │
│       ↓                                                                      │
│  2. Count events per bucket                                                 │
│       ↓                                                                      │
│  3. Color mapping:                                                           │
│     • Green (0 events)                                                       │
│     • Yellow (1-2 events)                                                    │
│     • Orange (3-5 events)                                                    │
│     • Red (>5 events)                                                        │
│       ↓                                                                      │
│  4. Overlay colored bars on video timeline                                  │
│       ↓                                                                      │
│  SUBMODULE 10B: Event Timeline Chart                                        │
│  • X-axis: Time (seconds)                                                   │
│  • Y-axis: Event type                                                       │
│  • Markers: Individual events with confidence                               │
│       ↓                                                                      │
│  SUBMODULE 10C: Distribution Charts                                         │
│  • Pie chart: Events by type                                                │
│  • Bar chart: Events over time                                              │
│  • Line chart: Confidence trends                                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODULE 11: REPORT GENERATION                              │
│                                                                              │
│  web_app/report_generator.py                                                │
│                                                                              │
│  PROCESS:                                                                    │
│  1. Query database for session data                                         │
│       ↓                                                                      │
│  2. Generate visualizations (charts, heatmaps)                              │
│       ↓                                                                      │
│  3. Calculate statistics and scores                                         │
│       ↓                                                                      │
│  4. Load HTML template                                                      │
│     (web_app/templates/report_template.html)                                │
│       ↓                                                                      │
│  5. Populate template with data                                             │
│       ↓                                                                      │
│  6. Convert HTML to PDF                                                     │
│     (using WeasyPrint or ReportLab)                                         │
│       ↓                                                                      │
│  7. Save PDF to reports/ directory                                          │
│       ↓                                                                      │
│  8. Return download link to user                                            │
│                                                                              │
│  REPORT SECTIONS:                                                            │
│  • Session Summary (date, duration, student info)                           │
│  • Event Timeline (detailed list with timestamps)                           │
│  • Behavior Analysis (score, interpretation)                                │
│  • Visualizations (charts, heatmaps)                                        │
│  • Recommendations (based on patterns detected)                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MODULE 12: USER INTERFACE                              │
│                                                                              │
│  SUBMODULE 12A: Dashboard (analytics_home.html)                             │
│  • Session list with thumbnails                                             │
│  • Quick statistics cards                                                   │
│  • Search and filter functionality                                          │
│       ↓                                                                      │
│  SUBMODULE 12B: Video Player (analytics_player.html)                        │
│  • Video playback with controls                                             │
│  • Heatmap overlay on timeline                                              │
│  • Event markers with jump-to functionality                                 │
│       ↓                                                                      │
│  SUBMODULE 12C: Reports Page (analytics_reports.html)                       │
│  • Session selection                                                         │
│  • Report preview                                                            │
│  • PDF export button                                                         │
│       ↓                                                                      │
│  SUBMODULE 12D: Live Monitoring (index.html)                                │
│  • Live webcam feed                                                          │
│  • Real-time event detection display                                        │
│  • Start/stop recording controls                                            │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 🔄 Module Interaction Summary

### **Sequential Processing Flow**

```
Module 1 (Video Input)
    ↓
Module 2 (Preprocessing)
    ↓
Module 3 (Detection Engine) → 3A (YOLO11) + 3B (MediaPipe) + 3C (Motion)
    ↓
Module 4 (Feature Extraction)
    ↓
Module 5 (LSTM Analysis)
    ↓
Module 6 (Event Filtering)
    ↓
Module 7 (Database Storage)
    ↓
Module 8 (Real-Time Notification)
    ↓
Module 9 (Analytics Engine)
    ↓
Module 10 (Visualization)
    ↓
Module 11 (Report Generation)
    ↓
Module 12 (User Interface)
```

---

## 📊 Data Flow Between Modules

```
Frame (RGB Image)
    → Module 3A: YOLO11 → phone_confidence, bbox
    → Module 3B: MediaPipe → head_angle, direction
    → Module 3C: Motion → motion_magnitude, location
    ↓
Feature Vector [phone_conf, head_angle, motion_mag]
    → Module 4 → Normalized features
    ↓
Sequence [10 feature vectors]
    → Module 5: LSTM → cheating_probability
    ↓
Detection Result {type, confidence, timestamp, bbox}
    → Module 6 → Filtered Event
    ↓
Event Object {session_id, type, conf, time, frame, bbox}
    → Module 7 → Database Record
    ↓
WebSocket Message {type, data}
    → Module 8 → Frontend Update
    ↓
Database Query Results
    → Module 9 → Statistics
    → Module 10 → Visualizations
    → Module 11 → PDF Report
    ↓
HTML/WebSocket
    → Module 12 → User Interface
```

---

## 🎯 Timing Diagram (Real-Time Processing)

```
t=0ms     Module 1: Frame Capture
            ↓
t=5ms     Module 2: Preprocessing (resize, color convert)
            ↓
t=10ms    Module 3A: YOLO11 Detection (parallel)
t=10ms    Module 3B: MediaPipe Pose (parallel)
t=10ms    Module 3C: Motion Analysis (parallel)
            ↓
t=20ms    Module 4: Feature Extraction (merge results)
            ↓
t=30ms    Module 5: LSTM Classification (every 10 frames)
            ↓
t=35ms    Module 6: Event Filtering (threshold + dedup)
            ↓
t=40ms    Module 7: Database INSERT
            ↓
t=45ms    Module 8: WebSocket Broadcast
            ↓
t=50ms    Module 9: Dashboard Update (analytics)
            ↓
Total: 50ms per detection cycle (20 FPS throughput)
```

---

## 🔧 Module Dependencies

| Module | Depends On | Provides To |
|--------|-----------|-------------|
| **Module 1** | Hardware (webcam/file) | Raw frames → Module 2 |
| **Module 2** | Module 1 | Preprocessed frames → Module 3 |
| **Module 3A** | Module 2, YOLO11 model | Phone detections → Module 4 |
| **Module 3B** | Module 2, MediaPipe | Head angles → Module 4 |
| **Module 3C** | Module 2, OpenCV | Motion data → Module 4 |
| **Module 4** | Module 3A/B/C | Feature vectors → Module 5 |
| **Module 5** | Module 4, LSTM model | Probabilities → Module 6 |
| **Module 6** | Module 5 | Filtered events → Module 7 |
| **Module 7** | Module 6, SQLite | Stored records → Module 9 |
| **Module 8** | Module 6, WebSocket | Live updates → Module 12 |
| **Module 9** | Module 7 | Statistics → Module 10, 11 |
| **Module 10** | Module 9 | Visualizations → Module 11, 12 |
| **Module 11** | Module 9, 10 | PDF reports → Module 12 |
| **Module 12** | Module 8, 9, 10, 11 | User interaction → Module 1 |

---

## 📁 Module File Mapping

```
Module 1:  web_app/app.py, web_app/upload_routes.py
Module 2:  detection/video_processor.py
Module 3A: detection/phone_detector.py
Module 3B: detection/head_turning_detector.py
Module 3C: detection/motion_detector.py
Module 4:  lstm_model/feature_extractor.py
Module 5:  lstm_model/lstm_classifier.py
Module 6:  detection/event_manager.py
Module 7:  database/db_manager.py
Module 8:  web_app/websocket_handler.py
Module 9:  web_app/analytics_routes.py, analytics/statistics.py
Module 10: analytics/heatmap_generator.py
Module 11: web_app/report_generator.py
Module 12: web_app/templates/*.html
```

---

## 🚀 Processing Modes

### **Mode 1: Real-Time (Live Stream)**
```
Module 1 (Webcam) → Module 2 → Module 3 → Module 4 → Module 5 → Module 6 
→ Module 7 → Module 8 → Module 12 (Dashboard)
```

### **Mode 2: Batch (Uploaded Video)**
```
Module 1 (Upload) → Module 2 → Module 3 → Module 4 → Module 5 → Module 6 
→ Module 7 → Module 9 → Module 10 → Module 11 → Module 12 (Reports)
```

### **Mode 3: Analytics (Post-Processing)**
```
Module 7 (Database) → Module 9 → Module 10 → Module 11 → Module 12 (Reports)
```

---

## ⚡ Parallel vs Sequential Processing

### **Parallel Processing (Module 3)**
```
        Module 2 Output
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
Module 3A  Module 3B  Module 3C
(YOLO11)  (MediaPipe) (Motion)
    ↓         ↓         ↓
    └─────────┼─────────┘
              ↓
         Module 4
```

### **Sequential Processing (Modules 4-7)**
```
Module 4 → Module 5 → Module 6 → Module 7
(Feature) (LSTM)   (Filter)  (Database)
```

---

**Documentation Status:** Complete Module Breakdown  
**Last Updated:** October 26, 2025  
**System Version:** CheatGPT v3.0
