# Hotspot Overlay System - Usage Guide

## 🎯 Overview

The Hotspot Overlay System adds visual markers (⚠️ warning signs, colored dots) to video frames when suspicious events are detected. It also stores event data in a database for timeline analysis.

## 🔧 Features

### ✅ Visual Overlays
- **Warning markers** above detected persons
- **Color-coded events** (orange=looking, yellow=lean, magenta=gesture, red=phone/cheating)
- **Pulsing effects** for high-priority events
- **Event labels** with confidence scores
- **Mini-timeline** showing recent events

### ✅ Database Storage
- **SQLite database** storing all events with timestamps
- **Event metadata** including bbox, confidence, additional data
- **Timeline export** to JSON for analysis
- **Query support** for time-range filtering

### ✅ Video Recording
- **Real-time recording** with overlays burned into video
- **Session management** with automatic file naming
- **Recording indicators** and session stats
- **Export capabilities** for both video and timeline data

## 🚀 Quick Start

### 1. Basic Demo
```bash
cd "d:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3"
python demo_hotspot_overlay.py
```

### 2. Enhanced Detection with Overlays
```bash
python run_enhanced_detection.py
```

### 3. Recording Only (No Display)
```bash
python run_enhanced_detection.py --no-display --session-name my_session
```

### 4. Disable Overlays but Keep Recording
```bash
python run_enhanced_detection.py --no-overlay
```

## 📋 Controls

### Demo Controls
- **ESC or 'q'**: Quit
- **'s'**: Save timeline to file
- **'c'**: Clear all events
- **SPACE**: Pause/Resume

### Enhanced Detection Controls
- **'q'**: Quit
- **'s'**: Add manual cheating event
- **'r'**: Toggle recording on/off
- **'d'**: Toggle debug info

## 🎨 Visual Elements

### Event Colors
- 🟠 **Orange**: Suspicious Looking (`suspicious_looking`)
- 🟡 **Yellow**: Body Lean (`suspicious_lean`)
- 🟣 **Magenta**: Hand Gesture (`suspicious_gesture`)
- 🔴 **Red**: Phone Detection (`phone_detected`)
- 🔴 **Dark Red**: Cheating (`cheating`, `temporal_cheating`)

### Overlay Components
1. **Warning Marker**: Circle with exclamation mark above person
2. **Event Label**: Text showing event type and confidence
3. **Connecting Line**: Links marker to person's bounding box
4. **Pulsing Effect**: For high-priority cheating events
5. **Mini-Timeline**: Recent events list in bottom-left corner

## 💾 Database Schema

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,           -- Event timestamp (seconds)
    event_type TEXT NOT NULL,          -- Type of suspicious event
    person_id TEXT NOT NULL,           -- Person identifier
    confidence REAL,                   -- Detection confidence (0.0-1.0)
    bbox_x INTEGER,                    -- Bounding box X coordinate
    bbox_y INTEGER,                    -- Bounding box Y coordinate  
    bbox_w INTEGER,                    -- Bounding box width
    bbox_h INTEGER,                    -- Bounding box height
    additional_data TEXT,              -- JSON metadata
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 📊 Export Formats

### Timeline JSON Structure
```json
{
  "session_info": {
    "start_time": 1693123456.78,
    "end_time": 1693123556.78,
    "duration": 100.0,
    "total_events": 15
  },
  "events": [
    {
      "timestamp": 1693123460.12,
      "relative_time": 3.34,
      "event_type": "suspicious_looking",
      "person_id": "person_001",
      "confidence": 0.85,
      "bbox": {"x": 150, "y": 120, "width": 100, "height": 230},
      "additional_data": {"head_yaw": 15, "direction": "left"}
    }
  ]
}
```

## 🔗 Integration Examples

### Basic Integration
```python
from cheatgpt.overlays import HotspotOverlay

# Initialize overlay system
overlay = HotspotOverlay(db_path="session_events.db")

# Add event
overlay.add_event(
    event_type='suspicious_looking',
    person_id='person_001', 
    bbox=(100, 100, 50, 100),
    confidence=0.85
)

# Process frame with overlays
processed_frame = overlay.process_frame(frame, timestamp)
```

### With Video Recording
```python
from cheatgpt.overlays import OverlayVideoRecorder

# Initialize recorder with overlays
recorder = OverlayVideoRecorder(enable_overlay=True)

# Start recording
recorder.start_recording("my_session.mp4")

# Record frames with events
events = [{'event_type': 'suspicious_looking', 'person_id': 'person_001', ...}]
processed_frame = recorder.record_frame(frame, events)

# Stop and get summary
summary = recorder.stop_recording()
```

### Engine Integration
```python
from cheatgpt.overlays import EngineOverlayIntegration, HotspotOverlay

overlay = HotspotOverlay()
integration = EngineOverlayIntegration(overlay)

# Process detection results
processed_frame = integration.process_engine_events(
    events=detection_results,
    timestamp=current_time,
    frame=input_frame
)
```

## 📁 Output Files

### Generated Files
- **Video recordings**: `recordings/session_YYYYMMDD_HHMMSS.mp4`
- **Event timelines**: `recordings/timeline_session_YYYYMMDD_HHMMSS.json`
- **Event database**: `data/events.db`
- **Demo outputs**: `demo_*.json`, `demo_events.db`

### File Locations
```
cheatgptv3/
├── recordings/           # Video recordings and timelines
├── data/                # Event databases
├── demo_*.json         # Demo timeline exports
└── demo_events.db      # Demo event database
```

## 🛠️ Configuration

### Overlay Settings
```python
overlay = HotspotOverlay()
overlay.overlay_duration = 3.0      # Seconds to show overlay
overlay.colors['cheating'] = (0, 0, 255)  # Custom colors
```

### Recording Settings
```python
recorder = OverlayVideoRecorder()
recorder.fps = 30                   # Recording frame rate
recorder.fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 AVC1 codec for web compatibility
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd "d:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3"
   ```

2. **Database Locked**
   ```python
   # Close any open database connections
   overlay.event_db = EventDatabase("new_events.db")
   ```

3. **Video Recording Issues**
   ```python
   # Check codec support
   recorder.fourcc = cv2.VideoWriter_fourcc(*'XVID')
   ```

4. **Performance Issues**
   ```python
   # Disable timeline display
   overlay.process_frame(frame, timestamp, show_timeline=False)
   ```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Tips

1. **Reduce overlay duration** for better performance
2. **Disable timeline display** if not needed
3. **Use lower recording FPS** for smaller files
4. **Clear old events** periodically from database
5. **Use background recording** for non-interactive sessions

## 🎯 Use Cases

1. **Real-time Monitoring**: Live detection with visual feedback
2. **Evidence Recording**: Capture suspicious behavior with timestamps
3. **Behavior Analysis**: Export timelines for pattern analysis
4. **Training Data**: Generate labeled datasets with event markers
5. **Audit Trail**: Maintain complete event logs for review

This hotspot overlay system provides comprehensive visual feedback and data logging for your cheating detection system! 🎯
