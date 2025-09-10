# 🎬 CheatGPT Session Playback UI

A comprehensive video playback interface for reviewing CheatGPT detection sessions with interactive hotspot analysis.

## ✨ Features

### 🎥 Video Playback
- **HTML5 Video Player** with MP4 support
- **YouTube-like Controls**: Play, Pause, Forward (10s), Backward (10s)
- **Fullscreen Support** with keyboard shortcut (F)
- **Keyboard Shortcuts**:
  - `Space` - Play/Pause
  - `Arrow Left` - Skip backward 10s
  - `Arrow Right` - Skip forward 10s
  - `F` - Toggle fullscreen

### 🎯 Interactive Timeline
- **Clickable Timeline Bar** - Click anywhere to seek
- **Red Hotspot Markers** - Visual indicators for detected events
- **Draggable Handle** - Smooth seeking control
- **Event Color Coding**:
  - 🔴 Red: Phone detected
  - 🟣 Purple: Suspicious gesture  
  - 🟠 Orange: Suspicious looking
  - 🟡 Yellow: Suspicious lean

### ⚠️ Event Analysis
- **Real-time Event Overlay** - Shows event type during playback
- **Clickable Hotspot Markers** - Jump directly to event timestamps
- **Hover Tooltips** - Preview event details without clicking
- **Events Panel** - Complete list of all detected events with timestamps
- **Event Timeline Export** - JSON data available for analysis

### 📱 Responsive Design
- **Mobile-friendly** interface
- **Adaptive layout** for different screen sizes
- **Touch-friendly** controls

## 🚀 Quick Start

### 1. Record a Session
```bash
# Start detection with recording enabled
python run_enhanced_detection.py

# Let it run for a while to capture some events
# Press 'q' to stop recording
```

### 2. Launch Playback UI
```bash
# Start the playback server
python demo_playback_ui.py

# Or manually start the Flask server
python playback_server.py
```

### 3. Open in Browser
- **Homepage**: http://localhost:5000
- **Direct Playback**: http://localhost:5000/playback/session_YYYYMMDD_HHMMSS

## 📁 File Structure

```
recordings/
├── session_20250910_120533.mp4          # Recorded video with overlays
├── timeline_session_20250910_120533.json # Event timeline data
└── ...

videos/
├── session_session_20250910_120533_realtime_enhanced_detection_*.mp4
└── ...                                   # Engine-generated videos

templates/
├── base.html                            # Base template
├── session_list.html                   # Sessions overview
└── session_playback.html               # Main playback interface
```

## 🎮 Usage Guide

### Session List Page
1. **View All Sessions** - See all recorded detection sessions
2. **Session Stats** - Duration, event count, start time
3. **Quick Actions** - Watch recording or view raw timeline data

### Playback Interface
1. **Load Video** - Automatically loads the session's MP4 file
2. **Navigate Timeline** - Click timeline or use keyboard shortcuts
3. **Jump to Events** - Click red markers to jump to specific events
4. **View Event Details** - Hover over markers or check events panel
5. **Control Playback** - Use standard video controls or keyboard

### Event Types
The system detects and marks several types of suspicious activities:

- **🤚 Suspicious Gesture** - Hand movements near face (potential cheating gestures)
- **👀 Suspicious Looking** - Looking away from screen (potential distraction)
- **📱 Phone Detected** - Mobile device detected in frame
- **🔄 Suspicious Lean** - Body positioning suggesting cheating behavior

## 🛠️ Technical Details

### Data Format
**Timeline JSON Structure:**
```json
{
  "session_info": {
    "start_time": 1757477133.4196475,
    "end_time": 1757477145.1071799,
    "duration": 11.687532424926758,
    "total_events": 3
  },
  "events": [
    {
      "timestamp": 1757477135.2,
      "event_type": "suspicious_gesture",
      "person_id": "person_000",
      "confidence": 0.85,
      "bbox_x": 320,
      "bbox_y": 240,
      "bbox_w": 120,
      "bbox_h": 160
    }
  ]
}
```

### API Endpoints
- `GET /` - Session list page
- `GET /playback/<session_id>` - Playback interface
- `GET /api/sessions` - JSON list of all sessions
- `GET /api/session/<session_id>` - JSON data for specific session
- `GET /recordings/<filename>` - Serve recording files
- `GET /videos/<filename>` - Serve video files

### Browser Compatibility
- **Chrome/Edge**: Full support
- **Firefox**: Full support  
- **Safari**: Full support
- **Mobile Browsers**: Responsive design

## 🔧 Configuration

### Video Settings
The playback system automatically detects and serves:
- **Recording Videos**: `recordings/session_*.mp4` (with overlays)
- **Engine Videos**: `videos/session_*.mp4` (raw detection output)

### Timeline Synchronization  
Events are automatically synchronized with video playback using:
- **Relative timestamps** (event_time - session_start_time)
- **Real-time event detection** during playback
- **1-second tolerance** for event display

## 📊 Analytics Integration

The playback UI provides data for analysis:

1. **Event Distribution** - See which types of events occur most
2. **Temporal Patterns** - Identify when cheating typically happens
3. **Session Comparison** - Compare event patterns across sessions
4. **Confidence Analysis** - Review detection accuracy

## 🐛 Troubleshooting

### Common Issues

**Video Not Playing:**
- Check file path in browser network tab
- Ensure MP4 file exists in recordings/ or videos/
- Verify browser supports H.264 codec

**Hotspots Not Appearing:**
- Check if timeline JSON contains events
- Verify event timestamps are within video duration
- Check browser console for JavaScript errors

**Timeline Sync Issues:**
- Ensure event timestamps are relative to session start
- Check video duration matches timeline duration
- Verify playback rate is 1.0x

### Debug Mode
Enable debug logging:
```bash
export FLASK_DEBUG=1
python playback_server.py
```

## 🚀 Advanced Usage

### Custom Event Analysis
```javascript
// Access session data in browser console
console.log(SESSION_DATA.events);

// Filter events by type
const gestures = SESSION_DATA.events.filter(e => e.event_type === 'suspicious_gesture');

// Calculate event frequency
const eventRate = SESSION_DATA.events.length / SESSION_DATA.duration;
```

### Integration with External Tools
Export timeline data for analysis in:
- **Excel/CSV** - Convert JSON to spreadsheet
- **Python/Pandas** - Load JSON for data analysis  
- **Video Editors** - Use timestamps for manual review

---

## 💡 Tips for Best Results

1. **Record Longer Sessions** - More events = better timeline visualization
2. **Good Lighting** - Improves detection accuracy and video quality
3. **Stable Camera** - Reduces false positives from camera movement
4. **Clear Audio** - Future versions may include audio analysis

**Happy reviewing! 🎬✨**
