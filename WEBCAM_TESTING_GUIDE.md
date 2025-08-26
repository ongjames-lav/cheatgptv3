# CheatGPT3 Real-time Webcam Testing Guide

## 🎯 Overview

Test the complete CheatGPT3 Engine using your webcam for real-time exam proctoring simulation. This will demonstrate all the detection and analysis capabilities in a live environment.

## 🚀 Quick Start

### 1. Prerequisites Check
```bash
# Verify webcam access
python test_webcam_simple.py
```

### 2. Run Real-time Test
```bash
# Start the full CheatGPT3 webcam test
python test_webcam_realtime.py
```

## 🎮 Interactive Controls

Once the webcam test is running, you can use these keyboard controls:

| Key | Action |
|-----|--------|
| **ESC** or **q** | Quit the test |
| **SPACE** | Pause/Resume processing |
| **s** | Save current frame manually |
| **r** | Reset engine statistics |
| **h** | Toggle help overlay |

## 🎭 Testing Behaviors

Try these behaviors to see CheatGPT3 in action:

### ✅ Normal Behavior (Green)
- Sit upright and face forward
- Keep hands visible
- Maintain normal posture

### ⚠️ Suspicious Behavior (Yellow)
- **Look around significantly** - Turn head left/right
- **Lean forward/backward** - Change posture dramatically
- **Move hands near face** - But without phone

### 🚨 Cheating Behavior (Orange/Red)
- **Hold up a phone** - Any rectangular object works
- **Phone near ear** - Simulate phone call
- **Multiple violations** - Combine suspicious behaviors

## 📊 What You'll See

### Visual Indicators
- **Green Boxes**: Normal students
- **Yellow Boxes**: Minor violations (L=Lean, H=Head turn)
- **Orange Boxes**: Phone detected
- **Red Boxes**: Active cheating behavior
- **Track IDs**: Persistent person identification
- **Performance Stats**: FPS, processing time, violations

### Real-time Feedback
- **Console Events**: Violations printed to terminal
- **Evidence Saving**: Automatic capture when cheating detected
- **Performance Metrics**: Live FPS and processing statistics

## 📁 Output Files

The test will generate:

### Evidence Files (Automatic)
```
uploads/evidence/
├── cheating_webcam_test_20250826_143022_person_001.jpg
└── (saved when violations detected)
```

### Manual Screenshots
```
webcam_screenshots/
├── manual_screenshot_20250826_143055.jpg
└── (saved when you press 's')
```

## 🔧 Technical Details

### Performance Expectations
- **FPS**: 3-6 frames per second (suitable for monitoring)
- **Latency**: ~150-300ms processing time per frame
- **Detection**: Real-time person and phone detection
- **Tracking**: Consistent ID assignment across frames

### System Requirements
- **Webcam**: Any USB or built-in camera
- **CPU**: Modern processor (GPU optional but helps)
- **RAM**: 4GB+ recommended
- **Storage**: Space for evidence files

## 🛠️ Troubleshooting

### Camera Issues
```
❌ Cannot access webcam!
```
**Solutions:**
- Close other applications using the camera (Zoom, Teams, etc.)
- Check camera permissions in Windows Settings
- Try a different camera (change `camera_id = 1` in script)
- Restart your computer

### Performance Issues
```
⚠️ Low FPS (< 2 FPS)
```
**Solutions:**
- Close other resource-intensive applications
- Reduce camera resolution (edit webcam script)
- Ensure good lighting for better detection
- Check CPU usage in Task Manager

### Detection Issues
```
🤔 No detections showing
```
**Solutions:**
- Ensure good lighting
- Position yourself clearly in frame
- Make sure you're the only person in view
- Try more pronounced behaviors

## 📈 Expected Results

### Successful Test Session
```
🎓 CheatGPT3 Webcam Test Complete!
==================================================
📊 Session Statistics:
   Runtime: 45.2 seconds
   Frames processed: 168
   Average FPS: 3.7
   Average processing time: 267.3ms
   Total tracks: 1
   Total violations: 3
   Evidence frames saved: 1
   Manual screenshots: 2
==================================================
```

### What This Means
- **Frames processed**: Total frames analyzed
- **Average FPS**: Processing speed
- **Total tracks**: People detected and tracked
- **Total violations**: Policy violations detected
- **Evidence frames**: Automatic cheating captures
- **Manual screenshots**: User-saved frames

## 🎓 Educational Value

This test demonstrates:

1. **Real-time Processing**: Live video analysis
2. **Behavior Detection**: Pose-based behavior analysis
3. **Policy Enforcement**: Rule-based violation detection
4. **Evidence Collection**: Automatic suspicious activity capture
5. **Visual Monitoring**: Professional surveillance interface
6. **Performance Optimization**: Real-time computer vision

## 🚀 Next Steps

After successful webcam testing:

1. **Integration**: Connect to web dashboard
2. **Deployment**: Set up with actual exam cameras
3. **Scaling**: Multi-camera monitoring setup
4. **Customization**: Adjust sensitivity and rules
5. **Alerts**: Connect to notification systems

## 💡 Tips for Best Results

### Lighting
- Use good, even lighting
- Avoid backlighting or shadows
- Natural lighting works best

### Positioning
- Sit 2-3 feet from camera
- Keep upper body in frame
- Avoid cluttered backgrounds

### Testing
- Try each behavior type deliberately
- Watch console output for events
- Check evidence files are generated
- Monitor performance statistics

---

**Ready to test? Run: `python test_webcam_realtime.py`** 🎯
