# 🧠 LSTM Integration in CheatGPT System - How It Works

## 📋 Overview

The LSTM (Long Short-Term Memory) neural network is a **temporal behavior analyzer** that examines sequences of behavioral features over time to predict suspicious exam behaviors. It's the "brain" that learns patterns from past behavior to identify cheating attempts.

## 🔄 LSTM Trigger Flow

```
📹 Frame Input → 🔍 YOLO Detection → 🧍 Pose Analysis → 📊 Feature Extraction → 🧠 LSTM Analysis → 🚨 Event Generation
```

## 🎯 When LSTM is Triggered

### **1. Sequence Length Condition**
```python
if len(self.behavior_history) >= self.sequence_length and self.lstm_classifier.is_loaded:
```
- **Default sequence_length**: 10 frames
- **Trigger**: Only when we have at least 10 behavioral feature vectors
- **Why**: LSTM needs temporal context to identify patterns

### **2. Feature Accumulation Process**
```
Frame 1: [lean_flag, look_flag, phone_flag, gesture_flag, ...]  ← Added to history
Frame 2: [lean_flag, look_flag, phone_flag, gesture_flag, ...]  ← Added to history
Frame 3: [lean_flag, look_flag, phone_flag, gesture_flag, ...]  ← Added to history
...
Frame 10: [lean_flag, look_flag, phone_flag, gesture_flag, ...] ← LSTM TRIGGERED!
```

## 🧠 Two LSTM Systems Available

### **A. Enhanced LSTM (88.64% Accuracy) - Our Realistic Model**
- **Classes**: `['normal', 'suspicious_gesture', 'suspicious_looking', 'mixed_suspicious']`
- **Features**: 12-dimensional vectors with COCO spatial enhancements
- **Model**: `realistic_lstm_behavior.pth`
- **Usage**: `predict_enhanced()` method

### **B. Standard LSTM (Fallback)**
- **Classes**: `['normal', 'suspicious']`
- **Features**: Basic behavioral flags
- **Model**: `lstm_behavior.pth`
- **Usage**: `predict()` method

## 📊 Feature Vector Creation

Each frame creates a behavioral feature vector:

```python
# Example feature vector (12 dimensions)
behavior_vector = [
    lean_flag,          # 0 or 1
    look_flag,          # 0 or 1  
    phone_flag,         # 0 or 1
    gesture_flag,       # 0 or 1
    lean_angle,         # degrees
    head_turn_angle,    # degrees
    confidence,         # 0.0-1.0
    center_x,           # normalized position
    center_y,           # normalized position
    bbox_area,          # normalized area
    combined_suspicious, # composite score
    spatial_offset      # spatial consistency
]
```

## 🔄 LSTM Processing Pipeline

### **Step 1: History Management**
```python
# Add current frame features
self.behavior_history.append(behavior_vector)

# Maintain sliding window
if len(self.behavior_history) > max_history:
    self.behavior_history = self.behavior_history[-max_history:]
```

### **Step 2: Multi-Scale Analysis**
```python
# Test different sequence lengths for robustness
sequence_lengths = [10, 15]  # frames
for seq_len in sequence_lengths:
    sequence = np.array(self.behavior_history[-seq_len:])
    prediction = lstm_classifier.predict(sequence)
```

### **Step 3: Confidence Gating**
```python
# Only trust high-confidence predictions
if prediction['confidence'] > 0.65:
    # Process the prediction
    predicted_label = prediction['predicted_label']
    confidence = prediction['confidence']
```

## 🚨 Event Generation from LSTM

### **Severity Mapping**
```python
severity_map = {
    'normal': None,                    # No event
    'suspicious_gesture': 'orange',    # Hand near face
    'suspicious_looking': 'yellow',    # Head turning
    'mixed_suspicious': 'orange',      # Multiple behaviors
    'phone_use': 'red',               # Phone detected
    'cheating': 'red'                 # High confidence cheating
}
```

### **Dynamic Severity Adjustment**
- **High Confidence (>0.9)**: Escalate yellow → orange
- **Low Confidence (<0.75)**: De-escalate red → orange
- **Multiple Behaviors**: Escalate severity level

## 🎮 Real-Time Usage Example

```python
# During webcam processing:
for frame in webcam_feed:
    # 1. Extract features from current frame
    detections = yolo_detector.detect(frame)
    pose_data = pose_detector.estimate(frame)
    behavior_vector = extract_features(detections, pose_data)
    
    # 2. Add to temporal history
    engine.behavior_history.append(behavior_vector)
    
    # 3. LSTM triggers when sequence_length reached
    if len(engine.behavior_history) >= 10:
        prediction = lstm_classifier.predict(recent_sequence)
        
        # 4. Generate events based on prediction
        if prediction['predicted_label'] != 'normal':
            event = create_violation_event(prediction)
            alert_proctor(event)
```

## 📈 Performance Characteristics

### **Enhanced LSTM (Realistic Model)**
- **Accuracy**: 88.64% (realistic, not suspicious 100%)
- **Parameters**: 228,008
- **Inference Time**: ~5-10ms per prediction
- **Memory**: ~50MB GPU memory

### **Confidence Thresholds**
- **Low**: 0.65-0.75 (Yellow alerts)
- **Medium**: 0.75-0.90 (Orange alerts) 
- **High**: 0.90+ (Red alerts)

## 🔧 Configuration Options

### **In `.env` file:**
```bash
# LSTM sequence length (frames needed before prediction)
LSTM_SEQUENCE_LENGTH=10

# Confidence threshold for triggering events
LSTM_CONFIDENCE_THRESHOLD=0.65

# Enhanced LSTM model path
ENHANCED_LSTM_MODEL=weights/realistic_lstm_behavior.pth
```

## 🎯 Why LSTM is Essential

1. **Temporal Pattern Recognition**: Detects behavioral sequences, not just single frames
2. **False Positive Reduction**: Ignores brief accidental movements
3. **Context Awareness**: Considers behavior history for better accuracy
4. **Cheating Pattern Learning**: Trained on real exam scenarios
5. **Multi-Task Learning**: Predicts multiple behavior types simultaneously

## 🚀 Enhanced Integration Features

- **COCO Spatial Enhancement**: Uses COCO dataset patterns for improved accuracy
- **Multi-Task Outputs**: Predicts gesture + looking behaviors separately
- **Auxiliary Heads**: Additional neural network branches for specific tasks
- **Confidence Calibration**: Realistic confidence scores (not overconfident)
- **Temporal Smoothing**: Reduces noise in predictions over time

---

**💡 Key Insight**: The LSTM acts as the "temporal memory" of the system, remembering how behaviors evolve over time to make intelligent predictions about exam integrity violations.
