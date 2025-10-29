# CheatGPT System Benchmarking Guide

## Overview
This guide explains how to benchmark the CheatGPT detection system with accuracy, precision, recall, and F1-score metrics.

## Quick Start

### Run Benchmark (Without Ground Truth)
```bash
cd "d:\CHEATGPT CAPSTONE\Cheatgpt4\cheatgptv3"
python benchmark_system.py
```

This will:
- Analyze all sessions in the database
- Calculate system-level metrics based on confidence scores
- Generate component-level performance metrics
- Save results to `benchmark_results.json`

## Metrics Explained

### 1. **Precision**
- **Definition**: Of all cheating events detected, how many were actually cheating?
- **Formula**: `TP / (TP + FP)`
- **Interpretation**: 
  - High precision (>0.85): Few false alarms
  - Low precision (<0.65): Many false positives

### 2. **Recall (Sensitivity)**
- **Definition**: Of all actual cheating events, how many did we detect?
- **Formula**: `TP / (TP + FN)`
- **Interpretation**:
  - High recall (>0.85): Catches most cheating
  - Low recall (<0.65): Misses many cheating instances

### 3. **F1-Score**
- **Definition**: Harmonic mean of precision and recall
- **Formula**: `2 × (Precision × Recall) / (Precision + Recall)`
- **Interpretation**: Balanced measure of detection performance

### 4. **Accuracy**
- **Definition**: Overall correctness of all predictions
- **Formula**: `(TP + TN) / (TP + TN + FP + FN)`
- **Interpretation**: Percentage of correct classifications

## Using Ground Truth Data

For accurate benchmarking, you need ground truth labels (manual annotations).

### Step 1: Create Ground Truth File

1. Copy `ground_truth_template.json` to `ground_truth.json`
2. Watch your recorded sessions
3. Manually label each timestamp:
   - `has_event: true` - Cheating behavior occurred
   - `has_event: false` - Normal behavior
   - Include `event_type` and `notes` for documentation

Example ground truth entry:
```json
{
  "session_id": "single_1761452718",
  "timestamp": 15.5,
  "has_event": true,
  "event_type": "phone_detection",
  "notes": "Student clearly using phone to photograph exam"
}
```

### Step 2: Label Multiple Timestamps

Create labels every 5-10 seconds throughout each video:
- Normal behavior timestamps (no cheating)
- Cheating behavior timestamps (with specific event types)

Aim for:
- **Minimum**: 50 labels across 3-5 sessions
- **Good**: 200+ labels across 10+ sessions
- **Excellent**: 500+ labels across 20+ sessions

### Step 3: Run Benchmark with Ground Truth

```bash
python benchmark_system.py
# When prompted, type 'y' to use ground truth
```

## Component Metrics

The benchmark analyzes each detection component:

### Phone Detection
- Total detections
- Average confidence
- High confidence rate (>0.75)

### Head Turning Detection
- Based on pose keypoints
- Angle threshold: ≥40°
- Confidence scores

### Posture/Leaning Detection
- Body angle changes
- Sustained leaning patterns

### Gesture Detection
- Hand movements
- Suspicious gestures

### Temporal Detection
- Multiple violations within time window
- Pattern recognition

## LSTM Model Metrics

From training log (`weights/training_log.txt`):
- **Final Accuracy**: 83.87%
- **F1 Score**: 0.7651
- **Precision (suspicious class)**: 0.84
- **Recall (suspicious class)**: 1.00

## Expected Performance Ranges

### Excellent Performance
- Precision: >0.85
- Recall: >0.80
- F1-Score: >0.82
- Accuracy: >0.90

### Good Performance
- Precision: 0.75-0.85
- Recall: 0.70-0.80
- F1-Score: 0.72-0.82
- Accuracy: 0.85-0.90

### Acceptable Performance
- Precision: 0.65-0.75
- Recall: 0.60-0.70
- F1-Score: 0.62-0.72
- Accuracy: 0.80-0.85

### Needs Improvement
- Any metric below 0.60

## Current System Metrics (Estimated)

Based on confidence distribution and detection patterns:

### Detection Metrics (System-based)
Without ground truth, metrics are estimated based on:
- **High confidence detections** (≥0.75): Likely true positives
- **Medium confidence** (0.50-0.75): Mixed (60% TP, 40% FP)
- **Low confidence** (<0.50): Likely false positives

### Component Breakdown
1. **Phone Detection**: Highest precision (hardware detection)
2. **Head Turning**: Good precision (pose-based)
3. **Posture Changes**: Medium precision (angle-based)
4. **Gestures**: Lower precision (motion patterns)
5. **Temporal**: High precision (pattern confirmation)

## Interpreting Results

### High Precision, Low Recall
- System is conservative (few false alarms)
- Missing some actual cheating events
- **Solution**: Lower detection thresholds

### Low Precision, High Recall
- System is aggressive (many false alarms)
- Catching most cheating but too sensitive
- **Solution**: Increase confidence thresholds

### Balanced (High Precision & Recall)
- Optimal performance
- Good balance between catching cheating and avoiding false alarms

## Benchmark Output Files

### 1. `benchmark_results.json`
Complete benchmark data including:
- Session statistics
- Detection metrics
- Component performance
- Overall performance summary

### 2. Console Output
Human-readable report with:
- Visual metrics display
- Performance breakdown
- System rating

## Tips for Accurate Benchmarking

1. **Diverse Test Data**
   - Include both cheating and non-cheating sessions
   - Various lighting conditions
   - Different camera angles
   - Multiple students

2. **Consistent Labeling**
   - Use same criteria for all labels
   - Document ambiguous cases
   - Include borderline behaviors

3. **Multiple Reviewers**
   - Have 2-3 people label the same videos
   - Calculate inter-rater reliability
   - Resolve disagreements

4. **Regular Benchmarking**
   - Run after system updates
   - Track metrics over time
   - Document configuration changes

## Advanced: Custom Metrics

You can extend `benchmark_system.py` to calculate:
- Per-event-type precision/recall
- Time-to-detection latency
- False positive rate per hour
- Detection consistency across sessions

## Troubleshooting

### No sessions found
- Check database path
- Ensure sessions have been processed
- Verify database connection

### All metrics are 0
- No events detected in database
- Check if detection system is working
- Review confidence thresholds

### Unrealistic metrics
- System-based estimates may not be accurate
- Create ground truth data for true metrics
- Review confidence calibration

## Research-Based Thresholds

Current system uses research-backed thresholds:
- **Head turning**: ≥40° (research-validated)
- **Phone confidence**: ≥0.25 (balanced sensitivity)
- **Temporal window**: 30 frames (1 second at 30 FPS)
- **Confirmation required**: 2+ detections

## Export for Reports

Use `benchmark_results.json` for:
- Thesis/dissertation documentation
- Research papers (performance section)
- System demonstrations
- Client presentations

## Questions?

The benchmark system provides hard numbers for:
✅ Detection accuracy rates
✅ False positive rates
✅ Component-specific performance
✅ Overall system effectiveness

For detailed analysis, create ground truth data and run comprehensive benchmarking.
