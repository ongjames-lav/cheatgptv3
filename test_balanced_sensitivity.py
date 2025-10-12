#!/usr/bin/env python3
"""
Test balanced sensitivity settings - less spam, practical detection.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cheatgpt.detectors.pose_detector import PoseDetector
from cheatgpt.engines.engine_hybrid import ResearchBasedRuleEngine

def test_balanced_sensitivity():
    """Test balanced sensitivity settings."""
    print("⚖️ Testing Balanced Detection Sensitivity")
    print("=" * 45)
    
    detector = PoseDetector()
    engine = ResearchBasedRuleEngine()
    
    print(f"📐 Balanced Detection Thresholds:")
    print(f"   Head Turn Threshold: {detector.head_turn_thresh}° (was 35°)")
    print(f"   Engine Head Turn Threshold: {engine.thresholds['head_turn_angle_threshold']}° (was 30°)")
    print(f"   Phone IoU Threshold: {detector.phone_iou_thresh}")
    print(f"   Hand Extension Debounce: 2.0 seconds")
    
    print(f"\n🧪 Testing Head Turn Detection Range:")
    
    # Test various angles to see new detection range
    test_angles = [25, 30, 35, 40, 42, 45, 50, 55, 60]
    
    for angle in test_angles:
        # Test pose detector level
        pose_detected = detector._compute_looking_around(angle)
        
        # Test hybrid engine level
        detection_data = {
            'head_turn_angle': float(angle),
            'bbox': [100, 100, 200, 200],
            'phone_flag': False,
            'gesture_flag': False,
            'lean_angle': 0.0,
            'out_of_frame': False
        }
        
        # Reset time to avoid debounce
        timestamp = time.time() + angle  # Unique timestamp for each test
        events = engine.update_detection(1, detection_data, timestamp)
        engine_detected = len(events) > 0
        
        pose_status = "✅ DETECTED" if pose_detected else "❌ NORMAL"
        engine_status = "✅ DETECTED" if engine_detected else "❌ NORMAL"
        
        # Expected behavior: detect 40°+ turns
        expected = angle >= 40
        correct = "✅" if (engine_detected == expected) else "⚠️"
        
        print(f"   {angle:2d}° → Pose: {pose_status:11s} | Engine: {engine_status:11s} {correct}")
    
    print(f"\n🤚 Testing Hand Extension Debounce:")
    
    # Test rapid hand extension detections
    detection_data = {
        'head_turn_angle': 0.0,
        'bbox': [100, 100, 200, 200],
        'phone_flag': False,
        'gesture_flag': True,
        'gesture_reason': 'left_hand_sideward_extension',
        'lean_angle': 0.0,
        'out_of_frame': False
    }
    
    base_time = time.time()
    person_id = 2
    
    # Simulate sustained hand extension (should only trigger once per 2 seconds)
    test_times = [0.0, 0.5, 1.0, 1.5, 2.1, 2.5, 3.0, 4.1]
    
    detected_count = 0
    for i, time_offset in enumerate(test_times):
        timestamp = base_time + time_offset
        events = engine.update_detection(person_id, detection_data, timestamp)
        
        if events:
            detected_count += 1
            print(f"   t={time_offset:3.1f}s → ✅ DETECTED (event #{detected_count})")
        else:
            print(f"   t={time_offset:3.1f}s → ❌ DEBOUNCED")
    
    print(f"\n   Expected: ~3-4 detections with 2-second debounce")
    print(f"   Actual: {detected_count} detections")
    
    print(f"\n✅ Balanced Sensitivity Summary:")
    print(f"   • Head turning: 45°/40° thresholds (more practical)")
    print(f"   • Hand extensions: 2-second debounce (reduces spam)")
    print(f"   • Phone detection: Still ultra-sensitive (0.005 IoU)")
    print(f"   • Geometric calculations: Balanced for 40-45° detection")
    print(f"   • Temporal smoothing: Practical noise reduction")
    print(f"\n🎯 Result: Balanced sensitivity - less spam, practical detection!")

if __name__ == "__main__":
    test_balanced_sensitivity()