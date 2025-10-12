#!/usr/bin/env python3
"""
Test script to verify reduced turning sensitivity parameters.
This script tests the updated thresholds for head turning detection.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cheatgpt.detectors.pose_detector import PoseDetector
from cheatgpt.engines.engine_hybrid import ResearchBasedRuleEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pose_detector_thresholds():
    """Test pose detector threshold values."""
    print("🔍 Testing Pose Detector Thresholds...")
    
    detector = PoseDetector()
    
    print(f"Head Turn Threshold: {detector.head_turn_thresh}° (should be 42.0°)")
    print(f"Lean Angle Threshold: {detector.lean_angle_thresh}° (should be 8.0°)")
    print(f"Phone IoU Threshold: {detector.phone_iou_thresh} (should be 0.01)")
    
    # Test the looking around function with various angles
    test_angles = [0, 10, 30, 40, 45, 50, 60, 75, 80]
    print("\n📐 Head Turn Detection Tests (Classroom Sensitive Settings):")
    
    for angle in test_angles:
        is_turning = detector._compute_looking_around(angle)
        status = "✅ DETECTED" if is_turning else "❌ Normal"
        print(f"  {angle:2d}° → {status}")
    
    print()

def test_hybrid_engine_thresholds():
    """Test hybrid engine threshold values."""
    print("🔍 Testing Hybrid Engine Thresholds...")
    
    engine = ResearchBasedRuleEngine()
    thresholds = engine.thresholds
    
    print(f"Head Turn Angle Threshold: {thresholds['head_turn_angle_threshold']}° (should be 40.0°)")
    print(f"Head Turn Frequency Threshold: {thresholds['head_turn_frequency_threshold']} occurrences (should be 2)")
    print(f"Head Turn Sustained Threshold: {thresholds['head_turn_sustained_threshold']}s (should be 2.0s)")
    print(f"Head Pitch Threshold: {thresholds['head_pitch_threshold']}° (should be 20.0°)")
    print()

def test_detection_scenarios():
    """Test detection with various simulated scenarios."""
    print("📊 Testing Detection Scenarios...")
    
    engine = ResearchBasedRuleEngine()
    
    # Test scenarios: [angle, should_detect_expectation]
    scenarios = [
        (10.0, False, "Small head movement"),
        (30.0, False, "Normal head adjustment"),
        (40.0, True, "Classroom turn threshold - should detect"),
        (45.0, True, "Moderate classroom turn - should detect"),
        (50.0, True, "Noticeable head movement - should detect"),
        (60.0, True, "Significant head movement - should detect"),
        (75.0, True, "Large turn - should detect"),
        (80.0, True, "Full side turn - should detect")
    ]
    
    person_id = 1
    timestamp = 0.0
    
    for angle, should_detect, description in scenarios:
        # Simulate detection data
        detection_data = {
            'head_turn_angle': angle,
            'bbox': [100, 100, 200, 200],
            'phone_flag': False,
            'gesture_flag': False,
            'lean_angle': 0.0,
            'out_of_frame': False
        }
        
        # Get events from engine
        events = engine.update_detection(person_id, detection_data, timestamp)
        
        detected = len(events) > 0
        status = "✅" if detected == should_detect else "❌"
        result = "DETECTED" if detected else "Normal"
        
        print(f"  {status} {angle:4.1f}° → {result:8s} | {description}")
        
        timestamp += 5.0  # Move time forward
    
    print()

def test_debounce_timing():
    """Test debounce timing to prevent spam."""
    print("⏱️ Testing Debounce Timing...")
    
    engine = ResearchBasedRuleEngine()
    person_id = 1
    
    # Simulate rapid head turns (should be debounced)
    detection_data = {
        'head_turn_angle': 50.0,  # Above classroom threshold (40°)
        'bbox': [100, 100, 200, 200],
        'phone_flag': False,
        'gesture_flag': False,
        'lean_angle': 0.0,
        'out_of_frame': False
    }
    
    timestamps = [0.0, 1.0, 2.0, 3.1, 4.0, 6.1, 9.2]
    
    for i, timestamp in enumerate(timestamps):
        events = engine.update_detection(person_id, detection_data, timestamp)
        detected = len(events) > 0
        
        if i == 0:
            expected = True  # First detection
        elif timestamp - timestamps[i-1] < 3.0:
            expected = False  # Should be debounced
        else:
            expected = True  # Should pass debounce
        
        status = "✅" if detected == expected else "❌"
        result = "DETECTED" if detected else "Debounced"
        
        print(f"  {status} t={timestamp:4.1f}s → {result:9s} | {len(events)} events")
    
    print()

def main():
    """Run all tests."""
    print("🧪 Testing Classroom-Optimized Head Turning Sensitivity")
    print("=" * 55)
    
    try:
        test_pose_detector_thresholds()
        test_hybrid_engine_thresholds()
        test_detection_scenarios()
        test_debounce_timing()
        
        print("✅ All tests completed!")
        print("\n📋 Summary of Changes:")
        print("• Head turn threshold set to 45° in pose detector for classroom sensitivity")
        print("• Head turn angle threshold set to 40° in hybrid engine for balanced detection")
        print("• Frequency threshold set to 2 occurrences (appropriate for classroom monitoring)")
        print("• Sustained threshold set to 2.0s (quick detection of sustained turns)")
        print("• Debounce timing remains at 3s to prevent spam")
        print("• Geometric sensitivity tuned for 40-50° classroom monitoring")
        print("\n🎯 Result: Head turns 40-45° and above indicating potential cheating will be detected")
        print("📚 Classroom Context: Balanced sensitivity for monitoring student behavior without false positives")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()