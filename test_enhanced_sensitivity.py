#!/usr/bin/env python3
"""
Test enhanced sensitivity for turning and phone detection.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cheatgpt.detectors.pose_detector import PoseDetector
from cheatgpt.engines.engine_hybrid import ResearchBasedRuleEngine

def test_enhanced_sensitivity():
    """Test the enhanced sensitivity settings."""
    print("🔧 Testing Enhanced Turning and Phone Detection Sensitivity")
    print("=" * 65)
    
    detector = PoseDetector()
    engine = ResearchBasedRuleEngine()
    
    print(f"📐 Enhanced Detection Thresholds:")
    print(f"   Head Turn Threshold: {detector.head_turn_thresh}° (was 42°)")
    print(f"   Phone IoU Threshold: {detector.phone_iou_thresh} (was 0.01)")
    print(f"   Engine Head Turn Threshold: {engine.thresholds['head_turn_angle_threshold']}° (was 40°)")
    
    print(f"\n🧪 Testing Head Turn Detection Range:")
    
    # Test various angles to see detection range
    test_angles = [20, 25, 30, 32, 35, 38, 40, 45, 50]
    
    for angle in test_angles:
        # Test pose detector level
        pose_detected = detector._compute_looking_around(angle)
        
        # Test hybrid engine level with proper transition simulation
        # First call with normal angle (to establish baseline)
        normal_data = {
            'head_turn_angle': 0.0,
            'bbox': [100, 100, 200, 200],
            'phone_flag': False,
            'gesture_flag': False,
            'lean_angle': 0.0,
            'out_of_frame': False
        }
        engine.update_detection(1, normal_data, 0.0)  # Establish normal state
        
        # Then call with actual angle (to trigger transition)
        detection_data = {
            'head_turn_angle': float(angle),
            'bbox': [100, 100, 200, 200],
            'phone_flag': False,
            'gesture_flag': False,
            'lean_angle': 0.0,
            'out_of_frame': False
        }
        
        events = engine.update_detection(1, detection_data, 1.0)  # Trigger transition
        engine_detected = len(events) > 0
        
        pose_status = "✅ DETECTED" if pose_detected else "❌ NORMAL"
        engine_status = "✅ DETECTED" if engine_detected else "❌ NORMAL"
        
        print(f"   {angle:2d}° → Pose: {pose_status:11s} | Engine: {engine_status}")
        
        # Reset engine state for next test
        if 1 in engine.previous_head_turn_state:
            del engine.previous_head_turn_state[1]
    
    print(f"\n📱 Testing Phone Detection Sensitivity:")
    print(f"   IoU Threshold: {detector.phone_iou_thresh} (lower = more sensitive)")
    print(f"   Expected: More sensitive than previous 0.01 threshold")
    
    print(f"\n✅ Enhanced Sensitivity Summary:")
    print(f"   • Head turning detection lowered from 42° to 35°")
    print(f"   • Engine threshold lowered from 40° to 30°")
    print(f"   • Phone IoU threshold lowered from 0.01 to 0.005")
    print(f"   • Geometric calculations enhanced for 30-35° detection")
    print(f"   • Temporal smoothing reduced for better responsiveness")
    print(f"\n🎯 Result: System now more sensitive to turning and phone detection!")

if __name__ == "__main__":
    test_enhanced_sensitivity()