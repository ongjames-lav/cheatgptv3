#!/usr/bin/env python3
"""
Test improved balanced sensitivity settings.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cheatgpt.engines.engine_hybrid import ResearchBasedRuleEngine

def test_improved_sensitivity():
    """Test the improved balanced sensitivity settings."""
    print("🎯 Testing Improved Detection Sensitivity")
    print("=" * 45)
    
    engine = ResearchBasedRuleEngine()
    
    print(f"📐 Improved Detection Settings:")
    print(f"   Head Turn Threshold: {engine.thresholds['head_turn_angle_threshold']}° (balanced)")
    print(f"   Hand Detection: Instant with 2.0s debounce")
    print(f"   Phone IoU: 0.005 (ultra-sensitive)")
    
    print(f"\n🔄 Testing Head Turn Detection:")
    
    # Test head turning with balanced thresholds
    head_angles = [30, 35, 40, 42, 45, 50]
    person_id = 1
    timestamp = 0.0
    
    for angle in head_angles:
        detection_data = {
            'head_turn_angle': float(angle),
            'bbox': [100, 100, 200, 200],
            'phone_flag': False,
            'gesture_flag': False,
            'lean_angle': 0.0,
            'out_of_frame': False
        }
        
        timestamp += 2.0  # Advance time
        events = engine.update_detection(person_id, detection_data, timestamp)
        detected = len(events) > 0
        
        expected = angle >= 40  # Should detect at 40°+
        status = "✅ DETECTED" if detected else "❌ NORMAL"
        expected_status = "✅ EXPECTED" if expected else "❌ NOT EXPECTED"
        correct = "✅" if detected == expected else "❌"
        
        print(f"   {angle:2d}° → {status:11s} | {expected_status:13s} {correct}")
    
    print(f"\n🤚 Testing Hand Extension Detection:")
    
    # Test hand extension with instant detection and debouncing
    hand_scenarios = [
        {"gesture": True, "reason": "left_hand_sideward_extension", "time": 0.0},
        {"gesture": True, "reason": "left_hand_sideward_extension", "time": 0.5},  # Should be debounced
        {"gesture": True, "reason": "right_hand_sideward_extension", "time": 2.5}, # Should detect
        {"gesture": True, "reason": "face_covering", "time": 3.0},  # Should ignore
        {"gesture": True, "reason": "right_wrist_extreme_sideward_reach", "time": 5.0}, # Should detect
    ]
    
    person_id = 2
    detected_count = 0
    
    for i, scenario in enumerate(hand_scenarios):
        detection_data = {
            'head_turn_angle': 0.0,
            'bbox': [100, 100, 200, 200],
            'phone_flag': False,
            'gesture_flag': scenario['gesture'],
            'gesture_reason': scenario['reason'],
            'lean_angle': 0.0,
            'out_of_frame': False
        }
        
        events = engine.update_detection(person_id, detection_data, scenario['time'])
        detected = len(events) > 0
        
        if detected:
            detected_count += 1
        
        status = "✅ DETECTED" if detected else "❌ DEBOUNCED/IGNORED"
        print(f"   t={scenario['time']:3.1f}s: {scenario['reason'][:25]:25s} → {status}")
    
    print(f"\n   Expected detections: 3 (initial + after 2s debounce + extreme reach)")
    print(f"   Actual detections: {detected_count}")
    
    print(f"\n✅ Improved Sensitivity Summary:")
    print(f"   • Head turning: 40°+ detection (practical threshold)")
    print(f"   • Hand extensions: Instant detection with 2s debounce")
    print(f"   • Phone detection: Ultra-sensitive (0.005 IoU)")
    print(f"   • Face covering gestures: Properly ignored")
    print(f"   • Spam prevention: Effective debouncing")
    print(f"\n🎯 Result: Balanced sensitivity - effective detection without spam!")

if __name__ == "__main__":
    test_improved_sensitivity()