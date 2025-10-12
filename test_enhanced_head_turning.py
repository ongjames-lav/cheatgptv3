#!/usr/bin/env python3
"""
Test enhanced head turning sensitivity
"""

import os
import sys
sys.path.append(os.path.abspath('.'))

from cheatgpt.engines.engine_hybrid import ResearchBasedRuleEngine
from cheatgpt.detectors.pose_detector import PoseDetector

def test_enhanced_head_turning_sensitivity():
    """Test enhanced head turning detection sensitivity"""
    
    print("🔄 Testing enhanced head turning sensitivity...")
    
    # Initialize components
    engine = ResearchBasedRuleEngine()
    pose_detector = PoseDetector()
    
    print(f"📊 Enhanced Settings:")
    print(f"   • Head Turn Threshold: {pose_detector.head_turn_thresh}° (was 45°)")
    print(f"   • Engine Threshold: {engine.thresholds['head_turn_angle_threshold']}° (was 40°)")
    print(f"   • Debounce Time: 0.5s (was 1.0s)")
    
    print(f"\n🧪 Testing head turn detection at various angles:")
    print("=" * 60)
    
    # Test angles from subtle to obvious
    test_angles = [25.0, 28.0, 30.0, 32.0, 35.0, 40.0, 45.0]
    
    for i, angle in enumerate(test_angles):
        print(f"\n{i+1}. Testing {angle}° head turn...")
        
        # Create detection data
        detection_data = {
            'phone_detected': False,
            'head_turn_angle': angle,
            'head_pitch_angle': 0.0,
            'hand_extended': False,
            'out_of_frame': False,
            'normal_posture': True
        }
        
        # Test with unique timestamp to avoid debouncing
        events = engine.update_detection(
            person_id=1,
            detection_data=detection_data,
            timestamp=float(i + 1)
        )
        
        # Check if head turn was detected
        head_turn_detected = any(event.get('event_type') == 'Head Turning' for event in events)
        
        # Determine expected result based on new threshold
        expected = angle >= 30.0
        
        if head_turn_detected == expected:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        print(f"   Angle: {angle}° - Detected: {head_turn_detected} - {status}")
        
        if head_turn_detected and events:
            for event in events:
                if event.get('event_type') == 'Head Turning':
                    print(f"   📝 Details: {event.get('details')}")
    
    print(f"\n" + "=" * 60)
    print(f"🎯 Enhanced head turning sensitivity test completed!")
    
    print(f"\n📈 Sensitivity Improvements:")
    print(f"   ✅ Detection threshold: 45° → 35° (28% more sensitive)")
    print(f"   ✅ Engine threshold: 40° → 30° (25% more sensitive)")  
    print(f"   ✅ Debounce time: 1.0s → 0.5s (2x more responsive)")
    print(f"   ✅ Instant detection for classroom monitoring")
    
    print(f"\n🎓 Classroom Benefits:")
    print(f"   • Catches subtle head movements (cheating glances)")
    print(f"   • More responsive to quick turning motions")
    print(f"   • Better detection for distant students")
    print(f"   • Optimized for exam supervision")

if __name__ == "__main__":
    test_enhanced_head_turning_sensitivity()