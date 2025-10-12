#!/usr/bin/env python3
"""
Test to verify that head turning detection now works with inclusive 40° threshold
"""

import os
import sys
sys.path.append(os.path.abspath('.'))

from cheatgpt.engines.engine_hybrid import ResearchBasedRuleEngine

def test_inclusive_40_degree_threshold():
    """Test that exactly 40° head turn is now detected"""
    
    print("Testing inclusive 40° threshold...")
    
    # Initialize the research engine
    engine = ResearchBasedRuleEngine()
    
    # Test exact boundary cases
    test_angles = [39.0, 39.5, 40.0, 40.5, 41.0, 42.0]
    
    for i, angle in enumerate(test_angles):
        print(f"\n--- Testing {angle}° head turn ---")
        
        # Create detection data with specific head turn angle
        detection_data = {
            'phone_detected': False,
            'head_turn_angle': angle,
            'head_pitch_angle': 0.0,  # Normal pitch
            'hand_extended': False,
            'out_of_frame': False,
            'normal_posture': True
        }
        
        # Update detection with unique timestamp to avoid debouncing
        events = engine.update_detection(
            person_id=1,
            detection_data=detection_data,
            timestamp=float(i + 1)  # Use different timestamp for each test
        )
        
        # Check if head turn was detected (correct event type key and value)
        head_turn_detected = any(event.get('event_type') == 'Head Turning' for event in events)
        
        # Debug: Print all events to see what's being generated
        if events:
            print(f"Events generated: {[event.get('event_type', event.get('type', 'unknown')) for event in events]}")
        else:
            print("No events generated")
        
        print(f"Angle: {angle}° - Detected: {head_turn_detected}")
        
        # Verify expected behavior
        if angle >= 40.0:
            if head_turn_detected:
                print(f"✅ Correctly detected head turn at {angle}°")
            else:
                print(f"❌ Expected head turn detection at {angle}° but got none")
        else:
            if not head_turn_detected:
                print(f"✅ Correctly ignored head turn at {angle}°")
            else:
                print(f"❌ Unexpected head turn detection at {angle}°")
    
    print("\n🎯 Inclusive threshold test completed!")
    print("Testing shows head turning detection with >= 40° threshold")

if __name__ == "__main__":
    test_inclusive_40_degree_threshold()