#!/usr/bin/env python3
"""
Debug the gesture detection color logic by tracing execution
"""

import cv2
import numpy as np
from cheatgpt.engines.engine_hybrid import EngineHybrid
import time

def debug_gesture_detection():
    """Debug why gesture detection colors aren't working"""
    print("🐛 Debugging Gesture Detection Color Logic")
    print("=" * 50)
    
    # Initialize engine
    engine = EngineHybrid()
    
    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (250, 100), (390, 400), (100, 100, 100), -1)  # Body
    cv2.circle(test_frame, (320, 80), 30, (150, 150, 150), -1)  # Head
    
    # Mock normal person detection
    mock_detections = [{
        'bbox': [250, 100, 390, 400],  # x1, y1, x2, y2
        'confidence': 0.9,
        'person_id': 0,
        'class_name': 'person'
    }]
    
    print("📋 Step 1: Testing normal detection...")
    overlay_normal = engine._create_overlay(test_frame, mock_detections, [])  # No events
    print("✅ Normal overlay created")
    
    print("\n📋 Step 2: Setting up gesture event...")
    current_time = time.time()
    
    gesture_event = {
        'event_type': 'Suspicious Hand Activity',
        'gesture_type': 'left_wrist_extended_arm_absolute',
        'confidence': 0.9,
        'timestamp': current_time,
        'special_visual': True,
        'person_id': 0,  # This must match the detection person_id
        'severity': 'yellow',
        'details': 'Hand gesture detected: left_wrist_extended_arm_absolute for 1.0s'
    }
    
    print(f"✅ Created gesture event with person_id: {gesture_event['person_id']}")
    print(f"   Event type: '{gesture_event['event_type']}'")
    print(f"   Special visual: {gesture_event['special_visual']}")
    
    print("\n📋 Step 3: Creating gesture overlay with events parameter...")
    
    # Pass the event as a list to _create_overlay (this is the correct way!)
    overlay_gesture = engine._create_overlay(test_frame, mock_detections, [gesture_event])
    print("✅ Gesture overlay created with events parameter")
    
    # Check if the colors actually changed
    normal_sample = overlay_normal[100, 250]
    gesture_sample = overlay_gesture[100, 250]
    
    print(f"\n📊 Color Comparison:")
    print(f"Normal:  BGR{tuple(normal_sample)}")
    print(f"Gesture: BGR{tuple(gesture_sample)}")
    
    is_different = not np.array_equal(normal_sample, gesture_sample)
    print(f"Colors different: {is_different}")
    
    if is_different:
        print("🎉 SUCCESS: Colors are different!")
    else:
        print("❌ PROBLEM: Colors are identical")
        
        # Let's debug further by checking the detection parameters
        print("\n🔍 Deep debugging...")
        print("Checking if person_id matches between detection and events...")
        
        detection_person_id = mock_detections[0].get('person_id', 'unknown')
        print(f"Detection person_id: {detection_person_id}")
        print(f"Event person_id: {gesture_event['person_id']}")
        
        # Check the exact key format expected
        print("\nChecking key formats...")
        for key in engine.active_events.keys():
            print(f"Active event key: '{key}'")
        
        # Try different key formats
        alternative_keys = [0, "0", f"person_{0}"]
        for alt_key in alternative_keys:
            if alt_key in engine.active_events:
                print(f"✅ Found event with key: {alt_key}")
            else:
                print(f"❌ No event found with key: {alt_key}")

if __name__ == "__main__":
    try:
        debug_gesture_detection()
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()