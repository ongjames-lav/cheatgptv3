#!/usr/bin/env python3
"""
Deep debug the person_id matching issue
"""

import cv2
import numpy as np
from cheatgpt.engines.engine_hybrid import EngineHybrid
import time

def debug_person_id_matching():
    """Debug the exact person_id matching issue"""
    print("🔍 Deep Debugging Person ID Matching")
    print("=" * 50)
    
    # Initialize engine
    engine = EngineHybrid()
    
    # Create test frame
    test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(test_frame, (250, 100), (390, 400), (200, 200, 200), -1)
    cv2.circle(test_frame, (320, 80), 30, (150, 150, 150), -1)
    
    # Mock person detection with integer person_id
    mock_detections = [{
        'bbox': [250, 100, 390, 400],
        'confidence': 0.9,
        'person_id': 0,  # INTEGER person_id
        'class_name': 'person'
    }]
    
    # Create gesture event with string person_id (as the system creates)
    gesture_event = {
        'event_type': 'Suspicious Hand Activity',
        'gesture_type': 'left_wrist_extended_arm_absolute',
        'confidence': 0.9,
        'timestamp': time.time(),
        'special_visual': True,
        'person_id': 'person_000',  # STRING person_id (system format)
        'severity': 'yellow'
    }
    
    print(f"📋 Test Setup:")
    print(f"   Detection person_id: {mock_detections[0]['person_id']} (type: {type(mock_detections[0]['person_id'])})")
    print(f"   Event person_id: {gesture_event['person_id']} (type: {type(gesture_event['person_id'])})")
    
    # Manually inspect the _create_overlay process
    print(f"\n🔍 Tracing _create_overlay process...")
    
    # Step 1: Check how event_lookup is built
    events = [gesture_event]
    
    # Let me trace the exact logic from _create_overlay
    event_lookup = {}
    event_confidence_map = {}
    
    for event in events:
        try:
            person_id = event.get('person_id', 'unknown')
            confidence = event.get('confidence', 0.5)
            
            print(f"   Processing event with person_id: '{person_id}' (type: {type(person_id)})")
            
            # Priority: red > orange > yellow, but also consider confidence
            if person_id not in event_lookup:
                event_lookup[person_id] = event
                event_confidence_map[person_id] = confidence
                print(f"   Added to event_lookup: '{person_id}' -> {event.get('event_type', 'unknown')}")
            else:
                # Priority logic...
                current_priority = {'red': 3, 'orange': 2, 'yellow': 1}.get(
                    event_lookup[person_id].get('severity', 'yellow'), 1)
                new_priority = {'red': 3, 'orange': 2, 'yellow': 1}.get(
                    event.get('severity', 'yellow'), 1)
                
                if new_priority > current_priority or (new_priority == current_priority and confidence > event_confidence_map[person_id]):
                    event_lookup[person_id] = event
                    event_confidence_map[person_id] = confidence
                    print(f"   Updated event_lookup: '{person_id}' -> {event.get('event_type', 'unknown')}")
        except Exception as e:
            print(f"   ERROR processing event: {e}")
            continue
    
    print(f"\n📊 Event Lookup Built:")
    print(f"   event_lookup keys: {list(event_lookup.keys())}")
    print(f"   event_confidence_map keys: {list(event_confidence_map.keys())}")
    
    # Step 2: Check detection processing
    print(f"\n🔍 Tracing detection processing...")
    for detection in mock_detections:
        if 'bbox' not in detection:
            continue
            
        detection_person_id = detection.get('person_id', 'unknown')
        print(f"   Detection person_id: '{detection_person_id}' (type: {type(detection_person_id)})")
        
        # Apply the fix: ensure person_id is string formatted
        if isinstance(detection_person_id, int):
            formatted_person_id = f"person_{detection_person_id:03d}"
            print(f"   Formatted person_id: '{formatted_person_id}'")
        else:
            formatted_person_id = detection_person_id
            print(f"   Person_id already formatted: '{formatted_person_id}'")
        
        # Check if person has event
        if formatted_person_id in event_lookup:
            event = event_lookup[formatted_person_id]
            print(f"   ✅ MATCH FOUND! Event: {event.get('event_type', 'unknown')}")
            
            # Check gesture detection condition
            event_type = event.get('event_type', '')
            is_gesture_event = ('Suspicious Hand Activity' in event_type or 
                              'gesture_type' in event or 
                              'Hand' in event_type or
                              event.get('special_visual', False))
            
            print(f"   Is gesture event: {is_gesture_event}")
            if is_gesture_event:
                print(f"   🎨 MAGENTA COLOR SHOULD BE APPLIED!")
            
        else:
            print(f"   ❌ NO MATCH. Available keys: {list(event_lookup.keys())}")
    
    # Test the actual _create_overlay method
    print(f"\n🧪 Testing actual _create_overlay method...")
    overlay_result = engine._create_overlay(test_frame, mock_detections, events)
    
    # Check color at bounding box
    sample_color = overlay_result[100, 250]  # BGR
    print(f"   Result color: BGR{tuple(sample_color)}")
    
    is_magenta = sample_color[0] > 200 and sample_color[2] > 200 and sample_color[1] < 100
    print(f"   Is magenta: {is_magenta}")
    
    cv2.imwrite('debug_person_id_matching.jpg', overlay_result)
    print(f"\n💾 Debug image saved: debug_person_id_matching.jpg")

if __name__ == "__main__":
    try:
        debug_person_id_matching()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()