#!/usr/bin/env python3
"""
Test enhanced phone detection sensitivity for classroom settings
"""

import os
import sys
sys.path.append(os.path.abspath('.'))

from cheatgpt.engines.engine_hybrid import EngineHybrid
import numpy as np
import time

def test_enhanced_phone_sensitivity():
    """Test enhanced phone detection with classroom scenarios"""
    
    print("🎯 Testing enhanced phone detection sensitivity...")
    
    # Initialize engine
    engine = EngineHybrid()
    
    # Create mock frame
    mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    print("\n📱 Testing phone detection scenarios:")
    print("=" * 50)
    
    # We need to mock the detector outputs instead of last_detections
    # Let's test by directly calling the pose detector with mock phone detections
    
    # Scenario 1: Close range student with phone
    print("\n1. Testing close range student with low confidence phone...")
    
    # Mock person detection
    person_detections = [{
        'bbox': [500, 200, 600, 400],  # Large person (close)
        'conf': 0.8,
        'cls_name': 'person',
        'track_id': 1
    }]
    
    # Mock phone detection with low confidence
    phone_detections = [{
        'bbox': [520, 300, 550, 340],  # Phone near person
        'conf': 0.18,  # Low confidence phone
        'cls_name': 'cell phone'
    }]
    
    # Test if phone is detected near person using our enhanced sensitivity
    pose_results = engine.pose_detector.estimate(mock_frame, phone_detections)
    
    phone_detected = False
    for result in pose_results:
        if result.get('phone_flag', False):
            phone_detected = True
            break
    
    print(f"   Low confidence phone detection: {'✅ SUCCESS' if phone_detected else '❌ FAILED'}")
    if phone_detected:
        print("   📱 Enhanced sensitivity working - low confidence phone detected!")
    
    # Scenario 2: Very far student with tiny phone
    print("\n2. Testing very distant student with tiny phone...")
    
    phone_detections_far = [{
        'bbox': [822, 462, 838, 478],  # Very small phone
        'conf': 0.16,  # Very low confidence
        'cls_name': 'cell phone'
    }]
    
    # Mock very small person (far away)
    person_far = [{
        'bbox': [800, 400, 850, 500],  # Small person
        'conf': 0.7,
        'cls_name': 'person',
        'track_id': 2
    }]
    
    pose_results_far = engine.pose_detector.estimate(mock_frame, phone_detections_far)
    
    far_phone_detected = False
    for result in pose_results_far:
        if result.get('phone_flag', False):
            far_phone_detected = True
            break
    
    print(f"   Distant phone detection: {'✅ SUCCESS' if far_phone_detected else '❌ FAILED'}")
    if far_phone_detected:
        print("   🎯 Distance compensation working!")
    
    # Scenario 3: Test the actual detection pipeline end-to-end
    print("\n3. Testing full detection pipeline...")
    
    # Mock the YOLO detector outputs
    original_detect = engine.yolo_detector.detect
    
    def mock_detect(frame):
        return [
            {
                'bbox': [300, 150, 400, 350],  # Person
                'conf': 0.6,
                'cls_name': 'person'
            },
            {
                'bbox': [415, 200, 445, 230],  # Phone at angle
                'conf': 0.17,  # Very low confidence
                'cls_name': 'cell phone'
            }
        ]
    
    # Temporarily replace detector
    engine.yolo_detector.detect = mock_detect
    
    # Process multiple frames to trigger sustained detection
    events_generated = []
    for i in range(4):
        overlay_frame, events = engine.process_frame(mock_frame, ts=float(i))
        events_generated.extend(events)
        
        if events:
            phone_event = any(event.get('event_type') == 'Phone Usage' for event in events)
            if phone_event:
                print(f"   ✅ Phone usage event detected on frame {i+1}!")
                for event in events:
                    if event.get('event_type') == 'Phone Usage':
                        print(f"   📱 Details: {event.get('details')}")
                        print(f"   🚨 Alarm should trigger!")
                break
    
    # Restore original detector
    engine.yolo_detector.detect = original_detect
    
    total_phone_events = sum(1 for event in events_generated if event.get('event_type') == 'Phone Usage')
    print(f"   Full pipeline test: {'✅ SUCCESS' if total_phone_events > 0 else '❌ FAILED'}")
    
    print("\n" + "=" * 50)
    print("🎯 Enhanced sensitivity test completed!")
    print("📊 Summary of improvements:")
    print("   • Phone confidence threshold: 0.3 → 0.15")
    print("   • IoU threshold: 0.005 → 0.001") 
    print("   • Detection margins: 30%/20% → 50%/40%")
    print("   • Overlap threshold: 0.1 → 0.05")
    print("   • Distance compensation: 1.0x → 2.0x for far subjects")
    print("   • Enhanced for classroom monitoring at various angles")

if __name__ == "__main__":
    test_enhanced_phone_sensitivity()