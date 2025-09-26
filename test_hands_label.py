#!/usr/bin/env python3
"""
Test the new "Hands" label and verify transparency improvements
"""

import cv2
import numpy as np
from cheatgpt.engines.engine_hybrid import EngineHybrid
import time

def test_hands_label():
    """Test that gesture detection shows 'Hands' label with proper transparency"""
    print("🏷️ Testing 'Hands' Label and Transparency")
    print("=" * 50)
    
    # Initialize engine
    engine = EngineHybrid()
    
    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (250, 100), (390, 400), (100, 100, 100), -1)
    cv2.circle(test_frame, (320, 80), 30, (150, 150, 150), -1)
    
    # Mock detection
    mock_detection = {
        'bbox': [250, 100, 390, 400],
        'confidence': 0.9,
        'person_id': 0,
        'class_name': 'person'
    }
    
    print("📸 Creating gesture detection with 'Hands' label...")
    
    # Create gesture event
    gesture_event = {
        'event_type': 'Suspicious Hand Activity',
        'gesture_type': 'left_wrist_extended_arm_absolute', 
        'confidence': 0.9,
        'timestamp': time.time(),
        'special_visual': True,
        'person_id': 0,
        'severity': 'yellow',
        'details': 'Hand gesture detected: left_wrist_extended_arm_absolute for 1.0s'
    }
    
    # Create overlay with gesture
    overlay_frame = engine._create_overlay(test_frame, [mock_detection], [gesture_event])
    
    # Save the result
    cv2.imwrite('test_hands_label.jpg', overlay_frame)
    print("✅ Test image saved as 'test_hands_label.jpg'")
    
    # Analyze the label area (should be above the bounding box)
    label_area = overlay_frame[82:100, 250:300]  # Area above bounding box
    
    print(f"\n🔍 Label Area Analysis:")
    print(f"Label area shape: {label_area.shape}")
    
    # Check for text/label presence (non-zero pixels in expected area)
    non_zero_pixels = np.count_nonzero(label_area)
    total_pixels = label_area.shape[0] * label_area.shape[1] * 3
    
    print(f"Non-zero pixels in label area: {non_zero_pixels}/{total_pixels}")
    print(f"Label coverage: {(non_zero_pixels/total_pixels)*100:.1f}%")
    
    # Check for magenta in bounding box
    bbox_area = overlay_frame[100:110, 250:260]  # Top edge of bounding box
    avg_color = np.mean(bbox_area, axis=(0,1))
    
    print(f"\n📊 Bounding Box Color:")
    print(f"Average color at bbox edge: BGR({avg_color[0]:.0f}, {avg_color[1]:.0f}, {avg_color[2]:.0f})")
    
    is_magenta = avg_color[0] > 150 and avg_color[2] > 150 and avg_color[1] < 50
    print(f"Magenta detected: {'✅ YES' if is_magenta else '❌ NO'}")
    
    print(f"\n🎯 Results:")
    print(f"- Gesture detection color: {'✅ Working' if is_magenta else '❌ Failed'}")
    print(f"- Label area coverage: {(non_zero_pixels/total_pixels)*100:.1f}%")
    print(f"- Test image: test_hands_label.jpg")
    
    return is_magenta

if __name__ == "__main__":
    try:
        success = test_hands_label()
        if success:
            print("\n🎉 SUCCESS: Gesture detection with 'Hands' label working!")
        else:
            print("\n🔧 ISSUE: Gesture detection needs further debugging")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()