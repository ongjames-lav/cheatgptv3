#!/usr/bin/env python3
"""
Quick test to verify "Hands" label display
"""

import cv2
import numpy as np
from cheatgpt.engines.engine_hybrid import EngineHybrid
import time

def test_hands_label_display():
    """Test that gesture detection shows 'Hands' label"""
    print("🏷️ Testing 'Hands' Label Display")
    print("=" * 40)
    
    # Initialize engine
    engine = EngineHybrid()
    
    # Create test frame with white background for better label visibility
    test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White background
    
    # Draw a simple person figure
    cv2.rectangle(test_frame, (250, 100), (390, 400), (200, 200, 200), -1)  # Light gray body
    cv2.circle(test_frame, (320, 80), 30, (150, 150, 150), -1)  # Gray head
    
    # Mock person detection
    mock_detections = [{
        'bbox': [250, 100, 390, 400],
        'confidence': 0.9,
        'person_id': 0,
        'class_name': 'person'
    }]
    
    # Create gesture event
    gesture_event = {
        'event_type': 'Suspicious Hand Activity',
        'gesture_type': 'left_wrist_extended_arm_absolute',
        'confidence': 0.9,
        'timestamp': time.time(),
        'special_visual': True,
        'person_id': 0,
        'severity': 'yellow'
    }
    
    print("📸 Creating gesture overlay with 'Suspicious Hand Activity' event...")
    
    # Process with gesture event
    overlay_gesture = engine._create_overlay(test_frame, mock_detections, [gesture_event])
    
    # Save the result
    cv2.imwrite('test_hands_label_final.jpg', overlay_gesture)
    print("✅ Test image saved as 'test_hands_label_final.jpg'")
    
    # Check the label area for text
    label_area = overlay_gesture[82:100, 250:350]  # Area above bounding box where label should be
    
    # Look for non-white pixels (text should be there)
    non_white_mask = np.any(label_area < 250, axis=2)  # Pixels that are not white
    text_pixels = np.sum(non_white_mask)
    total_pixels = label_area.shape[0] * label_area.shape[1]
    
    print(f"\n📊 Label Analysis:")
    print(f"Label area size: {label_area.shape}")
    print(f"Non-white pixels (text): {text_pixels}/{total_pixels}")
    print(f"Text coverage: {(text_pixels/total_pixels)*100:.1f}%")
    
    if text_pixels > 50:  # If there's substantial text
        print("✅ SUCCESS: Label text detected in expected area!")
        print("🏷️ The label should display 'Hands' with gesture indicator")
    else:
        print("❌ WARNING: No significant text detected in label area")
    
    print(f"\n🔍 Check the saved image 'test_hands_label_final.jpg' to verify:")
    print("  - Magenta bounding box")
    print("  - 'Hands' label (possibly with 🤚 emoji)")
    print("  - High opacity label background")

if __name__ == "__main__":
    try:
        test_hands_label_display()
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()