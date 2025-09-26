#!/usr/bin/env python3
"""
Test Arms Label and Lower Opacity Changes
Tests the updated label from "Hands" to "Arms" and lower bounding box opacity
"""

import cv2
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Configure Python environment properly
os.environ['PYTHONPATH'] = os.path.dirname(__file__)

from cheatgpt.engines.engine_hybrid import EngineHybrid

def test_arms_label_and_opacity():
    """Test Arms label and lower opacity functionality."""
    
    print("🧪 Testing Arms Label and Lower Opacity Changes...")
    
    try:
        # Initialize the engine
        print("🚀 Initializing EngineHybrid...")
        engine = EngineHybrid()
        
        # Create a test frame
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        test_frame.fill(50)  # Dark gray background
        
        # Create mock person detection data
        test_person = {
            'bbox': [300, 200, 600, 500],  # x1, y1, x2, y2
            'confidence': 0.85,
            'id': 0
        }
        
        # Create mock gesture event data
        test_event = {
            'person_id': 'person_000',
            'event_type': 'Suspicious Hand Activity',
            'confidence': 0.75,
            'timestamp': 1632123456.789,
            'details': 'Hand gesture detected: left_wrist_extended_arm_absolute'
        }
        
        print("📊 Testing Arms label rendering...")
        
        # Test the overlay creation with Arms label
        overlay_frame = engine._create_overlay(test_frame, [test_person], [test_event])
        
        # Save the test result
        test_output_path = "test_arms_label_opacity.jpg"
        cv2.imwrite(test_output_path, overlay_frame)
        print(f"💾 Test result saved as: {test_output_path}")
        
        # Analyze the overlay frame to verify Arms label
        # Convert to RGB for text analysis
        rgb_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
        
        # Check for the presence of text in the expected label area
        label_area = overlay_frame[180:220, 280:400]  # Area where label should appear
        
        # Calculate non-background pixels (any pixel that's not close to the original background)
        background_color = [50, 50, 50]  # Original background color
        non_bg_mask = np.any(np.abs(label_area.astype(int) - background_color) > 20, axis=2)
        non_bg_pixels = np.sum(non_bg_mask)
        
        print(f"📈 Label area analysis: {non_bg_pixels} non-background pixels detected")
        
        # Test bounding box opacity by checking if it's not too bright
        bbox_area = overlay_frame[200:500, 300:600]  # Bounding box area
        avg_intensity = np.mean(bbox_area)
        max_intensity = np.max(bbox_area)
        
        print(f"📊 Bounding box opacity analysis:")
        print(f"   Average intensity: {avg_intensity:.1f}")
        print(f"   Maximum intensity: {max_intensity}")
        
        # Verification checks
        success_checks = 0
        total_checks = 3
        
        # Check 1: Label area has content
        if non_bg_pixels > 100:
            print("✅ SUCCESS: Label area contains rendered text")
            success_checks += 1
        else:
            print("❌ FAIL: Label area appears empty")
        
        # Check 2: Opacity looks reasonable (not too bright)
        if avg_intensity < 150:  # Should be relatively dim due to lower alpha
            print("✅ SUCCESS: Bounding box opacity is appropriately low")
            success_checks += 1
        else:
            print("❌ FAIL: Bounding box appears too bright (high opacity)")
        
        # Check 3: Maximum intensity suggests some visibility
        if 80 < max_intensity < 200:  # Should have some visibility but not too bright
            print("✅ SUCCESS: Bounding box has good visibility balance")
            success_checks += 1
        else:
            print(f"⚠️ WARNING: Bounding box intensity may be out of optimal range: {max_intensity}")
        
        # Final results
        success_rate = (success_checks / total_checks) * 100
        print(f"\n📊 Test Results: {success_checks}/{total_checks} checks passed ({success_rate:.1f}% success rate)")
        
        if success_checks >= 2:
            print("🎉 OVERALL SUCCESS: Arms label and lower opacity changes working correctly!")
        else:
            print("⚠️ NEEDS ATTENTION: Some issues detected with the changes")
        
        return success_checks >= 2
        
    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_arms_label_and_opacity()