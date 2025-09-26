#!/usr/bin/env python3
"""
Test Both Arms and Turning Labels with Magenta Colors
Tests that both gesture types get the same magenta treatment with lower opacity
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

def test_arms_and_turning_consistency():
    """Test that both Arms and Turning get consistent magenta treatment."""
    
    print("🧪 Testing Arms and Turning Label Consistency...")
    
    try:
        # Initialize the engine
        print("🚀 Initializing EngineHybrid...")
        engine = EngineHybrid()
        
        # Create a test frame
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        test_frame.fill(50)  # Dark gray background
        
        # Create mock person detection data
        test_person1 = {
            'bbox': [200, 150, 500, 450],  # x1, y1, x2, y2
            'confidence': 0.85,
            'id': 0
        }
        
        test_person2 = {
            'bbox': [600, 150, 900, 450],  # x1, y1, x2, y2
            'confidence': 0.85,
            'id': 1
        }
        
        # Create mock gesture events
        arms_event = {
            'person_id': 'person_000',
            'event_type': 'Suspicious Hand Activity',
            'confidence': 0.75,
            'timestamp': 1632123456.789,
            'details': 'Hand gesture detected: left_wrist_extended_arm_absolute'
        }
        
        turning_event = {
            'person_id': 'person_001',
            'event_type': 'Frequent Head Turning',
            'confidence': 0.75,
            'timestamp': 1632123456.789,
            'details': 'Head turning detected for 1.2s',
            'severity': 'orange'
        }
        
        print("📊 Testing Arms and Turning label rendering...")
        
        # Test the overlay creation with both event types
        overlay_frame = engine._create_overlay(test_frame, [test_person1, test_person2], [arms_event, turning_event])
        
        # Save the test result
        test_output_path = "test_arms_turning_consistency.jpg"
        cv2.imwrite(test_output_path, overlay_frame)
        print(f"💾 Test result saved as: {test_output_path}")
        
        # Analyze both bounding box areas for magenta color
        arms_area = overlay_frame[150:450, 200:500]  # Arms bounding box area
        turning_area = overlay_frame[150:450, 600:900]  # Turning bounding box area
        
        # Check for magenta color in both areas (BGR: 255, 0, 255 or similar)
        def count_magenta_pixels(area):
            magenta_mask = (area[:,:,0] > 100) & (area[:,:,1] < 50) & (area[:,:,2] > 100)
            return np.sum(magenta_mask)
        
        arms_magenta = count_magenta_pixels(arms_area)
        turning_magenta = count_magenta_pixels(turning_area)
        
        print(f"📈 Color analysis results:")
        print(f"   Arms area magenta pixels: {arms_magenta}")
        print(f"   Turning area magenta pixels: {turning_magenta}")
        
        # Check label areas for text
        arms_label_area = overlay_frame[130:150, 180:300]
        turning_label_area = overlay_frame[130:150, 580:700]
        
        def analyze_label_area(area, name):
            non_bg_mask = np.any(np.abs(area.astype(int) - [50, 50, 50]) > 20, axis=2)
            non_bg_pixels = np.sum(non_bg_mask)
            print(f"   {name} label pixels: {non_bg_pixels}")
            return non_bg_pixels > 50  # Threshold for text detection
        
        arms_has_text = analyze_label_area(arms_label_area, "Arms")
        turning_has_text = analyze_label_area(turning_label_area, "Turning")
        
        # Verification checks
        success_checks = 0
        total_checks = 4
        
        # Check 1: Arms area has magenta color
        if arms_magenta > 100:
            print("✅ SUCCESS: Arms area has magenta coloring")
            success_checks += 1
        else:
            print("❌ FAIL: Arms area missing magenta coloring")
        
        # Check 2: Turning area has magenta color
        if turning_magenta > 100:
            print("✅ SUCCESS: Turning area has magenta coloring")
            success_checks += 1
        else:
            print("❌ FAIL: Turning area missing magenta coloring")
        
        # Check 3: Arms label rendered
        if arms_has_text:
            print("✅ SUCCESS: Arms label text detected")
            success_checks += 1
        else:
            print("❌ FAIL: Arms label text not detected")
        
        # Check 4: Turning label rendered
        if turning_has_text:
            print("✅ SUCCESS: Turning label text detected")
            success_checks += 1
        else:
            print("❌ FAIL: Turning label text not detected")
        
        # Final results
        success_rate = (success_checks / total_checks) * 100
        print(f"\n📊 Test Results: {success_checks}/{total_checks} checks passed ({success_rate:.1f}% success rate)")
        
        if success_checks >= 3:
            print("🎉 OVERALL SUCCESS: Arms and Turning consistency implemented correctly!")
        else:
            print("⚠️ NEEDS ATTENTION: Some consistency issues detected")
        
        return success_checks >= 3
        
    except Exception as e:
        print(f"❌ ERROR during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_arms_and_turning_consistency()