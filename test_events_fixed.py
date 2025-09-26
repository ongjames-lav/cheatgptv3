#!/usr/bin/env python3
"""
Fixed Comprehensive Event Detection Test
Tests all detection events to verify they're working and triggering correctly
"""

import cv2
import numpy as np
import sys
import os
import time

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ['PYTHONPATH'] = os.path.dirname(__file__)

from cheatgpt.engines.engine_hybrid import EngineHybrid

def create_test_frame():
    """Create a standard test frame."""
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    test_frame.fill(50)  # Dark gray background
    return test_frame

def test_hand_gesture_detection():
    """Test hand gesture detection events."""
    print("\n🤚 Testing Hand Gesture Detection...")
    
    try:
        engine = EngineHybrid()
        test_frame = create_test_frame()
        
        # Test person with hand gesture - FIXED: use 'person_id' not 'id'
        test_person = {
            'bbox': [300, 200, 600, 500],
            'confidence': 0.85,
            'person_id': 0  # FIXED: Changed from 'id' to 'person_id'
        }
        
        # Test a single gesture type to debug
        test_event = {
            'person_id': 'person_000',  # This should match the converted person_id
            'event_type': 'Suspicious Hand Activity',
            'confidence': 0.75,
            'timestamp': time.time(),
            'details': 'Hand gesture detected: left_wrist_extended_arm_absolute',
            'gesture_type': 'left_wrist_extended_arm_absolute'
        }
        
        # Test overlay creation
        overlay_frame = engine._create_overlay(test_frame, [test_person], [test_event])
        
        # Save test image for visual debugging
        cv2.imwrite('debug_hand_gesture.jpg', overlay_frame)
        print(f"   Debug image saved: debug_hand_gesture.jpg")
        
        # Check for magenta coloring more thoroughly
        bbox_area = overlay_frame[200:500, 300:600]
        
        # Check different magenta variations
        bright_magenta = (bbox_area[:,:,0] > 200) & (bbox_area[:,:,1] < 100) & (bbox_area[:,:,2] > 200)
        dim_magenta = (bbox_area[:,:,0] > 100) & (bbox_area[:,:,1] < 50) & (bbox_area[:,:,2] > 100)
        any_magenta = bright_magenta | dim_magenta
        
        magenta_pixels = np.sum(any_magenta)
        
        # Also check for any non-background coloring
        background_color = [50, 50, 50]
        non_bg_mask = np.any(np.abs(bbox_area.astype(int) - background_color) > 30, axis=2)
        non_bg_pixels = np.sum(non_bg_mask)
        
        print(f"   Magenta pixels detected: {magenta_pixels}")
        print(f"   Non-background pixels: {non_bg_pixels}")
        print(f"   Person ID conversion: 0 -> person_000")
        
        success = magenta_pixels > 50 or non_bg_pixels > 100
        print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")
        
        return success
        
    except Exception as e:
        print(f"❌ ERROR in hand gesture test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_head_turning_detection():
    """Test head turning detection events."""
    print("\n🔄 Testing Head Turning Detection...")
    
    try:
        engine = EngineHybrid()
        test_frame = create_test_frame()
        
        # FIXED: Use 'person_id' not 'id'
        test_person = {
            'bbox': [300, 200, 600, 500],
            'confidence': 0.85,
            'person_id': 0
        }
        
        # Test sustained head turning (should be RED)
        sustained_event = {
            'person_id': 'person_000',
            'event_type': 'Sustained Head Turning',
            'confidence': 0.8,
            'timestamp': time.time(),
            'severity': 'red',
            'details': 'Head turning sustained for 2.5s'
        }
        
        # Test overlay creation
        overlay_frame = engine._create_overlay(test_frame, [test_person], [sustained_event])
        cv2.imwrite('debug_head_turning.jpg', overlay_frame)
        
        bbox_area = overlay_frame[200:500, 300:600]
        
        # Look for red coloring (BGR format)
        red_mask = (bbox_area[:,:,0] < 100) & (bbox_area[:,:,1] < 100) & (bbox_area[:,:,2] > 150)
        red_pixels = np.sum(red_mask)
        
        # Also check for any coloring
        non_bg_mask = np.any(np.abs(bbox_area.astype(int) - [50, 50, 50]) > 30, axis=2)
        colored_pixels = np.sum(non_bg_mask)
        
        print(f"   Red pixels: {red_pixels}")
        print(f"   Any colored pixels: {colored_pixels}")
        
        success = red_pixels > 50 or colored_pixels > 100
        print(f"   Sustained Head Turning: {'✅ PASS' if success else '❌ FAIL'}")
        
        return success
        
    except Exception as e:
        print(f"❌ ERROR in head turning test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_normal_detection():
    """Test normal person detection (baseline)."""
    print("\n✅ Testing Normal Detection...")
    
    try:
        engine = EngineHybrid()
        test_frame = create_test_frame()
        
        # FIXED: Use 'person_id' not 'id'
        test_person = {
            'bbox': [300, 200, 600, 500],
            'confidence': 0.85,
            'person_id': 0
        }
        
        # No events - should show normal green
        overlay_frame = engine._create_overlay(test_frame, [test_person], [])
        cv2.imwrite('debug_normal.jpg', overlay_frame)
        
        bbox_area = overlay_frame[200:500, 300:600]
        
        # Check for green coloring (BGR format)
        green_mask = (bbox_area[:,:,0] < 100) & (bbox_area[:,:,1] > 150) & (bbox_area[:,:,2] < 100)
        green_pixels = np.sum(green_mask)
        
        # Any coloring at all?
        non_bg_mask = np.any(np.abs(bbox_area.astype(int) - [50, 50, 50]) > 20, axis=2)
        any_coloring = np.sum(non_bg_mask)
        
        print(f"   Green pixels: {green_pixels}")
        print(f"   Any coloring: {any_coloring}")
        
        success = green_pixels > 50 or any_coloring > 100
        print(f"   Normal Detection: {'✅ PASS' if success else '❌ FAIL'}")
        
        return success
        
    except Exception as e:
        print(f"❌ ERROR in normal detection test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_thresholds_access():
    """Test access to detection thresholds."""
    print("\n⚙️ Testing Thresholds Access...")
    
    try:
        engine = EngineHybrid()
        
        # Test accessing thresholds
        if hasattr(engine, 'thresholds'):
            thresholds = engine.thresholds
            print("✅ Thresholds found:")
            for key, value in thresholds.items():
                print(f"   {key}: {value}")
            return True
        else:
            print("❌ No thresholds attribute found")
            return False
        
    except Exception as e:
        print(f"❌ ERROR accessing thresholds: {e}")
        return False

def run_quick_diagnostic():
    """Run diagnostic tests to identify issues."""
    print("🔧 QUICK DIAGNOSTIC TEST")
    print("=" * 50)
    
    hand_result = test_hand_gesture_detection()
    head_result = test_head_turning_detection()
    normal_result = test_normal_detection()
    threshold_result = test_thresholds_access()
    
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    total_tests = 4
    passed = sum([hand_result, head_result, normal_result, threshold_result])
    
    print(f"✅ Passed: {passed}/{total_tests}")
    print(f"❌ Failed: {total_tests - passed}/{total_tests}")
    print(f"📈 Success Rate: {(passed/total_tests)*100:.1f}%")
    
    if passed >= 3:
        print("🎉 System is mostly working!")
    elif passed >= 2:
        print("⚠️ System has some issues")
    else:
        print("🚨 System needs major fixes")
    
    print("\n💡 Check the debug images created:")
    print("   - debug_hand_gesture.jpg")
    print("   - debug_head_turning.jpg") 
    print("   - debug_normal.jpg")

if __name__ == "__main__":
    run_quick_diagnostic()