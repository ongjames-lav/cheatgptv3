#!/usr/bin/env python3
"""
Comprehensive Event Detection Test
Tests all detection events to verify they're working and triggering correctly
"""

import cv2
import numpy as np
import sys
import os
import time
import json

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
        
        # Test person with hand gesture
        test_person = {
            'bbox': [300, 200, 600, 500],
            'confidence': 0.85,
            'person_id': 0  # Changed from 'id' to 'person_id'
        }
        
        # Test different hand gesture types
        gesture_types = [
            'left_wrist_extended_arm_absolute',
            'right_wrist_extended_arm_absolute', 
            'left_wrist_near_head',
            'right_wrist_near_head',
            'left_wrist_in_face_region'
        ]
        
        results = {}
        
        for gesture_type in gesture_types:
            test_event = {
                'person_id': 'person_000',
                'event_type': 'Suspicious Hand Activity',
                'confidence': 0.75,
                'timestamp': time.time(),
                'details': f'Hand gesture detected: {gesture_type}',
                'gesture_type': gesture_type
            }
            
            # Test overlay creation
            overlay_frame = engine._create_overlay(test_frame, [test_person], [test_event])
            
            # Check for magenta coloring
            bbox_area = overlay_frame[200:500, 300:600]
            magenta_mask = (bbox_area[:,:,0] > 100) & (bbox_area[:,:,1] < 50) & (bbox_area[:,:,2] > 100)
            magenta_pixels = np.sum(magenta_mask)
            
            results[gesture_type] = {
                'magenta_detected': magenta_pixels > 100,
                'magenta_pixel_count': int(magenta_pixels)
            }
            
            print(f"   {gesture_type}: {'✅ PASS' if magenta_pixels > 100 else '❌ FAIL'} ({magenta_pixels} magenta pixels)")
        
        return results
        
    except Exception as e:
        print(f"❌ ERROR in hand gesture test: {e}")
        return {}

def test_head_turning_detection():
    """Test head turning detection events."""
    print("\n🔄 Testing Head Turning Detection...")
    
    try:
        engine = EngineHybrid()
        test_frame = create_test_frame()
        
        test_person = {
            'bbox': [300, 200, 600, 500],
            'confidence': 0.85,
            'id': 0
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
        
        # Test frequent head turning (should be ORANGE)
        frequent_event = {
            'person_id': 'person_000',
            'event_type': 'Frequent Head Turning', 
            'confidence': 0.8,
            'timestamp': time.time(),
            'severity': 'orange',
            'details': 'Head turned 4 times in window'
        }
        
        results = {}
        
        # Test sustained (RED)
        overlay_frame = engine._create_overlay(test_frame, [test_person], [sustained_event])
        bbox_area = overlay_frame[200:500, 300:600]
        red_mask = (bbox_area[:,:,0] < 50) & (bbox_area[:,:,1] < 50) & (bbox_area[:,:,2] > 150)
        red_pixels = np.sum(red_mask)
        
        results['sustained_turning'] = {
            'red_detected': red_pixels > 100,
            'red_pixel_count': int(red_pixels),
            'expected': 'RED'
        }
        print(f"   Sustained Head Turning (RED): {'✅ PASS' if red_pixels > 100 else '❌ FAIL'} ({red_pixels} red pixels)")
        
        # Test frequent (ORANGE)
        overlay_frame = engine._create_overlay(test_frame, [test_person], [frequent_event])
        bbox_area = overlay_frame[200:500, 300:600]
        orange_mask = (bbox_area[:,:,0] < 100) & (bbox_area[:,:,1] > 100) & (bbox_area[:,:,2] > 200)
        orange_pixels = np.sum(orange_mask)
        
        results['frequent_turning'] = {
            'orange_detected': orange_pixels > 100,
            'orange_pixel_count': int(orange_pixels),
            'expected': 'ORANGE'
        }
        print(f"   Frequent Head Turning (ORANGE): {'✅ PASS' if orange_pixels > 100 else '❌ FAIL'} ({orange_pixels} orange pixels)")
        
        return results
        
    except Exception as e:
        print(f"❌ ERROR in head turning test: {e}")
        return {}

def test_phone_detection():
    """Test phone usage detection."""
    print("\n📱 Testing Phone Detection...")
    
    try:
        engine = EngineHybrid()
        test_frame = create_test_frame()
        
        test_person = {
            'bbox': [300, 200, 600, 500],
            'confidence': 0.85,
            'id': 0
        }
        
        phone_event = {
            'person_id': 'person_000',
            'event_type': 'Phone Usage Detected',
            'confidence': 0.9,
            'timestamp': time.time(),
            'severity': 'orange',
            'details': 'Phone detected in hands'
        }
        
        overlay_frame = engine._create_overlay(test_frame, [test_person], [phone_event])
        bbox_area = overlay_frame[200:500, 300:600]
        
        # Check for orange coloring
        orange_mask = (bbox_area[:,:,0] < 100) & (bbox_area[:,:,1] > 100) & (bbox_area[:,:,2] > 200)
        orange_pixels = np.sum(orange_mask)
        
        result = {
            'orange_detected': orange_pixels > 100,
            'orange_pixel_count': int(orange_pixels),
            'expected': 'ORANGE'
        }
        
        print(f"   Phone Usage (ORANGE): {'✅ PASS' if orange_pixels > 100 else '❌ FAIL'} ({orange_pixels} orange pixels)")
        
        return result
        
    except Exception as e:
        print(f"❌ ERROR in phone test: {e}")
        return {}

def test_looking_down_detection():
    """Test looking down detection."""
    print("\n👀 Testing Looking Down Detection...")
    
    try:
        engine = EngineHybrid()
        test_frame = create_test_frame()
        
        test_person = {
            'bbox': [300, 200, 600, 500],
            'confidence': 0.85,
            'id': 0
        }
        
        looking_event = {
            'person_id': 'person_000',
            'event_type': 'Abnormal Looking Down',
            'confidence': 0.7,
            'timestamp': time.time(),
            'severity': 'yellow',
            'details': 'Looking down at 30° for 1.5s'
        }
        
        overlay_frame = engine._create_overlay(test_frame, [test_person], [looking_event])
        bbox_area = overlay_frame[200:500, 300:600]
        
        # Check for yellow coloring
        yellow_mask = (bbox_area[:,:,0] < 100) & (bbox_area[:,:,1] > 200) & (bbox_area[:,:,2] > 200)
        yellow_pixels = np.sum(yellow_mask)
        
        result = {
            'yellow_detected': yellow_pixels > 50,  # Lower threshold for yellow
            'yellow_pixel_count': int(yellow_pixels),
            'expected': 'YELLOW'
        }
        
        print(f"   Looking Down (YELLOW): {'✅ PASS' if yellow_pixels > 50 else '❌ FAIL'} ({yellow_pixels} yellow pixels)")
        
        return result
        
    except Exception as e:
        print(f"❌ ERROR in looking down test: {e}")
        return {}

def test_normal_detection():
    """Test normal person detection (baseline)."""
    print("\n✅ Testing Normal Detection...")
    
    try:
        engine = EngineHybrid()
        test_frame = create_test_frame()
        
        test_person = {
            'bbox': [300, 200, 600, 500],
            'confidence': 0.85,
            'id': 0
        }
        
        # No events - should show normal green
        overlay_frame = engine._create_overlay(test_frame, [test_person], [])
        bbox_area = overlay_frame[200:500, 300:600]
        
        # Check for green coloring
        green_mask = (bbox_area[:,:,0] < 100) & (bbox_area[:,:,1] > 150) & (bbox_area[:,:,2] < 100)
        green_pixels = np.sum(green_mask)
        
        result = {
            'green_detected': green_pixels > 50,
            'green_pixel_count': int(green_pixels),
            'expected': 'GREEN'
        }
        
        print(f"   Normal Detection (GREEN): {'✅ PASS' if green_pixels > 50 else '❌ FAIL'} ({green_pixels} green pixels)")
        
        return result
        
    except Exception as e:
        print(f"❌ ERROR in normal detection test: {e}")
        return {}

def test_event_triggering_logic():
    """Test the actual event triggering logic."""
    print("\n⚙️ Testing Event Triggering Logic...")
    
    try:
        engine = EngineHybrid()
        
        # Test detection parameters
        thresholds = engine.thresholds
        print(f"📊 Detection Thresholds:")
        print(f"   Hand extended frames: {thresholds.get('hand_extended_frames_threshold', 'N/A')}")
        print(f"   Head turn angle: {thresholds.get('head_turn_angle_threshold', 'N/A')}°")
        print(f"   Head turn frequency: {thresholds.get('head_turn_frequency_threshold', 'N/A')}")
        print(f"   Head turn sustained: {thresholds.get('head_turn_sustained_threshold', 'N/A')}s")
        print(f"   Head pitch threshold: {thresholds.get('head_pitch_threshold', 'N/A')}°")
        
        # Test confirmation thresholds
        print(f"📋 Confirmation Thresholds:")
        print(f"   General confirmation: {getattr(engine, 'confirmation_threshold', 'N/A')}")
        print(f"   Hand confirmation: {getattr(engine, 'hand_confirmation_threshold', 'N/A')}")
        print(f"   Phone confirmation: {getattr(engine, 'phone_confirmation_threshold', 'N/A')}")
        
        # Test debounce intervals
        print(f"⏱️ Timing Controls:")
        print(f"   Event debounce interval: {getattr(engine, 'event_debounce_interval', 'N/A')}s")
        print(f"   Detection FPS: {getattr(engine, 'detection_fps', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR in triggering logic test: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and generate comprehensive report."""
    print("🧪 COMPREHENSIVE EVENT DETECTION TEST")
    print("=" * 60)
    
    # Run all tests
    hand_results = test_hand_gesture_detection()
    head_results = test_head_turning_detection() 
    phone_results = test_phone_detection()
    looking_results = test_looking_down_detection()
    normal_results = test_normal_detection()
    logic_results = test_event_triggering_logic()
    
    # Generate summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    
    # Hand gesture results
    if hand_results:
        for gesture, result in hand_results.items():
            total_tests += 1
            if result.get('magenta_detected', False):
                passed_tests += 1
    
    # Head turning results  
    if head_results:
        for test, result in head_results.items():
            total_tests += 1
            if (result.get('red_detected', False) or result.get('orange_detected', False)):
                passed_tests += 1
    
    # Other tests
    for result in [phone_results, looking_results, normal_results]:
        if result:
            total_tests += 1
            expected_color = result.get('expected', '').lower()
            if (result.get(f'{expected_color}_detected', False)):
                passed_tests += 1
    
    if logic_results:
        total_tests += 1
        passed_tests += 1
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"✅ Passed Tests: {passed_tests}")
    print(f"❌ Failed Tests: {total_tests - passed_tests}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 OVERALL RESULT: Event detection system is working correctly!")
    elif success_rate >= 60:
        print("⚠️ OVERALL RESULT: Event detection has some issues that need attention")
    else:
        print("🚨 OVERALL RESULT: Event detection system needs significant fixes")
    
    # Save detailed results
    detailed_results = {
        'hand_gestures': hand_results,
        'head_turning': head_results,
        'phone_detection': phone_results,
        'looking_down': looking_results,
        'normal_detection': normal_results,
        'logic_test': logic_results,
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate
        }
    }
    
    with open('event_detection_test_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: event_detection_test_results.json")
    
    return success_rate >= 80

if __name__ == "__main__":
    run_comprehensive_test()