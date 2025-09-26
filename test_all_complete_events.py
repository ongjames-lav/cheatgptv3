#!/usr/bin/env python3
"""
Complete Event Detection Test - All Events and Colors
Tests ALL possible event types to verify complete system functionality
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

def test_all_event_types():
    """Test ALL possible event types and their colors."""
    print("🎯 TESTING ALL EVENT TYPES AND COLORS")
    print("=" * 60)
    
    try:
        engine = EngineHybrid()
        test_frame = create_test_frame()
        
        # Create test persons at different positions
        persons = []
        for i in range(6):
            x_offset = (i % 3) * 400 + 100
            y_offset = (i // 3) * 300 + 100
            persons.append({
                'bbox': [x_offset, y_offset, x_offset + 200, y_offset + 200],
                'confidence': 0.85,
                'person_id': i
            })
        
        # Define ALL possible events to test
        all_events = [
            # HAND GESTURES (Should be MAGENTA)
            {
                'person_id': 'person_000',
                'event_type': 'Suspicious Hand Activity',
                'confidence': 0.75,
                'timestamp': time.time(),
                'details': 'Hand gesture detected: left_wrist_extended_arm_absolute',
                'gesture_type': 'left_wrist_extended_arm_absolute',
                'severity': 'orange'
            },
            
            # HEAD TURNING - SUSTAINED (Should be RED)
            {
                'person_id': 'person_001',
                'event_type': 'Sustained Head Turning',
                'confidence': 0.8,
                'timestamp': time.time(),
                'severity': 'red',
                'details': 'Head turning sustained for 2.5s'
            },
            
            # HEAD TURNING - FREQUENT (Should be ORANGE)
            {
                'person_id': 'person_002',
                'event_type': 'Frequent Head Turning',
                'confidence': 0.7,
                'timestamp': time.time(),
                'severity': 'orange',
                'details': 'Head turned 4 times in window'
            },
            
            # PHONE DETECTION (Should be ORANGE)
            {
                'person_id': 'person_003',
                'event_type': 'Phone Usage Detected',
                'confidence': 0.9,
                'timestamp': time.time(),
                'severity': 'orange',
                'details': 'Phone detected in hands for 2.0s'
            },
            
            # LOOKING DOWN (Should be YELLOW)
            {
                'person_id': 'person_004',
                'event_type': 'Abnormal Looking Down',
                'confidence': 0.6,
                'timestamp': time.time(),
                'severity': 'yellow',
                'details': 'Looking down at 30° for 1.5s'
            },
            
            # NORMAL (Should be GREEN - no event)
            # person_005 will have no event
        ]
        
        # Test each event individually first
        results = {}
        
        for i, event in enumerate(all_events):
            person = [persons[i]]  # Single person for this test
            test_events = [event] if event else []
            
            # Create overlay
            overlay_frame = engine._create_overlay(test_frame, person, test_events)
            
            # Save individual debug images
            event_name = event['event_type'].replace(' ', '_').lower()
            filename = f'debug_{event_name}.jpg'
            cv2.imwrite(filename, overlay_frame)
            
            # Analyze colors in the bounding box area
            person_bbox = person[0]['bbox']
            x1, y1, x2, y2 = person_bbox
            bbox_area = overlay_frame[y1:y2, x1:x2]
            
            # Check for different colors
            colors_found = analyze_colors(bbox_area)
            
            expected_color = get_expected_color(event)
            actual_color = get_dominant_color(colors_found)
            
            results[event['event_type']] = {
                'expected': expected_color,
                'actual': actual_color,
                'colors_detected': colors_found,
                'success': expected_color.lower() in actual_color.lower(),
                'filename': filename
            }
            
            status = "✅ PASS" if results[event['event_type']]['success'] else "❌ FAIL"
            print(f"   {event['event_type']}: {status}")
            print(f"      Expected: {expected_color} | Detected: {actual_color}")
            print(f"      Colors found: {colors_found}")
            print(f"      Debug image: {filename}")
            print()
        
        # Test normal detection (no events)
        normal_person = [persons[5]]  # Last person
        normal_overlay = engine._create_overlay(test_frame, normal_person, [])
        cv2.imwrite('debug_normal_detection.jpg', normal_overlay)
        
        person_bbox = normal_person[0]['bbox']
        x1, y1, x2, y2 = person_bbox
        bbox_area = normal_overlay[y1:y2, x1:x2]
        colors_found = analyze_colors(bbox_area)
        
        results['Normal Detection'] = {
            'expected': 'GREEN',
            'actual': get_dominant_color(colors_found),
            'colors_detected': colors_found,
            'success': 'green' in get_dominant_color(colors_found).lower(),
            'filename': 'debug_normal_detection.jpg'
        }
        
        status = "✅ PASS" if results['Normal Detection']['success'] else "❌ FAIL"
        print(f"   Normal Detection: {status}")
        print(f"      Expected: GREEN | Detected: {get_dominant_color(colors_found)}")
        print(f"      Colors found: {colors_found}")
        print()
        
        # Create comprehensive test image with all events
        comprehensive_overlay = engine._create_overlay(test_frame, persons, all_events)
        cv2.imwrite('debug_all_events_comprehensive.jpg', comprehensive_overlay)
        print(f"💾 Comprehensive test image saved: debug_all_events_comprehensive.jpg")
        
        return results
        
    except Exception as e:
        print(f"❌ ERROR in comprehensive test: {e}")
        import traceback
        traceback.print_exc()
        return {}

def analyze_colors(bbox_area):
    """Analyze what colors are present in a bounding box area."""
    colors_found = {}
    
    # Based on actual color analysis results:
    # MAGENTA: BGR(106, 30, 106) - purple-like with blue and red dominance
    magenta_mask = (bbox_area[:,:,0] > 80) & (bbox_area[:,:,1] < 50) & (bbox_area[:,:,2] > 80)
    colors_found['magenta'] = int(np.sum(magenta_mask))
    
    # RED: BGR(10, 10, 173) - high red, very low blue/green
    red_mask = (bbox_area[:,:,0] < 30) & (bbox_area[:,:,1] < 30) & (bbox_area[:,:,2] > 120)
    colors_found['red'] = int(np.sum(red_mask))
    
    # ORANGE: BGR(15, 96, 140) and BGR(15, 119, 175) - medium green, high red, low blue
    orange_mask = (bbox_area[:,:,0] < 50) & (bbox_area[:,:,1] > 70) & (bbox_area[:,:,1] < 200) & (bbox_area[:,:,2] > 100)
    colors_found['orange'] = int(np.sum(orange_mask))
    
    # YELLOW: BGR(30, 91, 91) - equal green and red, low blue
    yellow_mask = (bbox_area[:,:,0] < 50) & (bbox_area[:,:,1] > 70) & (bbox_area[:,:,2] > 70) & (np.abs(bbox_area[:,:,1].astype(int) - bbox_area[:,:,2].astype(int)) < 30)
    colors_found['yellow'] = int(np.sum(yellow_mask))
    
    # Check for GREEN (Normal) - More flexible detection
    green_mask = (bbox_area[:,:,0] < 50) & (bbox_area[:,:,1] > 100) & (bbox_area[:,:,2] < 50)
    colors_found['green'] = int(np.sum(green_mask))
    
    # Check for any coloring (non-background)
    background_color = [50, 50, 50]
    non_bg_mask = np.any(np.abs(bbox_area.astype(int) - background_color) > 30, axis=2)
    colors_found['any_color'] = int(np.sum(non_bg_mask))
    
    return colors_found

def get_expected_color(event):
    """Get the expected color for an event type."""
    event_type = event['event_type']
    severity = event.get('severity', 'unknown')
    
    if 'Suspicious Hand Activity' in event_type:
        return 'MAGENTA'
    elif 'Sustained Head Turning' in event_type:
        return 'RED'
    elif 'Frequent Head Turning' in event_type or 'Phone Usage' in event_type:
        return 'ORANGE'
    elif 'Looking Down' in event_type:
        return 'YELLOW'
    else:
        return 'GREEN'

def get_dominant_color(colors_dict):
    """Get the dominant color from the colors dictionary."""
    # Remove 'any_color' from consideration
    color_counts = {k: v for k, v in colors_dict.items() if k != 'any_color' and v > 0}
    
    if not color_counts:
        return 'NONE'
    
    dominant = max(color_counts, key=color_counts.get)
    return dominant.upper()

def print_summary(results):
    """Print a comprehensive summary of results."""
    print("=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r['success'])
    
    print(f"✅ Total Events Tested: {total_tests}")
    print(f"✅ Passed Tests: {passed_tests}")
    print(f"❌ Failed Tests: {total_tests - passed_tests}")
    print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print()
    
    print("🎨 COLOR CODING VERIFICATION:")
    expected_colors = {}
    for event_type, result in results.items():
        expected = result['expected']
        if expected not in expected_colors:
            expected_colors[expected] = []
        expected_colors[expected].append((event_type, result['success']))
    
    for color, events in expected_colors.items():
        passed = sum(1 for _, success in events if success)
        total = len(events)
        print(f"   {color}: {passed}/{total} working")
        for event_type, success in events:
            status = "✅" if success else "❌"
            print(f"      {status} {event_type}")
    
    print()
    print("🖼️ Debug Images Created:")
    for event_type, result in results.items():
        print(f"   - {result['filename']}")
    print("   - debug_all_events_comprehensive.jpg (ALL events in one image)")
    
    if passed_tests == total_tests:
        print("\n🎉 PERFECT! All event detection and coloring is working!")
    elif passed_tests >= total_tests * 0.8:
        print("\n✅ EXCELLENT! Most event detection is working correctly!")
    elif passed_tests >= total_tests * 0.6:
        print("\n⚠️ GOOD! Some issues need attention but mostly working!")
    else:
        print("\n🚨 ISSUES DETECTED! Several events need fixing!")

if __name__ == "__main__":
    results = test_all_event_types()
    if results:
        print_summary(results)