#!/usr/bin/env python3
"""
Direct test of enhanced phone proximity detection
"""

import os
import sys
sys.path.append(os.path.abspath('.'))

from cheatgpt.detectors.pose_detector import PoseDetector
import numpy as np

def test_phone_proximity_detection():
    """Test the enhanced phone proximity detection directly"""
    
    print("🔍 Testing enhanced phone proximity detection...")
    
    # Initialize pose detector
    detector = PoseDetector()
    
    print(f"📊 Current settings:")
    print(f"   Phone IoU threshold: {detector.phone_iou_thresh}")
    print(f"   Debug mode: {detector.debug_mode}")
    
    # Test scenarios
    test_cases = [
        {
            'name': 'Close range - phone near person',
            'person_bbox': [500, 200, 600, 400],  # Large person
            'phone_bbox': [520, 300, 550, 340],   # Phone nearby
            'expected': True
        },
        {
            'name': 'Far range - small person, small phone',
            'person_bbox': [800, 400, 850, 500],  # Small person (far)
            'phone_bbox': [822, 462, 838, 478],   # Small phone
            'expected': True  # Should work with distance compensation
        },
        {
            'name': 'Angled view - phone offset',
            'person_bbox': [300, 150, 400, 350],  # Medium person
            'phone_bbox': [415, 200, 445, 230],   # Phone at angle
            'expected': True  # Should work with larger margins
        },
        {
            'name': 'Phone too far - should fail',
            'person_bbox': [100, 100, 200, 300],  # Person
            'phone_bbox': [500, 400, 530, 430],   # Phone far away
            'expected': False
        }
    ]
    
    print("\n🧪 Running test cases:")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        
        # Create mock phone detection
        phone_detections = [{
            'bbox': case['phone_bbox'],
            'conf': 0.18,  # Low confidence
            'cls_name': 'cell phone'
        }]
        
        # Test proximity detection
        print(f"   🔍 Testing with phone detections: {phone_detections}")
        is_near = detector._compute_phone_near(case['person_bbox'], phone_detections)
        print(f"   📊 Method returned: {is_near} (type: {type(is_near)})")
        
        # Check result
        if is_near == case['expected']:
            print(f"   ✅ PASS - Phone proximity: {is_near}")
        else:
            print(f"   ❌ FAIL - Expected: {case['expected']}, Got: {is_near}")
            print(f"   🐛 Debug: Return value does not match expected result!")
        
        # Print detailed debug info
        person_width = case['person_bbox'][2] - case['person_bbox'][0]
        person_height = case['person_bbox'][3] - case['person_bbox'][1]
        print(f"   📏 Person size: {person_width}x{person_height}")
        print(f"   📱 Phone bbox: {case['phone_bbox']}")
        
        # Manual calculation for debugging
        distance_factor = detector._get_classroom_distance_factor(case['person_bbox'], 720)
        base_margin_x = person_width * 0.5
        base_margin_y = person_height * 0.4
        margin_x = base_margin_x * distance_factor
        margin_y = base_margin_y * distance_factor
        
        x1, y1, x2, y2 = case['person_bbox']
        expanded_bbox = [x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y]
        
        print(f"   📐 Distance factor: {distance_factor:.2f}")
        print(f"   📦 Expanded bbox: [{expanded_bbox[0]:.1f}, {expanded_bbox[1]:.1f}, {expanded_bbox[2]:.1f}, {expanded_bbox[3]:.1f}]")
        
        # Calculate IoU and overlap
        iou = detector._calculate_iou(expanded_bbox, case['phone_bbox'])
        overlap = detector._calculate_overlap_ratio(case['person_bbox'], case['phone_bbox'])
        overlap_threshold = 0.05 / distance_factor
        
        print(f"   📊 IoU: {iou:.6f} (threshold: {detector.phone_iou_thresh})")
        print(f"   📊 Overlap: {overlap:.6f} (threshold: {overlap_threshold:.6f})")
        print(f"   ✓ IoU condition: {iou > detector.phone_iou_thresh}")
        print(f"   ✓ Overlap condition: {overlap > overlap_threshold}")
        print(f"   ✓ Final result: {iou > detector.phone_iou_thresh or overlap > overlap_threshold}")
    
    print("\n" + "=" * 60)
    print("🎯 Phone proximity test completed!")

if __name__ == "__main__":
    test_phone_proximity_detection()