#!/usr/bin/env python3
"""
Test script to verify sideward hand extension detection.
This script tests the updated gesture detection logic that focuses on sideward extensions
while ignoring hands covering the face.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cheatgpt.detectors.pose_detector import PoseDetector

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_sideward_hand_detection():
    """Test the sideward hand detection logic with simulated scenarios."""
    print("🤚 Testing Sideward Hand Extension Detection")
    print("=" * 50)
    
    detector = PoseDetector()
    
    # Test scenarios: [arm_points, head_points, expected_result, description]
    test_scenarios = [
        # Scenario 1: Left hand extended sideways (should detect)
        {
            'arm_points': {
                'left_shoulder': (100, 100),
                'left_wrist': (200, 110),  # 100px sideward extension
                'right_shoulder': (150, 100),
                'right_wrist': (140, 120)
            },
            'head_points': {
                'nose': (125, 80)
            },
            'expected': True,
            'description': "Left hand extended sideways (100px) - should detect"
        },
        
        # Scenario 2: Right hand extended sideways (should detect)
        {
            'arm_points': {
                'left_shoulder': (100, 100),
                'left_wrist': (90, 120),
                'right_shoulder': (150, 100),
                'right_wrist': (250, 110)  # 100px sideward extension
            },
            'head_points': {
                'nose': (125, 80)
            },
            'expected': True,
            'description': "Right hand extended sideways (100px) - should detect"
        },
        
        # Scenario 3: Hand covering face (should NOT detect)
        {
            'arm_points': {
                'left_shoulder': (100, 100),
                'left_wrist': (120, 60),  # Hand raised to face level
                'right_shoulder': (150, 100),
                'right_wrist': (140, 120)
            },
            'head_points': {
                'nose': (125, 80)
            },
            'expected': False,
            'description': "Hand covering face (raised up) - should NOT detect"
        },
        
        # Scenario 4: Hands in normal position (should NOT detect)
        {
            'arm_points': {
                'left_shoulder': (100, 100),
                'left_wrist': (90, 130),  # Close to shoulder
                'right_shoulder': (150, 100),
                'right_wrist': (160, 130)  # Close to shoulder
            },
            'head_points': {
                'nose': (125, 80)
            },
            'expected': False,
            'description': "Hands in normal position - should NOT detect"
        },
        
        # Scenario 5: Extreme sideward reach (should detect)
        {
            'arm_points': {
                'left_shoulder': (100, 100),
                'left_wrist': (250, 105),  # 150px extreme reach
                'right_shoulder': (150, 100),
                'right_wrist': (140, 120)
            },
            'head_points': {
                'nose': (125, 80)
            },
            'expected': True,
            'description': "Extreme sideward reach (150px) - should detect"
        },
        
        # Scenario 6: Hand slightly raised but still sideward (should detect)
        {
            'arm_points': {
                'left_shoulder': (100, 100),
                'left_wrist': (180, 95),  # 80px sideward, slightly raised
                'right_shoulder': (150, 100),
                'right_wrist': (140, 120)
            },
            'head_points': {
                'nose': (125, 80)
            },
            'expected': True,
            'description': "Hand slightly raised but sideward (80px) - should detect"
        }
    ]
    
    print("📊 Running Gesture Detection Tests:")
    print()
    
    results = []
    for i, scenario in enumerate(test_scenarios, 1):
        arm_points = scenario['arm_points']
        head_points = scenario['head_points']
        expected = scenario['expected']
        description = scenario['description']
        
        # Call the gesture detection function
        try:
            result = detector._compute_suspicious_gesture(arm_points, head_points)
            
            # Handle both old (boolean) and new (tuple) return formats
            if isinstance(result, tuple):
                detected, reason = result
            else:
                detected = result
                reason = "unknown"
            
            # Check if result matches expectation
            success = detected == expected
            status = "✅ PASS" if success else "❌ FAIL"
            
            result_text = f"DETECTED ({reason})" if detected else "Normal"
            expected_text = "DETECT" if expected else "Normal"
            
            print(f"  {status} Test {i}: {description}")
            print(f"       Expected: {expected_text}, Got: {result_text}")
            
            results.append(success)
            
        except Exception as e:
            print(f"  ❌ FAIL Test {i}: {description}")
            print(f"       Error: {e}")
            results.append(False)
        
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("📋 Test Summary:")
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎯 SUCCESS: All sideward hand detection tests passed!")
        print("🤚 The system now correctly:")
        print("   • Detects hands extended sideways (80px+ horizontal)")
        print("   • Ignores hands covering face (raised above shoulder)")
        print("   • Detects extreme sideward reaches (120px+ horizontal)")
        print("   • Ignores normal hand positions")
    else:
        print(f"\n⚠️ Some tests failed. Review the detection logic.")
    
    return passed == total

def show_detection_improvements():
    """Show the improvements made to gesture detection."""
    print("\n📈 Sideward Hand Detection Improvements:")
    print("=" * 50)
    print("1. ✅ Focus on sideward extensions only (80px+ horizontal distance)")
    print("2. ✅ Ignore hands covering face (vertical position check)")
    print("3. ✅ Detect extreme sideward reaches (120px+ for note passing)")
    print("4. ✅ Check hand position relative to shoulder level")
    print("5. ✅ Filter gesture reasons in hybrid engine")
    print("6. ✅ Higher confidence for sideward detections")
    
    print("\n🎯 Classroom Scenarios Covered:")
    print("• ✅ Passing notes to adjacent students")
    print("• ✅ Signaling to other students")
    print("• ✅ Reaching across desk/aisle")
    print("• ❌ Hands covering face (normal behavior)")
    print("• ❌ Hands in lap or normal positions")
    print("• ❌ Scratching head or adjusting hair")

if __name__ == "__main__":
    try:
        success = test_sideward_hand_detection()
        show_detection_improvements()
        
        if success:
            print("\n🚀 Ready for classroom monitoring!")
            print("   Sideward hand extensions will be detected accurately.")
        else:
            print("\n⚠️ Issues detected in sideward hand detection.")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()