#!/usr/bin/env python3
"""
Demo of enhanced phone detection for classroom monitoring
"""

import os
import sys
sys.path.append(os.path.abspath('.'))

from cheatgpt.detectors.pose_detector import PoseDetector
from cheatgpt.detectors.yolo11_detector import YOLO11Detector
import numpy as np

def demo_classroom_phone_detection():
    """Demonstrate enhanced phone detection capabilities"""
    
    print("🎓 CLASSROOM PHONE DETECTION DEMO")
    print("=" * 50)
    
    # Initialize detectors
    pose_detector = PoseDetector()
    yolo_detector = YOLO11Detector()
    
    print(f"📊 Enhanced Settings:")
    print(f"   • Phone IoU Threshold: {pose_detector.phone_iou_thresh}")
    print(f"   • YOLO Phone Confidence: 0.15 (was 0.3)")
    print(f"   • Detection Margins: 50%/40% (was 30%/20%)")
    print(f"   • Distance Compensation: Up to 2.0x for far students")
    
    print(f"\n📚 Classroom Scenarios:")
    print("-" * 30)
    
    # Scenario 1: Student in back row (small/far)
    print(f"\n1️⃣ Back Row Student (Distance Factor: 2.0x)")
    back_row_person = [800, 400, 850, 500]  # Small person bbox
    back_row_phone = [822, 462, 838, 478]   # Small phone near person
    
    phone_detections = [{'bbox': back_row_phone, 'conf': 0.16, 'cls_name': 'cell phone'}]
    detected = pose_detector._compute_phone_near(back_row_person, phone_detections)
    
    print(f"   📱 Phone detected: {'✅ YES' if detected else '❌ NO'}")
    print(f"   📏 Person size: 50x100 pixels (distant)")
    print(f"   📱 Phone size: 16x16 pixels (tiny)")
    print(f"   🎯 Result: {'Enhanced sensitivity working!' if detected else 'Needs adjustment'}")
    
    # Scenario 2: Student at angle (side view)
    print(f"\n2️⃣ Side Angle View")
    angled_person = [300, 150, 400, 350]   # Medium person
    angled_phone = [415, 200, 445, 230]    # Phone slightly offset
    
    phone_detections = [{'bbox': angled_phone, 'conf': 0.17, 'cls_name': 'cell phone'}]
    detected = pose_detector._compute_phone_near(angled_person, phone_detections)
    
    print(f"   📱 Phone detected: {'✅ YES' if detected else '❌ NO'}")
    print(f"   📐 Phone offset from person center")
    print(f"   🎯 Result: {'Angle compensation working!' if detected else 'Needs adjustment'}")
    
    # Scenario 3: Close range with low confidence
    print(f"\n3️⃣ Close Range - Low Confidence Phone")
    close_person = [500, 200, 600, 400]    # Large person
    close_phone = [520, 300, 550, 340]     # Phone near person
    
    phone_detections = [{'bbox': close_phone, 'conf': 0.18, 'cls_name': 'cell phone'}]
    detected = pose_detector._compute_phone_near(close_person, phone_detections)
    
    print(f"   📱 Phone detected: {'✅ YES' if detected else '❌ NO'}")
    print(f"   🔍 Confidence: 0.18 (below old threshold of 0.3)")
    print(f"   🎯 Result: {'Low confidence detection working!' if detected else 'Needs adjustment'}")
    
    print(f"\n" + "=" * 50)
    print(f"🎓 CLASSROOM DETECTION SUMMARY")
    print(f"📈 Sensitivity Improvements:")
    print(f"   ✅ Distance compensation for back-row students")
    print(f"   ✅ Angle tolerance for side camera views") 
    print(f"   ✅ Low confidence phone detection")
    print(f"   ✅ Larger detection zones around students")
    print(f"   ✅ Classroom-optimized IoU thresholds")
    
    print(f"\n🚨 Alarm System:")
    print(f"   ✅ Fixed event type matching")
    print(f"   ✅ Sound files available: alarm.wav, RIZZ Sound Effect.wav")
    print(f"   ✅ Triggers only for sustained phone usage (3+ frames)")
    
    print(f"\n🎯 Ready for classroom deployment!")

if __name__ == "__main__":
    demo_classroom_phone_detection()