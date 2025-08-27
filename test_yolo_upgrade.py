"""
Test YOLO Model Upgrade
======================
Test the upgraded YOLOv11m models for better phone detection accuracy.
"""

import cv2
import numpy as np
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cheatgpt.detectors.yolo11_detector import YOLO11Detector
from cheatgpt.detectors.pose_detector import PoseDetector

def test_yolo_models():
    """Test both YOLO models to verify upgrade."""
    print("🎯 Testing YOLO Model Upgrade to YOLOv11m")
    print("=" * 50)
    
    # Test object detector
    print("\n📱 Testing Object Detector (YOLOv11m)...")
    try:
        obj_detector = YOLO11Detector()
        print(f"✅ Object detector initialized")
        print(f"   Model path: {obj_detector.weights_path}")
        print(f"   Device: {obj_detector.device}")
        
        # Test with dummy image
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        start_time = time.time()
        detections = obj_detector.detect(dummy_image)
        inference_time = (time.time() - start_time) * 1000
        
        print(f"   Inference time: {inference_time:.1f}ms")
        print(f"   Detections: {len(detections)}")
        
    except Exception as e:
        print(f"❌ Object detector failed: {e}")
    
    # Test pose detector
    print("\n🧍 Testing Pose Detector (YOLOv11m-pose)...")
    try:
        pose_detector = PoseDetector()
        print(f"✅ Pose detector initialized")
        print(f"   Model path: {pose_detector.weights_path}")
        print(f"   Device: {pose_detector.device}")
        
        # Test with dummy image
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        start_time = time.time()
        pose_data = pose_detector.estimate(dummy_image)
        inference_time = (time.time() - start_time) * 1000
        
        print(f"   Inference time: {inference_time:.1f}ms")
        print(f"   Pose detections: {len(pose_data) if pose_data else 0}")
        
    except Exception as e:
        print(f"❌ Pose detector failed: {e}")
    
    print("\n🎯 Model Upgrade Test Complete!")
    print("=" * 50)
    print("📊 Expected improvements:")
    print("   • Better phone detection accuracy (fewer false positives)")
    print("   • More robust pose detection")
    print("   • Higher confidence scores for correct detections")
    print("   • Slightly slower inference (trade-off for accuracy)")

def test_phone_detection_confidence():
    """Test phone detection with different confidence levels."""
    print("\n📱 Testing Phone Detection Confidence Levels")
    print("-" * 40)
    
    try:
        detector = YOLO11Detector()
        
        # Create test image with various objects
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Test detection at different confidence thresholds
        confidence_levels = [0.3, 0.4, 0.5, 0.6]
        
        for conf in confidence_levels:
            # Note: The actual confidence filtering happens in the engine
            detections = detector.detect(test_image)
            
            # Filter by confidence (simulating engine behavior)
            phone_detections = [
                d for d in detections 
                if d.get('class_name') == 'cell phone' and d.get('confidence', 0) >= conf
            ]
            
            print(f"   Confidence {conf}: {len(phone_detections)} phone detections")
            
    except Exception as e:
        print(f"❌ Phone detection test failed: {e}")

if __name__ == "__main__":
    test_yolo_models()
    test_phone_detection_confidence()
