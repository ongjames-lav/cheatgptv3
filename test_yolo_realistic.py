#!/usr/bin/env python3
"""Test script for YOLO11 detector with a realistic test image."""

import os
import sys
import numpy as np
import cv2

# Add cheatgpt to path
sys.path.insert(0, os.path.dirname(__file__))

from cheatgpt.detectors.yolo11_detector import YOLO11Detector

def create_realistic_test_frame():
    """Create a more realistic test frame with person-like shapes."""
    # Create a 640x480 test image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add a background (sky blue)
    frame[:, :] = [135, 206, 235]  # Sky blue
    
    # Add ground (green)
    cv2.rectangle(frame, (0, 400), (640, 480), (34, 139, 34), -1)
    
    # Create person-like figure
    # Head (circle)
    cv2.circle(frame, (320, 150), 30, (255, 220, 177), -1)  # Skin color
    
    # Body (rectangle)
    cv2.rectangle(frame, (300, 180), (340, 280), (0, 0, 255), -1)  # Red shirt
    
    # Arms
    cv2.rectangle(frame, (270, 190), (300, 250), (255, 220, 177), -1)  # Left arm
    cv2.rectangle(frame, (340, 190), (370, 250), (255, 220, 177), -1)  # Right arm
    
    # Legs
    cv2.rectangle(frame, (305, 280), (320, 350), (0, 0, 139), -1)  # Left leg (blue jeans)
    cv2.rectangle(frame, (320, 280), (335, 350), (0, 0, 139), -1)  # Right leg (blue jeans)
    
    # Add some noise to make it more realistic
    noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    return frame

def main():
    """Test the YOLO11 detector with a realistic test image."""
    print("=== YOLO11 Detector Realistic Test ===")
    
    try:
        # Initialize detector
        print("\n--- Initializing YOLO11 Detector ---")
        detector = YOLO11Detector()
        
        # Print model info
        info = detector.get_model_info()
        print(f"Model Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Create realistic test frame
        print("\n--- Creating Realistic Test Frame ---")
        test_frame = create_realistic_test_frame()
        print(f"Test frame shape: {test_frame.shape}")
        
        # Save test frame for inspection
        cv2.imwrite("test_frame.jpg", test_frame)
        print("Saved test frame as 'test_frame.jpg'")
        
        # Run detection
        print("\n--- Running Detection ---")
        detections = detector.detect(test_frame)
        
        print(f"Found {len(detections)} detections:")
        
        if detections:
            for i, detection in enumerate(detections):
                print(f"  Detection {i+1}:")
                print(f"    Class: {detection['cls_name']}")
                print(f"    Confidence: {detection['conf']:.3f}")
                bbox = detection['bbox']
                print(f"    BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                
                # Draw bounding box on image
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(test_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(test_frame, f"{detection['cls_name']}: {detection['conf']:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("  No detections found")
        
        # Save result image
        cv2.imwrite("test_result.jpg", test_frame)
        print("Saved detection result as 'test_result.jpg'")
        
        # Test with a blank frame (should find nothing)
        print("\n--- Testing with Blank Frame ---")
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        blank_detections = detector.detect(blank_frame)
        print(f"Blank frame detections: {len(blank_detections)}")
        
        print("\n=== Test Completed Successfully ===")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
