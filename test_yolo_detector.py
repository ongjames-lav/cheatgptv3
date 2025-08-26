#!/usr/bin/env python3
"""Test script for YOLO11 detector to verify GPU usage and detection functionality."""

import os
import sys
import numpy as np
import cv2
import subprocess
import time

# Add cheatgpt to path
sys.path.insert(0, os.path.dirname(__file__))

from cheatgpt.detectors.yolo11_detector import YOLO11Detector

def check_gpu_activity():
    """Check if GPU is being used via nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("=== NVIDIA-SMI Output ===")
            print(result.stdout)
            return True
        else:
            print("nvidia-smi not available or failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("nvidia-smi command not found or timed out")
        return False

def create_test_frame():
    """Create a simple test frame for detection."""
    # Create a 640x480 test image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some random colors to make it more realistic
    frame[:, :] = [100, 150, 200]  # Light blue background
    
    # Add some simple shapes to simulate objects
    cv2.rectangle(frame, (100, 100), (200, 300), (255, 255, 255), -1)  # White rectangle
    cv2.circle(frame, (400, 200), 50, (0, 255, 0), -1)  # Green circle
    
    return frame

def main():
    """Test the YOLO11 detector."""
    print("=== YOLO11 Detector Test ===")
    
    # Check initial GPU state
    print("\n--- Initial GPU State ---")
    check_gpu_activity()
    
    try:
        # Initialize detector
        print("\n--- Initializing YOLO11 Detector ---")
        detector = YOLO11Detector()
        
        # Print model info
        info = detector.get_model_info()
        print(f"Model Info: {info}")
        
        # Create test frame
        print("\n--- Creating Test Frame ---")
        test_frame = create_test_frame()
        print(f"Test frame shape: {test_frame.shape}")
        
        # Run detection
        print("\n--- Running Detection ---")
        start_time = time.time()
        detections = detector.detect(test_frame)
        end_time = time.time()
        
        print(f"Detection completed in {end_time - start_time:.3f} seconds")
        print(f"Found {len(detections)} detections:")
        
        for i, detection in enumerate(detections):
            print(f"  Detection {i+1}:")
            print(f"    Class: {detection['cls_name']}")
            print(f"    Confidence: {detection['conf']:.3f}")
            print(f"    BBox: {detection['bbox']}")
        
        # Check GPU activity after detection
        print("\n--- GPU State After Detection ---")
        check_gpu_activity()
        
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
