#!/usr/bin/env python3
"""Comprehensive test script for YOLO11 detector functionality."""

import os
import sys
import numpy as np
import cv2
import requests
from PIL import Image
import io

# Add cheatgpt to path
sys.path.insert(0, os.path.dirname(__file__))

from cheatgpt.detectors.yolo11_detector import YOLO11Detector

def download_test_image():
    """Download a test image with people from the internet."""
    # Using a free test image with people
    url = "https://ultralytics.com/images/bus.jpg"  # Ultralytics test image
    
    try:
        print(f"Downloading test image from: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Convert to OpenCV format
        image = Image.open(io.BytesIO(response.content))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        return frame
    except Exception as e:
        print(f"Failed to download test image: {e}")
        return None

def test_env_configuration():
    """Test environment variable configuration."""
    print("=== Testing Environment Configuration ===")
    
    # Test with default settings
    detector1 = YOLO11Detector()
    info1 = detector1.get_model_info()
    print("Default configuration:")
    for key, value in info1.items():
        print(f"  {key}: {value}")
    
    # Test force CPU setting (if not already forced)
    if not info1['force_cpu']:
        print("\nTesting FORCE_CPU=true...")
        os.environ['FORCE_CPU'] = 'true'
        detector2 = YOLO11Detector()
        info2 = detector2.get_model_info()
        print("Force CPU configuration:")
        for key, value in info2.items():
            print(f"  {key}: {value}")
        
        # Reset environment
        os.environ['FORCE_CPU'] = 'false'
    
    return detector1

def main():
    """Comprehensive test of the YOLO11 detector."""
    print("=== Comprehensive YOLO11 Detector Test ===")
    
    try:
        # Test 1: Environment configuration
        detector = test_env_configuration()
        
        # Test 2: Download and test with real image
        print("\n=== Testing with Real Image ===")
        test_image = download_test_image()
        
        if test_image is not None:
            print(f"Test image shape: {test_image.shape}")
            
            # Save original image
            cv2.imwrite("original_test_image.jpg", test_image)
            print("Saved original test image as 'original_test_image.jpg'")
            
            # Run detection
            print("\n--- Running Detection on Real Image ---")
            detections = detector.detect(test_image)
            
            print(f"Found {len(detections)} detections:")
            
            if detections:
                result_image = test_image.copy()
                
                for i, detection in enumerate(detections):
                    print(f"  Detection {i+1}:")
                    print(f"    Class: {detection['cls_name']}")
                    print(f"    Confidence: {detection['conf']:.3f}")
                    bbox = detection['bbox']
                    print(f"    BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                    
                    # Draw bounding box on image
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(result_image, f"{detection['cls_name']}: {detection['conf']:.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save result image
                cv2.imwrite("detection_result.jpg", result_image)
                print("Saved detection result as 'detection_result.jpg'")
            else:
                print("  No target objects (person/cell phone) found in the image")
        
        # Test 3: Performance timing
        print("\n=== Performance Test ===")
        if test_image is not None:
            import time
            
            # Warm up
            _ = detector.detect(test_image)
            
            # Time multiple runs
            times = []
            num_runs = 5
            for i in range(num_runs):
                start = time.time()
                _ = detector.detect(test_image)
                end = time.time()
                times.append(end - start)
            
            avg_time = sum(times) / len(times)
            print(f"Average detection time over {num_runs} runs: {avg_time:.3f} seconds")
            print(f"FPS: {1/avg_time:.1f}")
        
        # Test 4: Error handling
        print("\n=== Error Handling Test ===")
        
        # Test with None frame
        none_result = detector.detect(None)
        print(f"None frame result: {none_result}")
        
        # Test with empty frame
        empty_frame = np.array([])
        empty_result = detector.detect(empty_frame)
        print(f"Empty frame result: {empty_result}")
        
        # Test with invalid frame
        invalid_frame = np.ones((10, 10), dtype=np.uint8)  # 2D instead of 3D
        try:
            invalid_result = detector.detect(invalid_frame)
            print(f"Invalid frame result: {invalid_result}")
        except Exception as e:
            print(f"Invalid frame handled gracefully: {type(e).__name__}")
        
        print("\n=== All Tests Completed Successfully ===")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
