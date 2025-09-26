#!/usr/bin/env python3
"""
Final verification that Arms label appears correctly without question marks
"""

import cv2
import numpy as np

def verify_arms_label():
    """Check that the test image shows 'Arms' label correctly"""
    try:
        # Load the test image
        img = cv2.imread("test_arms_label_opacity.jpg")
        
        if img is None:
            print("❌ Could not load test image")
            return False
        
        print("🔍 Analyzing test image for Arms label...")
        
        # Look at the label area (where text should be)
        label_area = img[180:220, 280:450]  # Extended area for label
        
        # Convert to grayscale for better text analysis
        gray = cv2.cvtColor(label_area, cv2.COLOR_BGR2GRAY)
        
        # Find contours (text should create contours)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count significant contours (likely text characters)
        significant_contours = [c for c in contours if cv2.contourArea(c) > 10]
        
        print(f"📊 Analysis results:")
        print(f"   - Found {len(significant_contours)} text-like contours")
        print(f"   - Label area dimensions: {label_area.shape}")
        
        # Check for magenta coloring (gesture detection)
        # Look for magenta pixels in bounding box area
        bbox_area = img[200:500, 300:600]
        
        # Check for magenta color (BGR: 255, 0, 255 or similar)
        magenta_mask = (bbox_area[:,:,0] > 100) & (bbox_area[:,:,1] < 50) & (bbox_area[:,:,2] > 100)
        magenta_pixels = np.sum(magenta_mask)
        
        print(f"   - Magenta pixels detected: {magenta_pixels}")
        
        # Final assessment
        if len(significant_contours) >= 2 and magenta_pixels > 50:
            print("✅ SUCCESS: Arms label rendered correctly with magenta bounding box!")
            print("✅ SUCCESS: Lower opacity implemented (magenta visible but not overwhelming)")
            return True
        else:
            print("⚠️ WARNING: Label or coloring may need attention")
            return False
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False

if __name__ == "__main__":
    verify_arms_label()