"""
Simple Webcam Test
==================

Quick test to verify your webcam is working before running the full CheatGPT3 test.
"""

import cv2
import sys

def test_webcam():
    """Test if webcam is accessible."""
    print("🔍 Testing webcam access...")
    
    # Try to open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot access webcam!")
        print("   Please check:")
        print("   - Camera is connected")
        print("   - Camera is not being used by another application")
        print("   - Camera permissions are granted")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Webcam accessible: {width}x{height} @ {fps}fps")
    print("📹 Starting preview... Press 'q' or ESC to stop")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            break
            
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Add text overlay
        cv2.putText(frame, f"Webcam Test - Frame {frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' or ESC to quit", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Webcam Test', frame)
        
        frame_count += 1
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam test complete!")
    return True

if __name__ == "__main__":
    if test_webcam():
        print("\n🎉 Webcam is working! You can now run the full CheatGPT3 test:")
        print("   python test_webcam_realtime.py")
    else:
        print("\n❌ Please fix webcam issues before running CheatGPT3 test")
