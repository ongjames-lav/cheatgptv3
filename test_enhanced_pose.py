"""
Enhanced Pose Detection Test
============================

Test the ultra-sensitive pose detection for leaning and looking around.
"""

import sys
import os
import cv2
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cheatgpt.detectors.pose_detector import PoseDetector
from cheatgpt.detectors.yolo11_detector import YOLO11Detector

def test_enhanced_pose_detection():
    """Test enhanced pose detection with sample image."""
    print("🧪 Testing Enhanced Pose Detection")
    print("=" * 40)
    
    # Initialize detectors
    print("Initializing detectors...")
    yolo_detector = YOLO11Detector()
    pose_detector = PoseDetector()
    
    print(f"✅ Pose detector thresholds:")
    print(f"   - Lean angle: {pose_detector.lean_angle_thresh}°")
    print(f"   - Head turn: {pose_detector.head_turn_thresh}°")
    print(f"   - Phone IoU: {pose_detector.phone_iou_thresh}")
    print(f"   - Debug mode: {pose_detector.debug_mode}")
    print()
    
    # Try to load a test image or use webcam
    test_image_path = "test_frame.jpg"
    if os.path.exists(test_image_path):
        print(f"📷 Loading test image: {test_image_path}")
        frame = cv2.imread(test_image_path)
    else:
        print("📷 No test image found, capturing from webcam...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print("❌ Failed to capture from webcam")
                return
            # Save the captured frame for reference
            cv2.imwrite("pose_test_frame.jpg", frame)
            print("💾 Saved captured frame as pose_test_frame.jpg")
        else:
            print("❌ No webcam available")
            return
    
    if frame is None:
        print("❌ Could not load or capture image")
        return
    
    print(f"✅ Image loaded: {frame.shape}")
    print()
    
    # Run pose detection
    print("🔍 Running pose detection...")
    
    # First detect persons and phones
    detections = yolo_detector.detect(frame)
    phones = [det for det in detections if det['cls_name'] == 'cell phone']
    
    print(f"📱 Found {len(phones)} phones")
    
    # Run pose estimation
    pose_estimates = pose_detector.estimate(frame, phones)
    
    print(f"🤸 Found {len(pose_estimates)} pose estimates")
    print()
    
    # Display results
    for i, pose in enumerate(pose_estimates):
        print(f"👤 Person {i+1}:")
        print(f"   Bounding box: {pose['bbox']}")
        print(f"   🔄 Lean flag: {pose['lean_flag']}")
        print(f"   👁️ Look flag: {pose['look_flag']}")
        print(f"   📱 Phone flag: {pose['phone_flag']}")
        print(f"   📊 Confidence: {pose['confidence']:.3f}")
        
        # Show keypoint counts
        if 'keypoints' in pose:
            keypoints = pose['keypoints']
            valid_keypoints = np.sum(keypoints[:, 2] > 0.3) if keypoints.size > 0 else 0
            print(f"   🎯 Valid keypoints: {valid_keypoints}")
        print()
    
    # Create visualization
    print("🎨 Creating visualization...")
    vis_frame = frame.copy()
    
    for pose in pose_estimates:
        x1, y1, x2, y2 = [int(coord) for coord in pose['bbox']]
        
        # Choose color based on flags
        if pose['phone_flag']:
            color = (0, 0, 255)  # Red
            label = "PHONE"
        elif pose['lean_flag']:
            color = (0, 255, 255)  # Yellow
            label = "LEAN"
        elif pose['look_flag']:
            color = (0, 255, 255)  # Yellow
            label = "LOOK"
        else:
            color = (0, 255, 0)  # Green
            label = "NORMAL"
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Add labels
        label_text = f"{label}"
        if pose['lean_flag']:
            label_text += " L"
        if pose['look_flag']:
            label_text += " H"
        if pose['phone_flag']:
            label_text += " P"
        
        cv2.putText(vis_frame, label_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw keypoints if available
        if 'keypoints' in pose and pose['keypoints'].size > 0:
            keypoints = pose['keypoints']
            for j in range(len(keypoints)):
                if keypoints[j, 2] > 0.3:  # Confidence threshold
                    x, y = int(keypoints[j, 0]), int(keypoints[j, 1])
                    cv2.circle(vis_frame, (x, y), 3, (255, 0, 0), -1)
    
    # Save visualization
    output_path = "enhanced_pose_test.jpg"
    cv2.imwrite(output_path, vis_frame)
    print(f"💾 Saved visualization: {output_path}")
    
    # Display results summary
    total_violations = sum(1 for pose in pose_estimates 
                          if pose['lean_flag'] or pose['look_flag'] or pose['phone_flag'])
    print(f"📊 Detection Summary:")
    print(f"   Total people: {len(pose_estimates)}")
    print(f"   With violations: {total_violations}")
    print(f"   Leaning: {sum(1 for pose in pose_estimates if pose['lean_flag'])}")
    print(f"   Looking around: {sum(1 for pose in pose_estimates if pose['look_flag'])}")
    print(f"   Phone use: {sum(1 for pose in pose_estimates if pose['phone_flag'])}")
    
    print()
    print("✅ Enhanced pose detection test complete!")
    print(f"🖼️ Check the visualization: {output_path}")

if __name__ == "__main__":
    test_enhanced_pose_detection()
