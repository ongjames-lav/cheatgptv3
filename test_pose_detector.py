"""Test the pose detector implementation."""
import os
import sys
import cv2
import numpy as np

# Add the cheatgpt module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cheatgpt'))

try:
    from cheatgpt.detectors.pose_detector import PoseDetector
    from cheatgpt.detectors.yolo11_detector import YOLO11Detector
    
    def test_pose_detector():
        """Test the pose detector with a sample image."""
        print("Testing PoseDetector...")
        
        # Initialize detectors
        pose_detector = PoseDetector()
        yolo_detector = YOLO11Detector()
        
        print(f"Pose detector model info: {pose_detector.get_model_info()}")
        
        # Load test image
        test_image_path = "original_test_image.jpg"
        if os.path.exists(test_image_path):
            frame = cv2.imread(test_image_path)
            
            if frame is not None:
                print(f"Loaded test image: {frame.shape}")
                
                # Get phone detections first
                phone_detections = yolo_detector.detect(frame)
                phone_only = [det for det in phone_detections if det['cls_name'] == 'cell phone']
                print(f"Found {len(phone_only)} phone detections")
                
                # Get pose estimates
                poses = pose_detector.estimate(frame, phone_only)
                print(f"Found {len(poses)} pose estimates")
                
                for i, pose in enumerate(poses):
                    print(f"Person {i+1}:")
                    print(f"  Track ID: {pose['track_id']}")
                    print(f"  Bbox: {pose['bbox']}")
                    print(f"  Yaw: {pose['yaw']:.2f}°")
                    print(f"  Pitch: {pose['pitch']:.2f}°")
                    print(f"  Leaning: {pose['lean_flag']}")
                    print(f"  Looking around: {pose['look_flag']}")
                    print(f"  Phone near: {pose['phone_flag']}")
                    print(f"  Confidence: {pose['confidence']:.3f}")
                    print()
                
                # Save annotated result
                annotated_frame = frame.copy()
                for pose in poses:
                    x1, y1, x2, y2 = [int(x) for x in pose['bbox']]
                    
                    # Draw bounding box
                    color = (0, 255, 0) if not any([pose['lean_flag'], pose['look_flag'], pose['phone_flag']]) else (0, 0, 255)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add labels
                    label = f"ID:{pose['track_id']}"
                    if pose['lean_flag']:
                        label += " LEAN"
                    if pose['look_flag']:
                        label += " LOOK"
                    if pose['phone_flag']:
                        label += " PHONE"
                    
                    cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imwrite("pose_detection_result.jpg", annotated_frame)
                print("Saved annotated result to pose_detection_result.jpg")
                
            else:
                print("Failed to load test image")
        else:
            print(f"Test image {test_image_path} not found")
    
    if __name__ == "__main__":
        test_pose_detector()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Required packages may not be installed. Please install:")
    print("pip install torch torchvision opencv-python numpy ultralytics python-dotenv")
except Exception as e:
    print(f"Error: {e}")
