"""Comprehensive test of the pose detector functionality."""
import os
import sys
import cv2
import numpy as np

# Add the cheatgpt module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cheatgpt'))

from cheatgpt.detectors.pose_detector import PoseDetector
from cheatgpt.detectors.yolo11_detector import YOLO11Detector

def test_pose_features():
    """Test specific pose detector features."""
    print("=== Comprehensive Pose Detector Test ===\n")
    
    # Initialize detectors
    pose_detector = PoseDetector()
    yolo_detector = YOLO11Detector()
    
    # Test configuration
    print("1. Configuration Test:")
    config = pose_detector.get_model_info()
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Test with original image
    print("2. Detection Test:")
    frame = cv2.imread("original_test_image.jpg")
    if frame is not None:
        print(f"   Image loaded: {frame.shape}")
        
        # Get all detections
        all_detections = yolo_detector.detect(frame)
        phone_detections = [det for det in all_detections if det['cls_name'] == 'cell phone']
        person_detections = [det for det in all_detections if det['cls_name'] == 'person']
        
        print(f"   YOLO detected {len(person_detections)} persons, {len(phone_detections)} phones")
        
        # Get pose estimates
        poses = pose_detector.estimate(frame, phone_detections)
        print(f"   Pose detector processed {len(poses)} poses")
        print()
        
        # Analyze behavior patterns
        print("3. Behavior Analysis:")
        behaviors = {
            'leaning': sum(1 for p in poses if p['lean_flag']),
            'looking_around': sum(1 for p in poses if p['look_flag']),
            'phone_near': sum(1 for p in poses if p['phone_flag']),
            'normal': sum(1 for p in poses if not any([p['lean_flag'], p['look_flag'], p['phone_flag']]))
        }
        
        for behavior, count in behaviors.items():
            print(f"   {behavior.replace('_', ' ').title()}: {count} people")
        print()
        
        # Detailed person analysis
        print("4. Detailed Person Analysis:")
        for i, pose in enumerate(poses):
            print(f"   Person {i+1} (ID: {pose['track_id']}):")
            print(f"     Bbox: [{pose['bbox'][0]:.1f}, {pose['bbox'][1]:.1f}, {pose['bbox'][2]:.1f}, {pose['bbox'][3]:.1f}]")
            print(f"     Head angles: Yaw={pose['yaw']:.1f}°, Pitch={pose['pitch']:.1f}°")
            print(f"     Behaviors: Lean={pose['lean_flag']}, Look={pose['look_flag']}, Phone={pose['phone_flag']}")
            print(f"     Confidence: {pose['confidence']:.3f}")
            
            # Risk assessment
            risk_factors = sum([pose['lean_flag'], pose['look_flag'], pose['phone_flag']])
            if risk_factors == 0:
                risk_level = "LOW"
            elif risk_factors == 1:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            print(f"     Risk Level: {risk_level} ({risk_factors} factors)")
            print()
        
        # Create visualization
        print("5. Visualization:")
        annotated_frame = frame.copy()
        
        for i, pose in enumerate(poses):
            x1, y1, x2, y2 = [int(x) for x in pose['bbox']]
            
            # Color coding based on risk
            risk_factors = sum([pose['lean_flag'], pose['look_flag'], pose['phone_flag']])
            if risk_factors == 0:
                color = (0, 255, 0)  # Green - safe
            elif risk_factors == 1:
                color = (0, 165, 255)  # Orange - caution
            else:
                color = (0, 0, 255)  # Red - alert
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Add comprehensive label
            labels = []
            if pose['lean_flag']:
                labels.append("LEAN")
            if pose['look_flag']:
                labels.append(f"LOOK({pose['yaw']:.0f}°)")
            if pose['phone_flag']:
                labels.append("PHONE")
            
            main_label = f"P{i+1}"
            if labels:
                main_label += f": {', '.join(labels)}"
            else:
                main_label += ": OK"
            
            # Draw label background
            label_size = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1-30), (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(annotated_frame, main_label, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw head direction indicator
            if abs(pose['yaw']) > 5:  # Only show if significant head turn
                center_x, center_y = int((x1 + x2) / 2), int(y1 + (y2 - y1) * 0.2)
                arrow_length = 30
                arrow_x = center_x + int(arrow_length * np.sin(np.radians(pose['yaw'])))
                arrow_y = center_y
                cv2.arrowedLine(annotated_frame, (center_x, center_y), (arrow_x, arrow_y), color, 3)
        
        # Add legend
        legend_y = 30
        cv2.putText(annotated_frame, "Legend: Green=Safe, Orange=Caution, Red=Alert", 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        output_path = "comprehensive_pose_result.jpg"
        cv2.imwrite(output_path, annotated_frame)
        print(f"   Saved comprehensive result to {output_path}")
        print()
        
        # Test edge cases
        print("6. Edge Case Tests:")
        
        # Test with empty frame
        empty_poses = pose_detector.estimate(None)
        print(f"   Empty frame handling: {len(empty_poses)} poses (expected: 0)")
        
        # Test with no phone detections
        poses_no_phone = pose_detector.estimate(frame, [])
        print(f"   No phone detections: {len(poses_no_phone)} poses")
        
        # Test with mock phone detection
        mock_phone = {
            'bbox': [poses[0]['bbox'][0], poses[0]['bbox'][1], 
                    poses[0]['bbox'][0] + 50, poses[0]['bbox'][1] + 100],
            'conf': 0.8,
            'cls_name': 'cell phone'
        }
        poses_with_phone = pose_detector.estimate(frame, [mock_phone])
        phone_flags = [p['phone_flag'] for p in poses_with_phone]
        print(f"   Mock phone near person 1: {phone_flags}")
        print()
        
        print("=== Test Completed Successfully ===")
        
    else:
        print("Failed to load test image")

if __name__ == "__main__":
    test_pose_features()
