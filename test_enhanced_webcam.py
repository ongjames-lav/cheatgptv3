"""
Enhanced CheatGPT3 Real-time Webcam Testing
===========================================

Test the complete enhanced CheatGPT3 system with realistic LSTM (88.64% accuracy) 
and COCO spatial enhancements using your webcam.

Features:
- Enhanced realistic LSTM integration
- COCO-enhanced spatial detection
- Real-time behavioral analysis
- Multi-task predictions (gesture + looking)
- Visual overlay with confidence scores
- Performance monitoring
- Keyboard controls

Controls:
- ESC or 'q': Quit
- SPACE: Toggle pause/resume
- 's': Save current frame manually
- 'r': Reset engine statistics
- 'h': Show/hide help overlay
- 'i': Show/hide detailed info
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from realistic_lstm_integration import RealisticLSTMClassifier
    print("✅ Enhanced LSTM integration imported successfully!")
    ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced LSTM not available: {e}")
    ENHANCED_AVAILABLE = False

# Try to import original engine as fallback
try:
    from cheatgpt.engine import Engine
    print("✅ Original CheatGPT Engine available as fallback!")
    ORIGINAL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Original engine not available: {e}")
    ORIGINAL_AVAILABLE = False

class EnhancedWebcamTester:
    def __init__(self):
        """Initialize the enhanced webcam tester."""
        print("🎓 Initializing Enhanced CheatGPT3 Real-time Webcam Tester...")
        
        # Initialize enhanced LSTM
        self.enhanced_lstm = None
        self.original_engine = None
        
        if ENHANCED_AVAILABLE:
            self.enhanced_lstm = RealisticLSTMClassifier()
            print(f"✅ Enhanced LSTM initialized: {self.enhanced_lstm.is_loaded}")
        
        if ORIGINAL_AVAILABLE:
            self.original_engine = Engine()
            print("✅ Original engine initialized with YOLO11-Pose detection")
        
        # Camera settings
        self.camera_id = 0
        self.cam_name = "enhanced_webcam_test"
        
        # Control flags
        self.paused = False
        self.show_help = True
        self.show_detailed_info = True
        self.frame_count = 0
        self.start_time = time.time()
        
        # Detection history for LSTM
        self.detection_history = []
        self.latest_overlay_frame = None
        
        # Performance tracking
        self.fps_history = []
        self.process_times = []
        self.prediction_history = []
        
        # Create screenshots directory
        self.screenshots_dir = "enhanced_webcam_screenshots"
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
    def initialize_camera(self):
        """Initialize the webcam."""
        print(f"📷 Initializing camera {self.camera_id}...")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_id}")
            return False
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera initialized: {width}x{height} @ {fps}fps")
        return True
    
    def extract_pose_features(self, frame):
        """Extract pose features from frame using the original CheatGPT engine."""
        if not self.original_engine:
            return self.create_dummy_detection()
        
        try:
            # Use the original engine to process the frame
            timestamp = time.time()
            overlay_frame, events = self.original_engine.process_frame(
                frame=frame,
                cam_id=self.cam_name,
                ts=timestamp
            )
            
            # Store the overlay frame for later use
            self.latest_overlay_frame = overlay_frame
            
            # Create realistic detection data based on whether events were triggered
            gesture_flag = 0
            look_flag = 0 
            lean_flag = 0
            phone_flag = 0
            
            # Check if any suspicious events were detected
            if events:
                for event in events:
                    event_type = event.get('event_type', '').lower()
                    if 'gesture' in event_type or 'hand' in event_type:
                        gesture_flag = 1
                    elif 'look' in event_type or 'head' in event_type:
                        look_flag = 1
                    elif 'lean' in event_type:
                        lean_flag = 1
                    elif 'phone' in event_type or 'object' in event_type:
                        phone_flag = 1
            
            # Create realistic feature vector
            return {
                'lean_flag': lean_flag,
                'look_flag': look_flag,
                'phone_flag': phone_flag,
                'gesture_flag': gesture_flag,
                'lean_angle': lean_flag * 15.0,  # Simulate angle
                'head_turn_angle': look_flag * 20.0,  # Simulate turn
                'confidence': 0.8 if events else 0.6,
                'center_x': 0.5 + (look_flag * 0.1),  # Simulate position shift
                'center_y': 0.5,
                'bbox_area': 0.3,
                'combined_suspicious': max(gesture_flag, look_flag, lean_flag, phone_flag),
                'spatial_center_offset': abs(0.5 - (0.5 + look_flag * 0.1))
            }
            
        except Exception as e:
            print(f"⚠️ Error extracting pose features: {e}")
            return self.create_dummy_detection()
    
    def create_dummy_detection(self):
        """Create dummy detection when pose detection fails."""
        return {
            'lean_flag': 0, 'look_flag': 0, 'phone_flag': 0, 'gesture_flag': 0,
            'lean_angle': 0, 'head_turn_angle': 0, 'confidence': 0.5,
            'center_x': 0.5, 'center_y': 0.5, 'bbox_area': 0.3,
            'combined_suspicious': 0, 'spatial_center_offset': 0
        }
    
    def process_frame_enhanced(self, frame):
        """Process frame using enhanced LSTM system with original engine detection."""
        if self.paused:
            return frame, {}
        
        process_start = time.time()
        
        # Extract pose features (this will also process the frame through the engine)
        detection_data = self.extract_pose_features(frame)
        
        # Use the overlay frame from pose feature extraction (has bounding boxes)
        overlay_frame = getattr(self, 'latest_overlay_frame', frame)
        
        # Add to detection history for LSTM
        self.detection_history.append(detection_data)
        if len(self.detection_history) > 15:
            self.detection_history.pop(0)
        
        # Get enhanced prediction if available
        prediction_result = {}
        if self.enhanced_lstm and self.enhanced_lstm.is_loaded and len(self.detection_history) >= 5:
            try:
                prediction_result = self.enhanced_lstm.predict_behavior_sequence(
                    self.detection_history[-10:]
                )
            except Exception as e:
                print(f"⚠️ LSTM prediction error: {e}")
        
        # Record processing time
        process_time = time.time() - process_start
        self.process_times.append(process_time)
        
        return overlay_frame, prediction_result
    
    def draw_enhanced_overlay(self, frame, prediction_result):
        """Draw enhanced overlay with LSTM predictions."""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        if prediction_result:
            # Main prediction info
            prediction = prediction_result.get('prediction', 'unknown')
            confidence = prediction_result.get('confidence', 0)
            
            # Color based on prediction
            if prediction == 'normal':
                color = (0, 255, 0)  # Green
            elif prediction in ['suspicious_gesture', 'suspicious_looking']:
                color = (0, 165, 255)  # Orange
            elif prediction == 'mixed_suspicious':
                color = (0, 255, 255)  # Yellow
            else:
                color = (128, 128, 128)  # Gray
            
            # Main prediction box
            cv2.rectangle(overlay, (10, h-150), (400, h-10), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Prediction text
            cv2.putText(frame, f"Prediction: {prediction}", (20, h-130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.3f}", (20, h-110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Model info
            cv2.putText(frame, f"Model: Realistic LSTM (88.64%)", (20, h-90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"COCO Enhanced: {prediction_result.get('coco_enhanced', False)}", 
                       (20, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Auxiliary predictions
            if 'auxiliary_predictions' in prediction_result:
                aux = prediction_result['auxiliary_predictions']
                cv2.putText(frame, f"Gesture: {aux.get('gesture_confidence', 0):.3f}", 
                           (20, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Looking: {aux.get('looking_confidence', 0):.3f}", 
                           (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_help_overlay(self, frame):
        """Draw help overlay."""
        if not self.show_help:
            return frame
            
        help_text = [
            "Enhanced CheatGPT3 Testing - Controls:",
            "ESC/q: Quit  |  SPACE: Pause/Resume",
            "s: Save Frame  |  r: Reset  |  h: Toggle Help  |  i: Toggle Info"
        ]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (650, 90), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        for i, text in enumerate(help_text):
            y = 30 + i * 20
            cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return frame
    
    def draw_performance_overlay(self, frame):
        """Draw performance statistics."""
        if not self.show_detailed_info:
            return frame
            
        h, w = frame.shape[:2]
        elapsed = time.time() - self.start_time
        
        # Calculate stats
        avg_fps = np.mean(self.fps_history[-30:]) if self.fps_history else 0
        avg_process_time = np.mean(self.process_times[-30:]) * 1000 if self.process_times else 0
        
        perf_text = [
            f"Runtime: {elapsed:.1f}s | Frames: {self.frame_count}",
            f"FPS: {avg_fps:.1f} | Process: {avg_process_time:.1f}ms",
            f"LSTM: {'Loaded' if self.enhanced_lstm and self.enhanced_lstm.is_loaded else 'Not Available'}",
            f"Detection History: {len(self.detection_history)}/15",
            f"Status: {'PAUSED' if self.paused else 'RUNNING'}"
        ]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-350, 10), (w-10, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        for i, text in enumerate(perf_text):
            y = 30 + i * 20
            cv2.putText(frame, text, (w-340, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
        return frame
    
    def save_enhanced_screenshot(self, frame, prediction_result):
        """Save screenshot with prediction info."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction = prediction_result.get('prediction', 'unknown') if prediction_result else 'no_prediction'
        confidence = prediction_result.get('confidence', 0) if prediction_result else 0
        
        filename = f"{self.screenshots_dir}/enhanced_{prediction}_{confidence:.3f}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Enhanced screenshot saved: {filename}")
    
    def run(self):
        """Main testing loop."""
        if not self.initialize_camera():
            return
            
        print("\n🎓 Enhanced CheatGPT3 Real-time Webcam Testing Started!")
        print("🧠 Using realistic LSTM (88.64% accuracy) + COCO enhancements")
        print("📹 Try different behaviors:")
        print("   - Sit normally → normal")
        print("   - Wave hands near face → suspicious_gesture") 
        print("   - Look left/right → suspicious_looking")
        print("   - Mix behaviors → mixed_suspicious")
        print("\n▶️  Starting enhanced live feed... Press ESC or 'q' to quit\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                    
                frame = cv2.flip(frame, 1)  # Mirror effect
                frame_start = time.time()
                
                # Process frame with enhanced system
                processed_frame, prediction_result = self.process_frame_enhanced(frame)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                if frame_time > 0:
                    fps = 1.0 / frame_time
                    self.fps_history.append(fps)
                
                # Add overlays
                processed_frame = self.draw_enhanced_overlay(processed_frame, prediction_result)
                processed_frame = self.draw_performance_overlay(processed_frame)
                processed_frame = self.draw_help_overlay(processed_frame)
                
                # Display
                cv2.imshow('Enhanced CheatGPT3 Real-time Testing', processed_frame)
                self.frame_count += 1
                
                # Track predictions
                if prediction_result:
                    self.prediction_history.append(prediction_result.get('prediction', 'unknown'))
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27 or key == ord('q'):  # ESC or 'q'
                    print("\n👋 Stopping enhanced webcam test...")
                    break
                elif key == ord(' '):  # SPACE
                    self.paused = not self.paused
                    print(f"⏯️  {'PAUSED' if self.paused else 'RESUMED'}")
                elif key == ord('s'):  # 's'
                    self.save_enhanced_screenshot(processed_frame, prediction_result)
                elif key == ord('r'):  # 'r'
                    self.detection_history.clear()
                    self.fps_history.clear()
                    self.process_times.clear()
                    self.prediction_history.clear()
                    self.frame_count = 0
                    self.start_time = time.time()
                    print("🔄 Enhanced system reset")
                elif key == ord('h'):  # 'h'
                    self.show_help = not self.show_help
                elif key == ord('i'):  # 'i'
                    self.show_detailed_info = not self.show_detailed_info
                    
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")
            
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.print_final_stats()
    
    def print_final_stats(self):
        """Print final enhanced testing statistics."""
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("🎓 Enhanced CheatGPT3 Webcam Test Complete!")
        print("="*60)
        print(f"📊 Session Statistics:")
        print(f"   Runtime: {elapsed:.1f} seconds")
        print(f"   Frames processed: {self.frame_count}")
        print(f"   Average FPS: {np.mean(self.fps_history):.2f}" if self.fps_history else "   Average FPS: N/A")
        print(f"   Average processing time: {np.mean(self.process_times)*1000:.1f}ms" if self.process_times else "   Average processing time: N/A")
        print(f"   Enhanced LSTM predictions: {len(self.prediction_history)}")
        
        if self.prediction_history:
            from collections import Counter
            prediction_counts = Counter(self.prediction_history)
            print(f"   Prediction distribution:")
            for pred, count in prediction_counts.items():
                percentage = count / len(self.prediction_history) * 100
                print(f"     {pred}: {count} ({percentage:.1f}%)")
        
        print(f"   Enhanced screenshots: {len([f for f in os.listdir(self.screenshots_dir) if f.endswith('.jpg')])}")
        print("="*60)

def main():
    """Main function."""
    print("🎓 Enhanced CheatGPT3 Real-time Webcam Tester")
    print("=" * 50)
    
    if not ENHANCED_AVAILABLE and not ORIGINAL_AVAILABLE:
        print("❌ No CheatGPT engines available!")
        return
    
    if not ORIGINAL_AVAILABLE:
        print("❌ Original CheatGPT engine required for pose detection!")
        print("   Please ensure the cheatgpt.engine module is available.")
        return
    
    # Check camera
    test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        print("❌ No webcam detected! Please ensure your camera is connected.")
        return
    test_cap.release()
    
    # Run enhanced test
    tester = EnhancedWebcamTester()
    tester.run()

if __name__ == "__main__":
    main()
