"""
CheatGPT3 Headless Webcam Testing
=================================

Test the complete CheatGPT3 Engine pipeline using your webcam without GUI display.
This version saves processed frames to disk instead of displaying them.

Features:
- Live webcam feed processing
- Real-time detection and tracking
- Behavior analysis and policy evaluation
- Automatic frame saving with annotations
- Performance monitoring
- GPU acceleration support
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
    from cheatgpt.engine import Engine
    print("✅ CheatGPT Engine imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import CheatGPT Engine: {e}")
    print("Make sure you're in the correct directory and all dependencies are installed.")
    sys.exit(1)

class HeadlessWebcamTester:
    def __init__(self):
        """Initialize the headless webcam tester with CheatGPT3 Engine."""
        print("🎓 Initializing CheatGPT3 Headless Webcam Tester...")
        
        # Initialize the engine
        self.engine = Engine()
        print(f"✅ Engine initialized: {self.engine}")
        
        # Camera settings
        self.camera_id = 0
        self.frame_width = 640
        self.frame_height = 480
        self.fps_target = 5  # Lower FPS for better processing
        
        # Output settings
        self.output_dir = "webcam_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_save_time = 0
        self.save_interval = 2.0  # Save every 2 seconds
        
        # Status tracking
        self.total_violations = 0
        self.last_violation_time = 0
        
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Target FPS: {self.fps_target}")
        print(f"💾 Save interval: {self.save_interval}s")
    
    def initialize_camera(self):
        """Initialize the webcam."""
        print("📹 Initializing webcam...")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)
        
        # Verify camera settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
        return True
    
    def add_status_overlay(self, frame):
        """Add status information overlay to the frame."""
        # Get current engine stats
        stats = self.engine.get_statistics()
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate FPS
        if elapsed_time > 0:
            current_fps = self.frame_count / elapsed_time
        else:
            current_fps = 0
        
        # Status information
        status_lines = [
            "CheatGPT3 - Real-time Monitoring",
            f"Time: {elapsed_time:.1f}s | Frame: {self.frame_count}",
            f"FPS: {current_fps:.1f} | Target: {self.fps_target}",
            f"Students: {stats.get('total_tracks', 0)}",
            f"Violations: {stats.get('active_violations', 0)}",
            f"Total Events: {stats.get('total_events', 0)}"
        ]
        
        # Add GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                status_lines.append(f"GPU: {gpu_name}")
        except:
            pass
        
        # Draw status overlay
        y_offset = 30
        for i, line in enumerate(status_lines):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)  # Green for title, white for others
            if "Violations:" in line and stats.get('active_violations', 0) > 0:
                color = (0, 0, 255)  # Red for active violations
            
            cv2.putText(frame, line, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def save_frame(self, frame, frame_type="regular"):
        """Save frame to disk with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{frame_type}_{timestamp}_frame_{self.frame_count:06d}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        cv2.imwrite(filepath, frame)
        print(f"💾 Saved {frame_type} frame: {filename}")
        return filepath
    
    def run(self):
        """Run the headless webcam test."""
        print("\n🚀 Starting CheatGPT3 Headless Webcam Test")
        print("=" * 50)
        
        if not self.initialize_camera():
            return
        
        print("▶️  Processing live feed... Press Ctrl+C to stop")
        print(f"📁 Frames will be saved to: {self.output_dir}")
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                
                current_time = time.time()
                
                # Process frame with CheatGPT3 Engine
                try:
                    camera_id = f"webcam_{self.camera_id}"
                    overlay_frame, events = self.engine.process_frame(frame, camera_id, current_time)
                    
                    # Add status overlay
                    overlay_frame = self.add_status_overlay(overlay_frame)
                    
                    # Check for violations and save important frames
                    has_violations = False
                    if events:
                        for event in events:
                            if event.get('severity') in ['orange', 'red']:
                                has_violations = True
                                self.total_violations += 1
                                self.last_violation_time = current_time
                                print(f"🚨 VIOLATION: {event.get('track_id')} - {event.get('event_type')}")
                    
                    # Save frames periodically or when violations occur
                    should_save = (
                        current_time - self.last_save_time >= self.save_interval or
                        has_violations or
                        self.frame_count % 50 == 0  # Save every 50th frame
                    )
                    
                    if should_save:
                        frame_type = "violation" if has_violations else "regular"
                        self.save_frame(overlay_frame, frame_type)
                        self.last_save_time = current_time
                    
                    # Print periodic status
                    if self.frame_count % 30 == 0:  # Every 30 frames
                        stats = self.engine.get_statistics()
                        elapsed = current_time - self.start_time
                        fps = self.frame_count / elapsed if elapsed > 0 else 0
                        print(f"📊 Frame {self.frame_count}: {fps:.1f}fps | "
                              f"Tracks: {stats.get('total_tracks', 0)} | "
                              f"Violations: {stats.get('active_violations', 0)}")
                
                except Exception as e:
                    print(f"⚠️  Frame processing error: {e}")
                    # Save the raw frame for debugging
                    self.save_frame(frame, "error")
                
                self.frame_count += 1
                
                # Control frame rate
                time.sleep(1.0 / self.fps_target)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping webcam test...")
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        
        finally:
            # Cleanup
            self.cap.release()
            
            # Final statistics
            end_time = time.time()
            total_time = end_time - self.start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            
            print("\n📊 Final Statistics:")
            print("=" * 30)
            print(f"⏱️  Total time: {total_time:.1f}s")
            print(f"🎬 Frames processed: {self.frame_count}")
            print(f"📈 Average FPS: {avg_fps:.2f}")
            print(f"🚨 Total violations: {self.total_violations}")
            
            # Engine statistics
            stats = self.engine.get_statistics()
            print(f"👥 Total students tracked: {stats.get('total_tracks', 0)}")
            print(f"📋 Total events: {stats.get('total_events', 0)}")
            
            print(f"📁 Output saved to: {self.output_dir}")
            print("✅ CheatGPT3 Headless Test Complete!")

def main():
    """Main function to run the headless webcam test."""
    print("🎓 CheatGPT3 Headless Webcam Tester")
    print("=" * 40)
    
    # Check for GPU support
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"🔋 CUDA version: {torch.version.cuda}")
        else:
            print("💻 Running on CPU")
    except ImportError:
        print("💻 PyTorch not available, running basic mode")
    
    tester = HeadlessWebcamTester()
    tester.run()

if __name__ == "__main__":
    main()
