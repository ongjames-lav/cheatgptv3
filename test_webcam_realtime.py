"""
CheatGPT3 Real-time Webcam Testing
==================================

Test the complete CheatGPT3 Engine pipeline using your webcam for real-time exam proctoring simulation.

Features:
- Live webcam feed processing
- Real-time detection and tracking
- Behavior analysis and policy evaluation
- Visual overlay with bounding boxes
- Automatic evidence saving
- Performance monitoring
- Keyboard controls for interaction

Controls:
- ESC or 'q': Quit
- SPACE: Toggle pause/resume
- 's': Save current frame manually
- 'r': Reset engine statistics
- 'h': Show/hide help overlay
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

class WebcamTester:
    def __init__(self):
        """Initialize the webcam tester with CheatGPT3 Engine."""
        print("🎓 Initializing CheatGPT3 Real-time Webcam Tester...")
        
        # Initialize the engine
        self.engine = Engine()
        print(f"✅ Engine initialized: {self.engine}")
        
        # Camera settings
        self.camera_id = 0  # Default webcam
        self.cam_name = "webcam_test"
        
        # Control flags
        self.paused = False
        self.show_help = True
        self.frame_count = 0
        self.start_time = time.time()
        
        # Performance tracking
        self.fps_history = []
        self.process_times = []
        
        # Create screenshots directory
        self.screenshots_dir = "webcam_screenshots"
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
    def initialize_camera(self):
        """Initialize the webcam."""
        print(f"📷 Initializing camera {self.camera_id}...")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_id}")
            return False
            
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera initialized: {width}x{height} @ {fps}fps")
        return True
        
    def draw_help_overlay(self, frame):
        """Draw help text overlay on the frame."""
        if not self.show_help:
            return frame
            
        help_text = [
            "CheatGPT3 Real-time Testing - Controls:",
            "ESC/q: Quit  |  SPACE: Pause/Resume",
            "s: Save Frame  |  r: Reset Stats  |  h: Toggle Help"
        ]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 90), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw help text
        for i, text in enumerate(help_text):
            y = 30 + i * 20
            cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return frame
        
    def draw_performance_overlay(self, frame):
        """Draw performance statistics overlay."""
        # Get current stats
        stats = self.engine.get_statistics()
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calculate average FPS
        if self.fps_history:
            avg_fps = np.mean(self.fps_history[-30:])  # Last 30 frames
        else:
            avg_fps = 0
            
        # Calculate average processing time
        if self.process_times:
            avg_process_time = np.mean(self.process_times[-30:]) * 1000  # Convert to ms
        else:
            avg_process_time = 0
        
        # Performance text
        perf_text = [
            f"Runtime: {elapsed:.1f}s | Frames: {self.frame_count}",
            f"FPS: {avg_fps:.1f} | Process: {avg_process_time:.1f}ms",
            f"Tracks: {stats['total_tracks_created']} | Violations: {stats['active_violations']}",
            f"Status: {'PAUSED' if self.paused else 'RUNNING'}"
        ]
        
        # Draw performance overlay (top right)
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-320, 10), (w-10, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        for i, text in enumerate(perf_text):
            y = 30 + i * 20
            cv2.putText(frame, text, (w-310, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
        return frame
        
    def save_manual_screenshot(self, frame):
        """Save a manual screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.screenshots_dir}/manual_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
        
    def process_frame(self, frame):
        """Process a single frame through the CheatGPT3 pipeline."""
        if self.paused:
            return frame, []
            
        # Record processing start time
        process_start = time.time()
        
        # Generate timestamp as float (Unix timestamp)
        timestamp = time.time()
        
        try:
            # Process frame through CheatGPT3 Engine
            overlay_frame, events = self.engine.process_frame(
                frame=frame,
                cam_id=self.cam_name,
                ts=timestamp
            )
            
            # Record processing time
            process_time = time.time() - process_start
            self.process_times.append(process_time)
            
            # Print events if any
            if events:
                for event in events:
                    severity = event.get('severity', 'Unknown')
                    event_type = event.get('event_type', 'Unknown')
                    track_id = event.get('track_id', 'Unknown')
                    print(f"🚨 Event: {event_type} | Severity: {severity} | Track: {track_id}")
                    
            return overlay_frame, events
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return frame, []
            
    def run(self):
        """Main testing loop."""
        if not self.initialize_camera():
            return
            
        print("\n🎓 CheatGPT3 Real-time Webcam Testing Started!")
        print("📹 Point your webcam at yourself and try different behaviors:")
        print("   - Sit normally (should be green)")
        print("   - Look around (should turn yellow)")
        print("   - Lean significantly (should turn yellow)")
        print("   - Hold up a phone (should turn orange/red)")
        print("\n▶️  Starting live feed... Press ESC or 'q' to quit\n")
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                    
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Record frame time for FPS calculation
                frame_start = time.time()
                
                # Process frame through CheatGPT3
                overlay_frame, events = self.process_frame(frame)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                if frame_time > 0:
                    fps = 1.0 / frame_time
                    self.fps_history.append(fps)
                
                # Add overlays
                overlay_frame = self.draw_performance_overlay(overlay_frame)
                overlay_frame = self.draw_help_overlay(overlay_frame)
                
                # Display frame
                cv2.imshow('CheatGPT3 Real-time Testing', overlay_frame)
                
                self.frame_count += 1
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27 or key == ord('q'):  # ESC or 'q'
                    print("\n👋 Stopping webcam test...")
                    break
                elif key == ord(' '):  # SPACE
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "RESUMED"
                    print(f"⏯️  {status}")
                elif key == ord('s'):  # 's'
                    self.save_manual_screenshot(overlay_frame)
                elif key == ord('r'):  # 'r'
                    self.engine.reset()
                    self.fps_history.clear()
                    self.process_times.clear()
                    self.frame_count = 0
                    self.start_time = time.time()
                    print("🔄 Engine statistics reset")
                elif key == ord('h'):  # 'h'
                    self.show_help = not self.show_help
                    
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")
            
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self.print_final_stats()
            
    def print_final_stats(self):
        """Print final testing statistics."""
        stats = self.engine.get_statistics()
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*50)
        print("🎓 CheatGPT3 Webcam Test Complete!")
        print("="*50)
        print(f"📊 Session Statistics:")
        print(f"   Runtime: {elapsed:.1f} seconds")
        print(f"   Frames processed: {self.frame_count}")
        print(f"   Average FPS: {np.mean(self.fps_history):.2f}" if self.fps_history else "   Average FPS: N/A")
        print(f"   Average processing time: {np.mean(self.process_times)*1000:.1f}ms" if self.process_times else "   Average processing time: N/A")
        print(f"   Total tracks: {stats['total_tracks_created']}")
        print(f"   Active violations: {stats['active_violations']}")
        print(f"   Evidence frames saved: {len([f for f in os.listdir('uploads/evidence') if f.endswith('.jpg')]) if os.path.exists('uploads/evidence') else 0}")
        print(f"   Manual screenshots: {len([f for f in os.listdir(self.screenshots_dir) if f.endswith('.jpg')])}")
        print("="*50)

def main():
    """Main function to run the webcam test."""
    print("🎓 CheatGPT3 Real-time Webcam Tester")
    print("=" * 40)
    
    # Check if camera is available
    test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        print("❌ No webcam detected! Please ensure your camera is connected and not in use.")
        return
    test_cap.release()
    
    # Run the test
    tester = WebcamTester()
    tester.run()

if __name__ == "__main__":
    main()
