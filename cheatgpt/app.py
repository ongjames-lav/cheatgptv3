"""Enhanced CheatGPT Application with Video Recording and Session Management.

This application demonstrates the complete CheatGPT system with:
- Real-time webcam processing
- Video recording with overlays
- Session management with database storage
- Hotspot tracking and analysis
"""

import cv2
import time
import logging
import os
from datetime import datetime
from .engine import Engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CheatGPTApp:
    """Main CheatGPT Application with video recording capabilities."""
    
    def __init__(self):
        """Initialize the CheatGPT application."""
        logger.info("🎓 Initializing CheatGPT Application...")
        
        # Initialize engine
        self.engine = Engine()
        
        # Camera settings
        self.camera_id = 0
        self.cam_name = "main_camera"
        
        # Control flags
        self.running = False
        self.paused = False
        
        logger.info("✅ CheatGPT Application initialized")
    
    def start_webcam_session(self, duration_minutes: int = 5, record_video: bool = True):
        """Start a webcam monitoring session with optional video recording.
        
        Args:
            duration_minutes: Duration to run the session (0 = infinite)
            record_video: Whether to record video with overlays
        """
        logger.info(f"🎥 Starting webcam session (duration: {duration_minutes}min, recording: {record_video})")
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            logger.error(f"❌ Failed to open camera {self.camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual frame size for video recording
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        
        logger.info(f"📷 Camera initialized: {width}x{height}")
        
        # Start session
        session_id = self.engine.start_session(
            cam_id=self.cam_name,
            frame_size=frame_size if record_video else None
        )
        
        if not session_id:
            logger.error("❌ Failed to start session")
            cap.release()
            return
        
        # Session timing
        start_time = time.time()
        max_duration = duration_minutes * 60 if duration_minutes > 0 else float('inf')
        frame_count = 0
        
        logger.info(f"▶️ Session {session_id} started - Press 'q' to quit, SPACE to pause/resume")
        
        self.running = True
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.error("❌ Failed to capture frame")
                    break
                
                # Mirror effect for user convenience
                frame = cv2.flip(frame, 1)
                
                # Process frame through CheatGPT engine (unless paused)
                if not self.paused:
                    overlay_frame, events = self.engine.process_frame(
                        frame=frame,
                        cam_id=self.cam_name,
                        ts=time.time()
                    )
                    frame_count += 1
                    
                    # Log significant events
                    if events:
                        critical_events = [e for e in events if e.get('severity') == 'red']
                        if critical_events:
                            logger.warning(f"🚨 CRITICAL EVENTS DETECTED: {len(critical_events)}")
                            for event in critical_events:
                                logger.warning(f"   - {event['event_type']} (confidence: {event['confidence']:.3f})")
                else:
                    overlay_frame = frame
                
                # Add session info overlay
                overlay_frame = self._add_session_overlay(overlay_frame, session_id, frame_count, start_time)
                
                # Display frame
                cv2.imshow('CheatGPT - Live Monitoring', overlay_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("👋 User requested quit")
                    break
                elif key == ord(' '):
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "RESUMED"
                    logger.info(f"⏯️ Session {status}")
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"screenshots/session_{session_id}_{timestamp}.jpg"
                    os.makedirs("screenshots", exist_ok=True)
                    cv2.imwrite(screenshot_path, overlay_frame)
                    logger.info(f"📸 Screenshot saved: {screenshot_path}")
                
                # Check duration limit
                elapsed_time = time.time() - start_time
                if elapsed_time >= max_duration:
                    logger.info(f"⏰ Session duration limit reached ({duration_minutes}min)")
                    break
        
        except KeyboardInterrupt:
            logger.info("🛑 Session interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Stop session and get final stats
            session_info = self.engine.stop_session()
            
            logger.info("📊 Session Summary:")
            logger.info(f"   Session ID: {session_id}")
            logger.info(f"   Duration: {session_info.get('duration', 0):.1f} seconds")
            logger.info(f"   Frames processed: {session_info.get('frame_count', frame_count)}")
            logger.info(f"   Video recorded: {session_info.get('video_recorded', False)}")
            logger.info(f"   Hotspots detected: {session_info.get('hotspot_count', 0)}")
            
            if session_info.get('video_recorded'):
                logger.info(f"   Video file: {session_info.get('video_path', 'Unknown')}")
    
    def _add_session_overlay(self, frame, session_id: str, frame_count: int, start_time: float):
        """Add session information overlay to frame."""
        try:
            # Calculate session stats
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Get session status
            session_status = self.engine.get_session_status()
            recording_status = session_status.get('recording', {})
            
            # Session info
            info_lines = [
                f"Session: {session_id[:12]}...",
                f"Time: {elapsed:.1f}s | Frames: {frame_count}",
                f"FPS: {fps:.1f} | Status: {'PAUSED' if self.paused else 'RUNNING'}",
                f"Recording: {'ON' if recording_status.get('recording', False) else 'OFF'}",
                f"Hotspots: {session_status.get('hotspot_count', 0)}"
            ]
            
            # Draw overlay background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 140), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Draw session info
            for i, line in enumerate(info_lines):
                color = (0, 255, 0) if not self.paused else (0, 255, 255)
                cv2.putText(frame, line, (15, 35 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add controls info
            cv2.putText(frame, "Controls: Q=Quit, SPACE=Pause, S=Screenshot", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            logger.error(f"Failed to add session overlay: {e}")
        
        return frame
    
    def stop(self):
        """Stop the application."""
        logger.info("🛑 Stopping CheatGPT Application...")
        self.running = False
    
    def demo_batch_processing(self, video_path: str):
        """Demonstrate batch processing of a pre-recorded video with session recording.
        
        Args:
            video_path: Path to input video file
        """
        logger.info(f"🎬 Starting batch processing demo: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"❌ Failed to open video: {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📹 Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
        
        # Start session with recording
        session_id = self.engine.start_session(
            cam_id="batch_processing",
            frame_size=(width, height)
        )
        
        if not session_id:
            logger.error("❌ Failed to start batch processing session")
            cap.release()
            return
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                overlay_frame, events = self.engine.process_frame(
                    frame=frame,
                    cam_id="batch_processing",
                    ts=time.time()
                )
                
                frame_count += 1
                
                # Log progress
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"🔄 Processing: {frame_count}/{total_frames} ({progress:.1f}%)")
                
                # Optional: Display processed frames (comment out for pure batch processing)
                # cv2.imshow('Batch Processing', overlay_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        
        except KeyboardInterrupt:
            logger.info("🛑 Batch processing interrupted")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Stop session
            session_info = self.engine.stop_session()
            
            processing_time = time.time() - start_time
            
            logger.info("📊 Batch Processing Complete:")
            logger.info(f"   Session ID: {session_id}")
            logger.info(f"   Frames processed: {frame_count}")
            logger.info(f"   Processing time: {processing_time:.1f}s")
            logger.info(f"   Processing FPS: {frame_count / processing_time:.1f}")
            logger.info(f"   Output video: {session_info.get('video_path', 'None')}")
            logger.info(f"   Hotspots detected: {session_info.get('hotspot_count', 0)}")

def main():
    """Main function demonstrating the CheatGPT application."""
    print("🎓 CheatGPT Application with Video Recording")
    print("=" * 50)
    
    app = CheatGPTApp()
    
    # Start webcam session with video recording
    app.start_webcam_session(duration_minutes=2, record_video=True)

if __name__ == "__main__":
    main()
