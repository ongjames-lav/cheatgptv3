#!/usr/bin/env python3
"""
Enhanced Detection Runner with Hotspot Overlay Integration
Run detection with real-time video recording and event overlays
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
import logging
import sys
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from cheatgpt.engine import Engine
from cheatgpt.overlays.overlay_recorder import OverlayVideoRecorder, DetectionRecorderIntegration

# Simple webcam manager class
class WebcamManager:
    """Simple webcam management class"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.is_active = False
    
    def start(self) -> bool:
        """Start the webcam"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_active = True
        logger.info(f"Camera {self.camera_id} started successfully")
        return True
    
    def get_frame(self) -> np.ndarray:
        """Get a frame from the webcam"""
        if not self.is_active or not self.cap:
            logger.warning("Camera not active or not initialized")
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            return None
        
        return frame
    
    def stop(self):
        """Stop the webcam"""
        if self.cap:
            self.cap.release()
        self.is_active = False
        logger.info("Camera stopped")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDetectionRunner:
    """Enhanced detection runner with overlay recording capabilities"""
    
    def __init__(self, enable_recording: bool = True, enable_overlay: bool = True,
                 output_dir: str = "recordings", enable_visual_overlays: bool = True):
        
        # Initialize detection engine
        self.engine = Engine()
        
        # Initialize recording system
        self.enable_recording = enable_recording
        self.enable_visual_overlays = enable_visual_overlays
        self.recorder = None
        self.recorder_integration = None
        
        if enable_recording:
            self.recorder = OverlayVideoRecorder(
                output_dir=output_dir,
                enable_overlay=enable_overlay,
                enable_visual_overlays=enable_visual_overlays
            )
            self.recorder_integration = DetectionRecorderIntegration(self.recorder)
        
        # Initialize camera
        self.camera = WebcamManager()
        
        # Runtime settings
        self.is_running = False
        self.display_overlay = enable_overlay
        self.show_debug_info = True
        self.engine_session_id = None
        
        logger.info(f"Enhanced detection runner initialized")
        logger.info(f"Recording: {'ON' if enable_recording else 'OFF'}")
        logger.info(f"Overlay: {'ON' if enable_overlay else 'OFF'}")
    
    def start_session(self, session_name: str = None) -> str:
        """Start a new detection session with optional recording"""
        logger.info("🚀 Starting enhanced detection session...")
        
        # Start camera
        if not self.camera.start():
            raise RuntimeError("Failed to start camera")
        
        # Get camera properties for the engine
        frame = self.camera.get_frame()
        if frame is not None:
            h, w = frame.shape[:2]
            frame_size = (w, h)
        else:
            frame_size = (640, 480)  # Default size
        
        # Start engine session with video recording
        engine_session_id = self.engine.start_session(
            cam_id=session_name or "enhanced_detection",
            frame_size=frame_size
        )
        
        if not engine_session_id:
            raise RuntimeError("Failed to start engine session")
        
        logger.info(f"🎬 Engine session started: {engine_session_id}")
        
        # Start overlay recording if enabled
        recording_file = None
        if self.enable_recording and self.recorder:
            recording_file = self.recorder.start_recording(session_name)
            logger.info(f"📹 Overlay recording started: {recording_file}")
        
        self.is_running = True
        self.engine_session_id = engine_session_id
        return recording_file or engine_session_id
    
    def stop_session(self) -> dict:
        """Stop the detection session and return summary"""
        logger.info("🛑 Stopping detection session...")
        
        self.is_running = False
        
        # Stop camera
        self.camera.stop()
        
        # Stop engine session and get summary
        engine_summary = {}
        if self.engine_session_id:
            engine_summary = self.engine.stop_session()
            logger.info(f"🎬 Engine session stopped: {self.engine_session_id}")
        
        # Stop overlay recording and get summary
        overlay_summary = {}
        if self.enable_recording and self.recorder:
            recording_summary = self.recorder.stop_recording()
            if recording_summary:
                overlay_summary.update(recording_summary)
                logger.info(f"📁 Overlay recording saved: {recording_summary['filename']}")
                if 'timeline_file' in recording_summary:
                    logger.info(f"📊 Timeline exported: {recording_summary['timeline_file']}")
        
        # Combine summaries
        combined_summary = {
            'engine_session': engine_summary,
            'overlay_recording': overlay_summary
        }
        
        return combined_summary
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """Process a single frame through the detection pipeline"""
        
        # Run detection using the existing engine
        overlay_frame, events = self.engine.process_frame(frame, cam_id="webcam", ts=time.time())
        
        # Convert events to the format expected by the overlay recorder
        converted_events = []
        if events:
            for event in events:
                # Convert engine event format to overlay format
                event_data = {
                    'event_type': event.get('type', 'unknown'),
                    'person_id': event.get('person_id', 'unknown'),
                    'bbox': event.get('bbox', (0, 0, 100, 100)),
                    'confidence': event.get('confidence', 0.5),
                    'additional_data': {
                        'severity': event.get('severity', 'yellow'),
                        'raw_event': event
                    }
                }
                converted_events.append(event_data)
        
        # Process through recorder if enabled
        if self.enable_recording and self.recorder_integration:
            final_frame = self.recorder_integration.process_detection_results(
                overlay_frame, {'events': converted_events}
            )
        else:
            final_frame = overlay_frame
        
        return final_frame, {'events': converted_events, 'raw_events': events}
    
    def add_debug_overlay(self, frame: np.ndarray, detection_results: dict, 
                         fps: float = 0.0) -> np.ndarray:
        """Add debug information overlay to frame"""
        if not self.show_debug_info:
            return frame
        
        h, w = frame.shape[:2]
        
        # Debug info background
        debug_bg_h = 140
        cv2.rectangle(frame, (10, 10), (320, debug_bg_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (320, debug_bg_h), (255, 255, 255), 2)
        
        # System info
        y_offset = 30
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        y_offset += 20
        cv2.putText(frame, f"Resolution: {w}x{h}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Detection info
        events = detection_results.get('events', [])
        raw_events = detection_results.get('raw_events', [])
        
        y_offset += 20
        cv2.putText(frame, f"Events: {len(events)}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(frame, f"Raw Events: {len(raw_events)}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Count events by severity
        if raw_events:
            red_events = len([e for e in raw_events if e.get('severity') == 'red'])
            yellow_events = len([e for e in raw_events if e.get('severity') == 'yellow'])
            
            y_offset += 20
            cv2.putText(frame, f"Red: {red_events}, Yellow: {yellow_events}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Engine session info
        if self.engine_session_id:
            y_offset += 20
            cv2.putText(frame, f"Session: {self.engine_session_id[-8:]}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Recording status
        if self.enable_recording and self.recorder and self.recorder.is_recording:
            stats = self.recorder.get_session_stats()
            if stats:
                y_offset += 20
                cv2.putText(frame, f"Recording: {stats['duration']:.1f}s", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return frame
    
    def run_realtime_detection(self, display: bool = True):
        """Run real-time detection with display"""
        
        if not self.is_running:
            self.start_session()
        
        frame_count = 0
        start_time = time.time()
        fps = 0.0
        consecutive_failures = 0
        max_failures = 30  # Stop after 30 consecutive frame failures
        
        logger.info("🎯 Starting real-time detection...")
        logger.info("Press 'q' to quit, 's' to save manual event, 'r' to toggle recording")
        
        try:
            while self.is_running:
                # Get frame from camera
                frame = self.camera.get_frame()
                if frame is None:
                    consecutive_failures += 1
                    logger.warning(f"Failed to get frame from camera (attempt {consecutive_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("Too many consecutive frame failures, stopping detection")
                        break
                    
                    time.sleep(0.1)  # Wait a bit before retrying
                    continue
                else:
                    consecutive_failures = 0  # Reset failure counter on successful frame
                
                # Log frame processing for debugging
                if frame_count < 5:  # Log first few frames
                    logger.info(f"Processing frame {frame_count + 1}, shape: {frame.shape}")
                
                # Process frame
                try:
                    processed_frame, detection_results = self.process_frame(frame)
                    
                    # Add debug overlay
                    if display:
                        display_frame = self.add_debug_overlay(processed_frame, detection_results, fps)
                    else:
                        display_frame = processed_frame
                    
                    # Display frame
                    if display:
                        cv2.imshow('Enhanced Cheating Detection with Overlays', display_frame)
                    
                    # Calculate FPS
                    frame_count += 1
                    if frame_count % 30 == 0:
                        current_time = time.time()
                        elapsed = current_time - start_time
                        fps = frame_count / elapsed
                        logger.info(f"📊 FPS: {fps:.1f}, Frames processed: {frame_count}")
                    
                    # Handle keyboard input
                    if display:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logger.info("User pressed 'q' - quitting")
                            break
                        elif key == ord('s') and self.recorder:
                            # Manual event - simulate cheating detection
                            self.recorder.add_manual_event(
                                event_type='manual_cheating',
                                person_id='person_manual',
                                bbox=(100, 100, 100, 200),
                                confidence=1.0,
                                additional_data={'source': 'manual_input'}
                            )
                            logger.info("Manual cheating event added")
                        elif key == ord('r') and self.recorder:
                            # Toggle recording
                            if self.recorder.is_recording:
                                summary = self.recorder.stop_recording()
                                logger.info(f"Recording stopped: {summary}")
                            else:
                                filename = self.recorder.start_recording()
                                logger.info(f"Recording started: {filename}")
                        elif key == ord('d'):
                            # Toggle debug info
                            self.show_debug_info = not self.show_debug_info
                    else:
                        # If no display, still need to check for stopping condition
                        # Add a small delay to prevent busy loop
                        time.sleep(0.033)  # ~30 FPS equivalent
                        
                        # For headless mode, add a way to exit after processing some frames
                        if frame_count > 1000:  # Stop after processing 1000 frames in headless mode
                            logger.info("Processed 1000 frames in headless mode, stopping")
                            break
                
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue processing instead of stopping
                    continue
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in detection loop: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            if display:
                cv2.destroyAllWindows()
            
            session_summary = self.stop_session()
            logger.info(f"Session completed. Summary: {session_summary}")
            
            return session_summary

def main():
    parser = argparse.ArgumentParser(description='Enhanced Cheating Detection with Hotspot Overlays')
    parser.add_argument('--no-recording', action='store_true', 
                       help='Disable video recording')
    parser.add_argument('--no-overlay', action='store_true', 
                       help='Disable hotspot overlays')
    parser.add_argument('--no-visual-overlays', action='store_true', 
                       help='Disable visual overlay markers (keeps database logging)')
    parser.add_argument('--no-display', action='store_true', 
                       help='Run without display (headless)')
    parser.add_argument('--output-dir', default='recordings', 
                       help='Output directory for recordings')
    parser.add_argument('--session-name', 
                       help='Custom session name for recording')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = EnhancedDetectionRunner(
        enable_recording=not args.no_recording,
        enable_overlay=not args.no_overlay,
        output_dir=args.output_dir,
        enable_visual_overlays=not args.no_visual_overlays
    )
    
    # Start session and run detection
    try:
        logger.info(f"Starting session with display: {not args.no_display}")
        
        # Instead of manually starting session, let run_realtime_detection handle it
        # This ensures the session is started at the right time
        summary = runner.run_realtime_detection(display=not args.no_display)
        
        print("\n🎯 SESSION COMPLETE")
        print("="*50)
        if summary:
            for key, value in summary.items():
                print(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
