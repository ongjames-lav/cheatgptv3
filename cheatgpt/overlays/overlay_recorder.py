#!/usr/bin/env python3
"""
Enhanced Video Recorder with Hotspot Overlay Integration
Records video with real-time suspicious event markers and timeline export
"""

import cv2
import numpy as np
import time
import threading
from pathlib import Path
from typing import Optional, Dict, List
import logging
from datetime import datetime
import json

from .hotspot_overlay import HotspotOverlay, EngineOverlayIntegration

logger = logging.getLogger(__name__)

class OverlayVideoRecorder:
    """Video recorder with integrated hotspot overlay system"""
    
    def __init__(self, output_dir: str = "recordings", 
                 enable_overlay: bool = True,
                 db_path: str = "data/events.db",
                 enable_visual_overlays: bool = True):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_overlay = enable_overlay
        self.enable_visual_overlays = enable_visual_overlays
        self.is_recording = False
        self.writer = None
        self.current_session = None
        
        # Overlay system
        if enable_overlay:
            self.hotspot_overlay = HotspotOverlay(db_path, enable_visual_overlays)
            self.overlay_integration = EngineOverlayIntegration(self.hotspot_overlay)
        else:
            self.hotspot_overlay = None
            self.overlay_integration = None
        
        # Recording settings
        self.fps = 30
        # Use H.264 codec for better web browser compatibility
        self.fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        self.frame_size = None
        
        # Session tracking
        self.session_start_time = None
        self.frame_count = 0
        
        logger.info(f"Overlay video recorder initialized (overlay={'ON' if enable_overlay else 'OFF'})")
    
    def start_recording(self, output_filename: Optional[str] = None) -> str:
        """Start recording with optional custom filename"""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return self.current_session['filename']
        
        # Generate filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"session_{timestamp}.mp4"
        
        output_path = self.output_dir / output_filename
        
        # Initialize session
        self.current_session = {
            'filename': str(output_path),
            'start_time': time.time(),
            'frame_count': 0,
            'events': []
        }
        
        self.session_start_time = self.current_session['start_time']
        self.frame_count = 0
        self.is_recording = True
        
        logger.info(f"🎥 Started recording: {output_filename}")
        return str(output_path)
    
    def stop_recording(self) -> Optional[Dict]:
        """Stop recording and return session summary"""
        if not self.is_recording:
            logger.warning("No recording in progress")
            return None
        
        if self.writer:
            self.writer.release()
            self.writer = None
        
        # Session summary
        end_time = time.time()
        duration = end_time - self.session_start_time
        
        session_summary = {
            'filename': self.current_session['filename'],
            'duration': duration,
            'frame_count': self.frame_count,
            'fps': self.frame_count / duration if duration > 0 else 0,
            'start_time': self.session_start_time,
            'end_time': end_time
        }
        
        # Export event timeline if overlay is enabled
        if self.enable_overlay and self.hotspot_overlay:
            timeline_path = self.output_dir / f"timeline_{Path(self.current_session['filename']).stem}.json"
            self.hotspot_overlay.export_event_timeline(
                str(timeline_path), 
                self.session_start_time, 
                end_time
            )
            session_summary['timeline_file'] = str(timeline_path)
        
        self.is_recording = False
        self.current_session = None
        
        logger.info(f"🎬 Recording stopped. Duration: {duration:.1f}s, Frames: {self.frame_count}")
        return session_summary
    
    def record_frame(self, frame: np.ndarray, events: Optional[List[Dict]] = None) -> np.ndarray:
        """Record a frame with optional event overlay"""
        if not self.is_recording:
            return frame
        
        # Initialize video writer if needed
        if self.writer is None:
            h, w = frame.shape[:2]
            self.frame_size = (w, h)
            self.writer = cv2.VideoWriter(
                self.current_session['filename'],
                self.fourcc,
                self.fps,
                self.frame_size
            )
            logger.info(f"Video writer initialized: {w}x{h} @ {self.fps}fps")
        
        # Calculate current timestamp relative to session start
        current_time = time.time()
        relative_timestamp = current_time - self.session_start_time
        
        # Process frame with overlay if enabled
        processed_frame = frame.copy()
        
        if self.enable_overlay and self.overlay_integration and events:
            processed_frame = self.overlay_integration.process_engine_events(
                events, relative_timestamp, processed_frame
            )
        elif self.enable_overlay and self.hotspot_overlay:
            # Just apply existing overlays without new events
            processed_frame = self.hotspot_overlay.process_frame(
                processed_frame, relative_timestamp
            )
        
        # Add recording indicator
        self._add_recording_indicator(processed_frame, relative_timestamp)
        
        # Write frame
        self.writer.write(processed_frame)
        self.frame_count += 1
        
        return processed_frame
    
    def _add_recording_indicator(self, frame: np.ndarray, timestamp: float):
        """Add recording indicator to frame"""
        h, w = frame.shape[:2]
        
        # Red recording dot (pulsing)
        pulse = int(abs(np.sin(timestamp * 4)) * 50) + 50  # Pulse between 50-100
        recording_color = (0, 0, pulse + 155)  # Red with pulsing intensity
        
        # Recording dot
        cv2.circle(frame, (w - 40, 30), 8, recording_color, -1)
        cv2.circle(frame, (w - 40, 30), 8, (255, 255, 255), 2)
        
        # Recording text
        rec_text = "REC"
        cv2.putText(frame, rec_text, (w - 80, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Timestamp
        time_text = f"{timestamp:.1f}s"
        cv2.putText(frame, time_text, (w - 120, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def add_manual_event(self, event_type: str, person_id: str, 
                        bbox: tuple, confidence: float = 1.0, 
                        additional_data: Optional[Dict] = None):
        """Manually add an event to the overlay system"""
        if not self.enable_overlay or not self.hotspot_overlay:
            logger.warning("Overlay system not enabled")
            return
        
        current_time = time.time()
        relative_timestamp = current_time - self.session_start_time if self.session_start_time else 0
        
        self.hotspot_overlay.add_event(
            event_type=event_type,
            person_id=person_id,
            bbox=bbox,
            confidence=confidence,
            additional_data=additional_data
        )
        
        logger.info(f"Manual event added: {event_type} for {person_id}")
    
    def get_session_stats(self) -> Optional[Dict]:
        """Get current session statistics"""
        if not self.is_recording:
            return None
        
        current_time = time.time()
        duration = current_time - self.session_start_time
        
        stats = {
            'is_recording': True,
            'duration': duration,
            'frame_count': self.frame_count,
            'current_fps': self.frame_count / duration if duration > 0 else 0,
            'filename': self.current_session['filename'],
            'overlay_enabled': self.enable_overlay
        }
        
        # Add event stats if overlay is enabled
        if self.enable_overlay and self.hotspot_overlay:
            recent_events = self.hotspot_overlay.event_db.get_event_timeline(limit=50)
            session_events = [e for e in recent_events 
                            if e['timestamp'] >= (current_time - self.session_start_time)]
            
            stats['total_events'] = len(session_events)
            stats['event_types'] = list(set(e['event_type'] for e in session_events))
        
        return stats

# Integration with main detection system
class DetectionRecorderIntegration:
    """Integration class for main detection system"""
    
    def __init__(self, recorder: OverlayVideoRecorder):
        self.recorder = recorder
        self.last_frame_time = 0
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps
    
    def should_record_frame(self) -> bool:
        """Check if we should record this frame (FPS limiting)"""
        current_time = time.time()
        if current_time - self.last_frame_time >= self.frame_interval:
            self.last_frame_time = current_time
            return True
        return False
    
    def process_detection_results(self, frame: np.ndarray, 
                                 detection_results: Dict) -> np.ndarray:
        """Process detection results and record frame with overlays"""
        
        if not self.should_record_frame():
            return frame
        
        # Extract events from detection results
        events = []
        
        # Handle different result formats
        if 'events' in detection_results:
            # Already in events format
            events = detection_results['events']
        elif 'persons' in detection_results:
            # Convert from persons format
            for person_id, person_data in detection_results['persons'].items():
                bbox = person_data.get('bbox', (0, 0, 100, 100))
                
                # Check for various event types
                if person_data.get('looking_flag', False):
                    events.append({
                        'event_type': 'suspicious_looking',
                        'person_id': person_id,
                        'bbox': bbox,
                        'confidence': person_data.get('looking_confidence', 0.5),
                        'additional_data': {
                            'head_yaw': person_data.get('head_yaw', 0),
                            'head_pitch': person_data.get('head_pitch', 0)
                        }
                    })
                
                if person_data.get('lean_flag', False):
                    events.append({
                        'event_type': 'suspicious_lean',
                        'person_id': person_id,
                        'bbox': bbox,
                        'confidence': person_data.get('lean_confidence', 0.5),
                        'additional_data': {
                            'lean_angle': person_data.get('lean_angle', 0)
                        }
                    })
                
                if person_data.get('gesture_flag', False):
                    events.append({
                        'event_type': 'suspicious_gesture',
                        'person_id': person_id,
                        'bbox': bbox,
                        'confidence': person_data.get('gesture_confidence', 0.5)
                    })
                
                if person_data.get('phone_flag', False):
                    events.append({
                        'event_type': 'phone_detected',
                        'person_id': person_id,
                        'bbox': bbox,
                        'confidence': person_data.get('phone_confidence', 0.5)
                    })
                
                # Temporal cheating events
                if person_data.get('temporal_cheating', False):
                    events.append({
                        'event_type': 'temporal_cheating',
                        'person_id': person_id,
                        'bbox': bbox,
                        'confidence': 0.9,
                        'additional_data': {
                            'behaviors': person_data.get('active_behaviors', [])
                        }
                    })
        
        # Record frame with events
        return self.recorder.record_frame(frame, events)

if __name__ == "__main__":
    # Example usage
    recorder = OverlayVideoRecorder(enable_overlay=True)
    
    # Start recording
    output_file = recorder.start_recording("test_overlay_recording.mp4")
    print(f"Recording to: {output_file}")
    
    # Simulate some frames and events
    for i in range(100):
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        # Add some text
        cv2.putText(frame, f"Frame {i}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Simulate events
        events = []
        if i % 30 == 0:  # Every 30 frames
            events.append({
                'event_type': 'suspicious_looking',
                'person_id': 'person_001',
                'bbox': (100, 100, 100, 200),
                'confidence': 0.8
            })
        
        # Record frame
        processed_frame = recorder.record_frame(frame, events)
        
        # Small delay to simulate real-time
        time.sleep(0.033)  # ~30 FPS
    
    # Stop recording
    summary = recorder.stop_recording()
    print(f"Recording completed: {summary}")
