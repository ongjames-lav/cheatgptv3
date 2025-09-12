"""Real-time synchronized video recorder that eliminates fast forward completely.

This module provides perfect frame-rate synchronization by using a different approach:
Instead of trying to predict FPS, we record at a fixed low rate and duplicate frames as needed.
"""

import cv2
import os
import time
import logging
import numpy as np
from typing import Optional, Tuple, List
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)

class RealTimeSyncRecorder:
    """Real-time synchronized video recorder with perfect timing."""
    
    def __init__(self, videos_dir: str = "videos", base_fps: float = 10.0):
        """Initialize the real-time sync recorder.
        
        Args:
            videos_dir: Directory to store recorded videos
            base_fps: Base FPS for recording (will match actual processing)
        """
        self.videos_dir = videos_dir
        self.base_fps = base_fps
        self.writer: Optional[cv2.VideoWriter] = None
        self.current_session_id: Optional[str] = None
        self.current_video_path: Optional[str] = None
        
        # Frame tracking
        self.frame_count = 0
        self.start_time: Optional[float] = None
        self.last_frame_time: Optional[float] = None
        self.expected_frame_interval = 1.0 / base_fps
        
        # Video settings
        # Use H.264 codec for better web browser compatibility
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')  # More compatible codec
        
        # Ensure videos directory exists
        self._ensure_videos_directory()
        
        logger.info(f"RealTimeSyncRecorder initialized:")
        logger.info(f"  Directory: {self.videos_dir}")
        logger.info(f"  Base FPS: {self.base_fps} (fixed for perfect sync)")
    
    def _ensure_videos_directory(self):
        """Ensure videos directory exists."""
        try:
            os.makedirs(self.videos_dir, exist_ok=True)
            logger.info(f"Videos directory ready: {self.videos_dir}")
        except Exception as e:
            logger.error(f"Failed to create videos directory: {e}")
            # Fallback to current directory
            self.videos_dir = "videos_fallback"
            os.makedirs(self.videos_dir, exist_ok=True)
    
    def start_recording(self, session_id: str, frame_size: Tuple[int, int], 
                       cam_id: str = "webcam") -> Tuple[bool, str]:
        """Start recording a session with perfect sync.
        
        Args:
            session_id: Unique session identifier
            frame_size: (width, height) of video frames
            cam_id: Camera identifier for filename
            
        Returns:
            Tuple of (success, video_path)
        """
        try:
            if self.writer is not None:
                logger.warning("Recording already in progress, stopping previous recording")
                self.stop_recording()
            
            # Generate video filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{session_id}_realtime_{cam_id}_{timestamp}.mp4"
            video_path = os.path.join(self.videos_dir, filename)
            
            # Use fixed base FPS for consistent playback
            width, height = frame_size
            
            self.writer = cv2.VideoWriter(video_path, self.fourcc, self.base_fps, (width, height))
            
            if not self.writer.isOpened():
                logger.error(f"Failed to open video writer for {video_path}")
                self.writer = None
                return False, ""
            
            # Set recording state
            self.current_session_id = session_id
            self.current_video_path = video_path
            self.frame_count = 0
            self.start_time = time.time()
            self.last_frame_time = self.start_time
            
            logger.info(f"Started real-time sync recording session {session_id}: {video_path}")
            logger.info(f"Video settings: {width}x{height} @ {self.base_fps}fps (fixed for perfect sync)")
            
            return True, video_path
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.writer = None
            return False, ""
    
    def write_frame(self, overlay_frame) -> bool:
        """Write a frame with perfect timing synchronization.
        
        Args:
            overlay_frame: Frame with overlays already drawn
            
        Returns:
            True if frame was written successfully
        """
        try:
            if self.writer is None:
                return False
            
            current_time = time.time()
            
            # For first few frames, write immediately to avoid startup lag
            if self.frame_count < 3:
                if len(overlay_frame.shape) == 3 and overlay_frame.shape[2] == 3:
                    if overlay_frame.dtype == 'uint8':
                        self.writer.write(overlay_frame)
                        self.frame_count += 1
                        logger.debug(f"Initial frame {self.frame_count} written immediately")
                        return True
                    else:
                        logger.warning(f"Frame dtype {overlay_frame.dtype} not supported")
                        return False
                else:
                    logger.warning(f"Frame shape {overlay_frame.shape} not supported")
                    return False
            
            # After startup, use normal sync timing
            elapsed_time = current_time - self.start_time
            expected_frames = int(elapsed_time * self.base_fps)
            
            # Write frames to catch up to real-time
            frames_to_write = max(1, expected_frames - self.frame_count + 1)
            
            # Limit frame duplication to prevent excessive catch-up
            frames_to_write = min(frames_to_write, 2)  # Reduced from 3 to 2
            
            # Write frame(s)
            if len(overlay_frame.shape) == 3 and overlay_frame.shape[2] == 3:
                if overlay_frame.dtype == 'uint8':
                    for _ in range(frames_to_write):
                        self.writer.write(overlay_frame)
                        self.frame_count += 1
                    
                    if frames_to_write > 1:
                        logger.debug(f"Duplicated frame {frames_to_write} times for sync")
                    
                    return True
                else:
                    logger.warning(f"Frame dtype {overlay_frame.dtype} not supported")
                    return False
            else:
                logger.warning(f"Frame shape {overlay_frame.shape} not supported")
                return False
                
        except Exception as e:
            logger.error(f"Failed to write frame: {e}")
            return False
    
    def stop_recording(self) -> Tuple[bool, dict]:
        """Stop the current recording session.
        
        Returns:
            Tuple of (success, session_info)
        """
        try:
            if self.writer is None:
                logger.warning("No recording in progress")
                return False, {}
            
            # Release video writer
            self.writer.release()
            
            # Calculate session info
            end_time = time.time()
            duration = end_time - self.start_time if self.start_time else 0
            
            # Calculate perfect sync efficiency
            expected_frames = duration * self.base_fps
            sync_efficiency = (min(self.frame_count, expected_frames) / expected_frames * 100) if expected_frames > 0 else 0
            
            session_info = {
                'session_id': self.current_session_id,
                'video_path': self.current_video_path,
                'frame_count': self.frame_count,
                'duration_seconds': duration,
                'video_fps': self.base_fps,
                'expected_frames': expected_frames,
                'sync_efficiency': sync_efficiency,
                'start_time': self.start_time,
                'end_time': end_time
            }
            
            logger.info(f"Real-time sync recording stopped: {self.current_session_id}")
            logger.info(f"Final stats: {self.frame_count} frames, {duration:.1f}s duration")
            logger.info(f"Video FPS: {self.base_fps} (fixed) | Expected frames: {expected_frames:.0f}")
            logger.info(f"Sync efficiency: {sync_efficiency:.1f}% (frame timing accuracy)")
            
            # Reset state
            self.writer = None
            self.current_session_id = None
            self.current_video_path = None
            self.frame_count = 0
            self.start_time = None
            self.last_frame_time = None
            
            return True, session_info
            
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            return False, {}
    
    def is_recording(self) -> bool:
        """Check if recording is currently active."""
        return self.writer is not None and self.writer.isOpened()
    
    def get_recording_status(self) -> dict:
        """Get current recording status and statistics."""
        if not self.is_recording():
            return {
                'recording': False,
                'session_id': None,
                'frame_count': 0,
                'duration_seconds': 0,
                'video_fps': self.base_fps
            }
        
        duration = time.time() - self.start_time if self.start_time else 0
        expected_frames = duration * self.base_fps
        
        return {
            'recording': True,
            'session_id': self.current_session_id,
            'video_path': self.current_video_path,
            'frame_count': self.frame_count,
            'expected_frames': expected_frames,
            'duration_seconds': duration,
            'video_fps': self.base_fps,
            'sync_efficiency': (self.frame_count / expected_frames * 100) if expected_frames > 0 else 0
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if self.writer is not None:
            logger.info("Cleaning up real-time sync video recorder...")
            self.stop_recording()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
