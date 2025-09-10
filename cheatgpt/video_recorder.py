"""Video recorder for CheatGPT sessions with overlay support.

This module provides video recording functionality that captures frames
with overlays (bounding boxes, labels, etc.) for session evidence.
"""

import cv2
import os
import time
import logging
from typing import Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class VideoRecorder:
    """Video recorder for capturing sessions with overlays."""
    
    def __init__(self, videos_dir: str = "videos"):
        """Initialize the video recorder.
        
        Args:
            videos_dir: Directory to store recorded videos
        """
        self.videos_dir = videos_dir
        self.writer: Optional[cv2.VideoWriter] = None
        self.current_session_id: Optional[str] = None
        self.current_video_path: Optional[str] = None
        self.frame_count = 0
        self.start_time: Optional[float] = None
        self.frame_times = []  # Track frame timestamps for dynamic FPS
        
        # Video settings - Use realistic FPS to prevent fast forward effect
        self.fps = 15.0  # More realistic for webcam processing with overlays
        # Use H.264 codec for better web browser compatibility
        self.fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        
        # Ensure videos directory exists
        self._ensure_videos_directory()
        
        logger.info(f"VideoRecorder initialized with directory: {self.videos_dir}")
    
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
        """Start recording a session.
        
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
            filename = f"session_{session_id}_{cam_id}_{timestamp}.mp4"
            video_path = os.path.join(self.videos_dir, filename)
            
            # Initialize video writer
            width, height = frame_size
            self.writer = cv2.VideoWriter(video_path, self.fourcc, self.fps, (width, height))
            
            if not self.writer.isOpened():
                logger.error(f"Failed to open video writer for {video_path}")
                self.writer = None
                return False, ""
            
            # Set recording state
            self.current_session_id = session_id
            self.current_video_path = video_path
            self.frame_count = 0
            self.start_time = time.time()
            self.frame_times = []  # Reset frame times tracking
            
            logger.info(f"Started recording session {session_id}: {video_path}")
            logger.info(f"Video settings: {width}x{height} @ {self.fps}fps (realistic rate to prevent fast forward)")
            
            return True, video_path
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.writer = None
            return False, ""
    
    def write_frame(self, overlay_frame) -> bool:
        """Write a frame with overlays to the video.
        
        Args:
            overlay_frame: Frame with overlays already drawn
            
        Returns:
            True if frame was written successfully
        """
        try:
            if self.writer is None:
                return False
            
            # Track frame timing for FPS calculation
            current_time = time.time()
            self.frame_times.append(current_time)
            
            # Ensure frame is in correct format
            if len(overlay_frame.shape) == 3 and overlay_frame.shape[2] == 3:
                # Convert BGR to RGB if needed (OpenCV uses BGR)
                if overlay_frame.dtype == 'uint8':
                    self.writer.write(overlay_frame)
                    self.frame_count += 1
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
            
            # Calculate actual FPS from frame times
            actual_fps = self.frame_count / duration if duration > 0 else 0
            
            session_info = {
                'session_id': self.current_session_id,
                'video_path': self.current_video_path,
                'frame_count': self.frame_count,
                'duration_seconds': duration,
                'fps': self.fps,
                'actual_fps': actual_fps,
                'start_time': self.start_time,
                'end_time': end_time
            }
            
            logger.info(f"Recording stopped: {self.current_session_id}")
            logger.info(f"Final stats: {self.frame_count} frames, {duration:.1f}s duration")
            logger.info(f"Video FPS: {self.fps} (set) vs {actual_fps:.1f} (actual processing rate)")
            
            # Reset state
            self.writer = None
            self.current_session_id = None
            self.current_video_path = None
            self.frame_count = 0
            self.start_time = None
            self.frame_times = []
            
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
                'duration_seconds': 0
            }
        
        duration = time.time() - self.start_time if self.start_time else 0
        
        return {
            'recording': True,
            'session_id': self.current_session_id,
            'video_path': self.current_video_path,
            'frame_count': self.frame_count,
            'duration_seconds': duration,
            'fps': self.fps
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if self.writer is not None:
            logger.info("Cleaning up video recorder...")
            self.stop_recording()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
