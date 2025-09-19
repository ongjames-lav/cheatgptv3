"""Adaptive video recorder that adjusts FPS based on actual processing rate.

This module provides an adaptive video recording solution that prevents
fast forward playback by matching video FPS to actual frame processing rate.
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

class AdaptiveVideoRecorder:
    """Adaptive video recorder that matches FPS to processing rate."""
    
    def __init__(self, videos_dir: str = "videos", target_fps: float = 15.0):
        """Initialize the adaptive video recorder.
        
        Args:
            videos_dir: Directory to store recorded videos
            target_fps: Target FPS for video playback
        """
        self.videos_dir = videos_dir
        self.target_fps = target_fps
        self.writer: Optional[cv2.VideoWriter] = None
        self.current_session_id: Optional[str] = None
        self.current_video_path: Optional[str] = None
        
        # Frame tracking
        self.frame_count = 0
        self.start_time: Optional[float] = None
        self.frame_times: deque = deque(maxlen=30)  # Last 30 frame times for rolling average
        
        # Adaptive FPS calculation
        self.min_fps = 5.0   # Minimum FPS for very slow processing
        self.max_fps = 30.0  # Maximum FPS for very fast processing
        self.current_fps = target_fps
        
        # Video settings - prioritize H.264 AVC1 for best web compatibility
        self.fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 AVC1 codec for web compatibility
        
        # Ensure videos directory exists
        self._ensure_videos_directory()
        
        logger.info(f"AdaptiveVideoRecorder initialized:")
        logger.info(f"  Directory: {self.videos_dir}")
        logger.info(f"  Target FPS: {self.target_fps}")
        logger.info(f"  FPS Range: {self.min_fps}-{self.max_fps}")
    
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
    
    def _calculate_adaptive_fps(self) -> float:
        """Calculate FPS based on actual frame processing rate."""
        if len(self.frame_times) < 3:
            return self.target_fps
        
        # Calculate average time between frames using recent frames
        times = list(self.frame_times)
        intervals = [times[i] - times[i-1] for i in range(1, len(times))]
        
        # Use weighted average - give more weight to recent intervals
        if len(intervals) > 5:
            recent_intervals = intervals[-5:]  # Last 5 intervals
            avg_interval = np.mean(recent_intervals)
        else:
            avg_interval = np.mean(intervals)
        
        if avg_interval <= 0:
            return self.target_fps
        
        # Calculate actual processing FPS
        actual_fps = 1.0 / avg_interval
        
        # Be more conservative - use 90% of actual FPS to ensure no fast forward
        conservative_fps = actual_fps * 0.9
        
        # Clamp to reasonable range
        adaptive_fps = max(self.min_fps, min(self.max_fps, conservative_fps))
        
        # More conservative smooth transition
        if hasattr(self, 'current_fps'):
            max_change = 1.5  # Smaller max FPS change per calculation
            if abs(adaptive_fps - self.current_fps) > max_change:
                if adaptive_fps > self.current_fps:
                    adaptive_fps = self.current_fps + max_change
                else:
                    adaptive_fps = self.current_fps - max_change
        
        return round(adaptive_fps, 1)
    
    def start_recording(self, session_id: str, frame_size: Tuple[int, int], 
                       cam_id: str = "webcam") -> Tuple[bool, str]:
        """Start recording a session with adaptive FPS.
        
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
            filename = f"session_{session_id}_adaptive_{cam_id}_{timestamp}.mp4"
            video_path = os.path.join(self.videos_dir, filename)
            
            # Start with target FPS, will adapt as frames are processed
            width, height = frame_size
            self.current_fps = self.target_fps
            
            self.writer = cv2.VideoWriter(video_path, self.fourcc, self.current_fps, (width, height))
            
            if not self.writer.isOpened():
                logger.error(f"Failed to open video writer for {video_path}")
                self.writer = None
                return False, ""
            
            # Set recording state
            self.current_session_id = session_id
            self.current_video_path = video_path
            self.frame_count = 0
            self.start_time = time.time()
            self.frame_times.clear()
            
            logger.info(f"Started adaptive recording session {session_id}: {video_path}")
            logger.info(f"Video settings: {width}x{height} @ {self.current_fps}fps (adaptive)")
            
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
            
            # Track frame timing for adaptive FPS
            current_time = time.time()
            self.frame_times.append(current_time)
            
            # Calculate adaptive FPS every 10 frames after we have enough data
            if self.frame_count > 0 and self.frame_count % 10 == 0 and len(self.frame_times) >= 5:
                new_fps = self._calculate_adaptive_fps()
                
                # If FPS changes significantly, we need to restart the writer
                if abs(new_fps - self.current_fps) > 2.0:
                    logger.info(f"Significant FPS change detected: {self.current_fps} → {new_fps}")
                    logger.info("Restarting video writer with new FPS...")
                    
                    # Close current writer
                    self.writer.release()
                    
                    # Create new writer with updated FPS
                    height, width = overlay_frame.shape[:2]
                    self.writer = cv2.VideoWriter(
                        self.current_video_path, 
                        self.fourcc, 
                        new_fps, 
                        (width, height)
                    )
                    
                    if not self.writer.isOpened():
                        logger.error("Failed to restart video writer")
                        return False
                    
                    self.current_fps = new_fps
                    logger.info(f"Video writer restarted with FPS: {new_fps}")
                
                elif abs(new_fps - self.current_fps) > 0.5:  # Log smaller changes
                    logger.info(f"Adaptive FPS: {self.current_fps} → {new_fps} fps")
                    self.current_fps = new_fps
            
            # Write frame
            if len(overlay_frame.shape) == 3 and overlay_frame.shape[2] == 3:
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
            actual_processing_fps = self.frame_count / duration if duration > 0 else 0
            
            session_info = {
                'session_id': self.current_session_id,
                'video_path': self.current_video_path,
                'frame_count': self.frame_count,
                'duration_seconds': duration,
                'video_fps': self.current_fps,
                'actual_processing_fps': actual_processing_fps,
                'fps_efficiency': (actual_processing_fps / self.current_fps) * 100 if self.current_fps > 0 else 0,
                'start_time': self.start_time,
                'end_time': end_time
            }
            
            logger.info(f"Adaptive recording stopped: {self.current_session_id}")
            logger.info(f"Final stats: {self.frame_count} frames, {duration:.1f}s duration")
            logger.info(f"Video FPS: {self.current_fps} | Processing rate: {actual_processing_fps:.1f} fps")
            logger.info(f"Efficiency: {session_info['fps_efficiency']:.1f}% (closer to 100% = better sync)")
            
            # Reset state
            self.writer = None
            self.current_session_id = None
            self.current_video_path = None
            self.frame_count = 0
            self.start_time = None
            self.frame_times.clear()
            
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
                'current_fps': self.target_fps
            }
        
        duration = time.time() - self.start_time if self.start_time else 0
        actual_fps = self.frame_count / duration if duration > 0 else 0
        
        return {
            'recording': True,
            'session_id': self.current_session_id,
            'video_path': self.current_video_path,
            'frame_count': self.frame_count,
            'duration_seconds': duration,
            'video_fps': self.current_fps,
            'actual_processing_fps': actual_fps,
            'target_fps': self.target_fps
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if self.writer is not None:
            logger.info("Cleaning up adaptive video recorder...")
            self.stop_recording()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
