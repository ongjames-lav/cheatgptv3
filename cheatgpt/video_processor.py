"""
CheatGPT3 Video Processor
Handles video processing through the CheatGPT3 detection engine
"""

import os
import cv2
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from .engines.engine_hybrid import EngineHybrid
from .report_generator import ReportGenerator


class VideoProcessor:
    """Process videos using CheatGPT3 detection engine"""
    
    def __init__(self):
        self.engine = None
        self.report_generator = ReportGenerator()
        
    def process_video(self, input_path: str, session_id: str, 
                     output_dir: str = "results", progress_callback=None) -> Dict[str, Any]:
        """
        Process a video file through CheatGPT3 detection engine
        
        Args:
            input_path: Path to input video file
            session_id: Unique session identifier
            output_dir: Output directory for results
            
        Returns:
            Dict containing processing results and file paths
        """
        try:
            print(f"🎥 Starting video processing for session {session_id}")
            
            # Initialize engine
            if self.engine is None:
                self.engine = EngineHybrid()
            
            # Create output directory
            session_output_dir = os.path.join(output_dir, session_id)
            os.makedirs(session_output_dir, exist_ok=True)
            
            # Define output paths
            output_video_path = os.path.join(session_output_dir, f"processed_{session_id}.mp4")
            
            # Get video metadata
            video_metadata = self._get_video_metadata(input_path)
            
            # Process video
            all_events = self._process_video_file(
                input_path, 
                output_video_path, 
                session_id,
                progress_callback
            )
            
            # Generate event summary
            summary = self._generate_event_summary(all_events)
            
            # Generate hotspots from events
            hotspots = self._generate_hotspots_from_events(all_events, video_metadata)
            
            # Generate comprehensive reports
            report_paths = self.report_generator.generate_comprehensive_report(
                session_id=session_id,
                video_metadata=video_metadata,
                events=all_events,
                summary=summary
            )
            
            # Create visualization paths
            visualization_paths = self.report_generator.generate_visualizations(
                session_id=session_id,
                video_metadata=video_metadata,
                events=all_events,
                summary=summary
            )
            
            result = {
                'session_id': session_id,
                'status': 'completed',
                'video_metadata': video_metadata,
                'total_events': len(all_events),
                'events': all_events,  # Include raw events for database storage
                'hotspots': hotspots,  # Include hotspots for database storage
                'event_summary': summary,
                'output_paths': {
                    'processed_video': output_video_path,
                    'json_report': report_paths.get('json_report'),
                    'csv_report': report_paths.get('csv_report'),
                    'executive_summary': report_paths.get('executive_summary'),
                    'visualizations': visualization_paths
                }
            }
            
            print(f"✅ Video processing completed for session {session_id}")
            print(f"   - Total events detected: {len(all_events)}")
            print(f"   - Event summary: {summary}")
            
            return result
            
        except Exception as e:
            print(f"❌ Video processing failed for session {session_id}: {e}")
            raise e
    
    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata"""
        cap = cv2.VideoCapture(video_path)
        
        metadata = {
            'filename': os.path.basename(video_path),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return metadata
    
    def _process_video_file(self, input_path: str, output_path: str, 
                           session_id: str, progress_callback=None) -> List[Dict[str, Any]]:
        """Process video file frame by frame"""
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video properties: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Start engine session
        self.engine.start_session(session_id)
        
        all_events = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame through engine
                timestamp = frame_count / fps
                overlay_frame, events = self.engine.process_frame(
                    frame, 
                    cam_id="video_processing", 
                    ts=timestamp
                )
                
                # Collect events
                all_events.extend(events)
                
                # Write processed frame
                writer.write(overlay_frame)
                
                frame_count += 1
                
                # Progress logging and callback
                if frame_count % 50 == 0:  # Update every 50 frames for smoother progress
                    progress = (frame_count / total_frames) * 100
                    progress_message = f"Processing frame {frame_count}/{total_frames}"
                    print(f"📊 Processing progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                    
                    # Call progress callback if provided
                    if progress_callback:
                        # Progress ranges from 10% to 90% during video processing
                        adjusted_progress = 10 + (progress * 0.8)  # 10% + (0-100% * 80%)
                        progress_callback(adjusted_progress, progress_message)
        
        finally:
            # Cleanup
            cap.release()
            writer.release()
            self.engine.stop_session()
        
        print(f"🎬 Video processing completed: {len(all_events)} events detected")
        return all_events
    
    def _generate_event_summary(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Generate summary of events by type"""
        summary = {}
        
        for event in events:
            event_type = event.get('event_type', 'Unknown')
            summary[event_type] = summary.get(event_type, 0) + 1
        
        return summary

    def _generate_hotspots_from_events(self, events: List[Dict[str, Any]], 
                                     video_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hotspots from detected events"""
        hotspots = []
        
        # Group events by spatial location to create hotspots
        spatial_groups = {}
        
        for event in events:
            bbox = event.get('bbox', [])
            if len(bbox) >= 4:
                # Normalize bbox to create spatial key (grid approach)
                x1, y1, x2, y2 = bbox[:4]
                width = video_metadata.get('width', 1920)
                height = video_metadata.get('height', 1080)
                
                # Create grid cells (20x20 grid)
                grid_x = int((x1 + x2) / 2 / width * 20)
                grid_y = int((y1 + y2) / 2 / height * 20)
                spatial_key = f"{grid_x}_{grid_y}"
                
                if spatial_key not in spatial_groups:
                    spatial_groups[spatial_key] = {
                        'events': [],
                        'total_confidence': 0,
                        'bbox_sum': [0, 0, 0, 0],
                        'first_time': float('inf'),
                        'last_time': 0
                    }
                
                group = spatial_groups[spatial_key]
                group['events'].append(event)
                group['total_confidence'] += event.get('confidence', 0.0)
                group['first_time'] = min(group['first_time'], event.get('timestamp', 0))
                group['last_time'] = max(group['last_time'], event.get('timestamp', 0))
                
                # Accumulate bbox coordinates for averaging
                for i in range(4):
                    group['bbox_sum'][i] += bbox[i]
        
        # Create hotspots from spatial groups
        for spatial_key, group in spatial_groups.items():
            event_count = len(group['events'])
            if event_count >= 2:  # Only create hotspots for areas with multiple events
                # Average bbox coordinates
                avg_bbox = [coord / event_count for coord in group['bbox_sum']]
                x1, y1, x2, y2 = avg_bbox
                
                # Determine severity based on event count and confidence
                avg_confidence = group['total_confidence'] / event_count
                if event_count >= 10 or avg_confidence >= 0.9:
                    severity = 'high'
                elif event_count >= 5 or avg_confidence >= 0.7:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                hotspot = {
                    'x': x1,
                    'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'event_count': event_count,
                    'severity_level': severity,
                    'first_detection_time': group['first_time'],
                    'last_detection_time': group['last_time']
                }
                hotspots.append(hotspot)
        
        return hotspots
