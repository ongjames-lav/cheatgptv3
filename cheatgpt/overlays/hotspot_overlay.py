#!/usr/bin/env python3
"""
Hotspot Overlay System for Video Events
Adds visual markers (⚠️, red dots) on video frames when suspicious events are detected
"""

import cv2
import numpy as np
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class EventDatabase:
    """Database manager for storing event timeline data"""
    
    def __init__(self, db_path: str = "data/events.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the events database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                person_id TEXT NOT NULL,
                confidence REAL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_w INTEGER,
                bbox_h INTEGER,
                additional_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_person_id ON events(person_id)')
        
        conn.commit()
        conn.close()
        logger.info(f"Event database initialized: {self.db_path}")
    
    def store_event(self, timestamp: float, event_type: str, person_id: str, 
                   confidence: float = 0.0, bbox: Optional[Tuple[int, int, int, int]] = None,
                   additional_data: Optional[Dict] = None):
        """Store an event in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        bbox_x = bbox_y = bbox_w = bbox_h = None
        if bbox:
            bbox_x, bbox_y, bbox_w, bbox_h = bbox
        
        additional_json = json.dumps(additional_data) if additional_data else None
        
        cursor.execute('''
            INSERT INTO events (timestamp, event_type, person_id, confidence, 
                              bbox_x, bbox_y, bbox_w, bbox_h, additional_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, event_type, person_id, confidence, 
              bbox_x, bbox_y, bbox_w, bbox_h, additional_json))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Stored event: {event_type} for {person_id} at {timestamp:.2f}s")
    
    def get_events_in_timerange(self, start_time: float, end_time: float) -> List[Dict]:
        """Get all events within a time range"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM events 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        ''', (start_time, end_time))
        
        columns = [desc[0] for desc in cursor.description]
        events = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return events
    
    def get_event_timeline(self, limit: int = 100) -> List[Dict]:
        """Get recent events for timeline display"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM events 
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        events = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return events

class HotspotOverlay:
    """Video overlay system for marking suspicious events"""
    
    def __init__(self, db_path: str = "data/events.db", enable_visual_overlays: bool = True):
        self.event_db = EventDatabase(db_path)
        self.active_events = {}  # person_id -> event_info
        self.overlay_duration = 3.0  # seconds to show overlay
        self.current_timestamp = 0.0
        self.enable_visual_overlays = enable_visual_overlays  # Control visual overlay rendering
        
        # Visual settings
        self.colors = {
            'suspicious_looking': (0, 165, 255),    # Orange
            'suspicious_lean': (0, 255, 255),       # Yellow
            'suspicious_gesture': (255, 0, 255),    # Magenta
            'phone_detected': (0, 0, 255),          # Red
            'cheating': (0, 0, 139),                # Dark Red
            'temporal_cheating': (0, 0, 255)        # Bright Red
        }
        
        self.warning_emoji = "⚠️"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
    
    def update_timestamp(self, timestamp: float):
        """Update current timestamp and clean expired overlays"""
        self.current_timestamp = timestamp
        
        # Remove expired overlays
        expired_persons = []
        for person_id, event_info in self.active_events.items():
            if timestamp - event_info['start_time'] > self.overlay_duration:
                expired_persons.append(person_id)
        
        for person_id in expired_persons:
            del self.active_events[person_id]
    
    def add_event(self, event_type: str, person_id: str, bbox: Tuple[int, int, int, int],
                  confidence: float = 0.0, additional_data: Optional[Dict] = None):
        """Add a new suspicious event"""
        
        # Store in database
        self.event_db.store_event(
            timestamp=self.current_timestamp,
            event_type=event_type,
            person_id=person_id,
            confidence=confidence,
            bbox=bbox,
            additional_data=additional_data
        )
        
        # Add to active overlays
        self.active_events[person_id] = {
            'event_type': event_type,
            'bbox': bbox,
            'confidence': confidence,
            'start_time': self.current_timestamp,
            'additional_data': additional_data or {}
        }
        
        logger.info(f"🚨 Added hotspot overlay: {event_type} for {person_id}")
    
    def draw_warning_marker(self, frame: np.ndarray, x: int, y: int, 
                           color: Tuple[int, int, int], size: int = 20):
        """Draw a warning marker (red circle with exclamation)"""
        # Ensure coordinates are integers
        x, y = int(x), int(y)
        size = int(size)
        
        # Draw filled circle
        cv2.circle(frame, (x, y), size, color, -1)
        
        # Draw white border
        cv2.circle(frame, (x, y), size, (255, 255, 255), 2)
        
        # Draw exclamation mark
        # Vertical line
        cv2.line(frame, (x, y-12), (x, y+2), (255, 255, 255), 3)
        # Dot
        cv2.circle(frame, (x, y+8), 2, (255, 255, 255), -1)
    
    def draw_event_overlay(self, frame: np.ndarray, person_id: str, event_info: Dict):
        """Draw overlay for a specific event"""
        # If visual overlays are disabled, skip drawing but keep logging
        if not self.enable_visual_overlays:
            return
            
        bbox = event_info['bbox']
        event_type = event_info['event_type']
        confidence = event_info['confidence']
        
        # Convert all bbox coordinates to integers
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Calculate center coordinates  
        center_x = int(x + w // 2)
        center_y = int(y + h // 2)
        
        # Get color for event type
        color = self.colors.get(event_type, (0, 255, 0))  # Default green
        
        # Draw warning marker above the person
        marker_y = max(int(y - 30), 30)  # Position above bbox
        self.draw_warning_marker(frame, center_x, marker_y, color, size=15)
        
        # Draw event label
        label = event_type.replace('_', ' ').title()
        if confidence > 0:
            label += f" ({confidence:.0%})"
        
        # Calculate text size and position
        (text_w, text_h), baseline = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
        
        # Position text above the marker (ensure integers)
        text_x = max(int(center_x - text_w // 2), 5)
        text_y = max(int(marker_y - 25), text_h + 5)
        
        # Draw text background
        cv2.rectangle(frame, 
                     (text_x - 5, text_y - text_h - 5),
                     (text_x + text_w + 5, text_y + baseline),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, label, (text_x, text_y), self.font, 
                   self.font_scale, (255, 255, 255), self.thickness)
        
        # Draw connecting line from marker to person
        cv2.line(frame, (center_x, marker_y + 15), (center_x, y), color, 2)
        
        # Add pulsing effect for high-priority events
        if event_type in ['cheating', 'temporal_cheating']:
            pulse_intensity = int(abs(np.sin(self.current_timestamp * 8)) * 100)
            pulse_color = (pulse_intensity, pulse_intensity, 255)
            cv2.rectangle(frame, (x-5, y-5), (x+w+5, y+h+5), pulse_color, 3)
    
    def draw_timeline_overlay(self, frame: np.ndarray, events: List[Dict]):
        """Draw a mini-timeline of recent events on the frame"""
        if not events:
            return
        
        frame_h, frame_w = frame.shape[:2]
        timeline_x = 10
        timeline_y = frame_h - 100
        timeline_w = 300
        timeline_h = 80
        
        # Draw timeline background
        cv2.rectangle(frame, 
                     (timeline_x, timeline_y), 
                     (timeline_x + timeline_w, timeline_y + timeline_h),
                     (0, 0, 0), -1)
        
        cv2.rectangle(frame, 
                     (timeline_x, timeline_y), 
                     (timeline_x + timeline_w, timeline_y + timeline_h),
                     (255, 255, 255), 2)
        
        # Timeline title
        cv2.putText(frame, "Recent Events", 
                   (timeline_x + 10, timeline_y + 20),
                   self.font, 0.5, (255, 255, 255), 1)
        
        # Draw recent events
        max_events = 5
        recent_events = events[:max_events]
        
        for i, event in enumerate(recent_events):
            y_pos = timeline_y + 35 + i * 12
            event_type = event['event_type']
            person_id = event['person_id']
            
            # Event color dot
            color = self.colors.get(event_type, (0, 255, 0))
            cv2.circle(frame, (timeline_x + 15, y_pos), 3, color, -1)
            
            # Event text
            elapsed = self.current_timestamp - event['timestamp']
            text = f"{person_id}: {event_type.replace('_', ' ')} ({elapsed:.1f}s ago)"
            cv2.putText(frame, text, 
                       (timeline_x + 25, y_pos + 3),
                       self.font, 0.35, (255, 255, 255), 1)
    
    def process_frame(self, frame: np.ndarray, timestamp: float, 
                     show_timeline: bool = True) -> np.ndarray:
        """Process frame and add all overlays"""
        self.update_timestamp(timestamp)
        
        # Draw active event overlays
        for person_id, event_info in self.active_events.items():
            self.draw_event_overlay(frame, person_id, event_info)
        
        # Draw timeline if requested
        if show_timeline:
            recent_events = self.event_db.get_event_timeline(limit=10)
            self.draw_timeline_overlay(frame, recent_events)
        
        return frame
    
    def export_event_timeline(self, output_path: str, start_time: float = 0, 
                             end_time: float = None):
        """Export event timeline to JSON file"""
        if end_time is None:
            end_time = self.current_timestamp
        
        events = self.event_db.get_events_in_timerange(start_time, end_time)
        
        # Format for export
        timeline_data = {
            'session_info': {
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'total_events': len(events)
            },
            'events': []
        }
        
        for event in events:
            formatted_event = {
                'timestamp': event['timestamp'],
                'relative_time': event['timestamp'] - start_time,
                'event_type': event['event_type'],
                'person_id': event['person_id'],
                'confidence': event['confidence'],
                'bbox': {
                    'x': event['bbox_x'],
                    'y': event['bbox_y'],
                    'width': event['bbox_w'],
                    'height': event['bbox_h']
                } if event['bbox_x'] is not None else None,
                'additional_data': json.loads(event['additional_data']) if event['additional_data'] else None
            }
            timeline_data['events'].append(formatted_event)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(timeline_data, f, indent=2)
        
        logger.info(f"Event timeline exported to: {output_path}")
        return timeline_data

# Integration helper for engine
class EngineOverlayIntegration:
    """Helper class to integrate hotspot overlay with the main engine"""
    
    def __init__(self, overlay: HotspotOverlay):
        self.overlay = overlay
        self.last_events = {}  # Track last event per person to avoid duplicates
    
    def process_engine_events(self, events: List[Dict], timestamp: float, frame: np.ndarray):
        """Process events from the main detection engine"""
        self.overlay.update_timestamp(timestamp)
        
        for event in events:
            event_key = f"{event['person_id']}_{event['event_type']}"
            
            # Avoid duplicate events (debouncing)
            if (event_key in self.last_events and 
                timestamp - self.last_events[event_key] < 1.0):  # 1 second debounce
                continue
            
            self.last_events[event_key] = timestamp
            
            # Add event to overlay system
            self.overlay.add_event(
                event_type=event['event_type'],
                person_id=event['person_id'],
                bbox=event.get('bbox', (0, 0, 100, 100)),
                confidence=event.get('confidence', 0.0),
                additional_data=event.get('additional_data')
            )
        
        # Process frame with overlays
        return self.overlay.process_frame(frame, timestamp)

if __name__ == "__main__":
    # Example usage
    overlay = HotspotOverlay()
    
    # Simulate some events
    overlay.update_timestamp(10.0)
    overlay.add_event("suspicious_looking", "person_001", (100, 100, 50, 100), 0.85)
    
    overlay.update_timestamp(12.0)
    overlay.add_event("suspicious_gesture", "person_001", (100, 100, 50, 100), 0.75)
    
    overlay.update_timestamp(15.0)
    overlay.add_event("temporal_cheating", "person_001", (100, 100, 50, 100), 0.95)
    
    # Export timeline
    overlay.export_event_timeline("data/session_timeline.json")
    
    print("Hotspot overlay system demo completed!")
