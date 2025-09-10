"""Simple DB manager using sqlite3 for the scaffold."""
import sqlite3
import os
import logging
from typing import Optional

DB_PATH = os.getenv("DATABASE_URL", "cheatgpt.db")
logger = logging.getLogger(__name__)

class DBManager:
    def __init__(self, path=DB_PATH):
        # support sqlite file path or sqlite URI
        self.path = path
        self.conn = sqlite3.connect(self._sqlite_path())
        self._create_tables()

    def _sqlite_path(self):
        # naive conversion for example: sqlite:///path -> path
        if self.path.startswith("sqlite:///"):
            return self.path.replace("sqlite:///", "")
        return self.path

    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        try:
            cursor = self.conn.cursor()
            
            # Existing events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    cam_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    confidence REAL,
                    evidence_path TEXT,
                    bbox TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # New sessions table for video recording
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    video_path TEXT,
                    start_timestamp REAL NOT NULL,
                    end_timestamp REAL,
                    cam_id TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    frame_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # New hotspots table (for spatial analysis)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hotspots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    width REAL NOT NULL,
                    height REAL NOT NULL,
                    event_count INTEGER DEFAULT 1,
                    severity_level TEXT,
                    first_detection_time REAL,
                    last_detection_time REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            self.conn.commit()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")

    def store_event(self, timestamp: float, cam_id: str, track_id: str, 
                   event_type: str, confidence: float, evidence_path: Optional[str] = None, 
                   bbox: Optional[list] = None):
        """Store an event in the database."""
        try:
            cursor = self.conn.cursor()
            bbox_str = str(bbox) if bbox else None
            cursor.execute('''
                INSERT INTO events (timestamp, cam_id, track_id, event_type, confidence, evidence_path, bbox)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, cam_id, track_id, event_type, confidence, evidence_path, bbox_str))
            self.conn.commit()
            logger.debug(f"Event stored: {event_type} for {track_id}")
        except Exception as e:
            logger.error(f"Failed to store event: {e}")

    def get_events(self, limit: int = 100):
        """Get recent events from the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM events ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return []

    def create_session(self, session_id: str, cam_id: str, start_timestamp: float) -> bool:
        """Create a new session record."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO sessions (session_id, cam_id, start_timestamp, status)
                VALUES (?, ?, ?, 'active')
            ''', (session_id, cam_id, start_timestamp))
            self.conn.commit()
            logger.info(f"Session created: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False

    def update_session_video_path(self, session_id: str, video_path: str) -> bool:
        """Update session with video path."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE sessions 
                SET video_path = ?, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            ''', (video_path, session_id))
            self.conn.commit()
            logger.info(f"Video path updated for session {session_id}: {video_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to update video path: {e}")
            return False

    def end_session(self, session_id: str, end_timestamp: float, frame_count: int = 0) -> bool:
        """End a session and record final metadata."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE sessions 
                SET end_timestamp = ?, status = 'completed', frame_count = ?, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            ''', (end_timestamp, frame_count, session_id))
            self.conn.commit()
            logger.info(f"Session ended: {session_id} ({frame_count} frames)")
            return True
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            return False

    def store_hotspot(self, session_id: str, x: float, y: float, width: float, height: float, 
                     event_count: int = 1, severity_level: str = "yellow", 
                     detection_time: Optional[float] = None):
        """Store or update a hotspot location."""
        try:
            if detection_time is None:
                import time
                detection_time = time.time()
            
            cursor = self.conn.cursor()
            
            # Check if hotspot already exists in this area (within 50 pixels)
            cursor.execute('''
                SELECT id, event_count FROM hotspots 
                WHERE session_id = ? AND 
                      ABS(x - ?) < 50 AND ABS(y - ?) < 50
                ORDER BY 
                    (ABS(x - ?) + ABS(y - ?)) ASC
                LIMIT 1
            ''', (session_id, x, y, x, y))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing hotspot
                hotspot_id, current_count = existing
                new_count = current_count + event_count
                cursor.execute('''
                    UPDATE hotspots 
                    SET event_count = ?, severity_level = ?, last_detection_time = ?
                    WHERE id = ?
                ''', (new_count, severity_level, detection_time, hotspot_id))
                logger.debug(f"Updated hotspot {hotspot_id} (count: {new_count})")
            else:
                # Create new hotspot
                cursor.execute('''
                    INSERT INTO hotspots 
                    (session_id, x, y, width, height, event_count, severity_level, 
                     first_detection_time, last_detection_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, x, y, width, height, event_count, severity_level, 
                      detection_time, detection_time))
                logger.debug(f"Created new hotspot at ({x:.0f}, {y:.0f})")
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to store hotspot: {e}")

    def get_session_hotspots(self, session_id: str):
        """Get all hotspots for a session."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT x, y, width, height, event_count, severity_level, 
                       first_detection_time, last_detection_time
                FROM hotspots 
                WHERE session_id = ?
                ORDER BY event_count DESC
            ''', (session_id,))
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get session hotspots: {e}")
            return []

    def get_session_info(self, session_id: str):
        """Get session information."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT session_id, video_path, start_timestamp, end_timestamp, 
                       cam_id, status, frame_count, created_at
                FROM sessions 
                WHERE session_id = ?
            ''', (session_id,))
            return cursor.fetchone()
        except Exception as e:
            logger.error(f"Failed to get session info: {e}")
            return None

    def close(self):
        self.conn.close()

    def __repr__(self):
        return f"DBManager(path={self.path})"
