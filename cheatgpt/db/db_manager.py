"""Simple DB manager using sqlite3 for the scaffold."""
import sqlite3
import os
import time
import logging
import threading
from typing import Optional

DB_PATH = os.getenv("DATABASE_URL", "cheatgpt.db")
logger = logging.getLogger(__name__)

class DBManager:
    def __init__(self, path=DB_PATH):
        # support sqlite file path or sqlite URI
        self.path = path
        # Configure SQLite for thread safety
        self.conn = sqlite3.connect(self._sqlite_path(), check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL')  # Enable WAL mode for better concurrency
        self._lock = threading.Lock()  # Thread safety lock
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
            
            # Enhanced sessions table for both recorded and uploaded videos
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
                    session_type TEXT DEFAULT 'recorded',  -- 'recorded' or 'uploaded'
                    original_filename TEXT,
                    processed_video_path TEXT,
                    total_events INTEGER DEFAULT 0,
                    event_summary TEXT,  -- JSON string of event summary
                    video_metadata TEXT,  -- JSON string of video metadata
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add new columns to existing sessions table if they don't exist
            cursor.execute("PRAGMA table_info(sessions)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'session_type' not in columns:
                cursor.execute('ALTER TABLE sessions ADD COLUMN session_type TEXT DEFAULT "recorded"')
            if 'original_filename' not in columns:
                cursor.execute('ALTER TABLE sessions ADD COLUMN original_filename TEXT')
            if 'processed_video_path' not in columns:
                cursor.execute('ALTER TABLE sessions ADD COLUMN processed_video_path TEXT')
            if 'total_events' not in columns:
                cursor.execute('ALTER TABLE sessions ADD COLUMN total_events INTEGER DEFAULT 0')
            if 'event_summary' not in columns:
                cursor.execute('ALTER TABLE sessions ADD COLUMN event_summary TEXT')
            if 'video_metadata' not in columns:
                cursor.execute('ALTER TABLE sessions ADD COLUMN video_metadata TEXT')
            if 'updated_at' not in columns:
                cursor.execute('ALTER TABLE sessions ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP')
            
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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

    def create_uploaded_video_session(self, session_id: str, original_filename: str, 
                                     video_path: str, video_metadata: dict = None) -> bool:
        """Create a new session record for uploaded video."""
        with self._lock:
            try:
                cursor = self.conn.cursor()
                import json
                metadata_json = json.dumps(video_metadata) if video_metadata else None
                
                cursor.execute('''
                    INSERT INTO sessions (session_id, cam_id, start_timestamp, status, 
                                        session_type, original_filename, video_path, video_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, 'upload', time.time(), 'processing', 'uploaded', 
                      original_filename, video_path, metadata_json))
                self.conn.commit()
                logger.info(f"Uploaded video session created: {session_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to create uploaded video session: {e}")
                return False

    def update_processed_video_results(self, session_id: str, result: dict) -> bool:
        """Update session with processing results."""
        with self._lock:
            try:
                cursor = self.conn.cursor()
                import json
                
                # Extract data from result
                processed_video_path = result.get('output_paths', {}).get('processed_video')
                total_events = result.get('total_events', 0)
                event_summary = json.dumps(result.get('event_summary', {}))
                video_metadata = json.dumps(result.get('video_metadata', {}))
                end_timestamp = time.time()
                
                cursor.execute('''
                    UPDATE sessions 
                    SET processed_video_path = ?, total_events = ?, event_summary = ?, 
                        video_metadata = ?, end_timestamp = ?, status = 'completed', 
                        updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                ''', (processed_video_path, total_events, event_summary, 
                      video_metadata, end_timestamp, session_id))
                self.conn.commit()
                logger.info(f"Processing results updated for session {session_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to update processing results: {e}")
                return False

    def store_uploaded_video_events(self, session_id: str, events: list) -> bool:
        """Store events from uploaded video processing."""
        with self._lock:
            try:
                cursor = self.conn.cursor()
                for event in events:
                    # Extract event data
                    timestamp = event.get('timestamp', 0)
                    track_id = event.get('person_id', 'unknown')
                    event_type = event.get('event_type', 'unknown')
                    confidence = event.get('confidence', 0.0)
                    bbox = str(event.get('bbox', []))
                    
                    cursor.execute('''
                        INSERT INTO events (timestamp, cam_id, track_id, event_type, confidence, bbox)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (timestamp, 'upload', track_id, event_type, confidence, bbox))
                
                self.conn.commit()
                logger.info(f"Stored {len(events)} events for uploaded video session {session_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to store uploaded video events: {e}")
                return False

    def store_uploaded_video_hotspots(self, session_id: str, hotspots: list) -> bool:
        """Store hotspots from uploaded video processing."""
        with self._lock:
            try:
                cursor = self.conn.cursor()
                for hotspot in hotspots:
                    cursor.execute('''
                        INSERT INTO hotspots (session_id, x, y, width, height, event_count, 
                                            severity_level, first_detection_time, last_detection_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (session_id, hotspot['x'], hotspot['y'], hotspot['width'], 
                          hotspot['height'], hotspot['event_count'], hotspot['severity_level'],
                          hotspot['first_detection_time'], hotspot['last_detection_time']))
                
                self.conn.commit()
                logger.info(f"Stored {len(hotspots)} hotspots for uploaded video session {session_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to store uploaded video hotspots: {e}")
                return False

    def get_session_info(self, session_id: str):
        """Get comprehensive session information including uploaded videos."""
        try:
            cursor = self.conn.cursor()
            cursor.row_factory = sqlite3.Row  # This makes results accessible by column name
            cursor.execute('''
                SELECT session_id, video_path, processed_video_path, start_timestamp, 
                       end_timestamp, cam_id, status, frame_count, session_type, 
                       original_filename, total_events, event_summary, video_metadata, created_at
                FROM sessions 
                WHERE session_id = ?
            ''', (session_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)  # Convert Row object to dictionary
            return None
        except Exception as e:
            logger.error(f"Failed to get session info: {e}")
            return None

    def get_all_sessions(self, limit: int = 50):
        """Get all sessions including both recorded and uploaded videos."""
        try:
            cursor = self.conn.cursor()
            cursor.row_factory = sqlite3.Row  # This makes results accessible by column name
            cursor.execute('''
                SELECT session_id, video_path, processed_video_path, start_timestamp, 
                       end_timestamp, cam_id, status, frame_count, session_type, 
                       original_filename, total_events, event_summary, video_metadata, created_at
                FROM sessions 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]  # Convert Row objects to dictionaries
        except Exception as e:
            logger.error(f"Failed to get sessions: {e}")
            return []

    def get_session_events(self, session_id: str):
        """Get events for a specific session."""
        try:
            cursor = self.conn.cursor()
            cursor.row_factory = sqlite3.Row  # This makes results accessible by column name
            
            # For uploaded videos, we need to match by session timeline
            session_info = self.get_session_info(session_id)
            if not session_info:
                return []
            
            if session_info.get('session_type') == 'uploaded':
                # For uploaded videos, get events by cam_id = 'upload'
                cursor.execute('''
                    SELECT * FROM events 
                    WHERE cam_id = 'upload' 
                    ORDER BY timestamp ASC
                ''')
                rows = cursor.fetchall()
                return [dict(row) for row in rows]  # Convert Row objects to dictionaries
            else:
                # For recorded videos, use the existing logic
                start_timestamp = session_info.get('start_timestamp')
                end_timestamp = session_info.get('end_timestamp') or (start_timestamp + 3600)
                
                cursor.execute('''
                    SELECT * FROM events 
                    WHERE timestamp BETWEEN ? AND ? 
                    ORDER BY timestamp ASC
                ''', (start_timestamp, end_timestamp))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]  # Convert Row objects to dictionaries
        except Exception as e:
            logger.error(f"Failed to get session events: {e}")
            return []

    def get_session_hotspots(self, session_id: str):
        """Get hotspots for a specific session."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM hotspots 
                WHERE session_id = ?
                ORDER BY event_count DESC
            ''', (session_id,))
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get session hotspots: {e}")
            return []

    def close(self):
        self.conn.close()

    def get_uploaded_video_sessions(self, limit: int = 50):
        """Get only uploaded video sessions."""
        try:
            cursor = self.conn.cursor()
            cursor.row_factory = sqlite3.Row
            cursor.execute('''
                SELECT session_id, video_path, processed_video_path, start_timestamp, 
                       end_timestamp, cam_id, status, frame_count, session_type, 
                       original_filename, total_events, event_summary, video_metadata, created_at
                FROM sessions 
                WHERE session_type = 'uploaded'
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get uploaded video sessions: {e}")
            return []

    def __repr__(self):
        return f"DBManager(path={self.path})"
