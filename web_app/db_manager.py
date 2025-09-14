"""
Database Manager for CheatGPT Web App
Handles session and hotspot data storage
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "cheatgpt_sessions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    video_path TEXT,
                    start_ts REAL NOT NULL,
                    end_ts REAL,
                    duration REAL,
                    status TEXT DEFAULT 'recording',
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hotspots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp_offset REAL NOT NULL,
                    frame_no INTEGER,
                    bbox_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_session_id 
                ON sessions (session_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hotspots_session_id 
                ON hotspots (session_id)
            """)
            
            conn.commit()
    
    def create_session(self, session_id: str, start_ts: float, metadata: Dict = None) -> int:
        """Create a new recording session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO sessions (session_id, start_ts, metadata, status)
                    VALUES (?, ?, ?, 'recording')
                """, (session_id, start_ts, json.dumps(metadata or {})))
                
                session_pk = cursor.lastrowid
                conn.commit()
                logger.info(f"Created session {session_id} with ID {session_pk}")
                return session_pk
                
        except sqlite3.IntegrityError:
            logger.warning(f"Session {session_id} already exists")
            return self.get_session_pk(session_id)
        except Exception as e:
            logger.error(f"Error creating session {session_id}: {e}")
            raise
    
    def end_session(self, session_id: str, end_ts: float, video_path: str = None) -> bool:
        """End a recording session"""
        try:
            duration = None
            with sqlite3.connect(self.db_path) as conn:
                # Get start time to calculate duration
                cursor = conn.execute(
                    "SELECT start_ts FROM sessions WHERE session_id = ?", 
                    (session_id,)
                )
                result = cursor.fetchone()
                if result:
                    start_ts = result[0]
                    duration = end_ts - start_ts
                
                # Update session
                conn.execute("""
                    UPDATE sessions 
                    SET end_ts = ?, video_path = ?, duration = ?, status = 'completed'
                    WHERE session_id = ?
                """, (end_ts, video_path, duration, session_id))
                
                rows_affected = conn.total_changes
                conn.commit()
                
                if rows_affected > 0:
                    logger.info(f"Ended session {session_id}, duration: {duration:.2f}s")
                    return True
                else:
                    logger.warning(f"No session found with ID {session_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error ending session {session_id}: {e}")
            return False
    
    def add_hotspot(self, session_id: str, event_type: str, confidence: float, 
                   timestamp_offset: float, frame_no: int = None, bbox_data: Dict = None) -> int:
        """Add a hotspot event to a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO hotspots 
                    (session_id, event_type, confidence, timestamp_offset, frame_no, bbox_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (session_id, event_type, confidence, timestamp_offset, frame_no, 
                     json.dumps(bbox_data) if bbox_data else None))
                
                hotspot_id = cursor.lastrowid
                conn.commit()
                logger.debug(f"Added hotspot {hotspot_id} for session {session_id}: {event_type}")
                return hotspot_id
                
        except Exception as e:
            logger.error(f"Error adding hotspot to session {session_id}: {e}")
            return -1
    
    def get_sessions(self, limit: int = 50, status: str = None) -> List[Dict]:
        """Get all sessions with optional filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = """
                    SELECT s.*, 
                           COUNT(h.id) as hotspot_count,
                           MAX(h.confidence) as max_confidence
                    FROM sessions s
                    LEFT JOIN hotspots h ON s.session_id = h.session_id
                """
                params = []
                
                if status:
                    query += " WHERE s.status = ?"
                    params.append(status)
                
                query += """
                    GROUP BY s.id
                    ORDER BY s.created_at DESC
                    LIMIT ?
                """
                params.append(limit)
                
                cursor = conn.execute(query, params)
                sessions = []
                
                for row in cursor.fetchall():
                    session_data = dict(row)
                    if session_data['metadata']:
                        session_data['metadata'] = json.loads(session_data['metadata'])
                    sessions.append(session_data)
                
                return sessions
                
        except Exception as e:
            logger.error(f"Error getting sessions: {e}")
            return []
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a specific session by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT s.*, COUNT(h.id) as hotspot_count
                    FROM sessions s
                    LEFT JOIN hotspots h ON s.session_id = h.session_id
                    WHERE s.session_id = ?
                    GROUP BY s.id
                """, (session_id,))
                
                row = cursor.fetchone()
                if row:
                    session_data = dict(row)
                    if session_data['metadata']:
                        session_data['metadata'] = json.loads(session_data['metadata'])
                    return session_data
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None
    
    def get_hotspots(self, session_id: str) -> List[Dict]:
        """Get all hotspots for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT * FROM hotspots 
                    WHERE session_id = ?
                    ORDER BY timestamp_offset ASC
                """, (session_id,))
                
                hotspots = []
                for row in cursor.fetchall():
                    hotspot_data = dict(row)
                    if hotspot_data['bbox_data']:
                        hotspot_data['bbox_data'] = json.loads(hotspot_data['bbox_data'])
                    hotspots.append(hotspot_data)
                
                return hotspots
                
        except Exception as e:
            logger.error(f"Error getting hotspots for session {session_id}: {e}")
            return []
    
    def get_session_pk(self, session_id: str) -> Optional[int]:
        """Get the primary key for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT id FROM sessions WHERE session_id = ?", 
                    (session_id,)
                )
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Error getting session PK for {session_id}: {e}")
            return None
    
    def get_analytics_data(self) -> Dict:
        """Get analytics data for dashboard"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Total sessions and hotspots
                cursor = conn.execute("SELECT COUNT(*) as total FROM sessions")
                total_sessions = cursor.fetchone()['total']
                
                cursor = conn.execute("SELECT COUNT(*) as total FROM hotspots")
                total_hotspots = cursor.fetchone()['total']
                
                # Hotspots by event type
                cursor = conn.execute("""
                    SELECT event_type, COUNT(*) as count, AVG(confidence) as avg_confidence
                    FROM hotspots
                    GROUP BY event_type
                    ORDER BY count DESC
                """)
                event_types = [dict(row) for row in cursor.fetchall()]
                
                # Recent activity (last 7 days)
                cursor = conn.execute("""
                    SELECT DATE(created_at) as date, COUNT(*) as sessions
                    FROM sessions
                    WHERE created_at >= date('now', '-7 days')
                    GROUP BY DATE(created_at)
                    ORDER BY date DESC
                """)
                recent_activity = [dict(row) for row in cursor.fetchall()]
                
                # Average session duration
                cursor = conn.execute("""
                    SELECT AVG(duration) as avg_duration, 
                           MIN(duration) as min_duration,
                           MAX(duration) as max_duration
                    FROM sessions 
                    WHERE duration IS NOT NULL
                """)
                duration_stats = dict(cursor.fetchone() or {})
                
                return {
                    'total_sessions': total_sessions,
                    'total_hotspots': total_hotspots,
                    'event_types': event_types,
                    'recent_activity': recent_activity,
                    'duration_stats': duration_stats
                }
                
        except Exception as e:
            logger.error(f"Error getting analytics data: {e}")
            return {}

    def get_sessions_with_details(self, limit: int = 50) -> List[Dict]:
        """Get sessions with detailed information for analytics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        session_id, 
                        video_path,
                        start_ts,
                        end_ts,
                        duration,
                        status,
                        metadata,
                        created_at,
                        (SELECT COUNT(*) FROM hotspots WHERE hotspots.session_id = sessions.session_id) as hotspot_count
                    FROM sessions 
                    ORDER BY start_ts DESC 
                    LIMIT ?
                """, (limit,))
                
                sessions = []
                for row in cursor.fetchall():
                    metadata = json.loads(row[6]) if row[6] else {}
                    session = {
                        'session_id': row[0],
                        'video_file': row[1],
                        'start_time': row[2],
                        'end_time': row[3],
                        'duration': row[4] if row[4] else 0,
                        'status': row[5],
                        'metadata': metadata,
                        'created_at': row[7],
                        'hotspot_count': row[8],
                        'frame_count': metadata.get('frame_count', 0)
                    }
                    sessions.append(session)
                
                return sessions
                
        except Exception as e:
            logger.error(f"Error getting sessions with details: {e}")
            return []

    def get_session_events(self, session_id: str) -> List[Dict]:
        """Get all events/hotspots for a specific session formatted for analytics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        id,
                        event_type,
                        confidence,
                        timestamp_offset,
                        frame_no,
                        bbox_data,
                        created_at
                    FROM hotspots 
                    WHERE session_id = ? 
                    ORDER BY timestamp_offset ASC
                """, (session_id,))
                
                events = []
                for row in cursor.fetchall():
                    # Determine severity based on confidence
                    confidence = row[2]
                    if confidence >= 0.9:
                        severity = 'red'
                    elif confidence >= 0.7:
                        severity = 'orange'
                    else:
                        severity = 'yellow'
                    
                    event = {
                        'id': row[0],
                        'event_type': row[1],
                        'confidence': row[2],
                        'timestamp_seconds': row[3],
                        'frame_no': row[4],
                        'bbox_data': row[5],
                        'severity': severity,
                        'description': row[1],  # Use the formatted event_type directly
                        'created_at': row[6]
                    }
                    events.append(event)
                
                return events
                
        except Exception as e:
            logger.error(f"Error getting session events for {session_id}: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its associated hotspots"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First delete associated hotspots
                cursor = conn.execute("""
                    DELETE FROM hotspots WHERE session_id = ?
                """, (session_id,))
                hotspots_deleted = cursor.rowcount
                
                # Then delete the session
                cursor = conn.execute("""
                    DELETE FROM sessions WHERE session_id = ?
                """, (session_id,))
                sessions_deleted = cursor.rowcount
                
                conn.commit()
                
                if sessions_deleted > 0:
                    logger.info(f"Deleted session {session_id} and {hotspots_deleted} associated hotspots")
                    return True
                else:
                    logger.warning(f"No session found with ID {session_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up sessions older than specified days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM hotspots 
                    WHERE session_id IN (
                        SELECT session_id FROM sessions 
                        WHERE created_at < date('now', '-{} days')
                    )
                """.format(days_old))
                
                hotspots_deleted = cursor.rowcount
                
                cursor = conn.execute("""
                    DELETE FROM sessions 
                    WHERE created_at < date('now', '-{} days')
                """.format(days_old))
                
                sessions_deleted = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {sessions_deleted} sessions and {hotspots_deleted} hotspots")
                return sessions_deleted
                
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            return 0

    def get_session_details(self, session_id: int) -> Optional[Dict]:
        """Get detailed session information by ID (integer primary key)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        s.id,
                        s.session_id,
                        s.video_path,
                        s.start_ts,
                        s.end_ts,
                        s.duration,
                        s.status,
                        s.metadata,
                        s.created_at,
                        COUNT(h.id) as total_hotspots
                    FROM sessions s
                    LEFT JOIN hotspots h ON s.session_id = h.session_id
                    WHERE s.id = ?
                    GROUP BY s.id
                """, (session_id,))
                
                row = cursor.fetchone()
                if row:
                    metadata = {}
                    try:
                        if row[7]:
                            metadata = json.loads(row[7])
                    except:
                        pass
                    
                    return {
                        'id': row[0],
                        'session_id': row[1],
                        'video_path': row[2],
                        'video_title': metadata.get('video_title', f'Session {row[1]}'),
                        'started_at': row[3],
                        'ended_at': row[4],
                        'duration': row[5] or 0,
                        'status': row[6],
                        'total_hotspots': row[9],
                        'created_at': row[8]
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error getting session details for ID {session_id}: {e}")
            return None
    
    def get_session_analytics_by_session_id(self, session_id: str) -> Dict:
        """Get analytics data for a specific session using session_id string"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get session info
                cursor = conn.execute("""
                    SELECT id, session_id, created_at, status, duration, 
                           (SELECT COUNT(*) FROM hotspots WHERE session_id = s.session_id) as total_hotspots
                    FROM sessions s 
                    WHERE session_id = ?
                """, (session_id,))
                
                session_row = cursor.fetchone()
                if not session_row:
                    return {}
                
                session = {
                    'id': session_row[0],
                    'session_id': session_row[1],
                    'created_at': session_row[2],
                    'status': session_row[3],
                    'duration': session_row[4],
                    'total_hotspots': session_row[5]
                }
                
                # Get event type distribution
                cursor = conn.execute("""
                    SELECT event_type, COUNT(*) as count
                    FROM hotspots 
                    WHERE session_id = ?
                    GROUP BY event_type
                    ORDER BY count DESC
                """, (session_id,))
                
                event_types = {}
                for row in cursor.fetchall():
                    event_types[row[0]] = row[1]
                
                # Get confidence distribution
                cursor = conn.execute("""
                    SELECT 
                        CASE 
                            WHEN confidence >= 0.9 THEN 'high'
                            WHEN confidence >= 0.7 THEN 'medium'
                            ELSE 'low'
                        END as confidence_level,
                        COUNT(*) as count
                    FROM hotspots 
                    WHERE session_id = ?
                    GROUP BY confidence_level
                """, (session_id,))
                
                confidence_dist = {'high': 0, 'medium': 0, 'low': 0}
                for row in cursor.fetchall():
                    confidence_dist[row[0]] = row[1]
                
                # Get timeline data (events over time)
                cursor = conn.execute("""
                    SELECT 
                        CAST(timestamp_offset / 60 AS INTEGER) as minute,
                        COUNT(*) as count
                    FROM hotspots 
                    WHERE session_id = ?
                    GROUP BY minute
                    ORDER BY minute
                """, (session_id,))
                
                timeline = []
                for row in cursor.fetchall():
                    timeline.append({
                        'minute': row[0],
                        'events': row[1]
                    })
                
                return {
                    'summary': {
                        'total_events': session['total_hotspots'],
                        'duration_minutes': round((session['duration'] or 0) / 60, 1),
                        'events_per_minute': round(session['total_hotspots'] / max((session['duration'] or 0) / 60, 1), 2),
                        'session_date': session['created_at']
                    },
                    'event_types': event_types,
                    'confidence_distribution': confidence_dist,
                    'timeline': timeline,
                    'session_info': session
                }
                
        except Exception as e:
            logger.error(f"Error getting session analytics for session_id {session_id}: {e}")
            return {}

    def get_session_analytics(self, session_id: int) -> Dict:
        """Get analytics data for a specific session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get session info
                session = self.get_session_details(session_id)
                if not session:
                    return {}
                
                session_id_str = session['session_id']
                
                # Get event type distribution
                cursor = conn.execute("""
                    SELECT event_type, COUNT(*) as count
                    FROM hotspots 
                    WHERE session_id = ?
                    GROUP BY event_type
                    ORDER BY count DESC
                """, (session_id_str,))
                
                event_types = {}
                for row in cursor.fetchall():
                    event_types[row[0]] = row[1]
                
                # Get confidence distribution
                cursor = conn.execute("""
                    SELECT 
                        CASE 
                            WHEN confidence >= 0.9 THEN 'high'
                            WHEN confidence >= 0.7 THEN 'medium'
                            ELSE 'low'
                        END as confidence_level,
                        COUNT(*) as count
                    FROM hotspots 
                    WHERE session_id = ?
                    GROUP BY confidence_level
                """, (session_id_str,))
                
                confidence_dist = {'high': 0, 'medium': 0, 'low': 0}
                for row in cursor.fetchall():
                    confidence_dist[row[0]] = row[1]
                
                # Get timeline data (events over time)
                cursor = conn.execute("""
                    SELECT 
                        CAST(timestamp_offset / 60 AS INTEGER) as minute,
                        COUNT(*) as count
                    FROM hotspots 
                    WHERE session_id = ?
                    GROUP BY minute
                    ORDER BY minute
                """, (session_id_str,))
                
                timeline = []
                for row in cursor.fetchall():
                    timeline.append({
                        'minute': row[0],
                        'events': row[1]
                    })
                
                return {
                    'summary': {
                        'total_events': session['total_hotspots'],
                        'duration_minutes': round((session['duration'] or 0) / 60, 1),
                        'events_per_minute': round(session['total_hotspots'] / max((session['duration'] or 0) / 60, 1), 2),
                        'session_date': session['created_at']
                    },
                    'event_types': event_types,
                    'confidence_distribution': confidence_dist,
                    'timeline': timeline,
                    'session_info': session
                }
                
        except Exception as e:
            logger.error(f"Error getting session analytics for ID {session_id}: {e}")
            return {}


# Global database instance
db = DatabaseManager()

if __name__ == "__main__":
    # Test the database
    import time
    
    print("Testing DatabaseManager...")
    
    # Create test session
    session_id = f"test_{int(time.time())}"
    start_ts = time.time()
    
    db.create_session(session_id, start_ts, {"test": True})
    
    # Add test hotspots
    db.add_hotspot(session_id, "suspicious_gesture", 0.85, 5.2, 156)
    db.add_hotspot(session_id, "phone_detected", 0.92, 12.1, 363)
    
    # End session
    end_ts = time.time()
    db.end_session(session_id, end_ts, f"/videos/{session_id}.mp4")
    
    # Test queries
    sessions = db.get_sessions()
    print(f"Total sessions: {len(sessions)}")
    
    if sessions:
        session = sessions[0]
        print(f"Latest session: {session['session_id']}")
        
        hotspots = db.get_hotspots(session['session_id'])
        print(f"Hotspots in session: {len(hotspots)}")
    
    analytics = db.get_analytics_data()
    print(f"Analytics: {analytics}")
    
    print("Database test completed!")
