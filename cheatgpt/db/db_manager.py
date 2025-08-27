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
            self.conn.commit()
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

    def close(self):
        self.conn.close()

    def __repr__(self):
        return f"DBManager(path={self.path})"
