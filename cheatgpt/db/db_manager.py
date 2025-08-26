"""Simple DB manager using sqlite3 for the scaffold."""
import sqlite3
import os

DB_PATH = os.getenv("DATABASE_URL", "cheatgpt.db")

class DBManager:
    def __init__(self, path=DB_PATH):
        # support sqlite file path or sqlite URI
        self.path = path
        self.conn = sqlite3.connect(self._sqlite_path())

    def _sqlite_path(self):
        # naive conversion for example: sqlite:///path -> path
        if self.path.startswith("sqlite:///"):
            return self.path.replace("sqlite:///", "")
        return self.path

    def close(self):
        self.conn.close()

    def __repr__(self):
        return f"DBManager(path={self.path})"
