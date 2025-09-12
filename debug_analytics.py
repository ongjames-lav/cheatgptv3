import sqlite3
import os

# Check both databases to understand the actual data
db_path = "cheatgpt.db"
sessions_db_path = "cheatgpt_sessions.db"

print("=== DEBUGGING ANALYTICS DATA ===")

# Check main database
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n1. SESSIONS TABLE STRUCTURE:")
    cursor.execute("PRAGMA table_info(sessions)")
    columns = cursor.fetchall()
    for col in columns:
        print(f"   {col}")
    
    print("\n2. EVENTS TABLE STRUCTURE:")
    cursor.execute("PRAGMA table_info(events)")
    columns = cursor.fetchall()
    for col in columns:
        print(f"   {col}")
    
    print("\n3. SAMPLE SESSIONS DATA:")
    cursor.execute("SELECT session_id, duration FROM sessions LIMIT 10")
    sessions = cursor.fetchall()
    for session in sessions:
        print(f"   Session {session[0]}: Duration = {session[1]} seconds")
    
    print("\n4. EVENTS COUNT PER SESSION:")
    cursor.execute("""
        SELECT s.session_id, s.duration, COUNT(e.id) as event_count,
               (COUNT(e.id) * 1.0 / (s.duration / 60.0)) as events_per_minute
        FROM sessions s 
        LEFT JOIN events e ON s.session_id = e.session_id 
        GROUP BY s.session_id, s.duration
        LIMIT 10
    """)
    sessions_with_events = cursor.fetchall()
    for session in sessions_with_events:
        session_id, duration, event_count, events_per_min = session
        print(f"   Session {session_id}: {event_count} events in {duration}s = {events_per_min:.2f}/min")
    
    conn.close()
else:
    print(f"Main database {db_path} not found")

# Check sessions database
if os.path.exists(sessions_db_path):
    print(f"\n5. SESSIONS DATABASE ({sessions_db_path}):")
    conn = sqlite3.connect(sessions_db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"   Tables: {[table[0] for table in tables]}")
    
    for table in tables:
        table_name = table[0]
        print(f"\n   {table_name.upper()} TABLE STRUCTURE:")
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        for col in columns:
            print(f"      {col}")
            
        print(f"\n   SAMPLE {table_name.upper()} DATA:")
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        rows = cursor.fetchall()
        for row in rows:
            print(f"      {row}")
    
    conn.close()
else:
    print(f"Sessions database {sessions_db_path} not found")

print("\n=== END DEBUG ===")
