import sqlite3

# Check cheatgpt.db database
print('=== cheatgpt.db ===')
conn = sqlite3.connect('cheatgpt.db')
cursor = conn.cursor()

# Check tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('Tables:', [t[0] for t in tables])

# Check sessions table structure
cursor.execute("PRAGMA table_info(sessions)")
columns = cursor.fetchall()
print('\nSessions table columns:')
for col in columns:
    print(f'  {col[1]} ({col[2]})')

# Check events table structure  
cursor.execute("PRAGMA table_info(events)")
columns = cursor.fetchall()
print('\nEvents table columns:')
for col in columns:
    print(f'  {col[1]} ({col[2]})')

# Check some sessions data
cursor.execute('SELECT session_id, start_timestamp, end_timestamp FROM sessions LIMIT 5')
sessions = cursor.fetchall()
print('\nSessions:')
for session_id, start_ts, end_ts in sessions:
    duration = (end_ts - start_ts) if end_ts and start_ts else None
    print(f'  {session_id}: start={start_ts}, end={end_ts}, duration={duration}')

conn.close()
