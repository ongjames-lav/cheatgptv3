import sqlite3
import json
import os

# Check if database exists - use the correct database name
db_path = 'cheatgpt_sessions.db'
if not os.path.exists(db_path):
    print(f'❌ Database not found at {db_path}')
    exit(1)

conn = sqlite3.connect(db_path)

# First, check what tables exist
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f'Available tables: {[t[0] for t in tables]}')

# Check if we have the processing_sessions table instead
if ('processing_sessions',) in tables:
    print('\nUsing processing_sessions table...')
    cursor = conn.execute(
        'SELECT session_id, status, created_at, updated_at FROM processing_sessions WHERE session_id=?', 
        ('single_1761452718',)
    )
    row = cursor.fetchone()
    if row:
        print(f'Session ID: {row[0]}')
        print(f'Status: {row[1]}')
        print(f'Created: {row[2]}')
        print(f'Updated: {row[3]}')
    else:
        print('Session not found in processing_sessions')
        
# Check sessions table
if ('sessions',) in tables:
    print('\nUsing sessions table...')
    cursor = conn.execute(
        'SELECT session_id, duration, start_ts, end_ts, metadata FROM sessions WHERE session_id=?', 
        ('single_1761452718',)
    )
    row = cursor.fetchone()
    
    if row:
        print(f'Session ID: {row[0]}')
        print(f'Duration in DB: {row[1]} seconds ({row[1]/60:.1f} minutes)')
        print(f'Start TS: {row[2]}')
        print(f'End TS: {row[3]}')
        print(f'Calculated from timestamps: {row[3] - row[2]:.1f} seconds')
        
        meta = json.loads(row[4]) if row[4] else {}
        vm = meta.get('video_metadata', {})
        print(f'Video metadata duration: {vm.get("duration", "N/A")} seconds')
        print(f'Metadata keys: {list(meta.keys())}')
        
        if vm.get('duration'):
            print(f'\n✅ Video metadata has actual duration: {vm.get("duration"):.1f} seconds ({vm.get("duration")/60:.1f} minutes)')
            print(f'❌ But DB duration field shows: {row[1]} seconds ({row[1]/60:.1f} minutes)')
            print(f'\nThe issue: duration field in DB is {row[1]} but should be {vm.get("duration")}')
    else:
        print('Session not found in sessions table')

conn.close()
